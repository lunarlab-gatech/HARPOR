"""Reuse Stage A's VoMP predictions on the final geometry, without loading VoMP again.

Rigid predictions are in the canonical GLB's Blender frame. Per-part predictions
are in the early aligned frame. Both move into W, then the final alignment moves
them into S. Each occupied voxel of the final mesh (or
each link) takes its nearest prediction's material. Mass and inertia are recomputed
from those voxels, so they reflect the final geometry and scale. Material transfer
is a spatial approximation: newly created geometry inherits nearby materials.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from harpor import frames
from harpor.physics.align import load_w
from harpor.physics.mass import mass_properties
from harpor.physics.vomp_run import FIELDS, combine_links
from harpor.stage_a.place import canonical_glb_to_w


def prediction_centers(centers, pose, alignment):
    """Canonical Blender coordinates -> final simulator coordinates, including fitted scale."""
    world = canonical_glb_to_w(frames.w_to_gltf(centers), pose)
    return frames.transform_points(alignment, world)


def occupied_voxels(glb, resolution=64):
    """Filled surface voxelization in simulator axes, bounded by the longest dimension."""
    import trimesh

    vertices, faces = load_w(glb)
    h = float(np.ptp(vertices, axis=0).max()) / resolution
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"{glb} has no finite extent")
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    centers = mesh.voxelized(h).fill().points
    if not len(centers):
        raise ValueError(f"{glb} gives no occupied voxel")
    return centers, h


def load_predictions(job_dir):
    """(summary, {link: voxel arrays in W}); an empty link name denotes a rigid object."""
    J = Path(job_dir)
    articulated = json.loads((J / "job.json").read_text()).get("articulated") is True
    source = J / "run" / ("vomp_parts" if articulated else "vomp")
    if not (source / "summary.json").is_file():
        raise ValueError(f"Material predictions are missing; resume from the {source.name} stage")
    original = json.loads((source / "summary.json").read_text())
    if articulated:
        inverse = np.linalg.inv(json.loads((source / "aligned.json").read_text())["T_s_from_w"])
        paths = {name: source / "vomp" / name / "voxels.npz" for name in original["links"]}
    else:
        pose = json.loads((J / "run/placement.json").read_text())["pose"]
        paths = {"": source / "voxels.npz"}
    predictions = {}
    for name, path in paths.items():
        with np.load(path) as z:
            centers = (frames.transform_points(inverse, z["centers"]) if articulated else
                       prediction_centers(z["centers"], pose, np.eye(4)))
            predictions[name] = {"centers": centers, **{key: np.asarray(z[key], float) for _, key in FIELDS}}
    return original, predictions


def transfer(glb, out, original, predictions, transform, links=None):
    """Transfer world-frame predictions to a mesh in transform's frame, optionally per link.

    Known links use only their own predictions, preventing materials from leaking across
    touching parts. New or renamed links inherit the nearest prediction of the whole object.
    """
    glb, out = Path(glb), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").unlink(missing_ok=True)
    fields = ["centers", *(key for _, key in FIELDS)]
    all_values = {k: np.concatenate([v[k] for v in predictions.values()]) for k in fields}
    provenance = {"method": "nearest pre-refinement voxel; final geometry revoxelized",
                  "prediction_seconds": original.get("seconds", {})}

    def record(glb, dest, name=""):
        values = predictions.get(name, all_values)
        centers = frames.transform_points(transform, values["centers"])
        if not len(centers) or not np.isfinite(centers).all():
            raise ValueError("Material predictions contain no finite voxel centers")
        tree = cKDTree(centers)
        occupied, h = occupied_voxels(glb)
        distances, idx = tree.query(occupied)
        material = {key: values[key][idx] for _, key in FIELDS}
        dest.mkdir(parents=True, exist_ok=True)
        np.savez(dest / "voxels.npz", centers=occupied, h=h, **material)
        summary = {
            "glb": str(glb.resolve()), "config": original["config"],
            "voxels_occupied": len(occupied), "voxels_predicted": len(centers),
            "nearest_prediction_max_dist_voxels": float(distances.max() / h), "voxel_edge_m": h,
            "median": {field: float(np.median(material[key])) for field, key in FIELDS},
            "p05_p95": {field: np.percentile(material[key], [5, 95]).tolist() for field, key in FIELDS},
            "rigid_body": mass_properties(occupied, material["density"], h),
            "seconds": {"predict": 0.0}, "gpu_peak_gb": None, "material_transfer": provenance,
        }
        (dest / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary

    if links is not None:
        shutil.rmtree(out / "vomp", ignore_errors=True)
        if not links:
            raise ValueError("The aligned object has no link GLBs")
        done, skipped = [], {}
        for link in links:
            if link.stem in original.get("links_skipped", {}):
                skipped[link.stem] = original["links_skipped"][link.stem]
                continue
            try:
                record(link, out / "vomp" / link.stem, link.stem)
                done.append(link.stem)
            except (ValueError, RuntimeError) as e:
                skipped[link.stem] = f"{type(e).__name__}: {e}"
        if not done:
            raise ValueError(f"Material transfer gave no link a result: {skipped}")
        summary = combine_links(out, done, glb, original["config"], 0.0, skipped)
        summary["material_transfer"] = provenance
    else:
        summary = record(glb, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"Pre-refinement materials transferred: mass {summary['rigid_body']['mass']:.3f} kg")
    return summary


def transfer_job(job_dir):
    """Write final simulator properties using the predictions made before refinement."""
    J = Path(job_dir)
    original, predictions = load_predictions(J)
    out = J / "physics"
    alignment = json.loads((out / "aligned.json").read_text())
    articulated = json.loads((J / "job.json").read_text()).get("articulated") is True
    links = sorted((out / "links").glob("*.glb")) if articulated else None
    return transfer(out / "aligned.glb", out, original, predictions, alignment["T_s_from_w"], links)
