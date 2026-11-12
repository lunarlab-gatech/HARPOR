"""Reuse Stage A's VoMP predictions on the final geometry, without loading VoMP again.

VoMP's predictions are in the canonical GLB's Blender frame. They move into W with the placement's pose
(scale, rotation and shift), then the final alignment moves them into S. Each occupied voxel of the final
mesh takes its nearest prediction's material. Mass and inertia are recomputed from those voxels, so they
reflect the final geometry and scale. Material transfer is a spatial approximation: newly created
geometry inherits nearby materials.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from harpor import frames
from harpor.physics.align import load_w
from harpor.physics.mass import mass_properties
from harpor.physics.vomp_run import FIELDS
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
    """(summary, {"": voxel arrays in W}) of the job's VoMP run (run/vomp) on the canonical mesh."""
    J = Path(job_dir)
    source = J / "run" / "vomp"
    if not (source / "summary.json").is_file():
        raise ValueError("Material predictions are missing; resume from the vomp stage")
    original = json.loads((source / "summary.json").read_text())
    pose = json.loads((J / "run/placement.json").read_text())["pose"]
    with np.load(source / "voxels.npz") as z:
        centers = prediction_centers(z["centers"], pose, np.eye(4))
        predictions = {"": {"centers": centers, **{key: np.asarray(z[key], float) for _, key in FIELDS}}}
    return original, predictions


def transfer(glb, out, original, predictions, transform):
    """Transfer world-frame predictions to a mesh in transform's frame (see the module docstring)."""
    glb, out = Path(glb), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").unlink(missing_ok=True)
    fields = ["centers", *(key for _, key in FIELDS)]
    values = {k: np.concatenate([v[k] for v in predictions.values()]) for k in fields}
    provenance = {"method": "nearest pre-refinement voxel; final geometry revoxelized",
                  "prediction_seconds": original.get("seconds", {})}
    centers = frames.transform_points(transform, values["centers"])
    if not len(centers) or not np.isfinite(centers).all():
        raise ValueError("Material predictions contain no finite voxel centers")
    tree = cKDTree(centers)
    occupied, h = occupied_voxels(glb)
    distances, idx = tree.query(occupied)
    material = {key: values[key][idx] for _, key in FIELDS}
    np.savez(out / "voxels.npz", centers=occupied, h=h, **material)
    summary = {
        "glb": str(glb.resolve()), "config": original["config"],
        "voxels_occupied": len(occupied), "voxels_predicted": len(centers),
        "nearest_prediction_max_dist_voxels": float(distances.max() / h), "voxel_edge_m": h,
        "median": {field: float(np.median(material[key])) for field, key in FIELDS},
        "p05_p95": {field: np.percentile(material[key], [5, 95]).tolist() for field, key in FIELDS},
        "rigid_body": mass_properties(occupied, material["density"], h),
        "seconds": {"predict": 0.0}, "gpu_peak_gb": None, "material_transfer": provenance,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"Pre-refinement materials transferred: mass {summary['rigid_body']['mass']:.3f} kg")
    return summary


def transfer_job(job_dir):
    """Write final simulator properties using the predictions made before refinement."""
    J = Path(job_dir)
    original, predictions = load_predictions(J)
    out = J / "physics"
    alignment = json.loads((out / "aligned.json").read_text())
    return transfer(out / "aligned.glb", out, original, predictions, alignment["T_s_from_w"])
