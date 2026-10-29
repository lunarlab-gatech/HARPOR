"""Run VoMP v1.1 on one GLB: per-voxel materials and rigid-body mass properties (build_plan.md note 9).

Run it in the VoMP env from the VoMP clone, because inference.json names its weights by relative path:
    cd $VOMP && python $GATOR/harpor/physics/vomp_run.py --glb aligned.glb --out RUN_DIR [--links LINKS_DIR]
Writes into RUN_DIR:
  work/          VoMP's renders, render geometry, voxels and DINO features
  materials.npz  VoMP's save_materials format (voxel_data: x, y, z, youngs_modulus, poissons_ratio,
                 density, segment_id) at the predicted voxel centres
  voxels.npz     every occupied voxel centre with the nearest prediction's E (Pa), nu and rho (kg/m^3),
                 transform_center, transform_scale and the voxel edge h = transform_scale / 64
  summary.json   counts, medians, timings, and mass, COM and MuJoCo fullinertia
Coordinates are in Blender's world frame after the stock glTF import (z up), in the GLB's units.
VoMP's prediction does not depend on size (it rescales every mesh to a unit box), but mass does.

With --links (the per-link GLBs of harpor.physics.align --links-out, build_plan.md note 11) the model is
loaded once and runs on every LINKS_DIR/<link>.glb instead of the whole GLB, each on its own 64^3 grid,
into RUN_DIR/vomp/<link>/ with the files above. RUN_DIR/summary.json then describes the whole object:
rigid_body is the sum of the links (mass.combine) and rigid_body_covers names them; median and p05_p95
are over all the links' voxels, each weighted by its volume h^3, since every link has its own grid and a
small handle gives about as many voxels as the body; links holds each link's median, p05_p95, rigid_body,
voxels_occupied, voxel_edge_m and seconds. A link that VoMP cannot voxelize (a thin lid can give no
occupied voxel) goes to links_skipped with the reason and is missing from rigid_body;
harpor.physics.mjcf gives it a hull inertial, so mjcf.json's inertial is then the whole object's mass.
Each link costs about one VoMP predict.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harpor.physics.mass import combine, mass_properties  # noqa: E402

FIELDS = (("youngs_modulus_pa", "youngs_modulus"), ("poisson_ratio", "poisson_ratio"), ("density_kg_m3", "density"))


def run_glb(model, glb, out, config, load_s=None):
    """VoMP on one GLB into out (work/, materials.npz, voxels.npz, summary.json). Returns the summary.
    Its seconds hold load only when load_s is given (the single run)."""
    import torch
    from scipy.spatial import cKDTree
    from vomp.inference.utils import denormalize_coords, save_materials

    out = Path(out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    r = model.get_mesh_materials(str(Path(glb).resolve()), output_dir=str(work), query_points="voxel_centers",
                                 return_original_scale=True)
    t2 = time.time()
    save_materials(r, str(out / "materials.npz"))

    center, scale = np.asarray(r["transform_center"], float), float(r["transform_scale"])
    occupied = denormalize_coords(np.asarray(model._voxelize_mesh(str(Path(glb).resolve()), str(work))), center, scale)
    if not len(occupied):
        raise ValueError(f"{glb} gives no occupied voxel")
    pred = np.asarray(r["voxel_coords_world"], float)
    E, nu, rho = (np.asarray(r[k], float).reshape(-1) for k in ("youngs_modulus", "poisson_ratio", "density"))
    dist, idx = cKDTree(pred).query(occupied)
    h = scale / 64
    np.savez(out / "voxels.npz", centers=occupied, youngs_modulus=E[idx], poisson_ratio=nu[idx], density=rho[idx],
             transform_center=center, transform_scale=scale, h=h)

    summary = {
        "glb": str(Path(glb).resolve()),
        "config": config,
        "voxels_occupied": len(occupied),
        "voxels_predicted": len(pred),
        "nearest_prediction_max_dist_voxels": float(dist.max() / h),
        "voxel_edge_m": h,
        "median": {"youngs_modulus_pa": float(np.median(E)), "poisson_ratio": float(np.median(nu)),
                   "density_kg_m3": float(np.median(rho))},
        "p05_p95": {k: np.percentile(v, [5, 95]).tolist() for k, v in
                    (("youngs_modulus_pa", E), ("poisson_ratio", nu), ("density_kg_m3", rho))},
        "rigid_body": mass_properties(occupied, rho[idx], h),
        "seconds": {"predict": t2 - t1} if load_s is None else {"load": load_s, "predict": t2 - t1},
        "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def weighted_percentile(values, weights, q):
    """Percentiles q (0 to 100) of values that count with these weights: for each q, the smallest value whose
    cumulative weight share reaches q / 100 (numpy's inverted_cdf method when the weights are equal)."""
    order = np.argsort(values, kind="stable")
    v, cw = np.asarray(values, float)[order], np.cumsum(np.asarray(weights, float)[order])
    i = np.searchsorted(cw / cw[-1], np.asarray(q, float) / 100 - 1e-12)
    return v[np.minimum(i, len(v) - 1)]


def combine_links(out, names, glb, config, load_s, skipped=None, gpu_peak_gb=None):
    """Write and return the whole object's summary.json from out/vomp/<name>/summary.json and voxels.npz
    of each link in names (see the module docstring)."""
    out = Path(out)
    links, sums, values, volume = {}, [], {k: [] for _, k in FIELDS}, []
    for name in names:
        d = out / "vomp" / name
        s = json.loads((d / "summary.json").read_text())
        with np.load(d / "voxels.npz") as v:
            for _, k in FIELDS:
                values[k].append(np.asarray(v[k], float).reshape(-1))
            volume.append(np.full(len(values["density"][-1]), float(np.prod(np.broadcast_to(v["h"], (3,))))))
        links[name] = {k: s[k] for k in ("median", "p05_p95", "rigid_body", "voxels_occupied", "voxel_edge_m", "seconds")}
        sums.append(s)
    values = {k: np.concatenate(v) for k, v in values.items()}
    volume = np.concatenate(volume)  # each voxel weighs its own volume, whatever its link's grid
    summary = {
        "glb": str(Path(glb).resolve()),
        "config": config,
        "voxels_occupied": sum(s["voxels_occupied"] for s in sums),
        "voxels_predicted": sum(s["voxels_predicted"] for s in sums),
        "nearest_prediction_max_dist_voxels": max(s["nearest_prediction_max_dist_voxels"] for s in sums),
        "voxel_edge_m": max(s["voxel_edge_m"] for s in sums),  # the coarsest link grid
        "median": {f: float(weighted_percentile(values[k], volume, 50)) for f, k in FIELDS},
        "p05_p95": {f: weighted_percentile(values[k], volume, [5, 95]).tolist() for f, k in FIELDS},
        "rigid_body": combine([s["rigid_body"] for s in sums]),
        "rigid_body_covers": list(names),
        "seconds": {"load": load_s, "predict": sum(s["seconds"]["predict"] for s in sums)},
        "gpu_peak_gb": gpu_peak_gb,
        "links": links,
        "links_skipped": dict(skipped or {}),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="weights/vomp_1_1/inference.json")
    ap.add_argument("--links", help="folder of per-link GLBs: run on each instead of --glb")
    a = ap.parse_args()
    glbs = sorted(Path(a.links).glob("*.glb")) if a.links else []
    if a.links and not glbs:
        ap.error(f"{a.links} holds no .glb")

    import torch
    from vomp.inference import Vomp

    out = Path(a.out)
    t0 = time.time()
    model = Vomp.from_checkpoint(config_path=a.config)
    t1 = time.time()
    if not a.links:
        summary = run_glb(model, a.glb, out, a.config, load_s=t1 - t0)
        print(json.dumps(summary, indent=1))
        return
    done, skipped = [], {}
    for g in glbs:
        try:
            s = run_glb(model, g, out / "vomp" / g.stem, a.config)
            done.append(g.stem)
            print(f"link {g.stem}: {s['voxels_occupied']} voxels, {s['rigid_body']['mass']:.3f} kg, "
                  f"{s['seconds']['predict']:.0f} s", flush=True)
        except Exception as e:  # noqa: BLE001 - the link gets a hull inertial in harpor.physics.mjcf
            skipped[g.stem] = f"{type(e).__name__}: {e}"
            print(f"link {g.stem}: skipped ({skipped[g.stem]})", flush=True)
    if not done:
        raise SystemExit(f"VoMP gave no link a result: {json.dumps(skipped)}")
    gpu = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    summary = combine_links(out, done, a.glb, a.config, t1 - t0, skipped, gpu)
    print(json.dumps({k: summary[k] for k in ("voxels_occupied", "median", "rigid_body", "links_skipped")}, indent=1))


if __name__ == "__main__":
    main()
