"""Run VoMP v1.1 on one GLB: per-voxel materials and rigid-body mass properties (build_plan.md note 9).

Run it in the VoMP env from the VoMP clone, because inference.json names its weights by relative path:
    cd $VOMP && python <repo>/harpor/physics/vomp_run.py --glb canonical.glb --out RUN_DIR [--opaque-input]
Writes into RUN_DIR:
  work/          VoMP's renders, render geometry, voxels and DINO features
  materials.npz  VoMP's save_materials format (voxel_data: x, y, z, youngs_modulus, poissons_ratio,
                 density, segment_id) at the predicted voxel centres
  voxels.npz     every occupied voxel centre with the nearest prediction's E (Pa), nu and rho (kg/m^3),
                 transform_center, transform_scale and the voxel edge h = transform_scale / 64
  summary.json   counts, medians, timings, and mass, COM and MuJoCo fullinertia
Coordinates are in Blender's world frame after the stock glTF import (z up), in the GLB's units.
VoMP's prediction does not depend on size (it rescales every mesh to a unit box), but mass does.

--opaque-input predicts on an opaque copy (harpor.physics.align.make_opaque), since VoMP rejects transmissive or
partly transparent materials; the source GLB stays as it is.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harpor.physics.mass import mass_properties  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="weights/vomp_1_1/inference.json")
    ap.add_argument("--opaque-input", action="store_true", help="predict on an opaque copy; keep the source GLB intact")
    a = ap.parse_args()

    from vomp.inference import Vomp

    out = Path(a.out)
    t0 = time.time()
    model = Vomp.from_checkpoint(config_path=a.config)
    t1 = time.time()
    glb = a.glb
    if a.opaque_input:
        from harpor.physics.align import make_opaque, read_glb, write_glb

        out.mkdir(parents=True, exist_ok=True)
        doc, binary = read_glb(glb)
        glb = out / "input.glb"
        write_glb(glb, make_opaque(doc), binary)
    summary = run_glb(model, glb, out, a.config, load_s=t1 - t0)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
