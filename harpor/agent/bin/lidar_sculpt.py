"""The LiDAR sculpt helper inside Blender (README.md note 5, LiDAR geometry).

The agent runs it as `lidar_sculpt --receipt R --patch L1 --op plane|push [--strength S]`, which opens
/work/candidate.blend headless with this script:

    blender -b --factory-startup --disable-autoexec /work/candidate.blend --python-exit-code 1 \
        --python lidar_sculpt.py -- --receipt R --patch L1 --op plane --strength 1

It reads reviews/R/lidar.json (the region's seed, normal and bounds in W, its fitting correspondences at the reviewed
candidate, and the fingerprints of the .blend and the export that the review saw), checks that the saved scene is
still the reviewed one, resolves the patch in the current mesh and runs one pass of sculpt_math (plane or push). It
maps the W positions back into each object's local coordinates through the inverse of its world matrix, which also
holds for a nonuniform scale, and keeps UVs, materials, topology and object transforms. It keeps the reviewed .blend
in sculpt/ for a rollback, saves the result to a temporary sibling and replaces candidate.blend only after the checks
pass. Nothing is exported: review the result as usual.
"""
# The math is tested in tests/test_lidar_geometry.py; this file needs Blender 5.2.

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sculpt_math as S  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fail(msg):
    print(f"lidar_sculpt: {msg}")
    sys.exit(1)


def scene_mesh():
    """(V in W, triangles F, [(object, its first vertex row, vertex count)]) of the visible meshes."""
    V, F, owners, off = [], [], [], 0
    for o in bpy.context.scene.objects:
        if o.type != "MESH" or not o.visible_get() or not len(o.data.vertices):
            continue
        me = o.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        M = np.array(o.matrix_world)
        me.calc_loop_triangles()
        tri = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
        V.append(co @ M[:3, :3].T + M[:3, 3])
        F.append(tri.reshape(-1, 3) + off)
        owners.append((o, off, len(co)))
        off += len(co)
    if not V:
        fail("the scene has no visible mesh")
    return np.concatenate(V), np.concatenate(F), owners


def main():
    ap = argparse.ArgumentParser(prog="lidar_sculpt")
    ap.add_argument("--receipt", required=True, help="the review receipt whose lidar.json names the patch")
    ap.add_argument("--patch", required=True, help="a region ID of that review, such as L1")
    ap.add_argument("--op", required=True, choices=("plane", "push"))
    ap.add_argument("--strength", type=float, default=1.0, help="0 to 1 of the measured correction (default 1)")
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    blend = Path(bpy.data.filepath)
    work = blend.parent
    doc_path = work / "reviews" / a.receipt / "lidar.json"
    try:
        doc = json.loads(doc_path.read_text())
    except (OSError, ValueError):
        fail(f"no LiDAR review {doc_path.relative_to(work)}; run review first (a session with LiDAR evidence)")
    if doc.get("mode") != "sculpt":
        fail(f"this session's LiDAR mode is {doc.get('mode')!r}: the LiDAR evidence is for inspection only")
    if sha256(blend) != doc.get("blend_sha256"):
        fail(f"{blend.name} changed since review {a.receipt}: run review on the current scene first, then sculpt from "
             "its receipt")
    region = next((r for r in doc.get("regions") or [] if r.get("id") == a.patch), None)
    if region is None:
        fail(f"review {a.receipt} has no region {a.patch!r}; its regions: "
             f"{', '.join(r['id'] for r in doc.get('regions') or []) or 'none'}")
    if region.get("status") != "target":
        fail(f"region {a.patch} is {region.get('status')}, not a sculpt target: the LiDAR evidence does not support a "
             "local repair there")
    V, F, owners = scene_mesh()
    try:
        out, rep = S.sculpt(V, F, region, (doc.get("correspondences") or {}).get(a.patch) or {}, a.op, a.strength,
                            doc.get("max_displacement_m", 0.05))
    except S.SculptError as e:
        fail(str(e))
    moved = np.linalg.norm(out - V, axis=1) > 0
    if not moved.any():  # the surface already fits the returns: the scene stays as it is, so its review still holds
        print(f"lidar_sculpt: {a.op} on {a.patch}: nothing to move (the patch already fits the returns); "
              f"{blend.name} unchanged")
        return
    for o, off, n in owners:
        if not moved[off:off + n].any():
            continue
        if o.data.users > 1 or o.data.shape_keys is not None or len(o.modifiers):
            fail(f"object {o.name} has a shared mesh, shape keys or modifiers; apply them first")
        Minv = np.linalg.inv(np.array(o.matrix_world))
        local = out[off:off + n] @ Minv[:3, :3].T + Minv[:3, 3]
        o.data.vertices.foreach_set("co", local.astype(np.float32).ravel())
        o.data.update()
    keep = work / "sculpt" / f"{a.receipt}-before.blend"
    keep.parent.mkdir(exist_ok=True)
    if not keep.exists():  # the reviewed scene, for a rollback
        shutil.copyfile(blend, keep)
    tmp = blend.with_name(f".{blend.stem}.sculpt-{os.getpid()}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(tmp), copy=True)
    os.replace(tmp, blend)
    rep.update(receipt=a.receipt, patch=a.patch, strength=a.strength, rollback=str(keep.relative_to(work)))
    (work / "sculpt" / f"{a.receipt}-{a.patch}-{a.op}.json").write_text(json.dumps(rep, indent=1))
    print(f"lidar_sculpt: {a.op} on {a.patch}: {rep['patch_faces']} faces, {rep['movable']} movable vertices "
          f"({rep['pinned']} pinned), moved {rep['moved']}, at most {rep['max_move_m'] * 1000:.1f} mm "
          f"(median {rep['median_move_m'] * 1000:.1f} mm, cap {rep['cap_m'] * 1000:.1f} mm)")
    print(f"lidar_sculpt: saved {blend.name}; the reviewed scene is in {keep.relative_to(work)}. Review the result next.")


main()
