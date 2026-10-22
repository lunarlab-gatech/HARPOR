"""Stand-ins for the stages of a web-app job, so the whole app runs on a Mac without GPUs (WEBAPP_FAKE=1).

    python -m gator.webapp.fake_steps STAGE --job J [--key-file F] [--agent astra|opus]

Each stage writes the files its real command writes (steps.py), in a second or two, and prints a
few log lines and PROGRESS lines. The scene is synthetic and ignores the video's pixels: a box of
30 x 20 x 20 cm on a floor, filmed by a camera that circles it at 1 m, 50 cm above the floor, over
270 degrees from the first frame to the last. The masks, the Depth Anything 3 cameras and depth
and the posed mesh all come from that one scene, so the real view selection and the real gravity
alignment work on them.

Real code runs wherever it needs no GPU and no ICE env: topics, frames and extract (video.py, also
for a rosbag, since rosbags is pure Python, and for a Stray Scanner zip), prep (mvsam3d.prep), lidar
(lidar.py), select (views.py), align (physics_auto.py), cameras.json (mvsam3d.cameras) and package
(jobs.package). For a job with LiDAR (steps.lidar: a bag's LiDAR topic or a Stray Scanner capture's
depth), the fake DA3 is off by FAKE_DA3_SCALE: its depth, pointmaps and camera translations are 0.8 x
the scene's, so the real lidar stage must find 1.25 from the bag's true points or the capture's true
depth images, and place records metric_scale as the real place does. place sizes the posed box in the
run's DA3 units (run_scale), as MV-SAM3D would, so a scale that select did not apply shows in the
aligned size. The agent stage
checks the key file (mode 0600, an OPENAI_API_KEY line for astra, an ANTHROPIC_API_KEY line for
opus), never prints the key, and writes the result record of the agent it stands in for.

An articulated job (build_plan.md note 11) has a lid. propose lists "body" and "lid" for the VLM
modes (with the same key file check) and runs the real stage for the user's list. parts cuts the box
at 75% of its height: the list's base is below the cut and its next part above it (the other parts
are not found), with the part masks, check images, faces.npz, vote.json, parts.glb and parts.json
(the seam's lines from gator.parts.geometry) of the real stage. The agent hinges that part on the
base's back top edge, in frame W. vomp then runs per link of physics/links, and mjcf builds the
nested model with the real gator.physics.mjcf (one convex hull per link instead of CoACD) when the
aligned GLB has joints. usd writes the articulation's layout as text, and urdf runs the real
gator.physics.urdf on such an MJCF (for the box MJCF of a rigid job it writes one box link).
"""

import argparse
import importlib.util
import json
import re
import runpy
import shutil
import stat
import sys
import time
from pathlib import Path

import numpy as np

BOX_LO, BOX_HI = np.array([-0.15, -0.2, -0.1]), np.array([0.15, 0.0, 0.1])  # world: y down, floor at y = 0
TARGET = np.array([0.0, -0.1, 0.0])
RADIUS, HEIGHT, SWEEP = 1.0, 0.5, 1.5 * np.pi
FAR = 4.0
FOCAL = 0.9  # fx = fy = FOCAL * image width
DELAY = 0.05
FAKE_DA3_SCALE = 0.8  # a job with LiDAR: DA3's depth and camera translations are this times the scene's
CUT_Y = BOX_HI[1] + 0.75 * (BOX_LO[1] - BOX_HI[1])  # world y of the cut at 75% of the box's height
COLOR = [170, 120, 70, 255]
# The VLM's list in the VLM modes: the lid hinges on the body.
FAKE_PARTS = [{"name": "body", "parent": None, "joint": "fixed"},
              {"name": "lid", "parent": "body", "joint": "revolute", "count": 1,
               "axis_hint": "a hinge along the back top edge; the lid opens upward"}]


def pace(n=4):
    for i in range(1, n + 1):
        time.sleep(DELAY)
        print(f"PROGRESS {i / n:.2f}", flush=True)


def real(module, *args):
    """Run a real CLI in this process, as `python -m module args` would."""
    sys.argv = [module, *map(str, args)]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def camera(angle):
    """(R, C): OpenCV world-to-camera rotation and camera center of the circling camera."""
    C = np.array([RADIUS * np.sin(angle), -HEIGHT, -RADIUS * np.cos(angle)])
    z = (TARGET - C) / np.linalg.norm(TARGET - C)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z]), C


def intrinsics(w, h):
    return np.array([[FOCAL * w, 0.0, (w - 1) / 2], [0.0, FOCAL * w, (h - 1) / 2], [0.0, 0.0, 1.0]])


def render(w, h, angle):
    """Ray-cast the scene: depth (optical-axis z, h x w), the box mask, K and the 3x4 extrinsics."""
    R, C = camera(angle)
    K = intrinsics(w, h)
    v, u = np.mgrid[0:h, 0:w].astype(float)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)  # z = 1, so t is depth
    d = rays @ R  # world directions
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(d[..., 1] > 1e-9, -C[1] / d[..., 1], np.inf)
        t1, t2 = (BOX_LO - C) / d, (BOX_HI - C) / d
        near, far = np.nanmax(np.minimum(t1, t2), -1), np.nanmin(np.maximum(t1, t2), -1)
    t_box = np.where((far >= near) & (far > 0), np.maximum(near, 0), np.inf)
    mask = t_box < t_floor
    return np.minimum(np.minimum(t_floor, t_box), FAR), mask, K, np.hstack([R, (-R @ C)[:, None]])


def frame_angle(job_dir, name):
    meta = json.loads((Path(job_dir) / "frames" / "meta.json").read_text())
    n = len(meta["times"])
    return SWEEP * int(re.sub(r"\D", "", name)) / max(n - 1, 1)


def material():
    import trimesh

    return trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=COLOR, metallicFactor=0.0, roughnessFactor=0.8))


def box_mesh(R=np.eye(3), t=np.zeros(3), k=1.0):
    """The box with a PBR material, its world points mapped by x -> R (k x) + t (world to W), in glTF axes."""
    import trimesh

    from gator import frames

    box = trimesh.creation.box(extents=BOX_HI - BOX_LO)
    x = k * (box.vertices + (BOX_LO + BOX_HI) / 2) @ np.asarray(R).T + t
    return trimesh.Trimesh(frames.w_to_gltf(x), box.faces, process=False, visual=material())


def view0_pose(run, name=None):
    """(R, t) that map the scene's world to frame W, x -> R x + t: view 0's DA3 extrinsics. name: view 0,
    else the first camera of run/cameras.json."""
    run = Path(run)
    name = name or json.loads((run / "cameras.json").read_text())[0]["name"]
    with np.load(run / "da3_output.npz") as da:
        stem = {Path(str(f)).stem: i for i, f in enumerate(da["image_files"])}
        E0 = da["extrinsics"][stem[name]].astype(float)
    return E0[:, :3], E0[:, 3]


def run_scale(run, name):
    """The run's DA3 units per meter of the scene, to 4 decimals: view 0's (name's) camera distance from
    the world origin in run/da3_output.npz over the scene's, which is the same for every camera of the
    circle (camera). MV-SAM3D works in DA3's units, so the fake posed box takes this size: 1 after
    select applied the LiDAR scale, FAKE_DA3_SCALE when it was lost."""
    return round(float(np.linalg.norm(view0_pose(run, name)[1]) / np.hypot(RADIUS, HEIGHT)), 4)


def check_key_file(key_file, which):
    """SystemExit unless the key file is mode 0600 and starts with the line of the agent's provider.
    The key is never read into a message."""
    from gator.webapp.jobs import KEY_VAR

    f, var = Path(key_file or ""), KEY_VAR[which]
    if not key_file or not f.is_file() or stat.S_IMODE(f.stat().st_mode) != 0o600:
        raise SystemExit(f"key file {f} is missing or not mode 0600")
    if not f.read_text().startswith(f"{var}="):
        raise SystemExit(f"key file {f} has no {var} line")


def masks(J):
    from PIL import Image

    from gator.stage_a.mvsam3d import scene_images

    job = json.loads((J / "job.json").read_text())
    picks = (job.get("overrides") or {}).get("instance") or {}
    out = J / "scene" / f"sam3_{job['slug']}"
    (out / "check").mkdir(parents=True, exist_ok=True)
    views = []
    images = scene_images(J / "scene")
    for k, (name, path) in enumerate(images.items()):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        s = min(1.0, 480 / max(W, H))  # the silhouette at most 480 px wide, then scaled up
        m = render(round(W * s), round(H * s), frame_angle(J, name))[1]
        m = np.array(Image.fromarray(m).resize((W, H), Image.NEAREST))
        inst = []
        if m.any():
            r, c = np.nonzero(m)
            box = [float(c.min()), float(r.min()), float(c.max() + 1), float(r.max() + 1)]
            inst = [{"index": 0, "score": 0.93, "box_xyxy": box, "mask_px": int(m.sum())}]
        rec = {"name": name, "image": str(path), "size_wh": [W, H], "instances": inst, "picked": [0] if inst else [],
               "pick": "top_score", "mask": None, "mask_px": 0, "warnings": []}
        if picks.get(name) == "none":
            rec.update(picked=[], pick="user", warnings=[f"left out by --pick {name}=none"])
        elif not inst:
            rec["warnings"].append("no instance above score 0.5; no mask written")
        if rec["picked"]:
            Image.fromarray(np.where(m, 255, 0).astype(np.uint8)).save(out / f"{name}.png")
            rec.update(mask=f"{name}.png", mask_px=int(m.sum()))
        else:
            (out / f"{name}.png").unlink(missing_ok=True)
        check = np.asarray(img).astype(float)
        if rec["picked"]:
            check[m] = 0.6 * check[m] + 0.4 * np.array([0.0, 255.0, 0.0])
        Image.fromarray(check.astype(np.uint8)).save(out / "check" / f"{name}.jpg", quality=85)
        views.append(rec)
        if k % 4 == 0:
            print(f"PROGRESS {(k + 1) / len(images):.2f}", flush=True)
    (out / "sam3.json").write_text(json.dumps({"model": "fake", "prompt": job["prompt"], "threshold": 0.5,
                                               "mask_threshold": 0.5, "views": views}, indent=1))
    n = sum(v["mask"] is not None for v in views)
    print(f"{job['prompt']!r}: {n} of {len(views)} photos masked -> {out} (fake SAM 3)")


def prep(J):
    from gator.stage_a import mvsam3d

    job = json.loads((J / "job.json").read_text())
    for d in ("input", "masks"):
        shutil.rmtree(J / "pool" / d, ignore_errors=True)
    (J / "pool" / "prep.json").unlink(missing_ok=True)
    pace(2)
    mvsam3d.prep(J / "scene", f"sam3_{job['slug']}", J / "pool", 1024)


def da3(J):
    """DA3 rendered from the scene; for a job with LiDAR of either kind, off by FAKE_DA3_SCALE (depth,
    pointmaps and the extrinsics' translations)."""
    from PIL import Image

    from gator.stage_a.mvsam3d import view_names
    from gator.webapp import steps

    try:
        on = steps.lidar(json.loads((J / "job.json").read_text()))
    except ValueError:  # a malformed setting: the lidar stage fails and says why
        on = None
    off = FAKE_DA3_SCALE if on else 1.0
    names = view_names(J / "pool")
    W, H = Image.open(J / "pool" / "input" / "images" / f"{names[0]}.png").size
    s = min(504, max(W, H)) / max(W, H)
    w, h = round(W * s), round(H * s)
    out = {k: [] for k in ("depth", "mask", "K", "E")}
    for name in names:
        for k, x in zip(out, render(w, h, frame_angle(J, name))):
            out[k].append(x)
    depth, K = np.array(out["depth"], np.float32) * np.float32(off), np.array(out["K"], np.float32)
    E = np.array(out["E"], np.float32)
    E[:, :, 3] *= off
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.stack([(u - K[:, 0, 2, None, None]) * depth / K[:, 0, 0, None, None],
                    (v - K[:, 1, 2, None, None]) * depth / K[:, 1, 1, None, None], depth], -1)
    pace(3)
    np.savez(J / "pool" / "da3_output.npz", depth=depth, conf=np.ones_like(depth), pointmaps=pts,
             pointmaps_sam3d=pts.transpose(0, 3, 1, 2).copy(), extrinsics=E, intrinsics=K,
             image_files=np.array([str(J / "pool" / "input" / "images" / f"{n}.png") for n in names]), process_res=504)
    print(f"DA3 (fake): {len(names)} views at {w}x{h}" + (f", {off} x the scene's scale" if off != 1 else ""))


def sam3d(J):
    from gator.stage_a.mvsam3d import view_names

    pace(4)
    box_mesh().export(J / "run" / "canonical.glb")
    names = view_names(J / "run")
    pose = {"rotation_wxyz": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0],
            "frame": "fake: canonical box", "views": names, "view0": names[0], "seed": 42, "seconds": 0.2, "peak_gib": None}
    (J / "run" / "pose.json").write_text(json.dumps(pose, indent=1))
    print(f"MV-SAM3D (fake): a box from {len(names)} views")


def place(J):
    from PIL import Image

    from gator.stage_a import mvsam3d

    run = J / "run"
    cams = mvsam3d.cameras(run)
    (run / "cameras.json").write_text(json.dumps(cams, indent=1))
    R0, t0 = view0_pose(run, cams[0]["name"])
    da3_units = run_scale(run, cams[0]["name"])  # the box in DA3's units, as MV-SAM3D's would be
    box_mesh(R0, t0, da3_units).export(run / "posed.glb")
    for c in cams:
        img = Image.open(c["image"]).convert("RGB")
        k = 512 / max(img.size)
        size = (round(img.width * k), round(img.height * k))
        a = np.asarray(img.resize(size)).astype(float)
        m = np.array(Image.open(c["mask"]).resize(size, Image.NEAREST)) > 0
        a[m] = 0.55 * a[m] + 0.45 * np.array([255.0, 0.0, 0.0])
        Image.fromarray(a.astype(np.uint8)).save(run / f"overlay_{c['name']}.png")
    pace(3)
    center = R0 @ (da3_units * (BOX_LO + BOX_HI) / 2) + t0
    with np.load(run / "da3_output.npz") as da:  # select applied a LiDAR scale
        metric = float(da["metric_scale"]) if "metric_scale" in da.files else None
    units = "meters (LiDAR scale applied in select)" if metric is not None else "DA3 units (meters)"
    placement = {"frame_W": f"OpenCV camera of view 0: x right, y down, z forward, in {units}",
                 "glb_axes": "posed.glb vertices are glTF: (x, y, z)_W = (x, -z, y)_glTF (Prompt 2)",
                 "view0": cams[0]["name"], "metric_scale": metric, "units": units, "fake": True,
                 "pose": json.loads((run / "pose.json").read_text()), "scale_fit": {"applied": False},
                 "pose_fit": {"applied": False}, "posed_extent_W": (da3_units * (BOX_HI - BOX_LO)).tolist(),
                 "posed_center_W": center.tolist(),
                 "check": {"views": [{"name": c["name"], "mask_iou": 0.8} for c in cams], "mean_mask_iou": 0.8,
                           "view0_depth_to_mesh_median_over_diag": 0.01, "view0_depth_within_2pct_diag": 0.9}}
    (run / "placement.json").write_text(json.dumps(placement, indent=1))
    print(f"placement (fake): mean mask IoU 0.8 over {len(cams)} views")


def propose(J, key_file=None, which="astra"):
    """The VLM modes: FAKE_PARTS as the VLM's proposal, after the key file check. The user's list: the real
    stage (gator.parts.propose.run), which needs no key."""
    from gator.agent.controller import CLAUDE_MODEL, CODEX_MODEL
    from gator.parts import propose as P
    from gator.webapp import steps

    job = json.loads((J / "job.json").read_text())
    mode = steps.parts_mode(job)
    if mode == "user":
        P.run(J)
        return
    check_key_file(key_file, which)
    pace(2)
    try:
        parts = P.validate(FAKE_PARTS, job["prompt"])
    except ValueError as e:
        raise SystemExit(f"fake VLM: {e}") from None
    proposal = {"source": "vlm", "agent": which, "model": CLAUDE_MODEL if which == "opus" else CODEX_MODEL,
                "noun": job["prompt"], "base": parts[0]["name"], "parts": parts, "warnings": [], "seconds": 2 * DELAY,
                "attempts": 1, "answer": "fake VLM: a body and a lid", "mode": mode, "fake": True}
    P.write(J / "run" / "proposal.json", proposal)
    print(f"proposal (fake {proposal['model']}): base {parts[0]['name']!r}, {len(parts) - 1} moving parts")


def parts(J):
    """The base of the proposal and its next part, cut from the box at CUT_Y (see the module docstring)."""
    import trimesh
    from PIL import Image

    from gator import frames
    from gator.agent import controller, tools
    from gator.parts import geometry, render
    from gator.parts.vote import load_mesh
    from gator.stage_a.part_masks import COLORS, overlay_parts
    from gator.stage_a.sam3_masks import slug

    R = J / "run"
    for f in (R / "parts.json", R / "parts.glb"):
        f.unlink(missing_ok=True)
    shutil.rmtree(R / "parts", ignore_errors=True)
    out = R / "parts" / "sam3"
    proposal = json.loads((R / "proposal.json").read_text())
    cams = json.loads((R / "cameras.json").read_text())
    Rw, tw = view0_pose(R, cams[0]["name"])
    base = next(p for p in proposal["parts"] if p["name"] == proposal["base"])
    others = [p for p in proposal["parts"] if p is not base]
    found, lost = [base, *others[:1]], others[1:]
    names = [slug(p["name"]) for p in found]
    for d in ("check", *names):
        (out / d).mkdir(parents=True)

    box = trimesh.creation.box(bounds=np.stack([BOX_LO, BOX_HI])).subdivide()  # finer faces: four clean seam lines

    def capped_piece(m):  # the cut closed by a fan from its centre: trimesh's capping needs shapely, which envs/web lacks
        m = m.copy()
        m.merge_vertices()
        _, inv, n = np.unique(m.edges_sorted, axis=0, return_inverse=True, return_counts=True)
        edges = m.edges[n[inv.ravel()] == 1]  # the open edges, directed as in their faces
        c = len(m.vertices)
        return trimesh.Trimesh(np.vstack([m.vertices, m.vertices[np.unique(edges)].mean(0)]),
                               np.vstack([m.faces, np.column_stack([edges[:, 1], edges[:, 0], np.full(len(edges), c)])]),
                               process=False)

    def pieces(cap):  # the base below the cut (y is down), the part above it
        if len(found) == 1:
            return [box]
        # slice_faces_plane, not slice_mesh_plane, which imports shapely even without a cap (trimesh 5.1.1)
        cut = [trimesh.Trimesh(*trimesh.intersections.slice_faces_plane(box.vertices, box.faces, n, [0.0, CUT_Y, 0.0])[:2],
                               process=False) for n in ([0, 1, 0], [0, -1, 0])]
        return [capped_piece(m) for m in cut] if cap else cut

    def joined(meshes):  # (vertices in W, faces, piece of each face)
        V = np.concatenate([m.vertices @ Rw.T + tw for m in meshes])
        F = np.concatenate([m.faces + o for m, o in zip(meshes, np.cumsum([0] + [len(m.vertices) for m in meshes]))])
        return V, F, np.concatenate([np.full(len(m.faces), k) for k, m in enumerate(meshes)])

    capped, opened = pieces(True), pieces(False)
    Vc, Fc, Pc = joined(capped)
    Vo, Fo, Po = joined(opened)  # the seam is where the open pieces meet
    seams = geometry.seams(geometry.topology(Vo, Fo), Po, names)
    area = np.linalg.norm(np.cross(Vc[Fc[:, 1]] - Vc[Fc[:, 0]], Vc[Fc[:, 2]] - Vc[Fc[:, 0]]), axis=1) / 2

    # Part masks and check images: the capped pieces drawn under every selected camera.
    records = {p["name"]: {"name": p["name"], "slug": slug(p["name"]), "instances": 0, "views": {}}
               for p in proposal["parts"]}
    visibility = [{} for _ in found]
    for v, c in enumerate(cams):
        K, size = render.scaled(c["K"], (c["width"], c["height"]))
        ids = render.face_ids(Vc, Fc, K, c["w2c"], size)
        seen = np.zeros(len(Fc), bool)
        seen[ids[ids >= 0]] = True
        shown = []
        for k, (p, name) in enumerate(zip(found, names)):
            small = (ids >= 0) & (Pc[np.maximum(ids, 0)] == k)
            m = np.array(Image.fromarray(small).resize((c["width"], c["height"]), Image.NEAREST))
            recs = []
            if m.any():
                rows, cols = np.nonzero(m)
                f = f"{name}/{c['name']}_0.png"
                Image.fromarray(np.where(m, 255, 0).astype(np.uint8)).save(out / f)
                recs.append({"index": 0, "score": 0.9, "box_xyxy": [float(cols.min()), float(rows.min()),
                                                                     float(cols.max() + 1), float(rows.max() + 1)],
                             "mask_px": int(m.sum()), "file": f})
            records[p["name"]]["views"][c["name"]] = recs
            records[p["name"]]["instances"] += len(recs)
            shown.append((name, COLORS[k % len(COLORS)], [r["score"] for r in recs], [r["box_xyxy"] for r in recs],
                          [m] if recs else []))
            visibility[k][c["name"]] = round(float(area[seen & (Pc == k)].sum() / area[Pc == k].sum()), 4)
        for p in lost:
            records[p["name"]]["views"][c["name"]] = []
        overlay_parts(np.array(Image.open(c["image"]).convert("RGB")), shown).save(out / "check" / f"{c['name']}.jpg",
                                                                                   quality=90)
        if v % 4 == 0:
            print(f"PROGRESS {0.8 * (v + 1) / len(cams):.2f}", flush=True)
    (out / "part_masks.json").write_text(json.dumps(
        {"model": "fake", "threshold": 0.4, "mask_threshold": 0.5, "seconds": 0.1, "peak_gib": None,
         "views": [c["name"] for c in cams], "size_wh": {c["name"]: [c["width"], c["height"]] for c in cams},
         "parts": [records[p["name"]] for p in proposal["parts"]], "not_found": [p["name"] for p in lost]}, indent=1))

    # The vote on posed.glb's faces: a face belongs to the part when its centroid is above the cut.
    x_w, faces = load_mesh(R / "posed.glb")
    cen = x_w[faces].mean(1)
    above = (cen - (Rw @ [0.0, CUT_Y, 0.0] + tw)) @ (Rw @ [0.0, -1.0, 0.0]) > 0
    label = above.astype(np.int64) if len(found) == 2 else np.zeros(len(faces), np.int64)
    np.savez(R / "parts" / "faces.npz", part=label, names=np.array(names), centroids=cen, score=np.ones(len(faces)))
    why = f"fake: only {names[-1]} is cut from the box"
    share = [round(float(area[Pc == k].sum() / area.sum()), 5) for k in range(len(found))]
    views = [[n for n, x in vis.items() if x > 0] for vis in visibility]
    clusters = [{"name": n, "prompt": p["name"], "role": "base" if k == 0 else "part", "status": "found",
                 "faces": int((label == k).sum()), "area_share": share[k], "confidence": 0.9,
                 "visibility": visibility[k], "views": views[k]} for k, (p, n) in enumerate(zip(found, names))]
    clusters += [{"name": None, "prompt": p["name"], "role": "part", "status": "not_found", "why": why, "faces": 0,
                  "views": []} for p in lost]
    (R / "parts" / "vote.json").write_text(json.dumps({"clusters": clusters, "counts": {"faces": len(faces)},
                                                       "warnings": [], "seconds": 0.1, "fake": True}, indent=1))

    # The split: the capped pieces as named root nodes in glTF axes.
    scene = trimesh.Scene()
    for name, m in zip(names, capped):
        scene.add_geometry(trimesh.Trimesh(frames.w_to_gltf(m.vertices @ Rw.T + tw), m.faces, process=False,
                                           visual=material()), node_name=name, geom_name=name)
    scene.export(R / "parts.glb", file_type="glb")
    bad = controller.check_glb(R / "parts.glb")
    if bad:
        raise SystemExit(f"parts.glb: {bad}")
    caps = [len(c.faces) - len(o.faces) for c, o in zip(capped, opened)]
    split = {"faces_in": len(faces), "faces_npz": len(faces), "mapping": "fake: the box cut by a plane", "unmatched": 0,
             "welded_vertices": 0, "custom_normals_cleared": False, "warnings": [], "seconds": 0.1,
             "parts": [{"name": n, "faces": len(m.faces), "cap_faces": cap, "holes": int(cap > 0), "open_seam_edges": 0}
                       for n, m, cap in zip(names, capped, caps)], "names_again": sorted(names)}
    (R / "parts" / "split_report.json").write_text(json.dumps(split, indent=1))
    pace(1)

    out_parts = []
    for k, (p, n) in enumerate(zip(found, names)):
        pts = Vc[np.unique(Fc[Pc == k])]
        out_parts.append({"name": n, "role": clusters[k]["role"], "prompt": p["name"], "parent": names[0] if k else None,
                          "joint_guess": "fixed" if k == 0 else p["joint"], "axis_hint": p.get("axis_hint"),
                          "faces": clusters[k]["faces"], "area_share": share[k], "confidence": 0.9, "status": "found",
                          "aabb": geometry.aabb(pts), "obb": geometry.obb(pts), "visibility": visibility[k],
                          "views": views[k], "cap_faces": caps[k], "open_seam_edges": 0})
    out_parts += [{"name": None, "role": "part", "prompt": p["name"], "parent": p.get("parent"),
                   "joint_guess": p.get("joint", "unknown"), "axis_hint": p.get("axis_hint"), "faces": 0,
                   "status": "not_found", "why": why} for p in lost]
    doc = {"version": 1, "frame": "W", "glb": "parts.glb", "glb_sha256": tools.sha256(R / "parts.glb"),
           "noun": proposal.get("noun"), "base": names[0], "source": proposal.get("source"),
           "agent": proposal.get("agent"), "model": proposal.get("model"), "parts": out_parts, "seams": seams,
           "split": {k: split[k] for k in ("mapping", "unmatched", "welded_vertices", "custom_normals_cleared")},
           "warnings": [*proposal.get("warnings", []), *([f"{why}; not found: {', '.join(p['name'] for p in lost)}"]
                                                         if lost else [])], "fake": True}
    tools.write_json_atomic(R / "parts.json", doc)
    print(f"parts.json (fake): {len(names)} parts ({', '.join(names)}), {len(seams)} seams")
    for s in seams:
        print(f"  seam {s['parts'][0]}|{s['parts'][1]}: {s['edges']} edges, {s['length']:.3f} long, {len(s['lines'])} lines")


def hinge(J, child, parent):
    """child's joint in frame W: a hinge on the back top edge of the base (world z = BOX_HI z) at the cut,
    about world -x, so that the part's front edge goes up for q > 0 (world y is down)."""
    Rw, tw = view0_pose(J / "run")
    axis = Rw @ [-1.0, 0.0, 0.0]
    return {child: {"version": 1, "frame": "W", "parent": parent, "type": "revolute",
                    "origin": (Rw @ [0.0, CUT_Y, BOX_HI[2]] + tw).tolist(), "axis": (axis / np.linalg.norm(axis)).tolist(),
                    "lower": 0.0, "upper": 1.5}}


def articulate(J, out):
    """parts.glb with the hinge of its first moving part (run/parts.json) as out. Returns the number of
    joints (0 when the split found no moving part)."""
    from gator.agent import controller
    from gator.parts import joints
    from gator.physics.align import read_glb, write_glb

    rec = json.loads((J / "run" / "parts.json").read_text())
    moving = [p for p in rec["parts"] if p["status"] == "found" and p["parent"]]
    doc, binary = read_glb(J / "run" / "parts.glb")
    spec = hinge(J, moving[0]["name"], moving[0]["parent"]) if moving else {}
    problems = joints.validate(spec, joints.mesh_nodes(doc)) if spec else []
    if problems:
        raise SystemExit("fake agent: bad joint: " + "; ".join(problems))
    write_glb(out, joints.write_joints(doc, spec), binary)
    bad = controller.check_glb(out)
    if bad:
        raise SystemExit(f"fake agent: {bad}")
    return len(spec)


def agent(J, key_file, which="astra"):
    """A refinement that returns posed.glb, or for an articulated job (agent_step.mode) the parts with a hinge."""
    from gator.agent import controller, tools
    from gator.webapp.agent_step import AGENTS, mode

    check_key_file(key_file, which)
    root = J / "agent" / "s"
    shutil.rmtree(root, ignore_errors=True)
    (root / "work" / "exports").mkdir(parents=True)
    (J / "agent" / "result.json").unlink(missing_ok=True)
    glb = root / "work" / "exports" / "001.glb"
    articulation = mode(J, json.loads((J / "job.json").read_text())) == "articulation"
    if articulation:
        shutil.copyfile(J / "run" / "parts.glb", root / "original.glb")  # as prepare_session does
        n = articulate(J, glb)
    else:
        shutil.copyfile(J / "run" / "posed.glb", glb)
    pace(4)
    kind = AGENTS[which]  # the controller's name of the agent
    info = controller.agent_record(kind)
    result = {"mode": "refinement", "agent": kind, "model": info["model"], "effort": info["effort"], "source": "finish",
              "verdict": "complete", "reason": "fake agent: the posed mesh is returned as the refinement", "limitations": [],
              "glb": "work/exports/001.glb", "sha256": tools.sha256(glb), "exit": "finish_sigint", "returncode": 130,
              "cli_exit": controller.cli_exit(130, kind), "timings": {"wall_s": 4 * DELAY},
              "counts": {"exports": 1, "reviews": 1, "checkpoints": 0}, "rejected": [], "network_commands": [],
              "key_found_in": []}
    if articulation:
        rel = "work/exports/001.glb" if n else "original.glb"
        result.update(mode="articulation", verdict="articulated" if n else "unchanged",
                      reason="fake agent: the lid hinges on the back top edge of the body" if n
                      else "fake agent: the split has no part that moves", glb=rel, sha256=tools.sha256(root / rel),
                      counts={"exports": 1, "reviews": 0, "checkpoints": 0, "joints": n, "articulation_reviews": int(n > 0)},
                      joints_delivered=n)
    if kind == "claude":
        tokens = {"input_tokens": 41000, "output_tokens": 9000, "cache_read_input_tokens": 380000,
                  "cache_creation_input_tokens": 52000}
        result.update(turns=12, tokens=tokens, end_reason="interrupted",
                      cost_usd=round(sum(tokens[k] * controller.PRICES[k] for k in tokens) / 1e6, 4))
    tools.write_json_atomic(root / "result.json", result)
    shutil.copyfile(root / "result.json", J / "agent" / "result.json")
    print(f"agent (fake {info['model']}): finish {result['verdict']}, GLB {result['glb']}"
          + (f", {n} joints" if articulation else ""))


def grid(glb, n=12):
    """(centers, h, lo, hi): an n^3 voxel grid over the GLB's box in S, the fake VoMP's occupied voxels."""
    from gator.physics.align import load_w

    v, _ = load_w(glb)  # glTF -> W is Blender's import, so these are S coordinates (z up)
    lo, hi = v.min(0), v.max(0)
    h = (hi - lo) / n
    g = (np.arange(n) + 0.5) / n
    return lo + (hi - lo) * np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3), h, lo, hi


def vomp(J):
    """Softwood in every voxel: of aligned.glb, or of each physics/links GLB of an articulated job, with the
    combined summary.json of vomp_run.combine_links."""
    from gator.physics.mass import mass_properties
    from gator.physics.vomp_run import combine_links
    from gator.webapp import steps

    P = J / "physics"
    articulated = steps.articulated(json.loads((J / "job.json").read_text()))
    shutil.rmtree(P / "work", ignore_errors=True)  # as the real stage: VoMP needs a fresh folder
    for old in ("materials.npz", "voxels.npz", "summary.json"):
        (P / old).unlink(missing_ok=True)
    if articulated:
        shutil.rmtree(P / "vomp", ignore_errors=True)
    links = sorted((P / "links").glob("*.glb")) if articulated else []
    rho, E, nu = 600.0, 2.0e9, 0.3  # softwood

    def record(glb):
        centers, h, lo, hi = grid(glb)
        return {"glb": str(Path(glb).resolve()), "config": "fake", "fake": True, "voxels_occupied": len(centers),
                "voxels_predicted": len(centers), "voxel_edge_m": float(h.max()),
                "median": {"youngs_modulus_pa": E, "poisson_ratio": nu, "density_kg_m3": rho},
                "p05_p95": {"youngs_modulus_pa": [E, E], "poisson_ratio": [nu, nu], "density_kg_m3": [rho, rho]},
                "rigid_body": mass_properties(centers, np.full(len(centers), rho), h)}, centers, h, (lo + hi) / 2

    pace(3)
    if not links:
        summary = {**record(P / "aligned.glb")[0], "seconds": {"load": 0.0, "predict": 3 * DELAY}, "gpu_peak_gb": None}
        (P / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"VoMP (fake): mass {summary['rigid_body']['mass']:.2f} kg")
        return
    for glb in links:
        rec, centers, h, center = record(glb)
        d, n = P / "vomp" / glb.stem, len(centers)
        d.mkdir(parents=True)
        np.savez(d / "voxels.npz", centers=centers, youngs_modulus=np.full(n, E), poisson_ratio=np.full(n, nu),
                 density=np.full(n, rho), transform_center=center, transform_scale=float(12 * h.max()), h=h)
        (d / "summary.json").write_text(json.dumps({**rec, "nearest_prediction_max_dist_voxels": 0.0,
                                                    "seconds": {"predict": DELAY}, "gpu_peak_gb": None}, indent=1))
        print(f"link {glb.stem}: {n} voxels, {rec['rigid_body']['mass']:.3f} kg", flush=True)
    summary = {**combine_links(P, [g.stem for g in links], P / "aligned.glb", "fake", 0.0), "fake": True}
    (P / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"VoMP (fake): {len(links)} links, mass {summary['rigid_body']['mass']:.2f} kg")


def hull(vertices, faces, threshold):
    """The convex hull of a link: the fake's CoACD, for gator.physics.mjcf.build."""
    import trimesh

    h = trimesh.Trimesh(vertices, faces).convex_hull
    return [(np.asarray(h.vertices), np.asarray(h.faces))]


def mjcf(J):
    """A box body for a rigid object; the real nested model (one hull per link) when aligned.glb has joints."""
    from gator.webapp.physics_auto import sim_name

    job = json.loads((J / "job.json").read_text())
    name, P = sim_name(job["prompt"]), J / "physics"
    rb = json.loads((P / "summary.json").read_text())["rigid_body"]
    aligned = json.loads((P / "aligned.json").read_text())
    lo, hi = np.array(aligned["min_s"]), np.array(aligned["max_s"])
    shutil.rmtree(P / "mjcf", ignore_errors=True)
    if aligned.get("joints"):
        from gator.physics import mjcf as model

        report = model.build(P / "aligned.glb", P / "summary.json", P / "mjcf", name, decompose=hull)
        compiled = model.check(P / "mjcf", name) if importlib.util.find_spec("mujoco") else \
            {"mujoco": None, "fake": "mujoco is not installed, so scene.xml was not compiled"}
        pace(2)
        (P / "mjcf" / "mjcf.json").write_text(json.dumps({**report, "fake": True, "compiled": compiled}, indent=1))
        print(f"MJCF (fake hulls): {len(report['links'])} links, {len(report['joints'])} joints, "
              f"{P / 'mjcf' / 'scene.xml'}")
        return
    (P / "mjcf").mkdir()
    z0 = max(0.0, -float(lo[2])) + 0.005

    def vec(x):
        return " ".join(f"{v:.6g}" for v in x)

    (P / "mjcf" / f"{name}.xml").write_text(f"""\
<mujoco model="{name}">
  <worldbody>
    <body name="{name}" pos="0 0 {z0:.4f}">
      <freejoint/>
      <inertial pos="{vec(rb['com'])}" mass="{rb['mass']:.6g}" fullinertia="{vec(rb['fullinertia'])}"/>
      <geom type="box" pos="{vec((lo + hi) / 2)}" size="{vec((hi - lo) / 2)}"/>
    </body>
  </worldbody>
</mujoco>
""")
    (P / "mjcf" / "scene.xml").write_text(f"""\
<mujoco model="scene">
  <include file="{name}.xml"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
  </worldbody>
</mujoco>
""")
    pace(2)
    (P / "mjcf" / "mjcf.json").write_text(json.dumps({"name": name, "fake": True, "collision_hulls": 1, "start_z": z0,
                                                      "inertial": {k: rb[k] for k in ("mass", "com", "fullinertia")},
                                                      "compiled": True}, indent=1))
    print(f"MJCF (fake): {P / 'mjcf' / 'scene.xml'}")


USD_JOINT = {"revolute": "PhysicsRevoluteJoint", "prismatic": "PhysicsPrismaticJoint", "fixed": "PhysicsFixedJoint"}


def usd_links(name, mj):
    """(the asset's text, its check) for an MJCF with links: the layout of gator.physics.usd, masses only."""
    def vec(x):
        return "(" + ", ".join(f"{v:.6g}" for v in x) + ")"

    lines = [f'def Xform "{name}" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])', "{",
             '    def Xform "links"', "    {"]
    for link, rec in mj["links"].items():
        lines += [f'        def Xform "{link}" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])',
                  "        {", f"            float physics:mass = {rec['inertial']['mass']:.6g}", "        }"]
    lines += ["    }", '    def Scope "joints"', "    {"]
    joints = [*mj["joints"], *({"child": k, "parent": p, "type": "fixed", "pos": [0.0, 0.0, 0.0], "range": [0.0, 0.0]}
                               for k, p in (mj.get("welded") or {}).items())]
    for j in joints:
        lines += [f'        def {USD_JOINT[j["type"]]} "{j["child"]}"', "        {",
                  f'            rel physics:body0 = </{name}/links/{j["parent"]}>',
                  f'            rel physics:body1 = </{name}/links/{j["child"]}>',
                  f"            point3f physics:localPos0 = {vec(j['pos'])}",
                  f"            point3f physics:localPos1 = {vec(j['pos'])}"]
        if j["type"] != "fixed":
            k = 180 / np.pi if j["type"] == "revolute" else 1.0  # USD limits are degrees for a hinge
            lines += ['            uniform token physics:axis = "X"',
                      f"            float physics:lowerLimit = {j['range'][0] * k:.6g}",
                      f"            float physics:upperLimit = {j['range'][1] * k:.6g}"]
        lines.append("        }")
    if mj.get("fixed_base"):
        lines += ['        def PhysicsFixedJoint "world"', "        {",
                  f"            rel physics:body1 = </{name}/links/{mj['root']}>", "        }"]
    lines += ["    }", "}"]
    checked = {"mass": mj["inertial"]["mass"], "links": {k: {"mass": r["inertial"]["mass"]} for k, r in mj["links"].items()},
               "joints": [{"name": j["child"], "type": USD_JOINT[j["type"]], "body0": j["parent"], "body1": j["child"]}
                          for j in joints]}
    return "\n".join(lines) + "\n", checked


def usd(J):
    """A rigid body with the VoMP mass; for an MJCF with links, the articulation (usd_links)."""
    from gator.webapp.physics_auto import sim_name

    job = json.loads((J / "job.json").read_text())
    name, P = sim_name(job["prompt"]), J / "physics"
    rb = json.loads((P / "summary.json").read_text())["rigid_body"]
    mj = json.loads((P / "mjcf" / "mjcf.json").read_text())
    (P / "usd").mkdir(exist_ok=True)
    if "links" in mj:
        body, checked = usd_links(name, mj)
    else:
        body, checked = f"""\
def Xform "{name}" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
{{
    float physics:mass = {rb['mass']:.6g}
}}
""", {"mass": rb["mass"]}
    (P / "usd" / f"{name}.usda").write_text(f"""\
#usda 1.0
(
    defaultPrim = "{name}"
    metersPerUnit = 1
    upAxis = "Z"
)

""" + body)
    (P / "usd" / "scene.usda").write_text(f"""\
#usda 1.0

def Xform "World"
{{
    def "{name}" (prepend references = @./{name}.usda@)
    {{
    }}
}}
""")
    pace(2)
    (P / "usd" / "usd.json").write_text(json.dumps({"name": name, "fake": True, "checked": checked}, indent=1))
    print(f"USD (fake): {P / 'usd' / 'scene.usda'}")


def urdf(J):
    """The real URDF of an MJCF with links; for the box MJCF of a rigid job, one link with a box mesh."""
    import trimesh

    from gator.physics import urdf as U
    from gator.physics.mjcf import write_obj
    from gator.webapp.physics_auto import sim_name

    job = json.loads((J / "job.json").read_text())
    name, P = sim_name(job["prompt"]), J / "physics"
    out = P / "urdf"
    shutil.rmtree(out, ignore_errors=True)
    mj = json.loads((P / "mjcf" / "mjcf.json").read_text())
    if "links" in mj:
        report = U.build(P / "aligned.glb", P / "summary.json", P / "mjcf", out, name)
    else:
        aligned = json.loads((P / "aligned.json").read_text())
        rb = json.loads((P / "summary.json").read_text())["rigid_body"]
        (out / "meshes").mkdir(parents=True)
        box = trimesh.creation.box(bounds=np.array([aligned["min_s"], aligned["max_s"]]))
        write_obj(out / "meshes" / "link_0.obj", box.vertices, box.faces)
        links = [{"name": name, "frame": [0.0, 0.0, 0.0], "inertial": {k: rb[k] for k in ("mass", "com", "fullinertia")},
                  "mesh": "meshes/link_0.obj", "collisions": ["meshes/link_0.obj"]}]
        (out / "model.urdf").write_text(U.urdf_xml(name, links, [], False))
        report = {"glb": str((P / "aligned.glb").resolve()), "name": name, "root": name, "fixed_base": False,
                  "links": links, "joints": [], "excluded_pairs": []}
    pace(2)
    report.update(fake=True, checked=U.check(out))
    (out / "urdf.json").write_text(json.dumps(report, indent=1))
    print(f"URDF (fake): {report['checked']['links']} links, {report['checked']['joints']} joints, {out / 'model.urdf'}")


def main(argv=None):
    from gator.webapp import jobs, steps

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=steps.NAMES)
    ap.add_argument("--job", required=True)
    ap.add_argument("--key-file", help="agent and propose stages: the env file with the agent's API key")
    ap.add_argument("--agent", choices=("astra", "opus"), default="astra", help="agent and propose stages: the job's agent")
    a = ap.parse_args(argv)
    J = Path(a.job)
    print(f"fake {a.stage}: {J.name}", flush=True)
    if a.stage == "topics":
        real("gator.webapp.video", "topics", "--job", J)
    elif a.stage == "frames":
        real("gator.webapp.video", "analyze", "--job", J)
    elif a.stage == "extract":
        real("gator.webapp.video", "extract", "--job", J)
    elif a.stage == "lidar":
        real("gator.webapp.lidar", "scale", "--job", J)
    elif a.stage == "select":
        real("gator.webapp.views", "select", "--job", J)
    elif a.stage == "align":
        real("gator.webapp.physics_auto", "align", "--job", J)
    elif a.stage == "package":
        jobs.package(J)
    elif a.stage in ("agent", "propose"):
        globals()[a.stage](J, a.key_file, a.agent)
    else:
        globals()[a.stage](J)


if __name__ == "__main__":
    main()
