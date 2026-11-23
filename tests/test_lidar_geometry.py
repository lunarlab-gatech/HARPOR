"""LiDAR guided sculpting on a box of known geometry (harpor/agent/lidar_geometry.py, harpor/agent/bin/sculpt_math.py;
docs/lidar_sculpting_plan.md). The fixture (tests/lidar_fixture.py) measures the clean box analytically; the tests
check the scorer, the frozen evidence, the regions, the two sculpt operations, the host's acceptance check and the
delivery of the session's candidate with the fake agent."""

import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rtree")
import trimesh  # noqa: E402

import lidar_fixture as LF  # noqa: E402
from harpor import frames as Fr  # noqa: E402
from harpor.agent import controller as C  # noqa: E402
from harpor.agent import lidar_geometry as G  # noqa: E402
from harpor.agent import prompts, validate  # noqa: E402
from harpor.agent.bin import sculpt_math as S  # noqa: E402
from harpor.agent.fake_agent import sculpt_glb  # noqa: E402
from harpor.stage_a import mvsam3d  # noqa: E402
from harpor.webapp import agent_step  # noqa: E402

pytestmark = pytest.mark.xdist_group("lidar_geometry")
REPO = Path(__file__).resolve().parents[1]
FAKE_BLENDER = [sys.executable, "-m", "harpor.agent.fake_agent", "--blender"]
FAST = {"poll_s": 0.05, "kill_after_s": 0.5, "finish_wait_s": 0.5}


def prepare(J, requested="sculpt"):
    """The agent session of a fixture job, as agent_step prepares it, with its baseline."""
    job = json.loads((J / "job.json").read_text())
    root = J / "agent" / "s"
    if root.exists():
        shutil.rmtree(root)
    C.prepare_session(root, None, mvsam3d.session_views(J / "run"), source="fixture", metric=True,
                      original=J / "run" / "posed.glb", rounds=True,
                      lidar=agent_step.geometry_request(J, job, requested))
    G.baseline(root)
    return root


@pytest.fixture(scope="module")
def jobs(tmp_path_factory):
    base = tmp_path_factory.mktemp("lidar")
    out = {}
    for kind in ("clean", "bowed", "dent", "deleted", "shifted"):
        out[kind] = LF.job(base / kind, kind)
    out["misregistered"] = LF.job(base / "misregistered", "clean", misregister=2)
    out["uncalibrated"] = LF.job(base / "uncalibrated", "bowed", profile=None)
    return out


@pytest.fixture(scope="module")
def roots(jobs):
    with ThreadPoolExecutor(4) as ex:
        return dict(zip(jobs, ex.map(prepare, jobs.values())))


def evidence(root):
    return G.load_evidence(root / "control")


def baseline(root):
    return json.loads((root / "control" / "lidar" / "baseline.json").read_text())


def right_normal():
    W_from_D, R, _ = LF.world()
    return W_from_D[:3, :3] @ R @ np.array([1.0, 0, 0])


# ---- step 1: the known geometry ----

def test_the_fixture_returns_lie_on_the_clean_box(jobs, roots):
    """The scorer on the clean box finds every eligible return on its surface (floating point), and the analytic
    reference agrees with trimesh's first hits on the clean mesh."""
    root = roots["clean"]
    ev, m = evidence(root), G.manifest(root / "control")
    sc = G.score(G.Mesh(root / "original.glb"), ev)
    use = G.usable(ev, sc, m["settings"])
    assert use.sum() > 10000 and np.nanmax(np.abs(sc["err"][use])) < 1e-6
    d = (ev["end"] - ev["origin"]) / sc["range"][:, None]
    t, panel = LF.box_hit(ev["origin"], d)
    assert np.allclose(t[use], sc["range"][use], atol=1e-7)
    assert (panel[use] >= 0).all() and (panel[use] != LF.PANELS.index((1, -1))).all()  # the bottom is unmeasured


def test_the_bowed_panel_gives_a_local_error_of_known_sign(roots):
    root = roots["bowed"]
    ev, m = evidence(root), G.manifest(root / "control")
    sc = G.score(G.Mesh(root / "original.glb"), ev)
    use = G.usable(ev, sc, m["settings"])
    _, panel = LF.box_hit(ev["origin"], (ev["end"] - ev["origin"]) / sc["range"][:, None])
    right = use & (panel == LF.PANELS.index(LF.RIGHT))
    assert right.sum() > 500 and np.median(sc["err"][right]) < -0.008  # the mesh sticks out toward the sensors
    assert (sc["err"][right] <= 1e-6).all() and sc["err"][right].min() > -0.08  # never behind; at most 20 mm / cos
    # The other panels agree, but for the rays that the bulge now catches before an adjacent panel.
    others = np.abs(sc["err"][use & ~right])
    assert np.median(others) < 1e-6 and (others < 1e-6).mean() > 0.97


def test_the_deleted_panel_exposes_the_far_wall_or_misses(roots):
    root = roots["deleted"]
    ev = evidence(root)
    sc = G.score(G.Mesh(root / "original.glb"), ev)
    _, panel = LF.box_hit(ev["origin"], (ev["end"] - ev["origin"]) / sc["range"][:, None])
    right = ev["keep"] & (panel == LF.PANELS.index(LF.RIGHT))
    far = right & sc["hit"]
    assert right.sum() > 500
    # Each ray either misses or goes on through the opening to a farther wall's inside (a back face): a positive
    # error beyond the tolerance, which counts against the candidate.
    assert (sc["err"][far] > 0.005).all() and not sc["front"][far].any()


# ---- step 3: the evidence in frame W ----

def test_freeze_puts_the_returns_on_the_analytic_landmarks(tmp_path):
    """inverse(w2c) of each camera-frame return lands on the clean box in W, for another anchor view too, with the
    offset sensor's origin where the fixture put it; the index's metric scale is not applied again."""
    J = LF.job(tmp_path / "anchor", "clean", anchor=3)
    index = json.loads((J / "run" / "lidar" / "geometry" / "index.json").read_text())
    index["metric_scale"] = 1.25  # a nonunit scale that select applied to DA3: the meters stay
    (J / "run" / "lidar" / "geometry" / "index.json").write_text(json.dumps(index))
    root = prepare(J)
    ev = evidence(root)
    d = (ev["end"] - ev["origin"]) / np.linalg.norm(ev["end"] - ev["origin"], axis=1)[:, None]
    t, panel = LF.box_hit(ev["origin"], d)
    ok = panel >= 0
    assert ok.mean() > 0.99 and np.allclose(t[ok], np.linalg.norm(ev["end"] - ev["origin"], axis=1)[ok], atol=1e-7)
    _, _, cams = LF.world()
    off = ev["view"] == LF.OFFSET_VIEW
    c2w = np.linalg.inv(cams[LF.OFFSET_VIEW])
    assert np.allclose(ev["origin"][off], Fr.transform_points(c2w, np.array([[LF.OFFSET_M, 0, 0]])), atol=1e-9)
    assert np.allclose(ev["origin"][ev["view"] == 0], 0, atol=1e-12)  # view 0's camera is W's origin
    LF.ANCHOR[0] = 0


def test_node_transforms_are_applied(tmp_path):
    """A GLB whose panels sit under nonidentity node transforms (turns, nonuniform scales, shifts) scores as the flat
    one: the scorer maps every primitive into W."""
    J = LF.job(tmp_path / "nodes", "clean", nodes=True)
    root = prepare(J)
    ev, m = evidence(root), G.manifest(root / "control")
    sc = G.score(G.Mesh(root / "original.glb"), ev)
    use = G.usable(ev, sc, m["settings"])
    assert use.sum() > 10000 and np.abs(sc["err"][use]).max() < 1e-6


def test_the_split_reserves_viewpoint_groups(roots):
    root = roots["bowed"]
    m = G.manifest(root / "control")
    ev = evidence(root)
    roles = [g["role"] for g in m["groups"]]
    assert len(m["groups"]) == LF.N_VIEWS and roles.count(G.VERIFY) == 2 and m["groups"][0]["role"] == G.FITTING
    reserved = set(m["split"]["reserved_views"])
    assert set(np.unique(ev["view"][ev["verify"]])) == reserved
    with np.load(root / "work" / "input" / "lidar" / "fitting.npz") as f:
        assert not set(f["view"]) & reserved and len(f["ray"]) == int((ev["keep"] & ~ev["verify"]).sum())
    seen = json.loads((root / "work" / "input" / "lidar" / "manifest.json").read_text())
    assert seen["reserved_views"] == "kept by the host" and seen["mode"] == "sculpt"
    # Two views that share a scan are one viewpoint.
    views = [{"index": i, "time": {"window_ns": [0, 1], "scans": [7]}} for i in (0, 1)] + \
            [{"index": 2, "time": {"window_ns": [5, 6], "scans": [9]}}]
    _, _, cams = LF.world()
    groups, _, timed = G._groups(views, {i: {"w2c": cams[i]} for i in range(3)}, G.settings("synthetic"))
    assert groups == [[0, 1], [2]] and timed == [True, True]


def test_a_worker_cannot_change_the_hosts_evidence(roots):
    root = roots["bowed"]
    key = G.manifest(root / "control")["key"]
    np.savez(root / "work" / "input" / "lidar" / "fitting.npz", ray=np.zeros(1))
    assert G.manifest(root / "control")["key"] == key and len(evidence(root)["ray"]) > 10000


# ---- step 4: the scorer ----

def plane_glb(path, z=1.0, size=1.0, extra=None):
    """A square in W at depth z facing the origin (normal -z), as a GLB; extra: more (V, F) in W."""
    V = np.array([[-size, -size, z], [size, -size, z], [size, size, z], [-size, size, z]], float)
    F = np.array([[0, 2, 1], [0, 3, 2]])
    parts = [(V, F)] + list(extra or [])
    scene = trimesh.Scene()
    for k, (v, f) in enumerate(parts):
        scene.add_geometry(trimesh.Trimesh(Fr.w_to_gltf(v), f, process=False), node_name=f"n{k}")
    Path(path).write_bytes(scene.export(file_type="glb"))
    return path


def rays(origin, ends):
    ends = np.asarray(ends, float)
    return {"origin": np.repeat(np.asarray(origin, float)[None], len(ends), 0), "end": ends}


def test_signed_offsets_on_both_sides_occlusion_misses_and_off_axis(tmp_path):
    mesh = G.Mesh(plane_glb(tmp_path / "p.glb", z=1.0, extra=[(
        np.array([[0.15, -0.1, 0.5], [0.35, -0.1, 0.5], [0.35, 0.1, 0.5], [0.15, 0.1, 0.5]]),
        np.array([[0, 2, 1], [0, 3, 2]]))]))
    ev = rays([0, 0, 0], [[0, 0, 1.01], [0, 0, 0.99], [0.5, 0, 1.0], [0, 0, -1.0], [0.3, 0.2, 1.02]])
    sc = G.score(mesh, ev)
    assert sc["err"][0] == pytest.approx(-0.01) and sc["err"][1] == pytest.approx(0.01)  # too close; too far
    assert sc["err"][2] == pytest.approx(0.5 * np.sqrt(1.25) - np.sqrt(1.25))  # the nearer square is hit first
    assert not sc["hit"][3] and np.isnan(sc["err"][3])  # a miss
    # Off-axis: the range differs by 0.02 / cos, not the depth by 0.02.
    assert sc["err"][4] == pytest.approx(-0.02 * np.linalg.norm([0.3, 0.2, 1.02]) / 1.02, rel=1e-6)
    assert sc["front"][:3].all()


def test_reordered_triangles_and_transformed_primitives_score_the_same(tmp_path, roots):
    root = roots["bowed"]
    ev = evidence(root)
    a = G.score(G.Mesh(root / "original.glb"), ev)
    scene = trimesh.load(root / "original.glb", force="scene", process=False)
    rng = np.random.default_rng(0)
    for g in scene.geometry.values():
        g.faces = g.faces[rng.permutation(len(g.faces))]
    scene.export(tmp_path / "shuffled.glb")
    b = G.score(G.Mesh(tmp_path / "shuffled.glb"), ev)
    assert np.array_equal(a["hit"], b["hit"]) and np.allclose(a["err"][a["hit"]], b["err"][b["hit"]], atol=1e-9)


def test_scores_are_cached_and_repeatable(roots):
    root = roots["bowed"]
    ev, m = evidence(root), G.manifest(root / "control")
    a, mesh = G.cached_score(root / "control", root / "original.glb", ev, m["key"])
    b, again = G.cached_score(root / "control", root / "original.glb", ev, m["key"])
    assert again is None and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)


# ---- step 5: baseline and review outputs ----

def test_the_baseline_names_the_bowed_face_its_direction_and_the_unobserved_face(roots):
    rec = baseline(roots["bowed"])
    (r,) = rec["regions"]
    assert r["status"] == "target" and r["verifiable"] and r["suggested_op"] == "plane"
    assert np.dot(r["normal"], right_normal()) > 0.999
    assert r["median_signed_m"] < -0.008 and r["direction"].startswith("inward")
    assert len([x for x in r["support"] if x["rays"] >= 25]) >= 2 and r["verification"]
    (u,) = rec["unobserved"]
    W_from_D, R, _ = LF.world()
    assert np.dot(u["normal"], W_from_D[:3, :3] @ np.array([0, -1.0, 0])) > 0.99  # the bottom
    agent = json.loads((roots["bowed"] / "work" / "original" / "previews" / "lidar" / "summary.json").read_text())
    assert agent["regions"][0]["id"] == "L1" and agent["unobserved_surfaces"]
    assert "verification" not in agent["regions"][0] and all(not v["reserved"] for v in agent["views"])
    pngs = sorted(p.name for p in (roots["bowed"] / "work" / "original" / "previews" / "lidar").glob("*.png"))
    assert pngs == [f"lidar_view{i:02d}.png" for i in range(LF.N_VIEWS) if i not in (3, 6)]


def test_a_rounded_edge_divides_two_sides(tmp_path):
    """A connected box mesh with rounded edges, as a reconstruction gives, divides into its six sides. Each face turns
    at most a few degrees from the next, so a rule that compares only neighbors joins the whole box into one
    surface; the surface's mean normal cuts each edge."""
    sphere = trimesh.creation.icosphere(subdivisions=4)
    u = sphere.vertices
    glb = tmp_path / "rounded.glb"
    trimesh.Trimesh(np.sign(u) * np.abs(u) ** 0.25 * (LF.DIMS / 2), sphere.faces, process=False).export(glb)
    mesh = G.Mesh(glb)
    p = mesh.adjacency()
    turn = np.degrees(np.arccos(np.clip((mesh.normals[p[:, 0]] * mesh.normals[p[:, 1]]).sum(1), -1, 1)))
    assert turn.max() < 10 < G.BASE["crease_deg"]  # no sharp edge anywhere
    comp, _ = G.surfaces(mesh, G.BASE)
    ids = np.unique(comp)
    share = np.array([mesh.areas[comp == i].sum() for i in ids]) / mesh.areas.sum()
    top = ids[np.argsort(-share)[:6]]
    assert np.sort(share)[::-1][:6].sum() > 0.9
    axes = np.r_[np.eye(3), -np.eye(3)]  # W's axes are glTF's, permuted and signed
    dirs = []
    for i in top:
        sel = comp == i
        n = (mesh.normals[sel] * mesh.areas[sel, None]).sum(0)
        dirs.append(n / np.linalg.norm(n))
    near = np.array(dirs) @ axes.T
    assert sorted(np.argmax(near, 1).tolist()) == list(range(6)) and (near.max(1) > np.cos(np.radians(5))).all()
    # Each side's share of the area is about that of its face of the box (18 x 24, 36 x 24 and 36 x 18 cm, twice).
    d = LF.DIMS
    faces = np.repeat([d[1] * d[2], d[0] * d[2], d[0] * d[1]], 2)
    assert np.allclose(np.sort(share)[::-1][:6], np.sort(faces / faces.sum())[::-1], atol=0.015)


def test_a_correct_box_has_no_region(roots):
    rec = baseline(roots["clean"])
    assert rec["regions"] == [] and not rec["alignment"]["common_shift"] and rec["alignment"]["excluded"] == []
    assert G.summary_lines(rec)[-1].startswith("lidar: 1 surfaces have no returns")


def test_a_misregistered_view_is_an_alignment_finding_not_a_target(roots):
    rec = baseline(roots["misregistered"])
    assert rec["alignment"]["excluded"] == [2]
    assert not [r for r in rec["regions"] if r["status"] == "target"]


def test_a_common_shift_abstains(roots):
    rec = baseline(roots["shifted"])
    assert rec["alignment"]["common_shift"]
    assert not [r for r in rec["regions"] if r["status"] == "target"]


def test_review_outputs_hold_the_correspondences_and_fail_soft(roots, tmp_path):
    root = roots["bowed"]
    files, lines = G.review_outputs(root / "work", root / "control", root / "original.glb", None,
                                    {"blend_sha256": "x"})
    doc = json.loads(files["lidar.json"])
    corr = doc["correspondences"]["L1"]
    assert len(corr["end"]) == len(corr["hit"]) == len(corr["view"]) >= 500
    assert not set(corr["view"]) & set(G.manifest(root / "control")["split"]["reserved_views"])
    assert any(line.startswith("lidar L1 target (host-verifiable)") for line in lines)
    broken = tmp_path / "control"
    shutil.copytree(root / "control", broken)
    (broken / "lidar" / "evidence.npz").write_bytes(b"not an npz")
    (broken / "lidar" / "baseline.json").unlink()
    files, lines = G.review_outputs(root / "work", broken, root / "original.glb")
    assert files is None and lines[0].startswith("lidar: scoring failed")


# ---- step 6: the sculpt operations ----

def sculpted(root, op="plane", strength=1.0, tmp=None, region="L1", src=None, name=""):
    """One sculpt pass from a review of src (by default the original), as lidar_sculpt runs it: (GLB, report)."""
    src = src or root / "original.glb"
    files, _ = G.review_outputs(root / "work", root / "control", src, None, {})
    doc = json.loads(files["lidar.json"])
    reg = next(r for r in doc["regions"] if r["id"] == region)
    out = Path(tmp) / f"{root.parent.parent.name}_{op}{name}.glb"
    rep = sculpt_glb(src, out, reg, doc["correspondences"][region], op, strength)
    return out, rep


def panel_vertices(glb):
    """[(vertices in W)] per panel of a GLB, in the fixture's panel order (the node names), whatever order the export
    gave its nodes."""
    from harpor.agent.fake_agent import scene_w

    by = {p[0]: p[3] for p in scene_w(glb)[1]}
    return [by[f"panel{k}"] for k in range(len(by))]


def test_plane_flattens_the_bowed_panel_and_pins_the_rest(roots, tmp_path):
    root = roots["bowed"]
    out, rep = sculpted(root, tmp=tmp_path)
    assert rep["movable"] > 50 and rep["max_move_m"] <= LF.BOW_M + 1e-6
    before, after = panel_vertices(root / "original.glb"), panel_vertices(out)
    clean = [V for _, V, _, _ in LF.box_parts("clean")]
    k = LF.PANELS.index(LF.RIGHT)
    for i, (a, b, c) in enumerate(zip(before, after, clean)):
        if i != k:
            assert np.abs(b - a).max() < 1e-7  # other panels do not move
    rim = np.abs(before[k] - clean[k]).max(1) < 1e-6  # the bow is 0 there (float32 positions)
    assert rim.sum() > 20
    assert np.abs(after[k][rim] - before[k][rim]).max() < 1e-4  # the pinned rim moved under 0.1 mm
    assert np.abs(after[k] - clean[k]).max() < 0.0015 and np.median(np.abs(after[k] - clean[k]).max(1)) < 1e-5
    a, b = trimesh.load(root / "original.glb", force="scene", process=False), trimesh.load(out, force="scene", process=False)
    for g in a.geometry:
        assert np.array_equal(a.geometry[g].faces, b.geometry[g].faces)  # topology
        assert np.allclose(a.geometry[g].visual.uv, b.geometry[g].visual.uv)  # UVs


def test_push_fills_the_dent(roots, tmp_path):
    root = roots["dent"]
    (r,) = baseline(root)["regions"]
    assert r["direction"].startswith("outward") and r["median_signed_m"] > 0.0  # a dent: the mesh lies behind
    one, _ = sculpted(root, "push", tmp=tmp_path)
    two, _ = sculpted(root, "push", tmp=tmp_path, src=one, name="2")  # a second pass from a review of the first
    k = LF.PANELS.index(LF.RIGHT)
    clean = [V for _, V, _, _ in LF.box_parts("clean")][k]
    n = right_normal()
    err = [np.abs((panel_vertices(g)[k] - clean) @ n) for g in (root / "original.glb", one, two)]  # off the plane
    dented = err[0] > 0.003
    assert err[1].max() < 0.6 * err[0].max() and err[2].max() < 0.35 * err[0].max()  # each pass is bounded
    assert np.median(err[2][dented]) < 0.35 * np.median(err[0][dented])
    assert err[2][~dented].max() < 0.003  # the flat part of the panel stays flat


def test_a_correct_panel_does_not_move(roots, tmp_path):
    """A plane pass that a review region names on the clean box's right panel moves nothing measurable."""
    root = roots["clean"]
    files, _ = G.review_outputs(root / "work", root / "control", root / "original.glb", None, {})
    mesh = G.Mesh(root / "original.glb")
    ev = evidence(root)
    sc = G.score(mesh, ev)
    _, panel = LF.box_hit(ev["origin"], (ev["end"] - ev["origin"]) / sc["range"][:, None])
    sel = G.usable(ev, sc, G.manifest(root / "control")["settings"]) & (panel == LF.PANELS.index(LF.RIGHT))
    d = (ev["end"][sel] - ev["origin"][sel]) / sc["range"][sel, None]
    corr = {"end": ev["end"][sel].tolist(), "hit": (ev["origin"][sel] + d * sc["range"][sel, None]).tolist(),
            "normal": mesh.normals[sc["tri"][sel]].tolist(), "view": ev["view"][sel].tolist()}
    k = LF.PANELS.index(LF.RIGHT)
    clean = [V for _, V, _, _ in LF.box_parts("clean")][k]
    region = {"id": "X", "seed": clean.mean(0).tolist(), "normal": right_normal().tolist(),
              "bounds": [clean.min(0).tolist(), clean.max(0).tolist()]}
    V, F = panel_mesh(root / "original.glb")
    for op in ("plane", "push"):
        out, rep = S.sculpt(V, F, region, corr, op)
        assert np.abs(out - V).max() < 1e-6, op


def panel_mesh(glb):
    from harpor.agent.fake_agent import scene_w

    scene, parts = scene_w(glb)
    V = np.concatenate([p[3] for p in parts])
    off = np.cumsum([0] + [len(p[3]) for p in parts])
    F = np.concatenate([np.asarray(scene.geometry[g].faces) + o for (_, g, _, _), o in zip(parts, off)])
    return V, F


def test_two_nearby_surfaces_are_ambiguous_or_kept_apart():
    """Two separate squares facing the same way: a seed between them is ambiguous; a seed on one moves only it."""
    def grid(x0):
        s, t = np.meshgrid(np.linspace(0, 0.2, 11), np.linspace(0, 0.2, 11), indexing="ij")
        V = np.c_[x0 + s.ravel(), t.ravel(), np.zeros(s.size)]
        idx = np.arange(s.size).reshape(11, 11)
        q = np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()], 1)
        return V, np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])

    (Va, Fa), (Vb, Fb) = grid(0.0), grid(0.205)
    V, F = np.r_[Va, Vb], np.r_[Fa, Fb + len(Va)]
    end = np.c_[Va[:, :2], np.full(len(Va), 0.004)]
    corr = {"end": end.tolist(), "hit": Va.tolist(), "normal": [[0, 0, 1.0]] * len(Va), "view": [0, 1] * 60 + [0]}
    between = {"id": "A", "seed": [0.2025, 0.1, 0.0], "normal": [0, 0, 1.0], "bounds": [[0, 0, -0.01], [0.405, 0.2, 0.01]]}
    with pytest.raises(S.SculptError, match="ambiguous"):
        S.sculpt(V, F, between, corr, "plane")
    one = {**between, "seed": [0.1, 0.1, 0.0]}
    out, rep = S.sculpt(V, F, one, corr, "push")
    assert np.abs(out[len(Va):] - V[len(Va):]).max() == 0 and out[:len(Va), 2].max() > 0.002
    coarse = {"id": "C", "seed": [0.1, 0.1, 0], "normal": [0, 0, 1.0], "bounds": [[0, 0, -0.01], [0.2, 0.2, 0.01]]}
    sq = np.array([[0, 0, 0], [0.2, 0, 0], [0.2, 0.2, 0], [0, 0.2, 0]], float)
    with pytest.raises(S.SculptError, match="too coarse"):
        S.sculpt(sq, np.array([[0, 1, 2], [0, 2, 3]]), coarse, corr, "plane")


def test_sculpt_keeps_node_transforms(tmp_path):
    """The bowed box stored under nonuniform node transforms sculpts to the same W result."""
    J = LF.job(tmp_path / "nodes", "bowed", nodes=True)
    root = prepare(J)
    out, _ = sculpted(root, tmp=tmp_path)
    flat = panel_vertices(out)[LF.PANELS.index(LF.RIGHT)]
    clean = [V for _, V, _, _ in LF.box_parts("clean")][LF.PANELS.index(LF.RIGHT)]
    assert np.abs(flat - clean).max() < 0.0015
    a, b = trimesh.load(root / "original.glb", force="scene", process=False), trimesh.load(out, force="scene", process=False)
    assert all(np.allclose(a.graph[n][0], b.graph[n][0]) for n in a.graph.nodes_geometry)


# ---- step 7: the host's acceptance check ----

def report(root, glb):
    return validate.check(glb, root / "control", root / "original.glb")


def test_the_acceptance_targets_on_the_bowed_fixture(roots, tmp_path):
    """Synthetic targets: at least 75% less held-out median absolute error on the bowed interior, no lost reserved
    hits, and under 0.1 mm at pinned landmarks (the rim and the other panels)."""
    root = roots["bowed"]
    out, _ = sculpted(root, tmp=tmp_path)
    ev, m = evidence(root), G.manifest(root / "control")
    s = m["settings"]
    with np.load(root / "control" / "lidar" / "baseline.npz") as f:
        base = {k: f[k] for k in f.files}
    sc0, sc1 = G.score(G.Mesh(root / "original.glb"), ev), G.score(G.Mesh(out), ev)
    k = LF.PANELS.index(LF.RIGHT)
    clean = [V for _, V, _, _ in LF.box_parts("clean")][k]
    W_from_D, R, _ = LF.world()
    hit_box = Fr.transform_points(np.linalg.inv(W_from_D), ev["end"]) @ R  # the measured point in the box frame
    a, b = [i for i in range(3) if i != 0]
    interior = (np.abs(hit_box[:, a]) < 0.4 * LF.DIMS[a]) & (np.abs(hit_box[:, b]) < 0.4 * LF.DIMS[b])
    held = base["use"] & ev["verify"] & (base["ray_region"] == 0) & interior
    assert held.sum() > 200
    e0, e1 = np.median(np.abs(sc0["err"][held])), np.median(np.abs(sc1["err"][held]))
    assert e1 <= 0.25 * e0, (e0, e1)
    assert sc1["hit"][base["use"] & ev["verify"]].all()  # no lost reserved hits
    rep = report(root, out)
    assert rep["lidar"]["status"] == "improved" and rep["outcome"] == "pass" and rep["lidar"]["resolved"] == 1
    reg = rep["lidar"]["regions"][0]
    assert all(reg["after_m"][v] <= 0.25 * reg["before_m"][v] for v in reg["before_m"])
    print(json.dumps({"held_out_median_abs_before_mm": e0 * 1000, "after_mm": e1 * 1000,
                      "reduction": 1 - e1 / e0, "tol_mm": s["range_tol_m"] * 1000}))


def test_the_dent_is_caught_and_a_reversed_correction_is_refused(roots, tmp_path):
    root = roots["dent"]
    good, _ = sculpted(root, "push", tmp=tmp_path)
    assert report(root, good)["lidar"]["status"] == "improved"
    files, _ = G.review_outputs(root / "work", root / "control", root / "original.glb", None, {})
    doc = json.loads(files["lidar.json"])
    corr = doc["correspondences"]["L1"]
    flipped = {**corr, "end": (2 * np.array(corr["hit"]) - np.array(corr["end"])).tolist()}  # the sign reversed
    sculpt_glb(root / "original.glb", tmp_path / "reversed.glb", doc["regions"][0], flipped, "push")
    rep = report(root, tmp_path / "reversed.glb")
    assert rep["lidar"]["status"] == "regressed" and G.demonstrably_worse(rep)
    assert any(f["code"] == "lidar_regression" and validate.counts(f) for f in rep["findings"])


def test_the_original_and_a_texture_only_repair(roots, tmp_path):
    """On the clean box the original and a texture-only repair tie and pass; on the bowed box a texture-only repair
    leaves a host-verifiable target, which counts in sculpt mode and not in inspect mode."""
    clean = roots["clean"]
    LF.write_glb(tmp_path / "tinted_clean.glb", "clean", tint=(250, 250, 250))
    for glb in (clean / "original.glb", tmp_path / "tinted_clean.glb"):
        rep = report(clean, glb)
        assert rep["lidar"]["status"] == "tie" and rep["outcome"] == "pass", rep["findings"]
    bowed = roots["bowed"]
    LF.write_glb(tmp_path / "tinted_bowed.glb", "bowed", tint=(250, 250, 250))
    rep = report(bowed, tmp_path / "tinted_bowed.glb")
    assert [f["code"] for f in rep["findings"] if validate.counts(f)] == ["lidar_target"]
    insp = roots["uncalibrated"]  # an uncalibrated source runs inspect only, with nominal tolerances
    m = G.manifest(insp / "control")
    assert m["mode"] == {"requested": "sculpt", "effective": "inspect"} and not m["settings"]["calibrated"]
    rep = report(insp, tmp_path / "tinted_bowed.glb")
    assert rep["outcome"] == "pass" and not any(validate.counts(f) for f in rep["findings"])
    assert "lidar_sculpt" not in (insp / "session" / "prompt.txt").read_text()


def test_a_worse_mesh_and_missing_rays_regress(roots, tmp_path):
    bowed = roots["bowed"]
    LF.write_glb(tmp_path / "deleted.glb", "deleted")
    rep = report(bowed, tmp_path / "deleted.glb")
    assert rep["lidar"]["status"] == "regressed"
    assert rep["lidar"]["regions"][0]["coverage"] < 0.98 or rep["lidar"]["regions"][0]["status"] == "regressed"
    clean = roots["clean"]
    LF.write_glb(tmp_path / "bowed.glb", "bowed")
    assert report(clean, tmp_path / "bowed.glb")["lidar"]["regressed"]


@pytest.fixture(scope="module")
def see_root(tmp_path_factory):
    """The clean box whose right panel the laser goes through (as through a wire cage) to a surface SEE_M beyond."""
    return prepare(LF.job(tmp_path_factory.mktemp("see") / "see", "clean", see_through=LF.RIGHT))


def test_a_see_through_panel_may_be_opened(see_root, tmp_path):
    """The host marks the rays through the right panel as see-through. Opening that panel lets them go on to the box's
    inner walls, nearer their returns: no coverage loss, and the region improves. Deleting the top panel, which the
    returns lie on, still regresses. (A miss through the opening: the next test.)"""
    root = see_root
    LF.write_glb(tmp_path / "open.glb", "deleted")
    LF.write_glb(tmp_path / "no_top.glb", "no_top")
    rep = report(root, tmp_path / "open.glb")
    lid = rep["lidar"]
    assert lid["see_through"]["rays"] > 100, lid["see_through"]
    assert lid["status"] == "improved" and not lid["regressed"], [x for x in lid["regions"] + lid["protected"] if x.get("status") == "regressed"
                                  or x.get("regressed")]
    covs = [x["coverage"] for x in lid["regions"] + lid["protected"] if x["coverage"] is not None]
    assert covs and min(covs) >= 1 - G.BASE["coverage_tol"]
    assert not any(f["code"] == "lidar_regression" for f in rep["findings"])
    assert report(root, tmp_path / "no_top.glb")["lidar"]["regressed"]


def test_a_see_through_miss_is_neutral_and_a_solid_miss_counts(see_root, tmp_path, monkeypatch):
    """A candidate that misses every see-through ray keeps their original error and its coverage; one that also loses
    5% of the reserved hits on the top panel, where the returns lie, regresses on coverage."""
    root = see_root
    control, original = root / "control", root / "original.glb"
    ev, key = evidence(root), G.manifest(control)["key"]
    real = G.cached_score
    sc0, _ = real(control, original, ev, key)
    see = sc0["hit"] & (sc0["err"] < -G.BASE["max_displacement_m"])
    _, panel = LF.box_hit(ev["origin"], (ev["end"] - ev["origin"]) / np.linalg.norm(ev["end"] - ev["origin"],
                                                                                     axis=1)[:, None])
    top = np.flatnonzero(sc0["hit"] & ~see & ev["verify"] & (panel == LF.PANELS.index(LF.TOP)))
    assert see.sum() > 100 and len(top) > 100
    some_top = np.zeros(len(see), bool)
    some_top[top[::20]] = True

    def check(lose, name):
        glb = tmp_path / name
        shutil.copyfile(original, glb)

        def fake(control, path, ev, key, mesh=None):
            sc, m = real(control, original, ev, key)
            if Path(path) == glb:
                sc = {k: v.copy() for k, v in sc.items()}
                sc["hit"] &= ~lose
                sc["err"] = np.where(lose, np.nan, sc["err"])
                sc["tri"] = np.where(lose, -1, sc["tri"])
            return sc, m

        monkeypatch.setattr(G, "cached_score", fake)
        return G.evaluate(glb, control, original)

    opened = check(see, "opened.glb")
    assert opened["see_through"]["opened"] == opened["see_through"]["rays"] > 0
    assert not opened["regressed"] and opened["status"] != "regressed"
    for x in opened["regions"] + opened["protected"]:
        assert x["before_m"] == x["after_m"]  # each opened ray keeps the original's error
        assert x["coverage"] in (None, 1.0)
    lost = check(see | some_top, "lost.glb")
    assert lost["regressed"]
    assert min(x["coverage"] for x in lost["protected"] if x["coverage"] is not None) < 1 - G.BASE["coverage_tol"]


def test_a_cached_report_is_bound_to_the_evidence(roots, tmp_path):
    root = roots["bowed"]
    first = report(root, root / "original.glb")
    assert report(root, root / "original.glb")["cached"]
    m = json.loads((root / "control" / "lidar" / "manifest.json").read_text())
    copy = tmp_path / "s"
    shutil.copytree(root, copy)
    m2 = {**m, "key": "another calibration"}
    (copy / "control" / "lidar" / "manifest.json").write_text(json.dumps(m2))
    again = report(copy, copy / "original.glb")
    assert not again["cached"] and again["key"]["lidar"] == "another calibration" and first["key"]["lidar"] == m["key"]


def test_rank_terms_treat_gains_inside_the_tolerance_as_ties():
    rep = lambda g, n=1: {"lidar": {"status": "improved", "resolved": n, "gain_m": g, "settings": {"range_tol_m": 0.002}}}
    assert G.rank_terms(rep(0.0110)) == G.rank_terms(rep(0.0105)) == (1, 5)
    assert G.rank_terms(rep(0.0110)) > G.rank_terms(rep(0.0090))
    assert G.rank_terms(rep(0.001, 2)) > G.rank_terms(rep(0.03, 1)) and G.rank_terms({}) == (0, 0)


def test_prompts_name_the_mode():
    assert prompts.lidar_text(None) == prompts.lidar_text("unavailable") == ""
    assert "lidar_sculpt" in prompts.lidar_text("sculpt") and "sculpt helper is off" in prompts.lidar_text("inspect")
    assert "lidar_sculpt" in prompts.round_prompt(450, "", None, "evidence/round_01.json", "sculpt")


# ---- delivery with the fake agent ----

def tool(*args):
    return {"do": "tool", "args": list(args)}


REVIEW = tool("review")
CHECKPOINT = tool("checkpoint", "--receipt", "{receipt}", "--defect", "none open")


def finish(verdict="complete", reason="L1 flattened to the LiDAR returns (measured)"):
    return tool("finish", "--receipt", "{receipt}", "--verdict", verdict, "--reason", reason)


def use(path):
    return {"do": "use_glb", "path": str(path)}


def run_scripted(J, script, requested="sculpt"):
    root = prepare(J, requested)
    (J / "script.json").write_text(json.dumps(script))
    with tempfile.TemporaryDirectory(prefix="harpor-") as sock:
        env = {**os.environ, "PYTHONPATH": str(REPO), "HARPOR_TOOL_SOCKET": os.path.join(sock, "tools.sock")}
        cmd = [sys.executable, "-m", "harpor.agent.fake_agent", str(J / "script.json")]
        C.run_session(root, cmd, env, blender=FAKE_BLENDER, sock_dir=sock, budget_s=60, **FAST)
    return root, json.loads((root / "result.json").read_text())


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    base = tmp_path_factory.mktemp("sessions")
    LF.ANCHOR[0] = 0
    LF.write_glb(base / "deleted.glb", "deleted")
    LF.write_glb(base / "bowed.glb", "bowed")
    LF.write_glb(base / "tinted_bowed.glb", "bowed", tint=(250, 250, 250))
    LF.write_glb(base / "tinted_clean.glb", "clean", tint=(250, 250, 250))
    sculpt = {"do": "sculpt", "patch": "L1", "op": "plane"}
    plans = {
        "sculpt": ("bowed", [REVIEW, sculpt, REVIEW, CHECKPOINT, finish()]),
        "good_then_bad": ("bowed", [REVIEW, sculpt, REVIEW, CHECKPOINT, use(base / "deleted.glb"), REVIEW, finish()]),
        "only_worse": ("clean", [use(base / "bowed.glb"), REVIEW, finish()]),
        "texture_clean": ("clean", [use(base / "tinted_clean.glb"), REVIEW, finish("complete", "texture (inferred)")]),
        "texture_then_sculpt": ("bowed", {"rounds": [[use(base / "tinted_bowed.glb"), REVIEW, finish()],
                                                     [REVIEW, sculpt, REVIEW, finish()]]}),
        "misregistered": ("misregistered", [use(base / "tinted_clean.glb"), REVIEW, finish("complete", "texture")]),
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HARPOR_FAKE_GLB_SCENE", "1")
        jobs = {name: LF.job(base / name, "clean" if kind == "misregistered" else kind,
                             misregister=2 if kind == "misregistered" else None) for name, (kind, _) in plans.items()}
        with ThreadPoolExecutor(len(plans)) as ex:
            out = list(ex.map(lambda n: run_scripted(jobs[n], plans[n][1]), plans))
    return dict(zip(plans, out))


def test_a_session_delivers_the_sculpted_finish(sessions):
    root, r = sessions["sculpt"]
    assert (r["source"], r["verdict"]) == ("finish", "complete") and r["rounds"][0]["outcome"] == "pass"
    ev = json.loads((root / "control" / "evidence.json").read_text())
    win = ev["lidar"]["winner"]
    assert win["status"] == "improved" and win["resolved"] == 1 and win["sha256"] == r["sha256"]
    assert ev["lidar"]["mode"] == {"requested": "sculpt", "effective": "sculpt"}
    log = (root / "session" / "codex.jsonl").read_text()
    assert "lidar L1 target (host-verifiable)" in log and "now -0.0 mm" in log
    assert (root / "work" / "sculpt").is_dir() and list((root / "work" / "sculpt").glob("*-before.blend"))


def test_a_good_checkpoint_survives_a_later_bad_finish(sessions):
    root, r = sessions["good_then_bad"]
    assert r["source"] == "checkpoint" and r["verdict"] == "needs_review"
    cands = C.stored_candidates(root)
    fin = next(c for c in cands if c["source"] == "finish")
    assert C.eligibility(root, fin) == C.LIDAR_WORSE
    chk = next(c for c in cands if c["source"] == "checkpoint")
    assert r["sha256"] == chk["sha256"] and chk["report"]["lidar"]["status"] == "improved"


def test_a_worse_only_candidate_falls_back_to_the_original(sessions):
    root, r = sessions["only_worse"]
    assert r["verdict"] == "fallback_original" and "LiDAR" in r["reason"] and r["glb"] == "original.glb"


def test_a_texture_only_repair_passes_on_a_correct_box(sessions):
    root, r = sessions["texture_clean"]
    assert (r["source"], r["verdict"]) == ("finish", "complete") and r["rounds"][0]["outcome"] == "pass"


def test_a_texture_only_round_on_the_bowed_box_feeds_a_repair(sessions):
    root, r = sessions["texture_then_sculpt"]
    assert [x["outcome"] for x in r["rounds"]] == ["fail", "pass"]
    assert "lidar_target" in [f["code"] for f in r["rounds"][0]["findings"]]
    assert r["candidate"]["round"] == 2 and r["verdict"] == "complete"
    prompt = (root / "session" / "prompt.txt").read_text()
    assert "lidar_sculpt" in prompt and "lidar_target" in prompt


def test_a_misregistered_view_does_not_block_a_texture_repair(sessions):
    root, r = sessions["misregistered"]
    assert r["verdict"] == "complete"
    ev = json.loads((root / "control" / "evidence.json").read_text())
    assert ev["lidar"]["baseline"]["alignment"]["excluded"] == [2]
    assert any(f["code"] == "lidar_alignment" for f in ev["candidates"][0]["findings"])


# ---- the agent stage's request ----

def test_geometry_request_says_why_it_is_unavailable(jobs, tmp_path):
    J = tmp_path / "job"
    shutil.copytree(jobs["clean"], J, ignore=shutil.ignore_patterns("agent"))
    job = json.loads((J / "job.json").read_text())
    assert agent_step.geometry_request(J, job, "sculpt")["index"]
    assert agent_step.geometry_request(J, job, "off") == {"requested": "off", "index": None, "reason": None}
    assert "LiDAR is off" in agent_step.geometry_request(J, {**job, "lidar": None}, "sculpt")["reason"]
    (J / "run" / "placement.json").write_text(json.dumps({"metric_scale": 1.1}))
    assert "metric scale differs" in agent_step.geometry_request(J, job, "inspect")["reason"]
    (J / "run" / "placement.json").write_text(json.dumps({"metric_scale": 1.0}))
    rec = json.loads((J / "pool" / "lidar.json").read_text())
    (J / "pool" / "lidar.json").write_text(json.dumps({**rec, "geometry": None}))
    assert "before LiDAR geometry" in agent_step.geometry_request(J, job, "inspect")["reason"]
    (J / "pool" / "lidar.json").write_text(json.dumps(rec))
    (J / "lidar" / "geometry" / "index.json").write_text("{}")
    assert "changed since the scale record" in agent_step.geometry_request(J, job, "inspect")["reason"]
    root = J / "agent" / "s"
    C.prepare_session(root, None, mvsam3d.session_views(J / "run"), source="f", metric=True,
                      original=J / "run" / "posed.glb", lidar=agent_step.geometry_request(J, job, "sculpt"))
    m = G.manifest(root / "control")
    assert m["status"] == "unavailable" and m["mode"]["effective"] == "unavailable"
    assert "LiDAR geometry" not in (root / "session" / "prompt.txt").read_text()
