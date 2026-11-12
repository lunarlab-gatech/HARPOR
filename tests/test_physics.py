import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from harpor import frames
from harpor.physics.align import (gltf_matrix, make_opaque, read_glb, up_frame, up_from_normals, up_near, wrap_scene,
                                 write_glb)
from harpor.physics.mass import mass_properties


def box_mesh(size):
    """Closed box centred on the origin: 8 vertices, 12 triangles."""
    c = (np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")).reshape(3, -1).T - 0.5) * size
    f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return c, np.array(f)


def test_flat_up_frame_recovers_a_tilted_slab():
    # A 0.9 x 0.7 x 0.12 m slab, resting flat (z up), seen by a camera pitched 20 deg down and yawed.
    v, f = box_mesh([0.9, 0.7, 0.12])
    w_from_world = Rotation.from_euler("x", 90 + 20, degrees=True).as_matrix() @ Rotation.from_euler(
        "z", 30, degrees=True).as_matrix()
    v_w = v @ w_from_world.T + [0.2, 0.4, 3.0]
    T, eig = up_frame(v_w, f)
    up_w = w_from_world @ [0, 0, 1]
    assert np.allclose(T[2, :3], up_w, atol=1e-9) and up_w[1] < 0  # camera down is +y in W
    p = frames.transform_points(T, v_w)
    assert np.allclose(p.min(0), [-0.45, -0.35, 0]) and np.allclose(p.max(0), [0.45, 0.35, 0.12])
    assert np.isclose(np.linalg.det(T[:3, :3]), 1) and eig[0] < eig[1] < eig[2]


def test_up_from_normals_finds_the_floor_under_an_open_frame():
    # A 1.0 x 0.8 m floor slab under four thin 0.5 m posts: the box is taller than wide in no axis,
    # but the floor's normals give up. Seen by a camera pitched 20 deg down and yawed.
    parts = [box_mesh([1.0, 0.8, 0.01])]
    for sx in (-0.4, 0.4):
        for sy in (-0.3, 0.3):
            v, f = box_mesh([0.03, 0.03, 0.5])
            parts.append((v + [sx, sy, 0.25], f))
    v = np.concatenate([p[0] for p in parts])
    f = np.concatenate([p[1] + 8 * i for i, p in enumerate(parts)])
    w_from_world = Rotation.from_euler("x", 90 + 20, degrees=True).as_matrix() @ Rotation.from_euler(
        "z", 30, degrees=True).as_matrix()
    v_w = v @ w_from_world.T + [0.2, 0.4, 3.0]
    up = up_from_normals(v_w, f)
    assert np.allclose(up, w_from_world @ [0, 0, 1], atol=1e-9)
    T, _ = up_frame(v_w, f, up)
    p = frames.transform_points(T, v_w)
    assert np.isclose(p[:, 2].min(), 0) and np.allclose(p[:, 2].max(), 0.505)
    assert np.allclose(np.abs(T[0, :3] @ w_from_world), [1, 0, 0], atol=1e-6)  # x on the slab's 1.0 m side


def test_footprint_x_puts_the_long_side_on_x():
    # An open 0.9 x 0.7 x 0.5 m frame (corner posts and four top rails): its PCA axis is a diagonal.
    parts = []
    for sx in (-0.45, 0.45):
        for sy in (-0.35, 0.35):
            parts.append((box_mesh([0.04, 0.04, 0.5])[0] + [sx, sy, 0.25], box_mesh([1, 1, 1])[1]))
    parts.append((box_mesh([0.94, 0.04, 0.04])[0] + [0, 0.35, 0.5], box_mesh([1, 1, 1])[1]))
    parts.append((box_mesh([0.94, 0.04, 0.04])[0] + [0, -0.35, 0.5], box_mesh([1, 1, 1])[1]))
    parts.append((box_mesh([0.04, 0.74, 0.04])[0] + [0.45, 0, 0.5], box_mesh([1, 1, 1])[1]))
    parts.append((box_mesh([0.04, 0.74, 0.04])[0] + [-0.45, 0, 0.5], box_mesh([1, 1, 1])[1]))
    v = np.concatenate([p[0] for p in parts])
    f = np.concatenate([p[1] + 8 * i for i, p in enumerate(parts)])
    w_from_world = Rotation.from_euler("x", 90 + 15, degrees=True).as_matrix() @ Rotation.from_euler(
        "z", 40, degrees=True).as_matrix()
    v_w = v @ w_from_world.T
    T, _ = up_frame(v_w, f, w_from_world @ [0, 0, 1], x_rule="footprint")
    p = frames.transform_points(T, v_w)
    assert np.allclose(p.max(0) - p.min(0), [0.94, 0.74, 0.52], atol=1e-9)


def test_gltf_matrix_and_wrap_round_trip(tmp_path):
    T = frames.rigid(Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix(), [0.1, 0.2, 0.3])
    M = np.array(gltf_matrix(T)).reshape(4, 4).T  # column-major in the file
    p_w = np.random.default_rng(0).normal(size=(5, 3))
    q = frames.w_to_gltf(p_w) @ M[:3, :3].T + M[:3, 3]
    assert np.allclose(frames.gltf_to_w(q), frames.transform_points(T, p_w))

    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}], "nodes": [{"mesh": 0}, {"mesh": 1}]}
    write_glb(tmp_path / "a.glb", doc, b"\x01\x02\x03")
    d, b = read_glb(tmp_path / "a.glb")
    assert d == doc and b == b"\x01\x02\x03\0"
    w = wrap_scene(doc, gltf_matrix(T))
    assert w["scenes"][0]["nodes"] == [2] and w["nodes"][2]["children"] == [0, 1] and doc["nodes"][-1] == {"mesh": 1}
    assert (tmp_path / "a.glb").stat().st_size % 4 == 0 and json.loads(json.dumps(w)) == w


def test_up_near_picks_the_long_axis_of_a_tall_glass():
    # A 0.08 x 0.08 x 0.18 m "glass" (closed box, no dominant horizontal face) seen by a pitched camera,
    # with a hint 8 deg off its true up: the thin-axis rule would lay it on its side.
    v, f = box_mesh([0.08, 0.08, 0.18])
    w_from_world = Rotation.from_euler("x", 90 + 20, degrees=True).as_matrix()
    v_w = v @ w_from_world.T
    hint = Rotation.from_euler("x", 90 + 28, degrees=True).as_matrix() @ [0, 0, 1]
    up, axis, angle = up_near(v_w, f, hint)
    assert axis == "pca-long" and np.isclose(angle, 8) and np.allclose(up, w_from_world @ [0, 0, 1])


def test_make_opaque_strips_transmission():
    ext = ["KHR_materials_transmission", "KHR_materials_volume", "KHR_materials_ior", "KHR_materials_specular"]
    doc = {"extensionsUsed": ext, "materials": [
        {"alphaMode": "BLEND", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 0.3]},
         "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1}, "KHR_materials_ior": {"ior": 1.5}}},
        {"extensions": {"KHR_materials_specular": {}}}]}
    o = make_opaque(doc)
    assert o["extensionsUsed"] == ["KHR_materials_specular"] and "extensions" not in o["materials"][0]
    assert "alphaMode" not in o["materials"][0] and o["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"][3] == 1
    assert o["materials"][1]["extensions"] == {"KHR_materials_specular": {}} and doc["materials"][0]["alphaMode"] == "BLEND"


def test_mass_properties_of_a_uniform_box():
    # 4 x 6 x 2 voxels of 0.1 m at 500 kg/m^3: a 0.4 x 0.6 x 0.2 m box of 24 kg.
    h = 0.1
    idx = np.array(np.meshgrid(range(4), range(6), range(2), indexing="ij")).reshape(3, -1).T
    c = (idx + 0.5) * h + [1, 2, 3]
    r = mass_properties(c, np.full(len(c), 500.0), h)
    a, b, d, M = 0.4, 0.6, 0.2, 24.0
    assert np.isclose(r["mass"], M) and np.allclose(r["com"], [1.2, 2.3, 3.1])
    assert np.allclose(r["inertia"], np.diag([M / 12 * (b * b + d * d), M / 12 * (a * a + d * d), M / 12 * (a * a + b * b)]))
    k = np.array([2.0, 1.0, 1.0])  # the same voxels stretched 2x along x: a per-axis voxel edge
    s = mass_properties((c - r["com"]) * k, np.full(len(c), 500.0), h * k)
    assert np.isclose(s["mass"], 2 * M) and np.isclose(s["inertia"][2][2], 2 * M / 12 * (0.8 ** 2 + b * b))


def test_mjcf_round_trip_through_mujoco(tmp_path):
    # A textured 1.2 x 0.8 x 0.5 m box in frame S (bottom at z = 0), saved as a glTF y-up GLB.
    trimesh = pytest.importorskip("trimesh")
    mujoco = pytest.importorskip("mujoco")
    from PIL import Image

    from harpor.physics.mjcf import build, check

    box = trimesh.creation.box([1.2, 0.8, 0.5])
    box.apply_translation([0, 0, 0.25])
    box = box.subdivide()  # some vertices away from the corners, so the UV check has non-trivial values
    uv = np.random.default_rng(0).random((len(box.vertices), 2))
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (16, 16, 3), dtype=np.uint8))
    vis = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=img))
    g = trimesh.Trimesh(frames.w_to_gltf(box.vertices), box.faces, visual=vis, process=False)
    trimesh.Scene(g).export(tmp_path / "box.glb")
    rb = {"mass": 12.0, "com": [0.01, -0.02, 0.25], "fullinertia": [0.9, 1.5, 2.0, 0.05, -0.02, 0.01]}
    (tmp_path / "summary.json").write_text(json.dumps({"config": "test", "rigid_body": rb}))

    report = build(tmp_path / "box.glb", tmp_path / "summary.json", tmp_path / "mjcf", "box", decompose=hull_parts)
    c = check(tmp_path / "mjcf", "box")
    assert np.isclose(c["mass"], 12.0) and np.allclose(c["com"], rb["com"])
    assert np.allclose(c["fullinertia"], rb["fullinertia"], atol=1e-9) and c["ntex"] == 2

    m = mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "scene.xml"))
    # The floor carries the same friction, or MuJoCo's max rule would use the floor's default of 1.
    assert np.allclose(m.geom_friction[m.geom("floor").id], m.geom_friction[m.geom("box_collision_0").id])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    z0 = report["start_z"]
    for name in ("box_visual_0", "box_collision_0"):
        gid = m.geom(name).id
        mid = m.geom_dataid[gid]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = d.geom_xpos[gid] + v @ d.geom_xmat[gid].reshape(3, 3).T
        assert np.allclose(w.min(0), [-0.6, -0.4, z0], atol=1e-5) and np.allclose(w.max(0), [0.6, 0.4, 0.5 + z0], atol=1e-5)
    # MuJoCo's texcoords use a top-left origin, as glTF does: the stored glTF v is 1 - trimesh's v.
    mid = m.geom_dataid[m.geom("box_visual_0").id]
    tc = m.mesh_texcoord[m.mesh_texcoordadr[mid]:m.mesh_texcoordadr[mid] + m.mesh_texcoordnum[mid]]
    want = np.c_[uv[:, 0], 1 - uv[:, 1]]
    assert np.allclose(np.sort(np.round(tc, 5), axis=0), np.sort(np.round(want, 5), axis=0), atol=1e-5)


def test_usd_round_trip(tmp_path):
    # The same textured box as the MJCF test, then the USD from its MJCF folder: mass, COM and the full
    # inertia (through diagonalInertia and principalAxes) must come back, with one convex collider.
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdPhysics

    from harpor.physics.mjcf import build as build_mjcf
    from harpor.physics.usd import build, check

    box = trimesh.creation.box([1.2, 0.8, 0.5])
    box.apply_translation([0, 0, 0.25])
    uv = np.random.default_rng(0).random((len(box.vertices), 2))
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (16, 16, 3), dtype=np.uint8))
    mr = Image.fromarray(np.random.default_rng(2).integers(0, 255, (8, 8, 3), dtype=np.uint8))
    vis = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=img, metallicRoughnessTexture=mr, metallicFactor=0.5))
    trimesh.Scene(trimesh.Trimesh(frames.w_to_gltf(box.vertices), box.faces, visual=vis, process=False)).export(tmp_path / "box.glb")
    rb = {"mass": 12.0, "com": [0.01, -0.02, 0.25], "fullinertia": [0.9, 1.5, 2.0, 0.05, -0.02, 0.01]}
    (tmp_path / "summary.json").write_text(json.dumps({"rigid_body": rb}))

    build_mjcf(tmp_path / "box.glb", tmp_path / "summary.json", tmp_path / "mjcf", "box", decompose=hull_parts)
    build(tmp_path / "box.glb", tmp_path / "summary.json", tmp_path / "mjcf", tmp_path / "usd", "box")
    c = check(tmp_path / "usd", "box")
    assert c["rigid_body"] and c["scene_reference_is_rigid_body"] and c["colliders"] == 1
    assert np.isclose(c["mass"], 12.0) and np.allclose(c["com"], rb["com"], atol=1e-6)
    assert np.allclose(c["fullinertia"], rb["fullinertia"], atol=1e-6)
    assert c["up_axis"] == "Z" and c["meters_per_unit"] == 1 and c["kilograms_per_unit"] == 1
    assert c["textures"] == 2 and c["textures_resolved"]

    stage = Usd.Stage.Open(str(tmp_path / "usd" / "box.usd"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/box/visuals/visual_0"))
    pts = np.array(mesh.GetPointsAttr().Get())
    assert np.allclose(pts.min(0), [-0.6, -0.4, 0], atol=1e-6) and np.allclose(pts.max(0), [0.6, 0.4, 0.5], atol=1e-6)
    assert len(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()) == len(pts)
    assert not mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    col = stage.GetPrimAtPath("/box/collisions/collision_0")
    assert UsdPhysics.MeshCollisionAPI(col).GetApproximationAttr().Get() == "convexHull"
    assert UsdGeom.Imageable(col).GetPurposeAttr().Get() == "guide"
    # metallic and roughness read the metallicRoughness texture's B and G channels, scaled by the factors.
    from pxr import UsdShade
    surf = UsdShade.Shader(stage.GetPrimAtPath("/box/materials/visual_0/PreviewSurface"))
    (src_m, out_m, _), = [s for s in [surf.GetInput("metallic").GetConnectedSource()]]
    (src_r, out_r, _), = [s for s in [surf.GetInput("roughness").GetConnectedSource()]]
    assert (src_m.GetPath().name, out_m, out_r) == ("MetallicRoughness", "b", "g")
    mrs = UsdShade.Shader(stage.GetPrimAtPath("/box/materials/visual_0/MetallicRoughness"))
    assert mrs.GetInput("sourceColorSpace").Get() == "raw" and tuple(mrs.GetInput("scale").Get()) == (1, 1, 0.5, 1)


def test_load_w_needs_to_geometry(tmp_path, monkeypatch):
    """trimesh 5.1.1 in envs/sam3d and envs/mujoco has Scene.to_geometry, so load_w has no fallback to Scene.dump."""
    trimesh = pytest.importorskip("trimesh")
    from harpor.physics.align import load_w

    box = trimesh.creation.box([0.3, 0.2, 0.1])
    trimesh.Scene(trimesh.Trimesh(frames.w_to_gltf(box.vertices), box.faces)).export(tmp_path / "box.glb")
    v, f = load_w(tmp_path / "box.glb")
    assert np.allclose(np.ptp(v, axis=0), [0.3, 0.2, 0.1]) and len(f) == 12
    monkeypatch.delattr(trimesh.Scene, "to_geometry")
    with pytest.raises(AttributeError, match="to_geometry"):
        load_w(tmp_path / "box.glb")


def hull_parts(V, F, threshold):
    trimesh = pytest.importorskip("trimesh")
    h = trimesh.Trimesh(V, F).convex_hull
    return [(h.vertices, h.faces)]



def body_and_lid(tmp_path):
    """A 0.6 x 0.4 x 0.3 m body box with a textured 0.6 x 0.4 x 0.04 m lid on top, as the two glTF mesh nodes body
    and lid of one rigid GLB in frame S (bottom at z = 0). summary.json holds the mass properties of their voxels
    on a 2 cm grid, the body at 600 kg/m^3 and the lid at 300. Returns (glb, summary)."""
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image

    scene, centers, rho = trimesh.Scene(), [], []
    for name, size, z0, density in (("body", [0.6, 0.4, 0.3], 0.0, 600.0), ("lid", [0.6, 0.4, 0.04], 0.3, 300.0)):
        box = trimesh.creation.box(size)
        box.apply_translation([0, 0, z0 + size[2] / 2])
        visual = None
        if name == "lid":
            uv = np.random.default_rng(0).random((len(box.vertices), 2))
            img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (16, 16, 3), dtype=np.uint8))
            visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=img))
        scene.add_geometry(trimesh.Trimesh(frames.w_to_gltf(box.vertices), box.faces, visual=visual, process=False),
                           node_name=name, geom_name=name)
        n = np.round(np.array(size) / 0.02).astype(int)
        idx = np.array(np.meshgrid(*map(range, n), indexing="ij")).reshape(3, -1).T
        c = (idx + 0.5) * 0.02 + [-size[0] / 2, -size[1] / 2, z0]
        centers.append(c)
        rho.append(np.full(len(c), density))
    scene.export(tmp_path / "object.glb")
    rb = mass_properties(np.concatenate(centers), np.concatenate(rho), 0.02)
    (tmp_path / "summary.json").write_text(json.dumps({"config": "test", "rigid_body": rb,
                                                       "median": {"density_kg_m3": 600.0}}))
    return tmp_path / "object.glb", tmp_path / "summary.json"


def test_align_turns_a_posed_object_into_s_and_keeps_its_bytes(tmp_path):
    """The two nodes of body_and_lid posed in W by a camera pitched 20 deg down and yawed: align with an exact up
    hint gives back S (up to the free sign of x), with the bottom at z = 0. The BIN chunk is copied byte for byte,
    the nodes keep their matrices under one new root node, and the lid keeps its texture."""
    trimesh = pytest.importorskip("trimesh")
    from harpor.physics import align
    from harpor.physics.align import load_w

    glb_s, _ = body_and_lid(tmp_path)
    T_ws = frames.rigid(Rotation.from_euler("x", 90 + 20, degrees=True).as_matrix()
                        @ Rotation.from_euler("z", 30, degrees=True).as_matrix(), [0.2, 0.4, 3.0])
    doc, binary = read_glb(glb_s)
    G = np.eye(4)
    G[:3, :3] = frames.GLTF_FROM_W
    doc["nodes"] = [{**n, "matrix": (G @ T_ws @ G.T).T.reshape(-1).tolist()} for n in doc["nodes"]]
    write_glb(tmp_path / "posed.glb", doc, binary)
    up_w = T_ws[:3, :3] @ [0, 0, 1]
    align.main(["--glb", str(tmp_path / "posed.glb"), "--out", str(tmp_path / "aligned.glb"),
                "--up-near=" + ",".join(map(str, up_w))])
    rep = json.loads((tmp_path / "aligned.json").read_text())
    D = np.array(rep["T_s_from_w"]) @ T_ws  # S to S: the identity, or a half turn about z
    assert np.allclose(D[2], [0, 0, 1, 0], atol=1e-9) and np.allclose(D[:3, 3], 0, atol=1e-9)
    assert np.allclose(np.abs(D[:3, :3]), np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(D[:3, :3]), 1)
    assert rep["up_rule"] == "near" and rep["up_near"]["angle_deg"] < 1e-6 and np.allclose(rep["up_in_w"], up_w)
    assert np.isclose(rep["camera_tilt_deg"], 20) and not rep["opaque"]
    assert np.allclose(rep["extent_s"], [0.6, 0.4, 0.34]) and np.allclose(rep["min_s"], [-0.3, -0.2, 0], atol=1e-9)

    out_doc, out_binary = read_glb(tmp_path / "aligned.glb")
    assert out_binary == binary and out_doc["nodes"][:-1] == doc["nodes"]
    assert out_doc["nodes"][-1]["name"] == "gravity_align" and out_doc["scenes"][0]["nodes"] == [len(doc["nodes"])]
    v, _ = load_w(tmp_path / "aligned.glb")
    assert np.allclose([v[:, 2].min(), v[:, 2].max()], [0, 0.34], atol=1e-9)
    lid_visual = trimesh.load(tmp_path / "aligned.glb", force="scene").geometry["lid"].visual
    assert len(lid_visual.uv) == 8 and lid_visual.material.baseColorTexture.size == (16, 16)


def test_a_model_without_textures_writes_no_png(tmp_path):
    """build(..., textures=False) writes no PNG and no UVs, and its model has no texture; the inertial is still the
    summary's."""
    mujoco = pytest.importorskip("mujoco")
    from harpor.physics.mjcf import build, check

    glb, summary = body_and_lid(tmp_path)
    rb = json.loads(summary.read_text())["rigid_body"]
    report = build(glb, summary, tmp_path / "mjcf", "chest", decompose=hull_parts, textures=False)
    assets = tmp_path / "mjcf" / "assets"
    assert not list(assets.glob("*.png")) and all(v["texture"] is None for v in report["visuals"])
    assert len(report["visuals"]) == 2 and not any(
        line.startswith("vt ") for v in report["visuals"] for line in (assets / f"{v['mesh']}.obj").read_text().splitlines())
    assert mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "chest.xml")).ntex == 0  # scene.xml has the floor's
    assert np.isclose(check(tmp_path / "mjcf", "chest")["mass"], rb["mass"])


def test_mjcf_command_line_compiles_the_model_and_refuses_a_bad_name(tmp_path, monkeypatch):
    """main() writes mjcf.json with the compiled model: one body on a free joint with the summary's inertial, both
    visuals and the lid's texture (and the floor's). A --name that is not a plain identifier, or is 'scene', stops it."""
    pytest.importorskip("trimesh")
    pytest.importorskip("mujoco")
    from harpor.physics import mjcf

    glb, summary = body_and_lid(tmp_path)
    rb = json.loads(summary.read_text())["rigid_body"]
    build = mjcf.build
    monkeypatch.setattr(mjcf, "build", lambda *a, **k: build(*a, decompose=hull_parts, **k))  # no CoACD here
    argv = ["mjcf", "--glb", str(glb), "--summary", str(summary), "--out", str(tmp_path / "out")]
    monkeypatch.setattr("sys.argv", argv + ["--name", "chest"])
    mjcf.main()
    report = json.loads((tmp_path / "out" / "mjcf.json").read_text())
    c = report["compiled"]
    assert (c["nbody"], c["njnt"], c["nmesh"], c["ntex"]) == (2, 1, 3, 2) and report["collision_hulls"] == 1
    assert np.isclose(c["mass"], rb["mass"]) and np.allclose(c["com"], rb["com"], atol=1e-6)
    assert np.allclose(c["fullinertia"], rb["fullinertia"], rtol=1e-5, atol=1e-9)
    assert sorted(v["name"] for v in report["visuals"]) == ["body", "lid"] and report["inertial"]["mass"] == rb["mass"]
    for bad in ("scene", "1chest", "a b"):
        monkeypatch.setattr("sys.argv", argv + ["--name", bad])
        with pytest.raises(SystemExit):
            mjcf.main()


def test_vomp_run_glb_gives_each_occupied_voxel_its_nearest_prediction(tmp_path, monkeypatch):
    """run_glb gives every occupied voxel the material of its nearest VoMP prediction and sums those voxels into
    the rigid body, with the voxel edge h = transform_scale / 64. VoMP and torch are stand-ins: the voxeliser
    returns normalized coordinates, which denormalize_coords scales back. No occupied voxel is an error."""
    import sys
    import types

    from harpor.physics.vomp_run import run_glb

    center, scale = np.array([0.1, -0.03, 0.02]), 0.64
    h = scale / 64
    idx = np.array(np.meshgrid(range(10), range(6), range(4), indexing="ij")).reshape(3, -1).T
    occupied = (idx + 0.5) * h + center
    pred = occupied[::2] + [0, 0, 0.2 * h]  # each z pair (0, 1), (2, 3) shares the prediction just above its low voxel
    rng = np.random.default_rng(4)
    E, nu, rho = rng.uniform(1e8, 1e10, len(pred)), rng.uniform(0.2, 0.4, len(pred)), rng.uniform(200, 900, len(pred))

    class Model:
        voxels = (occupied - center) / scale

        def get_mesh_materials(self, glb, output_dir, query_points, return_original_scale):
            assert query_points == "voxel_centers" and return_original_scale and Path(output_dir).is_dir()
            return {"transform_center": center.tolist(), "transform_scale": scale, "voxel_coords_world": pred,
                    "youngs_modulus": E, "poisson_ratio": nu, "density": rho}

        def _voxelize_mesh(self, glb, work):
            return self.voxels

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    utils = types.ModuleType("vomp.inference.utils")
    utils.denormalize_coords = lambda c, ctr, s: np.asarray(c) * s + ctr
    utils.save_materials = lambda r, path: Path(path).write_bytes(b"materials")
    for name, module in (("torch", torch), ("vomp", types.ModuleType("vomp")),
                         ("vomp.inference", types.ModuleType("vomp.inference")), ("vomp.inference.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)

    s = run_glb(Model(), tmp_path / "in.glb", tmp_path / "run", "test.json", load_s=3.0)
    want = np.repeat(rho, 2)
    with np.load(tmp_path / "run" / "voxels.npz") as v:
        assert np.allclose(v["centers"], occupied) and np.allclose(v["density"], want) and float(v["h"]) == h
        assert np.allclose(v["youngs_modulus"], np.repeat(E, 2)) and np.allclose(v["poisson_ratio"], np.repeat(nu, 2))
    assert (s["voxels_occupied"], s["voxels_predicted"], s["voxel_edge_m"]) == (240, 120, h)
    assert np.isclose(s["nearest_prediction_max_dist_voxels"], 0.8)
    assert s["rigid_body"] == mass_properties(occupied, want, h) and s["config"] == "test.json"
    assert s["median"]["density_kg_m3"] == np.median(rho) and s["p05_p95"]["density_kg_m3"] == np.percentile(rho, [5, 95]).tolist()
    assert s["seconds"]["load"] == 3.0 and set(s["seconds"]) == {"load", "predict"} and s["gpu_peak_gb"] is None
    assert json.loads((tmp_path / "run" / "summary.json").read_text()) == s
    assert (tmp_path / "run" / "materials.npz").read_bytes() == b"materials"

    Model.voxels = np.zeros((0, 3))
    with pytest.raises(ValueError, match="gives no occupied voxel"):
        run_glb(Model(), tmp_path / "in.glb", tmp_path / "empty", "test.json")
