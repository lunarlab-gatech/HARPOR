import json

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


def fake_vomp_link(d, centers, density, h):
    """A VoMP-like run of one link in folder d: voxels.npz and summary.json at a uniform density."""
    d.mkdir(parents=True)
    n = len(centers)
    np.savez(d / "voxels.npz", centers=centers, youngs_modulus=np.full(n, 1e9), poisson_ratio=np.full(n, 0.3),
             density=np.full(n, density), h=h)
    stats = {"youngs_modulus_pa": 1e9, "poisson_ratio": 0.3, "density_kg_m3": density}
    (d / "summary.json").write_text(json.dumps({
        "voxels_occupied": n, "voxels_predicted": n, "nearest_prediction_max_dist_voxels": 0.0,
        "voxel_edge_m": h, "median": stats, "p05_p95": {k: [v, v] for k, v in stats.items()},
        "rigid_body": mass_properties(centers, np.full(n, density), h), "seconds": {"predict": 1.0}}))


def two_box_hinge(tmp_path, rho=(600.0, 300.0)):
    """A 0.6 x 0.4 x 0.3 m body box with a textured 0.6 x 0.4 x 0.04 m lid on top, as the glTF nodes body
    and lid of parts.glb in frame S. The lid is hinged on the body's back top edge (y = 0.2, z = 0.3) about
    -x, so that it opens upward, from 0 to 1.5 rad. Each box gets a VoMP-like run on a 2 cm voxel grid at
    density rho in physics/vomp/<box>/, combined into physics/summary.json by vomp_run.combine_links.
    Returns (glb, summary, hinge, {box: voxel centres})."""
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image

    from harpor.parts.joints import write_joints
    from harpor.physics.vomp_run import combine_links

    hinge = {"version": 1, "frame": "S", "parent": "body", "type": "revolute", "origin": [0.0, 0.2, 0.3],
             "axis": [-1.0, 0.0, 0.0], "lower": 0.0, "upper": 1.5}
    scene, centers = trimesh.Scene(), {}
    for name, size, z0, density in (("body", [0.6, 0.4, 0.3], 0.0, rho[0]), ("lid", [0.6, 0.4, 0.04], 0.3, rho[1])):
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
        centers[name] = c
        fake_vomp_link(tmp_path / "physics" / "vomp" / name, c, density, 0.02)
    scene.export(tmp_path / "parts.glb")
    doc, binary = read_glb(tmp_path / "parts.glb")
    write_glb(tmp_path / "parts.glb", write_joints(doc, {"lid": hinge}), binary)
    combine_links(tmp_path / "physics", ["body", "lid"], tmp_path / "parts.glb", "test", 2.0)
    return tmp_path / "parts.glb", tmp_path / "physics" / "summary.json", hinge, centers


def fix_to_world(glb):
    """Add a fixed joint from the world to the body link of glb, as two_lid_chest has."""
    from harpor.parts.joints import read_joints, write_joints

    doc, binary = read_glb(glb)
    joints = {**read_joints(doc), "body": {"version": 1, "frame": "S", "parent": "world", "type": "fixed"}}
    write_glb(glb, write_joints(doc, joints), binary)


def hull_parts(V, F, threshold):
    trimesh = pytest.importorskip("trimesh")
    h = trimesh.Trimesh(V, F).convex_hull
    return [(h.vertices, h.faces)]


def inflated_hull(V, F, threshold):
    """hull_parts grown 2% about its centre, as CoACD's slightly inflated hulls: neighbours overlap at rest."""
    ((v, f),) = hull_parts(V, F, threshold)
    c = v.mean(0)
    return [(c + 1.02 * (v - c), f)]


def two_lid_chest(tmp_path):
    """The body box of two_box_hinge fixed to the world, with two half lids side by side on top (lid_left at
    x < 0, lid_right at x > 0), each hinged on the back top edge as there, and a knob node parented to
    lid_left's node without a joint. summary.json has no links, so every link gets a hull inertial at
    500 kg/m^3. Returns (glb, summary)."""
    trimesh = pytest.importorskip("trimesh")
    from harpor.parts.joints import write_joints

    scene = trimesh.Scene()
    for name, size, center, parent in (("body", [0.6, 0.4, 0.3], [0, 0, 0.15], None),
                                       ("lid_left", [0.3, 0.4, 0.04], [-0.15, 0, 0.32], None),
                                       ("lid_right", [0.3, 0.4, 0.04], [0.15, 0, 0.32], None),
                                       ("knob", [0.04, 0.04, 0.04], [-0.15, -0.15, 0.36], "lid_left")):
        box = trimesh.creation.box(size)
        box.apply_translation(center)
        scene.add_geometry(trimesh.Trimesh(frames.w_to_gltf(box.vertices), box.faces, process=False),
                           node_name=name, geom_name=name, parent_node_name=parent)
    scene.export(tmp_path / "parts.glb")
    hinge = {"version": 1, "frame": "S", "parent": "body", "type": "revolute", "axis": [-1.0, 0.0, 0.0],
             "lower": 0.0, "upper": 1.5}
    joints = {"body": {"version": 1, "frame": "S", "parent": "world", "type": "fixed"},
              "lid_left": {**hinge, "origin": [-0.15, 0.2, 0.3]}, "lid_right": {**hinge, "origin": [0.15, 0.2, 0.3]}}
    doc, binary = read_glb(tmp_path / "parts.glb")
    write_glb(tmp_path / "parts.glb", write_joints(doc, joints), binary)
    (tmp_path / "summary.json").write_text(json.dumps({"median": {"density_kg_m3": 500.0}}))
    return tmp_path / "parts.glb", tmp_path / "summary.json"


def test_combine_moves_each_inertia_to_the_common_com():
    # Two voxel boxes of different density: their combined body equals the mass properties of the union.
    from harpor.physics.mass import combine

    rng = np.random.default_rng(3)
    a = (np.array(np.meshgrid(range(5), range(3), range(4), indexing="ij")).reshape(3, -1).T + 0.5) * 0.1
    b = (np.array(np.meshgrid(range(2), range(6), range(2), indexing="ij")).reshape(3, -1).T + 0.5) * 0.1 + [0.7, -0.2, 0.3]
    ra, rb = rng.uniform(200, 900, len(a)), rng.uniform(200, 900, len(b))
    c = combine([mass_properties(a, ra, 0.1), mass_properties(b, rb, 0.1)])
    u = mass_properties(np.concatenate([a, b]), np.concatenate([ra, rb]), 0.1)
    for k in ("mass", "volume", "com", "inertia", "fullinertia"):
        assert np.allclose(c[k], u[k]), k
    # A body given by fullinertia only (as mjcf.json stores it) combines the same way.
    slim = {k: mass_properties(b, rb, 0.1)[k] for k in ("mass", "com", "fullinertia")}
    assert np.allclose(combine([mass_properties(a, ra, 0.1), slim])["inertia"], u["inertia"])


def test_vomp_combine_links_sums_the_links(tmp_path):
    glb, summary, _, centers = two_box_hinge(tmp_path)
    s = json.loads(summary.read_text())
    union = mass_properties(np.concatenate([centers["body"], centers["lid"]]),
                            np.r_[np.full(len(centers["body"]), 600.0), np.full(len(centers["lid"]), 300.0)], 0.02)
    assert np.isclose(s["rigid_body"]["mass"], union["mass"]) and np.allclose(s["rigid_body"]["inertia"], union["inertia"])
    assert s["voxels_occupied"] == 30 * 20 * 15 + 30 * 20 * 2 and s["median"]["density_kg_m3"] == 600.0
    assert s["p05_p95"]["density_kg_m3"] == [300.0, 600.0] and s["seconds"] == {"load": 2.0, "predict": 2.0}
    assert set(s["links"]) == {"body", "lid"} and s["links_skipped"] == {}
    assert np.isclose(s["links"]["lid"]["rigid_body"]["mass"], 0.6 * 0.4 * 0.04 * 300)


def test_align_carries_the_joints_to_s_and_writes_one_glb_per_link(tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    from harpor.parts import joints as J
    from harpor.physics import align
    from harpor.physics.align import load_w

    glb_s, _, hinge, _ = two_box_hinge(tmp_path)
    # The same parts posed in W by a camera pitched 20 deg down and yawed, with the joint in W.
    T_ws = frames.rigid(Rotation.from_euler("x", 90 + 20, degrees=True).as_matrix()
                        @ Rotation.from_euler("z", 30, degrees=True).as_matrix(), [0.2, 0.4, 3.0])
    doc, binary = read_glb(glb_s)
    G = np.eye(4)
    G[:3, :3] = frames.GLTF_FROM_W
    doc["nodes"] = [{**n, "matrix": (G @ T_ws @ G.T).T.reshape(-1).tolist()} for n in doc["nodes"]]
    doc = J.write_joints(doc, J.transform({"lid": hinge}, T_ws, "W"))
    write_glb(tmp_path / "posed.glb", doc, binary)
    up_w = T_ws[:3, :3] @ [0, 0, 1]
    monkeypatch.setattr("sys.argv", ["align", "--glb", str(tmp_path / "posed.glb"), "--out", str(tmp_path / "aligned.glb"),
                                     "--up-near=" + ",".join(map(str, up_w)), "--links-out", str(tmp_path / "links")])
    (tmp_path / "links").mkdir()
    (tmp_path / "links" / "stale.glb").write_bytes(b"old")
    align.main()
    rep = json.loads((tmp_path / "aligned.json").read_text())
    T = np.array(rep["T_s_from_w"])
    lid = rep["joints"]["lid"]
    assert rep["links"] == ["body", "lid"] and rep["fixed_base"] is False and lid["frame"] == "S"
    assert lid == J.transform(J.read_joints(doc), T)["lid"] == J.load(tmp_path / "aligned.glb")[0]["lid"]
    assert np.isclose(lid["origin"][2], 0.3) and np.isclose(lid["axis"][2], 0, atol=1e-9)  # on the body's top edge
    assert np.isclose(abs(lid["origin"][0] if abs(lid["axis"][1]) > 0.5 else lid["origin"][1]), 0.2)
    assert sorted(p.name for p in (tmp_path / "links").iterdir()) == ["body.glb", "lid.glb"]
    for name, (lo, hi) in (("body", (0.0, 0.3)), ("lid", (0.3, 0.34))):
        scene = trimesh.load(tmp_path / "links" / f"{name}.glb", force="scene")
        assert list(scene.geometry) == [name]
        v, _ = load_w(tmp_path / "links" / f"{name}.glb")
        assert np.allclose([v[:, 2].min(), v[:, 2].max()], [lo, hi], atol=1e-9)
    lid_visual = trimesh.load(tmp_path / "links" / "lid.glb", force="scene").geometry["lid"].visual
    assert len(lid_visual.uv) == 8 and lid_visual.material.baseColorTexture.size == (16, 16)  # the texture stays

    # A rigid GLB: the report says so, and its node names need not be valid links.
    monkeypatch.setattr("sys.argv", ["align", "--glb", str(tmp_path / "links" / "body.glb"), "--out",
                                     str(tmp_path / "rigid.glb")])
    align.main()
    rep = json.loads((tmp_path / "rigid.json").read_text())
    assert (rep["joints"], rep["links"], rep["fixed_base"]) == ({}, [], False)


def test_articulated_mjcf_nests_the_lid_on_a_hinge(tmp_path):
    pytest.importorskip("trimesh")
    mujoco = pytest.importorskip("mujoco")
    from harpor.physics.mjcf import build, check

    glb, summary, hinge, _ = two_box_hinge(tmp_path)
    s = json.loads(summary.read_text())
    report = build(glb, summary, tmp_path / "mjcf", "chest", decompose=hull_parts)
    c = check(tmp_path / "mjcf", "chest")
    assert c["nbody"] == 3 and c["njnt"] == 2 and report["root"] == "body" and not report["fixed_base"]
    assert np.isclose(c["bodies"]["chest"], s["links"]["body"]["rigid_body"]["mass"])
    assert np.isclose(c["bodies"]["chest_lid"], s["links"]["lid"]["rigid_body"]["mass"])
    assert np.isclose(c["mass"], s["links"]["body"]["rigid_body"]["mass"])
    assert np.allclose(c["fullinertia"], s["links"]["body"]["rigid_body"]["fullinertia"], atol=1e-9)
    assert c["joints"] == [{"name": "chest_lid_joint", "type": "hinge", "body": "chest_lid", "range": [0.0, 1.5]}]
    lid_visuals = [v["mesh"] for v in report["visuals"] if v["name"] == "lid"]
    assert report["links"]["lid"]["source"] == "vomp" and report["links"]["lid"]["visuals"] == lid_visuals
    assert report["joints"] == [{"child": "lid", "parent": "body", "type": "revolute", "pos": hinge["origin"],
                                 "axis": hinge["axis"], "range": [0.0, 1.5]}]
    assert np.isclose(report["inertial"]["mass"], s["rigid_body"]["mass"])
    # At the upper limit the lid's front edge has swung up about the back edge.
    (p,) = c["probe"]
    z0 = report["start_z"]
    assert np.isclose(p["rest"][1][2], 0.34 + z0, atol=1e-6)
    lift = 0.3 + 0.4 * np.sin(1.5) + 0.04 * np.cos(1.5) + z0
    assert np.isclose(p["upper"][1][2], lift, atol=1e-6) and p["upper"][0][1] > 0.15

    # The textured lid keeps its texture, and the hinge sits at the joint's origin in the lid's frame.
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "scene.xml"))
    assert m.ntex == 2 and np.allclose(m.jnt_pos[m.joint("chest_lid_joint").id], hinge["origin"])
    assert np.allclose(m.jnt_axis[m.joint("chest_lid_joint").id], hinge["axis"])
    assert (tmp_path / "mjcf" / "assets" / "collision_lid_0.obj").is_file()

    # --fixed-base keeps only the hinge.
    build(glb, summary, tmp_path / "flag", "chest", decompose=hull_parts, fixed_base=True)
    c = check(tmp_path / "flag", "chest")
    assert c["njnt"] == 1 and json.loads((tmp_path / "flag" / "mjcf.json").read_text())["fixed_base"]

    # A world joint on the root keeps only the hinge.
    fix_to_world(glb)
    build(glb, summary, tmp_path / "fixed", "chest", decompose=hull_parts)
    c = check(tmp_path / "fixed", "chest")
    assert c["njnt"] == 1 and json.loads((tmp_path / "fixed" / "mjcf.json").read_text())["fixed_base"]


@pytest.mark.parametrize("flag, njnt", [([], 2), (["--fixed-base"], 1)])
def test_mjcf_command_line_fixes_the_base_only_with_the_flag(tmp_path, monkeypatch, flag, njnt):
    pytest.importorskip("trimesh")
    pytest.importorskip("mujoco")
    from harpor.physics import mjcf

    glb, summary, _, _ = two_box_hinge(tmp_path)
    build = mjcf.build
    monkeypatch.setattr(mjcf, "build", lambda *a, **k: build(*a, decompose=hull_parts, **k))  # no CoACD here
    monkeypatch.setattr("sys.argv", ["mjcf", "--glb", str(glb), "--summary", str(summary), "--out",
                                     str(tmp_path / "out"), "--name", "chest", *flag])
    mjcf.main()
    report = json.loads((tmp_path / "out" / "mjcf.json").read_text())
    assert report["compiled"]["njnt"] == njnt and report["fixed_base"] is bool(flag)


def test_a_link_without_vomp_voxels_gets_a_hull_inertial(tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("mujoco")
    from harpor.physics.mjcf import build, check

    glb, summary, _, _ = two_box_hinge(tmp_path)
    s = json.loads(summary.read_text())
    del s["links"]["lid"]
    summary.write_text(json.dumps(s))
    report = build(glb, summary, tmp_path / "mjcf", "chest", decompose=hull_parts)
    lid = report["links"]["lid"]
    assert lid["source"] == "hull" and any("lid has no VoMP result" in w for w in report["warnings"])
    assert np.isclose(lid["inertial"]["mass"], 0.6 * 0.4 * 0.04 * 600.0)  # the median density of summary.json
    assert np.allclose(lid["inertial"]["com"], [0, 0, 0.32])
    I = 0.6 * 0.4 * 0.04 * 600.0 / 12 * np.array([0.4 ** 2 + 0.04 ** 2, 0.6 ** 2 + 0.04 ** 2, 0.6 ** 2 + 0.4 ** 2])
    assert np.allclose(lid["inertial"]["fullinertia"][:3], I) and np.allclose(lid["inertial"]["fullinertia"][3:], 0)
    assert np.isclose(check(tmp_path / "mjcf", "chest")["bodies"]["chest_lid"], lid["inertial"]["mass"])


def test_articulated_usd_has_links_and_a_revolute_joint(tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    from harpor.physics.mjcf import build as build_mjcf
    from harpor.physics.usd import build, check

    glb, summary, hinge, _ = two_box_hinge(tmp_path)
    s = json.loads(summary.read_text())
    build_mjcf(glb, summary, tmp_path / "mjcf", "chest", decompose=hull_parts)
    build(glb, summary, tmp_path / "mjcf", tmp_path / "usd", "chest")
    c = check(tmp_path / "usd", "chest")
    assert c["articulation_root"] and c["scene_reference_is_articulation_root"] and set(c["links"]) == {"body", "lid"}
    for k in ("body", "lid"):
        assert np.isclose(c["links"][k]["mass"], s["links"][k]["rigid_body"]["mass"])
        assert np.allclose(c["links"][k]["fullinertia"], s["links"][k]["rigid_body"]["fullinertia"], atol=1e-6)
    (j,) = c["joints"]
    assert j["type"] == "PhysicsRevoluteJoint" and j["body0"] == ["/chest/links/body"] and j["body1"] == ["/chest/links/lid"]
    assert np.allclose(j["axis"], hinge["axis"], atol=1e-6) and np.allclose(j["pos0"], hinge["origin"])
    assert np.isclose(j["lower"], 0) and np.isclose(j["upper"], np.degrees(1.5), atol=1e-4)
    assert c["colliders"] == 2 and c["textures"] == 1 and c["textures_resolved"]
    stage = Usd.Stage.Open(str(tmp_path / "usd" / "chest.usd"))
    assert not stage.GetPrimAtPath("/chest").HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/chest/links/lid/collisions/collision_0").HasAPI(UsdPhysics.CollisionAPI)


def test_vomp_combine_links_weights_each_voxel_by_its_volume(tmp_path):
    # A wooden body on a 10 cm grid and a steel handle on a 1 mm grid give as many voxels each, but the
    # handle is a millionth of the volume: the object's median density is the wood's.
    from harpor.physics.vomp_run import combine_links

    grid = np.array(np.meshgrid(range(10), range(10), range(10), indexing="ij")).reshape(3, -1).T + 0.5
    fake_vomp_link(tmp_path / "vomp" / "body", grid * 0.1, 600.0, 0.1)
    fake_vomp_link(tmp_path / "vomp" / "handle", grid * 0.001 + [1.0, 0, 0], 7800.0, 0.001)
    s = combine_links(tmp_path, ["body", "handle"], tmp_path / "parts.glb", "test", 1.0, {"lid": "ValueError: no voxel"})
    assert s["median"]["density_kg_m3"] == 600.0 and s["p05_p95"]["density_kg_m3"] == [600.0, 600.0]
    assert s["voxels_occupied"] == 2000 and s["voxel_edge_m"] == 0.1
    assert s["rigid_body_covers"] == ["body", "handle"] and s["links_skipped"] == {"lid": "ValueError: no voxel"}

    from harpor.physics.vomp_run import weighted_percentile

    v = np.array([3.0, 1.0, 2.0, 4.0])
    assert weighted_percentile(v, np.ones(4), [0, 25, 50, 75, 100]).tolist() == [1, 1, 2, 3, 4]
    assert weighted_percentile(v, np.ones(4), 50) == np.percentile(v, 50, method="inverted_cdf")
    assert weighted_percentile(v, [1, 1, 1, 7], 50) == 4.0


def test_fixed_base_links_do_not_collide_with_their_parent_or_touching_neighbours(tmp_path):
    pytest.importorskip("trimesh")
    mujoco = pytest.importorskip("mujoco")
    import re

    from harpor.parts import joints as J
    from harpor.physics.mjcf import _body_points, build, check, rest_contacts

    glb, summary = two_lid_chest(tmp_path)
    report = build(glb, summary, tmp_path / "mjcf", "chest", decompose=inflated_hull)
    assert report["fixed_base"] and report["root"] == "body"
    assert report["loose"] == ["knob"] and report["welded"] == {"knob": "lid_left"}
    # Each lid and its parent, the knob (on lid_left) and the body, and the two lids, which overlap at rest.
    assert sorted(map(sorted, report["excluded_pairs"])) == [["body", "knob"], ["body", "lid_left"],
                                                             ["body", "lid_right"], ["lid_left", "lid_right"]]
    assert any("touch at rest" in w and "lid_left" in w and "lid_right" in w for w in report["warnings"])
    assert any("knob has no joint, so it is welded to lid_left" in w for w in report["warnings"])
    c = check(tmp_path / "mjcf", "chest")
    assert c["rest_contacts"] == [] and c["njnt"] == 2 and len(c["probe"]) == 2

    m = mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "scene.xml"))
    assert m.body_parentid[m.body("chest_knob").id] == m.body("chest_lid_left").id
    # Without the excludes the fixed body would push both lids, and the lids each other, at rest.
    xml = (tmp_path / "mjcf" / "chest.xml").read_text()
    (tmp_path / "mjcf" / "bare.xml").write_text(re.sub(r"  <contact>.*</contact>\n", "", xml, flags=re.S))
    bare = rest_contacts(mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "bare.xml")))
    assert ["chest", "chest_lid_left"] in bare and ["chest_lid_left", "chest_lid_right"] in bare
    # The lids stay shut for a second, and opening lid_left lifts the knob with it.
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 500)
    assert np.abs(d.qpos).max() < 1e-3
    d.qpos[:] = 0
    mujoco.mj_kinematics(m, d)
    rest = _body_points(m, d, m.body("chest_knob").id)[:, 2].max()
    d.qpos[m.jnt_qposadr[m.joint("chest_lid_left_joint").id]] = 1.5
    mujoco.mj_kinematics(m, d)
    assert _body_points(m, d, m.body("chest_knob").id)[:, 2].max() > rest + 0.2

    # Joints still in frame W (align skipped) are refused.
    doc, binary = read_glb(glb)
    write_glb(tmp_path / "w.glb", J.write_joints(doc, {k: {**j, "frame": "W"} for k, j in J.read_joints(doc).items()}),
              binary)
    with pytest.raises(ValueError, match="must be in frame S"):
        build(tmp_path / "w.glb", summary, tmp_path / "w", "chest", decompose=inflated_hull)


def test_each_hinge_gets_a_position_actuator_scaled_by_its_inertia(tmp_path):
    """kp = w^2 I, kv = 2 w I and joint damping 0.2 kv, with I the inertia about the hinge's axis line of the links
    that move with it: lid_left and the knob welded to it, but lid_right alone."""
    mujoco = pytest.importorskip("mujoco")
    from harpor.physics.mjcf import OMEGA, axis_inertia, build

    glb, summary = two_lid_chest(tmp_path)
    report = build(glb, summary, tmp_path / "mjcf", "chest", decompose=hull_parts)
    # A half lid is 0.3 x 0.4 x 0.04 m at 500 kg/m^3 with its centre 0.2 m in front of the hinge line and 0.02 m above
    # it; the knob is a 4 cm cube 0.35 m in front and 0.06 m above.
    lid, knob = 0.3 * 0.4 * 0.04 * 500, 0.04 ** 3 * 500
    I_lid = lid / 12 * (0.4 ** 2 + 0.04 ** 2) + lid * (0.2 ** 2 + 0.02 ** 2)
    I_knob = knob / 12 * 2 * 0.04 ** 2 + knob * (0.35 ** 2 + 0.06 ** 2)
    a = report["actuators"]
    assert list(a) == ["lid_left", "lid_right"] and a["lid_left"]["name"] == "chest_lid_left_joint"
    assert np.isclose(a["lid_left"]["inertia"], I_lid + I_knob) and np.isclose(a["lid_right"]["inertia"], I_lid)
    I = a["lid_left"]["inertia"]
    assert np.allclose([a["lid_left"][k] for k in ("kp", "kv", "damping")], [OMEGA ** 2 * I, 2 * OMEGA * I, 0.4 * OMEGA * I])
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "scene.xml"))
    k, j = m.actuator("chest_lid_left_joint").id, m.joint("chest_lid_left_joint").id
    assert m.actuator_trnid[k][0] == j and m.actuator_ctrllimited[k] and np.allclose(m.actuator_ctrlrange[k], [0, 1.5])
    assert np.isclose(m.actuator_gainprm[k][0], a["lid_left"]["kp"])
    assert np.allclose(m.actuator_biasprm[k][:3], [0, -a["lid_left"]["kp"], -a["lid_left"]["kv"]], rtol=1e-5)
    assert np.isclose(m.dof_damping[m.jnt_dofadr[j]], a["lid_left"]["damping"], rtol=1e-5)
    # A slide's gains take the mass of what moves with it.
    bodies = [{"mass": 2.0, "com": [1.0, 0, 0], "fullinertia": [1, 1, 1, 0, 0, 0]},
              {"mass": 3.0, "com": [0, 1.0, 0], "fullinertia": [1, 1, 1, 0, 0, 0]}]
    assert axis_inertia({"type": "prismatic", "origin": [0, 0, 0], "axis": [0, 0, 1.0]}, bodies) == 5.0


def test_a_model_without_textures_writes_no_png(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from harpor.physics.mjcf import build

    glb, summary, _, _ = two_box_hinge(tmp_path)
    report = build(glb, summary, tmp_path / "mjcf", "chest", decompose=hull_parts, textures=False)
    assert not list((tmp_path / "mjcf" / "assets").glob("*.png")) and all(v["texture"] is None for v in report["visuals"])
    assert mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "chest.xml")).ntex == 0  # scene.xml has the floor's


FLEX_SCENE = """<mujoco model="pad scene">
  <compiler angle="radian"/>
  <option integrator="discrete" solver="CG"/>
  <asset>
    <mesh name="block" vertex="-.2 -.2 0  .2 -.2 0  .2 .2 0  -.2 .2 0  0 0 .1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <body name="pad" pos="0 0 0.5">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
      <geom type="mesh" mesh="block"/>
      <body name="pad_seat">
        <joint name="pad_seat_joint" type="hinge" axis="0 0 1" range="-1 1" limited="true"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
        <geom type="mesh" mesh="block"/>
        <flexcomp name="cushion" type="direct" dim="3" mass="0.5" radius="0.002" dof="full"
                  point="0 0 0.1  0.2 0 0.1  0 0.2 0.1  0 0 0.3  0.2 0.2 0.3" element="0 1 2 3  1 2 3 4">
          <contact selfcollide="none"/>
          <edge damping="0.01"/>
          <elasticity young="30000" poisson="0.3"/>
          <pin id="0 1 2"/>
        </flexcomp>
      </body>
    </body>
    <body name="ball" pos="0.1 0.1 0.72">
      <freejoint/>
      <geom type="sphere" size="0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_check_probe_and_rest_contacts_leave_a_flex_out(tmp_path):
    """A flex adds a body and 3 slide joints per free node, and its contacts have geom -1. Pinned nodes
    name the seat, which keeps its hinge. A ball resting on the flex touches no other body of the model."""
    mujoco = pytest.importorskip("mujoco")
    from harpor.physics.mjcf import check, node_bodies, probe, rest_contacts

    (tmp_path / "scene.xml").write_text(FLEX_SCENE)
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    nodes = {m.body("cushion_3").id, m.body("cushion_4").id}
    assert node_bodies(m) == nodes and (m.nbody, m.njnt, m.flex_nodebodyid.size) == (6, 9, 0)
    assert m.body("pad_seat").id in m.flex_vertbodyid
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert any(c.geom2 < 0 for c in d.contact[:d.ncon])
    assert rest_contacts(m) == []
    (p,) = probe(m)
    assert (p["joint"], p["body"], p["q"]) == ("pad_seat_joint", "pad_seat", 1.0) and p["max_move_m"] > 0.1
    c = check(tmp_path, "pad")
    assert c["joints"] == [{"name": "pad_seat_joint", "type": "hinge", "body": "pad_seat", "range": [-1.0, 1.0]}]
    assert [q["joint"] for q in c["probe"]] == ["pad_seat_joint"] and c["rest_contacts"] == []
    assert sorted(c["bodies"]) == ["ball", "pad", "pad_seat"] and (c["nbody"], c["njnt"]) == (4, 3)
    assert np.isclose(c["soft_mass"]["cushion"], 0.5 * 2 / 5)  # the 3 pinned nodes drop their share


def soft_cushion_chest(tmp_path, frame="W", cushion=([0.5, 0.3, 0.06], [0, 0, 0.37], "lid"), extra=(), holes=0):
    """A 0.6 x 0.4 x 0.3 m body box with a lid on top, hinged on the back top edge, and a soft foam cushion on the
    lid, as the mesh nodes of parts.glb with their joints in frame (W by default). The numbers are those of S, so z
    is up. The cushion (size, centre, parent) is textured and subdivided, so that its skin has vertices across
    each face, and holes of its faces are left out. extra: more boxes (name, size, centre, parent node) without a
    joint. summary.json has no links, so every link gets a hull inertial. Returns (glb, summary)."""
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image

    from harpor.parts.joints import SOFT_MATERIALS, write_joints

    scene = trimesh.Scene()
    for name, size, center, parent in (("body", [0.6, 0.4, 0.3], [0, 0, 0.15], None),
                                       ("lid", [0.6, 0.4, 0.04], [0, 0, 0.32], None),
                                       ("cushion", *cushion[:2], None), *extra):
        box = trimesh.creation.box(size)
        box.apply_translation(center)
        visual = None
        if name == "cushion":
            box = box.subdivide().subdivide()
            box = trimesh.Trimesh(box.vertices, box.faces[holes:], process=False)
            uv = np.random.default_rng(0).random((len(box.vertices), 2))
            img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (16, 16, 3), dtype=np.uint8))
            visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=img))
        scene.add_geometry(trimesh.Trimesh(frames.w_to_gltf(box.vertices), box.faces, visual=visual, process=False),
                           node_name=name, geom_name=name, parent_node_name=parent)
    scene.export(tmp_path / "parts.glb")
    doc, binary = read_glb(tmp_path / "parts.glb")
    joints = {"lid": {"version": 1, "frame": frame, "parent": "body", "type": "revolute", "origin": [0.0, 0.2, 0.3],
                      "axis": [-1.0, 0.0, 0.0], "lower": 0.0, "upper": 1.5},
              "cushion": {"version": 1, "frame": frame, "parent": cushion[2], "type": "soft", "material": "foam",
                          **SOFT_MATERIALS["foam"], "support": "parent"}}
    write_glb(tmp_path / "parts.glb", write_joints(doc, joints), binary)
    (tmp_path / "summary.json").write_text(json.dumps({"median": {"density_kg_m3": 500.0}}))
    return tmp_path / "parts.glb", tmp_path / "summary.json"


def test_a_soft_part_passes_align_urdf_and_usd_as_a_rigid_link(tmp_path, monkeypatch):
    """Through align, URDF and USD a soft cushion is a link fixed to its parent. It rides on the lid's hinge,
    adds no joint and collides like a fixed link. Only the reports and the notes say that it is soft. The MJCF
    makes it a flex in the lid's body, which needs no exclude."""
    pytest.importorskip("trimesh")
    pytest.importorskip("mujoco")
    pytest.importorskip("pxr")
    from pxr import Usd

    from harpor.parts.joints import SOFT_AS_RIGID, load
    from harpor.physics import align, urdf, usd
    from harpor.physics.mjcf import build as build_mjcf
    from harpor.physics.mjcf import check as check_mjcf
    from harpor.webapp.fake_steps import usd_links

    glb, summary = soft_cushion_chest(tmp_path)
    (tmp_path / "links").mkdir()
    monkeypatch.setattr("sys.argv", ["align", "--glb", str(glb), "--out", str(tmp_path / "aligned.glb"), "--up-near=0,0,1",
                                     "--links-out", str(tmp_path / "links")])
    align.main()
    joints, links = load(tmp_path / "aligned.glb")
    soft = joints["cushion"]
    assert (soft["type"], soft["frame"], soft["material"], soft["young"], soft["support"]) == ("soft", "S", "foam", 3e4, "parent")
    assert links == ["body", "lid", "cushion"] and (tmp_path / "links" / "cushion.glb").is_file()

    mj = build_mjcf(tmp_path / "aligned.glb", summary, tmp_path / "mjcf", "chest", decompose=hull_parts)
    c = check_mjcf(tmp_path / "mjcf", "chest")
    assert [j["name"] for j in c["joints"]] == ["chest_lid_joint"] and c["njnt"] == 2 and c["nbody"] == 3
    assert [p["joint"] for p in c["probe"]] == ["chest_lid_joint"] and c["rest_contacts"] == []
    assert [(j["child"], j["type"]) for j in mj["joints"]] == [("lid", "revolute"), ("cushion", "soft")]
    assert "chest_cushion_joint" not in (tmp_path / "mjcf" / "chest.xml").read_text() and list(mj["soft"]) == ["cushion"]
    assert mj["excluded_pairs"] == [["body", "lid"]] and mj["soft_excluded_pairs"] == [["body", "cushion"]]

    report = urdf.build(tmp_path / "aligned.glb", summary, tmp_path / "mjcf", tmp_path / "urdf", "chest")
    checked = urdf.check(tmp_path / "urdf")
    assert checked["joint_types"] == {"lid_joint": "revolute", "cushion_joint": "fixed"} and checked["missing"] == []
    assert report["soft_as_rigid"] == {"cushion": SOFT_AS_RIGID}
    assert f"<!-- cushion: {SOFT_AS_RIGID} -->" in (tmp_path / "urdf" / "model.urdf").read_text()
    assert report["excluded_pairs"] == [["body", "lid"], ["body", "cushion"]]

    report = usd.build(tmp_path / "aligned.glb", summary, tmp_path / "mjcf", tmp_path / "usd", "chest")
    checked = usd.check(tmp_path / "usd", "chest")
    kinds = {j["name"]: j["type"] for j in checked["joints"]}
    assert kinds == {"lid": "PhysicsRevoluteJoint", "cushion": "PhysicsFixedJoint"}
    assert set(checked["links"]) == {"body", "lid", "cushion"} and report["soft_as_rigid"] == {"cushion": SOFT_AS_RIGID}
    # The rigid cushion still leaves out the body, as the lid does.
    filtered = sorted(sorted([a, t.rsplit("/", 1)[1]]) for a, rec in checked["links"].items() for t in rec["filtered"])
    assert filtered == [["body", "cushion"], ["body", "lid"]]
    assert report["filtered_pairs"] == [["body", "lid"], ["body", "cushion"]]
    stage = Usd.Stage.Open(str(tmp_path / "usd" / "chest.usd"))
    assert stage.GetPrimAtPath("/chest/links/cushion").GetCustomDataByKey("harpor:soft_as_rigid") is True
    assert not stage.GetPrimAtPath("/chest/links/lid").HasCustomDataKey("harpor")
    # The web app's fake USD writes the same layout.
    assert {"name": "cushion", "type": "PhysicsFixedJoint", "body0": "lid", "body1": "cushion"} in usd_links("chest", mj)[1]["joints"]


def test_urdf_comment_of_a_soft_link_stays_valid_xml():
    import xml.etree.ElementTree as ET

    from harpor.physics.urdf import urdf_xml

    for name in ("a--b", "a---b", "a----b", "-a-"):
        joint = {"name": f"{name}_joint", "type": "fixed", "parent": "base", "child": name, "xyz": [0.0, 0.0, 0.0]}
        assert ET.fromstring(urdf_xml("x", [], [joint], False, {name})).find("joint").get("name") == f"{name}_joint"


def test_articulated_usd_welds_the_knob_to_its_lid_and_filters_the_excluded_pairs(tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("pxr")
    from harpor.physics.mjcf import build as build_mjcf
    from harpor.physics.usd import build, check

    glb, summary = two_lid_chest(tmp_path)
    mj = build_mjcf(glb, summary, tmp_path / "mjcf", "chest", decompose=inflated_hull)
    build(glb, summary, tmp_path / "mjcf", tmp_path / "usd", "chest")
    c = check(tmp_path / "usd", "chest")
    joints = {j["name"]: j for j in c["joints"]}
    assert joints["knob"]["type"] == "PhysicsFixedJoint" and joints["knob"]["body0"] == ["/chest/links/lid_left"]
    assert joints["world"]["body0"] == [] and np.allclose(joints["world"]["pos0"], [0, 0, mj["start_z"]])
    filtered = sorted(sorted([a, t.rsplit("/", 1)[1]]) for a, rec in c["links"].items() for t in rec["filtered"])
    assert filtered == sorted(map(sorted, mj["excluded_pairs"]))


def soft_model(tmp_path, fixed_base=False, **kw):
    """soft_cushion_chest in frame S, built with one hull per link: (report, compiled MjModel of scene.xml)."""
    mujoco = pytest.importorskip("mujoco")
    from harpor.physics.mjcf import build

    glb, summary = soft_cushion_chest(tmp_path, "S", **kw)
    report = build(glb, summary, tmp_path / "mjcf", "chest", decompose=hull_parts, fixed_base=fixed_base)
    return report, mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "scene.xml"))


def test_a_soft_link_is_a_flex_in_its_parent_body(tmp_path):
    import hashlib

    mujoco = pytest.importorskip("mujoco")
    from harpor.physics.mjcf import build, check, node_bodies

    report, m = soft_model(tmp_path)
    soft = report["soft"]["cushion"]
    assert (soft["file"], soft["material"], soft["young"], soft["poisson"], soft["density"]) == \
        ("soft/cushion.npz", "foam", 3e4, 0.3, 120.0)
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_DISCRETE and m.opt.solver == mujoco.mjtSolver.mjSOL_CG
    assert (m.nflex, m.nskin, m.flex_vertnum[0], m.flex_group[0]) == (1, 1, soft["nodes"], 3)
    assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, 0) == "chest_cushion_flex"
    # The free nodes hang from the lid, and so do the pinned ones, which ride on its hinge.
    nodes = node_bodies(m)
    assert {int(m.body_parentid[b]) for b in nodes} == {m.body("chest_lid").id}
    assert [int(b) for b in m.flex_vertbodyid if b not in nodes] == [m.body("chest_lid").id] * soft["pinned"]
    # The cushion's inertial is its hull filled with foam, and the flex drops the pinned nodes' share.
    rec = report["links"]["cushion"]
    assert rec["source"] == "soft_table" and np.isclose(rec["inertial"]["mass"], 0.5 * 0.3 * 0.06 * 120)
    assert soft["mass"] == rec["inertial"]["mass"]
    c = check(tmp_path / "mjcf", "chest")
    assert np.isclose(c["soft_mass"]["cushion"], soft["mass"] * (1 - soft["pinned"] / soft["nodes"]))
    assert (c["nbody"], c["njnt"], sorted(c["bodies"])) == (3, 2, ["chest", "chest_lid"])
    # No mesh asset or body for it, but its hulls and visual OBJ stay for URDF and USD.
    assert m.nmesh == 4 and (tmp_path / "mjcf" / "assets" / "collision_cushion_0.obj").is_file() and rec["visuals"]
    with np.load(tmp_path / "mjcf" / soft["file"]) as b:
        assert str(b["asset_sha256"]) == hashlib.sha256((tmp_path / "parts.glb").read_bytes()).hexdigest()
        assert (str(b["link"]), str(b["flex"]), b["visuals"].tolist()) == ("cushion", "chest_cushion_flex", rec["visuals"])
        assert b["ranges"].tolist() == [[0, len(b["vertices"])]] and len(b["press"])
        assert np.allclose(b["direction"], [0, 0, -1])
    # A rigid model keeps its old head.
    glb, summary, _, _ = two_box_hinge(tmp_path / "rigid")
    build(glb, summary, tmp_path / "rigid" / "mjcf", "chest", decompose=hull_parts)
    xml = (tmp_path / "rigid" / "mjcf" / "chest.xml").read_text()
    assert "<option" not in xml and 'contype="3"' not in xml


@pytest.mark.parametrize("fixed_base", [False, True])
def test_a_soft_model_steps_with_a_free_root_and_a_fixed_base(tmp_path, fixed_base):
    mujoco = pytest.importorskip("mujoco")

    report, m = soft_model(tmp_path, fixed_base)
    free_nodes = m.flex_vertnum[0] - report["soft"]["cushion"]["pinned"]
    assert report["fixed_base"] is fixed_base and m.njnt == 1 + (not fixed_base) + 3 * free_nodes
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    root = m.body("chest").id
    rest = d.flexvert_xpos - d.xpos[root]
    mujoco.mj_step(m, d, 500)
    assert np.isfinite(d.qpos).all() and np.isfinite(d.flexvert_xpos).all()
    assert all(w.number == 0 for w in d.warning)
    # The foam sags a little under its own weight on the shut lid, and a free chest drops onto the floor.
    assert np.linalg.norm(d.flexvert_xpos - d.xpos[root] - rest, axis=1).max() < 0.002
    assert abs(d.xpos[root][2] - report["start_z"]) < (0.01 if not fixed_base else 1e-12)


def test_contact_bits_keep_the_flex_off_its_own_object(tmp_path):
    """<exclude> does not reach a flex, so contact bits do: links collide with links (2), not with the flex
    (0), and both with anything else (1). A ball dropped on the cushion rests on the flex."""
    mujoco = pytest.importorskip("mujoco")

    report, m = soft_model(tmp_path, fixed_base=True)

    def touch(a, b):
        return bool(a[0] & b[1] or b[0] & a[1])

    hull, floor = m.geom("chest_collision_lid_0").id, m.geom("floor").id
    link, flex = (m.geom_contype[hull], m.geom_conaffinity[hull]), (m.flex_contype[0], m.flex_conaffinity[0])
    other = (m.geom_contype[floor], m.geom_conaffinity[floor])
    assert (link, flex, other) == ((3, 2), (1, 0), (1, 1))
    assert touch(link, link) and touch(link, other) and touch(flex, other)
    assert not touch(link, flex) and not touch(flex, flex)
    # The hex stands proud into the lid at rest, yet nothing touches.
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert d.ncon == 0
    (tmp_path / "mjcf" / "ball.xml").write_text(
        '<mujoco>\n  <include file="chest.xml"/>\n  <worldbody>\n    <body name="ball" pos="0 0 0.55">\n'
        '      <freejoint/>\n      <geom type="sphere" size="0.03" density="300"/>\n    </body>\n  </worldbody>\n</mujoco>\n')
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "mjcf" / "ball.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 1000)
    top = d.flexvert_xpos[:, 2].max()
    assert any(min(c.geom) < 0 for c in d.contact[:d.ncon])  # the flex side of a contact has no geom
    assert abs(d.xpos[m.body("ball").id][2] - 0.03 - top) < 0.01 and top > 0.4 + report["start_z"]


def test_the_skin_follows_eq4_under_load(tmp_path):
    """Skin vertices from mjv_updateScene are V_rest + W (P_t - P_rest) at rest, under a press on the press
    nodes and with the lid opened, because each free node is a bone and the pinned ones one bone on the lid."""
    trimesh = pytest.importorskip("trimesh")
    mujoco = pytest.importorskip("mujoco")

    report, m = soft_model(tmp_path, fixed_base=True)
    b = dict(np.load(tmp_path / "mjcf" / "soft" / "cushion.npz"))
    d = mujoco.MjData(m)
    scene = mujoco.MjvScene(m, maxgeom=100)

    def skin():
        mujoco.mj_forward(m, d)
        mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.MjvCamera(),
                               mujoco.mjtCatBit.mjCAT_ALL, scene)
        start = scene.skinvertadr[0]
        return np.array(scene.skinvert).reshape(-1, 3)[start:start + scene.skinvertnum[0]]

    W = b["weights"]
    rows = b["tets"][b["tet_of_vertex"]]
    rest_nodes = b["nodes"] + [0, 0, report["start_z"]]
    rest = b["vertices"] + [0, 0, report["start_z"]]
    assert np.abs(skin() - rest).max() < 1e-6
    press = [m.body(f"chest_cushion_flex_{k}").id for k in b["press"]]
    d.xfrc_applied[press, 2] = -4.0
    mujoco.mj_step(m, d, 300)
    P = d.flexvert_xpos
    dent = np.linalg.norm(P - rest_nodes, axis=1).max()
    assert dent > 0.002 and np.abs(skin() - (rest + ((P - rest_nodes)[rows] * W[..., None]).sum(1))).max() < 1e-6
    d.qpos[m.jnt_qposadr[m.joint("chest_lid_joint").id]] = 1.0
    mujoco.mj_step(m, d, 20)
    assert np.abs(skin() - (rest + ((d.flexvert_xpos - rest_nodes)[rows] * W[..., None]).sum(1))).max() < 1e-6
    # The skin keeps the texture as the OBJ visuals do: the stored glTF v is 1 - trimesh's v.
    uv = trimesh.load(tmp_path / "parts.glb", force="scene").geometry["cushion"].visual.uv
    assert np.allclose(m.skin_texcoord, np.c_[uv[:, 0], 1 - uv[:, 1]], atol=1e-6)


def test_check_binding_catches_a_bad_file(tmp_path):
    from harpor.physics.mjcf import check_binding

    _, m = soft_model(tmp_path, fixed_base=True)
    npz, glb = tmp_path / "mjcf" / "soft" / "cushion.npz", tmp_path / "parts.glb"
    assert check_binding(m, npz, glb) == []
    good = dict(np.load(npz))
    lopsided = good["weights"].copy()
    lopsided[0] = [4.0, -3.0, 0.0, 0.0]
    for change, why in (({"flex": "chest_lid_flex"}, "the model has no flex chest_lid_flex"),
                        ({"nodes": good["nodes"][:-1]}, "nodes against"),
                        ({"tets": good["tets"] + len(good["nodes"])}, "an index is out of range"),
                        ({"weights": good["weights"][:-1]}, "weights of shape"),
                        ({"nodes": good["nodes"] + [0, 0, 1e-5]}, "the nodes are not where the flex is at rest"),
                        ({"start_z": good["start_z"] + 1e-3}, "the nodes are not where the flex is at rest"),
                        ({"weights": good["weights"] * 1.01}, "a weight row does not sum to 1"),
                        ({"weights": lopsided}, "a weight is above 3"),
                        ({"vertices": good["vertices"] + 1e-4}, "the weights do not rebuild the vertices"),
                        ({"link": "lid"}, "the vertices are not the GLB's")):
        np.savez(tmp_path / "bad.npz", **{**good, **change})
        assert any(why in p for p in check_binding(m, tmp_path / "bad.npz", glb)), why


def test_a_soft_part_at_the_bottom_starts_above_the_floor(tmp_path):
    # A foot pad under the body: its hex reaches below the hulls, so it sets the start height.
    mujoco = pytest.importorskip("mujoco")

    report, m = soft_model(tmp_path, cushion=([0.5, 0.3, 0.03], [0, 0, -0.015], "body"))
    with np.load(tmp_path / "mjcf" / "soft" / "cushion.npz") as b:
        low = b["nodes"][:, 2].min()
    assert low < -0.03 and np.isclose(report["lowest_collision_z"], low) and np.isclose(report["start_z"], 0.005 - low)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert np.isclose(d.flexvert_xpos[:, 2].min(), 0.005, atol=1e-4)


def test_a_link_held_by_a_soft_link_is_welded_to_its_parent(tmp_path):
    # A button parented to the cushion's node without a joint cannot hang from a flex: it rides on the lid.
    report, m = soft_model(tmp_path, extra=[("button", [0.04, 0.04, 0.02], [0, 0, 0.41], "cushion")])
    assert report["welded"] == {"button": "lid"} and report["loose"] == ["button"]
    assert "button is held by the soft link cushion, so it is welded to its parent" in report["warnings"]
    assert m.body_parentid[m.body("chest_button").id] == m.body("chest_lid").id


def test_a_link_named_as_a_flex_node_or_with_a_quote_still_compiles(tmp_path):
    # MuJoCo names the cushion's free node 7 chest_cushion_flex_7, the body of a link cushion_flex_7, so the flex
    # takes a number. A quote in a link name goes into its collision OBJ's file attribute.
    from harpor.physics.mjcf import check, check_binding, flex_of

    report, m = soft_model(tmp_path, extra=[("cushion_flex_7", [0.04, 0.04, 0.04], [0.35, 0, 0.15], None),
                                            ('knob"top', [0.04, 0.04, 0.04], [-0.35, 0, 0.15], None)])
    assert report["links"]["cushion"]["flex"] == "chest_cushion_flex1"
    with np.load(tmp_path / "mjcf" / "soft" / "cushion.npz") as b:
        assert 7 not in b["pinned"] and str(b["flex"]) == "chest_cushion_flex1"
    assert m.body_geomnum[m.body("chest_cushion_flex_7").id] and m.body("chest_cushion_flex1_7").id
    assert m.body_geomnum[m.body('chest_knob"top').id]
    assert list(check(tmp_path / "mjcf", "chest")["soft_mass"]) == ["cushion"]
    assert check_binding(m, tmp_path / "mjcf" / "soft" / "cushion.npz", tmp_path / "parts.glb") == []
    assert flex_of("o_a", ["o", "o_a_flex_1", "o_a_flex1_3", "o_a_flex2"]) == "o_a_flex2"


def test_an_open_soft_link_gets_a_warning(tmp_path):
    report, _ = soft_model(tmp_path, holes=4)
    assert "cushion is not closed, so its soft mesh is hollow" in report["warnings"] and report["soft"]


def test_a_soft_link_without_a_usable_mesh_stays_rigid(tmp_path):
    # A cushion floating 20 cm above the lid has no node near it: it is a fixed link, as before.
    report, m = soft_model(tmp_path, cushion=([0.5, 0.3, 0.06], [0, 0, 0.57], "lid"))
    assert report["soft"] == {} and report["soft_rigid"]["cushion"]["code"] == "soft_no_support"
    assert any(w.startswith("cushion stays rigid: no node lies within a pitch") for w in report["warnings"])
    assert m.nflex == 0 and m.body_parentid[m.body("chest_cushion").id] == m.body("chest_lid").id
    assert "<option" not in (tmp_path / "mjcf" / "chest.xml").read_text() and ["body", "cushion"] in report["excluded_pairs"]
    assert report["links"]["cushion"]["source"] == "soft_table"


def test_mjcf_command_line_checks_the_binding(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    from harpor.physics import mjcf

    glb, summary = soft_cushion_chest(tmp_path, "S")
    build = mjcf.build
    monkeypatch.setattr(mjcf, "build", lambda *a, **k: build(*a, decompose=hull_parts, **k))
    argv = ["mjcf", "--glb", str(glb), "--summary", str(summary), "--out", str(tmp_path / "out"), "--name", "chest"]
    monkeypatch.setattr("sys.argv", argv)
    mjcf.main()
    report = json.loads((tmp_path / "out" / "mjcf.json").read_text())
    assert list(report["compiled"]["soft_mass"]) == ["cushion"] and not any("binding" in w for w in report["warnings"])
    monkeypatch.setattr(mjcf, "check_binding", lambda m, npz, glb: ["a weight is above 3"])
    mjcf.main()
    report = json.loads((tmp_path / "out" / "mjcf.json").read_text())
    assert "the binding of cushion: a weight is above 3" in report["warnings"] and "compiled" in report
