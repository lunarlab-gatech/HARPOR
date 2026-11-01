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


STACK = {"table": [1.0, 1.0, 0.75], "pan": [0.5, 0.3, 0.08], "cup": [0.1, 0.1, 0.07]}  # sizes, bottom to top


def box_objects(tmp_path, sizes, com=None, usd=False):
    """A box per name of sizes {name: [x, y, z]}, bottom at z = 0: tmp_path/<name>.glb and <name>.json (1 kg, its
    COM at com, else at the box's centre), built into tmp_path/root/<name>/mjcf, and usd/ when usd is true."""
    trimesh = pytest.importorskip("trimesh")
    from harpor.physics.mjcf import build as build_mjcf

    for name, size in sizes.items():
        box = trimesh.creation.box(size)
        box.apply_translation([0, 0, size[2] / 2])
        trimesh.Scene(trimesh.Trimesh(frames.w_to_gltf(box.vertices), box.faces)).export(tmp_path / f"{name}.glb")
        rb = {"mass": 1.0, "com": com or [0, 0, size[2] / 2], "fullinertia": [0.1, 0.1, 0.1, 0, 0, 0]}
        (tmp_path / f"{name}.json").write_text(json.dumps({"rigid_body": rb}))
        d = tmp_path / "root" / name
        build_mjcf(tmp_path / f"{name}.glb", tmp_path / f"{name}.json", d / "mjcf", name, decompose=hull_parts)
        if usd:
            from harpor.physics.usd import build as build_usd

            build_usd(tmp_path / f"{name}.glb", tmp_path / f"{name}.json", d / "mjcf", d / "usd", name)


def test_combine_mjcf_places_objects_apart(tmp_path):
    mujoco = pytest.importorskip("mujoco")

    from harpor.physics.combine_mjcf import combine

    box_objects(tmp_path, {"big": [1.0, 0.8, 0.5], "small": [0.2, 0.1, 0.05]}, com=[0, 0, 0.1])
    layout = combine(tmp_path / "root", tmp_path / "root" / "all.xml", front="small", gap=0.3)
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "root" / "all.xml"))
    assert m.nbody == 3 and np.isclose(m.body("big").mass, 1.0) and m.nmesh == 4
    assert np.isclose(layout["big"][1] - layout["small"][1], 0.4 + 0.3 + 0.05)  # half depths plus the gap


def test_stack_mode_starts_a_falling_column_with_no_contacts(tmp_path):
    mujoco = pytest.importorskip("mujoco")

    from harpor.physics.combine_mjcf import stack

    box_objects(tmp_path, STACK)
    stack(tmp_path / "root", tmp_path / "root" / "stack.xml", "table", ["pan", "cup"], clearance=0.04, tilt=8)
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "root" / "stack.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert d.ncon == 0 and m.opt.timestep == 0.001
    # Each object's lowest collision vertex is 4 cm above the highest one of the object below it.
    lows, highs = [], []
    for body in ("table", "pan", "cup"):
        gid = m.geom(f"{body}_collision_0").id
        mid = m.geom_dataid[gid]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = d.geom_xpos[gid] + v @ d.geom_xmat[gid].reshape(3, 3).T
        lows.append(w[:, 2].min())
        highs.append(w[:, 2].max())
    assert np.allclose(np.array(lows[1:]) - np.array(highs[:-1]), 0.04, atol=1e-4)


def test_usd_scene_matches_the_mujoco_stack(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    from harpor.physics.combine_mjcf import stack
    from harpor.physics.usd_scene import build

    box_objects(tmp_path, STACK, usd=True)
    stack(tmp_path / "root", tmp_path / "root" / "stack.xml", "table", ["pan", "cup"])
    rep = build(tmp_path / "root" / "stack.xml", tmp_path / "root" / "stack.usda")
    assert rep["timesteps_per_second"] == 1000

    m = mujoco.MjModel.from_xml_path(str(tmp_path / "root" / "stack.xml"))
    dm = mujoco.MjData(m)
    mujoco.mj_forward(m, dm)
    stage = Usd.Stage.Open(str(tmp_path / "root" / "stack.usda"))
    for name in ("table", "pan", "cup"):
        prim = stage.GetPrimAtPath(f"/World/{name}")
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))  # row-vector form
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, dm.xquat[m.body(name).id])
        assert np.allclose(M[3, :3], dm.xpos[m.body(name).id], atol=1e-5)
        assert np.allclose(M[:3, :3].T, R.reshape(3, 3), atol=1e-5)


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

    # A world joint on the root keeps only the hinge.
    fix_to_world(glb)
    build(glb, summary, tmp_path / "fixed", "chest", decompose=hull_parts)
    c = check(tmp_path / "fixed", "chest")
    assert c["njnt"] == 1 and json.loads((tmp_path / "fixed" / "mjcf.json").read_text())["fixed_base"]


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
