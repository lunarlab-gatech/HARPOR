"""Write an Isaac Sim physics USD for one aligned GLB and its VoMP rigid-body numbers (build_plan.md note 9).

    python -m harpor.physics.usd --glb aligned.glb --summary summary.json --mjcf MJCF_DIR --out DIR --name NAME
        [--friction F]

It reuses the MJCF's convex collision hulls (MJCF_DIR/assets/collision_<j>.obj) and its start height and
friction (MJCF_DIR/mjcf.json), so the two models agree, except that omni.physx re-cooks a hull with more than
64 polygons (most CoACD hulls) into a slightly smaller one, 0.3 to 4 mm inward. Writes into DIR:
  NAME.usd     the asset (binary crate). Default prim /NAME carries RigidBodyAPI and MassAPI: mass,
               centerOfMass, and the full inertia from summary.json as diagonalInertia plus principalAxes
               (MassAPI has no full-tensor attribute). /NAME/visuals holds one Mesh per glTF primitive with
               normals, st UVs and a UsdPreviewSurface material (base colour, a vertex-colour mean or
               a texture; metallic and roughness as factors or from the metallicRoughness texture, B and G
               channels, times the factors); normal and occlusion maps are not exported. Visuals do not
               collide. /NAME/collisions holds the hulls as Meshes with
               CollisionAPI and MeshCollisionAPI approximation "convexHull", purpose "guide" (not
               rendered), bound to a physics material (static and dynamic friction F, restitution 0).
  scene.usda   PhysicsScene (gravity 9.81 m/s^2 along -z), a 20 x 20 m ground box with the same physics
               material, lights, and the asset referenced at the MJCF's start height. Open it in Isaac Sim.
  textures/    texture_<k>.png (base colour) and mr_<k>.png (metallicRoughness)
  usd.json     inputs, counts and the check below
The stage is z up, 1 m and 1 kg per unit, the frame S of harpor.physics.align and summary.json, so
Isaac Sim's Metrics Assembler has nothing to convert. PhysX averages two materials' friction by
default, so the ground carries the same value. VoMP gives no friction or restitution; F is an assumption.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from harpor.physics.mjcf import load_s


def read_obj(path):
    """Vertices and triangles of a plain OBJ (v and f lines; f may carry /vt/vn indices)."""
    v, f = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            v.append([float(c) for c in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(c.split("/")[0]) - 1 for c in line.split()[1:4]])
    return np.array(v), np.array(f, dtype=np.int64)


def principal_axes(fullinertia):
    """(diagonal, quaternion w x y z) with I = R diag(d) R^T, R a proper rotation (principal -> body)."""
    from scipy.spatial.transform import Rotation

    xx, yy, zz, xy, xz, yz = fullinertia
    I = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], float)
    d, R = np.linalg.eigh(I)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return d, (w, x, y, z)


def _mesh(stage, path, vertices, faces, normals=None, uv=None, double_sided=False):
    from pxr import UsdGeom, Sdf, Vt

    m = UsdGeom.Mesh.Define(stage, path)
    pts = Vt.Vec3fArray.FromNumpy(np.asarray(vertices, np.float32))
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, np.int32).reshape(-1)))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(pts))
    m.CreateDoubleSidedAttr(bool(double_sided))
    if normals is not None:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if uv is not None:
        UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
    return m


def _surface(stage, path, rgba, metallic, roughness, texture=None, mr_texture=None):
    """UsdPreviewSurface material. A base-colour texture is multiplied by rgba, and a glTF metallicRoughness
    texture (raw, roughness in G, metallic in B) by the factors, through UsdUVTexture's scale."""
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    color = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    metal = sh.CreateInput("metallic", Sdf.ValueTypeNames.Float)
    rough = sh.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    st = None
    if texture or mr_texture:
        st = UsdShade.Shader.Define(stage, f"{path}/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def uv_texture(name, file, space, scale):
        tex = UsdShade.Shader.Define(stage, f"{path}/{name}")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(space)
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*map(float, scale)))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        return tex

    if texture is None:
        color.Set(Gf.Vec3f(*map(float, rgba[:3])))
    else:
        tex = uv_texture("BaseColor", texture, "sRGB", (*rgba[:3], 1.0))
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        color.ConnectToSource(tex.ConnectableAPI(), "rgb")
    if mr_texture is None:
        metal.Set(float(metallic))
        rough.Set(float(roughness))
    else:
        mr = uv_texture("MetallicRoughness", mr_texture, "raw", (1.0, roughness, metallic, 1.0))
        mr.CreateOutput("g", Sdf.ValueTypeNames.Float)
        mr.CreateOutput("b", Sdf.ValueTypeNames.Float)
        rough.ConnectToSource(mr.ConnectableAPI(), "g")
        metal.ConnectToSource(mr.ConnectableAPI(), "b")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path, friction):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction))
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind(prim, mat, purpose=""):
    from pxr import UsdShade

    api = UsdShade.MaterialBindingAPI.Apply(prim)
    if purpose:
        api.Bind(mat, UsdShade.Tokens.weakerThanDescendants, purpose)
    else:
        api.Bind(mat)


def _stage(path):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    return stage


def build(glb, summary, mjcf_dir, out, name, friction=None):
    """Write the asset, scene.usda and usd.json (see the module docstring)."""
    from pxr import UsdGeom, UsdPhysics

    out, mjcf_dir = Path(out), Path(mjcf_dir)
    (out / "textures").mkdir(parents=True, exist_ok=True)
    mj = json.loads((mjcf_dir / "mjcf.json").read_text())
    friction = mj["friction"] if friction is None else friction
    rb = json.loads(Path(summary).read_text())["rigid_body"]
    hulls = [read_obj(mjcf_dir / "assets" / f"collision_{j}.obj") for j in range(mj["collision_hulls"])]

    stage = _stage(out / f"{name}.usd")
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    d, q = _mass(root.GetPrim(), rb)

    UsdGeom.Scope.Define(stage, f"/{name}/materials")
    visuals = []
    for k, (label, m) in enumerate(load_s(glb)):
        v = _visual(stage, out, name, k, label, m, f"/{name}/visuals/visual_{k}")
        if v:
            visuals.append(v)

    phys = _physics_material(stage, f"/{name}/materials/physics", friction)
    for j, (v, f) in enumerate(hulls):
        _collider(stage, f"/{name}/collisions/collision_{j}", v, f, phys)
    stage.GetRootLayer().Save()
    _scene(out, name, friction, mj["start_z"])

    report = {"glb": str(Path(glb).resolve()), "summary": str(Path(summary).resolve()), "mjcf": str(mjcf_dir.resolve()),
              "name": name, "friction": friction, "start_z": mj["start_z"], "visuals": visuals,
              "collision_hulls": len(hulls), "inertial": {k: rb[k] for k in ("mass", "com", "fullinertia")},
              "diagonal_inertia": d.tolist(), "principal_axes_wxyz": list(q)}
    (out / "usd.json").write_text(json.dumps(report, indent=1))
    return report


def _visual(stage, out, name, k, label, m, prim):
    """One glTF primitive as a Mesh at prim with its material /NAME/materials/visual_k. Returns its record,
    or None for a primitive with under 4 distinct vertices or no area."""
    if len(np.unique(m.vertices, axis=0)) < 4 or m.area < 1e-12:
        return None
    vis = getattr(m, "visual", None)
    mat = getattr(vis, "material", None)
    uv = getattr(vis, "uv", None)
    tex = getattr(mat, "baseColorTexture", None)
    mr = getattr(mat, "metallicRoughnessTexture", None)
    factor = getattr(mat, "baseColorFactor", None)
    if factor is None and vis is not None and getattr(vis, "kind", None) == "vertex":
        factor = np.asarray(vis.vertex_colors).mean(0)  # as harpor.physics.mjcf does
    rgba = np.ones(4) if factor is None else np.asarray(factor, float).reshape(-1)[:4]
    rgba = rgba / 255 if rgba.max() > 1 else rgba
    metallic = getattr(mat, "metallicFactor", None)
    roughness = getattr(mat, "roughnessFactor", None)
    metallic = 1.0 if metallic is None else metallic  # glTF defaults
    roughness = 1.0 if roughness is None else roughness
    has_uv = uv is not None and len(uv) == len(m.vertices)
    texture = mr_texture = None
    if has_uv and tex is not None:
        tex.convert("RGB").save(out / "textures" / f"texture_{k}.png")
        texture = f"./textures/texture_{k}.png"
    if has_uv and mr is not None:
        mr.convert("RGB").save(out / "textures" / f"mr_{k}.png")
        mr_texture = f"./textures/mr_{k}.png"
    mesh = _mesh(stage, prim, m.vertices, m.faces, m.vertex_normals,
                 np.asarray(uv) if texture or mr_texture else None, getattr(mat, "doubleSided", False))
    _bind(mesh.GetPrim(), _surface(stage, f"/{name}/materials/visual_{k}", rgba, metallic, roughness, texture,
                                   mr_texture))
    return {"name": label, "prim": prim, "texture": texture, "mr_texture": mr_texture, "rgba": rgba.tolist(),
            "metallic": float(metallic), "roughness": float(roughness)}


def _collider(stage, path, vertices, faces, phys):
    """A convex hull as a guide Mesh with CollisionAPI, approximation convexHull, bound to phys."""
    from pxr import UsdGeom, UsdPhysics

    h = _mesh(stage, path, vertices, faces)
    h.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(h.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(h.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    _bind(h.GetPrim(), phys, "physics")
    return h


def _mass(prim, inertial):
    """MassAPI on prim from an inertial {mass, com, fullinertia}. Returns (diagonal, quaternion w x y z)."""
    from pxr import Gf, UsdPhysics

    mass = UsdPhysics.MassAPI.Apply(prim)
    d, q = principal_axes(inertial["fullinertia"])
    mass.CreateMassAttr(float(inertial["mass"]))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, inertial["com"])))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, d)))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(*map(float, q)))
    return d, q


def _scene(out, name, friction, start_z):
    """scene.usda: physics scene, ground, lights and the asset at start_z (see the module docstring)."""
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    scene = _stage(out / "scene.usda")
    world = UsdGeom.Xform.Define(scene, "/World")
    scene.SetDefaultPrim(world.GetPrim())
    ps = UsdPhysics.Scene.Define(scene, "/World/physicsScene")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr(9.81)
    ground = UsdGeom.Cube.Define(scene, "/World/ground")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    ground.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.1))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.4, 0.45)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    _bind(ground.GetPrim(), _physics_material(scene, "/World/groundMaterial", friction), "physics")
    UsdLux.DistantLight.Define(scene, "/World/sun").CreateIntensityAttr(3000)
    UsdLux.DomeLight.Define(scene, "/World/sky").CreateIntensityAttr(500)
    obj = UsdGeom.Xform.Define(scene, f"/World/{name}")
    obj.GetPrim().GetReferences().AddReference(f"./{name}.usd")
    obj.AddTranslateOp().Set(Gf.Vec3d(0, 0, float(start_z)))
    scene.GetRootLayer().Save()


def _read_mass(prim):
    """(mass, com, fullinertia) of a prim's MassAPI, the inertia rebuilt from diagonalInertia and principalAxes."""
    from pxr import Gf, UsdPhysics

    mass = UsdPhysics.MassAPI(prim)
    q = mass.GetPrincipalAxesAttr().Get()
    R = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(q))).GetTranspose())  # Gf matrices act on row vectors
    I = R @ np.diag(np.array(mass.GetDiagonalInertiaAttr().Get(), float)) @ R.T
    full = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    return mass.GetMassAttr().Get(), list(mass.GetCenterOfMassAttr().Get()), full


def check(out, name):
    """Reopen both stages and compare the body's mass, COM and full inertia with what was written."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(Path(out) / f"{name}.usd"))
    root = stage.GetDefaultPrim()
    mass, com, full = _read_mass(root)
    coll = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    textures = []
    for p in stage.Traverse():
        if p.IsA(UsdShade.Shader) and UsdShade.Shader(p).GetIdAttr().Get() == "UsdUVTexture":
            textures.append(str(UsdShade.Shader(p).GetInput("file").Get().resolvedPath))
    scene = Usd.Stage.Open(str(Path(out) / "scene.usda"))
    placed = scene.GetPrimAtPath(f"/World/{name}")
    return {"usd": Usd.GetVersion(), "rigid_body": root.HasAPI(UsdPhysics.RigidBodyAPI),
            "mass": mass, "com": com, "fullinertia": full, "colliders": len(coll), "up_axis": UsdGeom.GetStageUpAxis(stage),
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "kilograms_per_unit": UsdPhysics.GetStageKilogramsPerUnit(stage),
            "textures_resolved": all(textures), "textures": len(textures),
            "scene_reference_is_rigid_body": placed.HasAPI(UsdPhysics.RigidBodyAPI)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--mjcf", required=True, help="the MJCF folder from harpor.physics.mjcf (hulls, start height, friction)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--friction", type=float)
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z_]\w*", a.name) or a.name == "World":
        ap.error("--name must be a plain identifier other than 'World'")
    out = Path(a.out)
    for old in (out / f"{a.name}.usd", out / "scene.usda", out / "usd.json"):  # only what build() writes
        old.unlink(missing_ok=True)
    shutil.rmtree(out / "textures", ignore_errors=True)
    report = build(a.glb, a.summary, a.mjcf, a.out, a.name, a.friction)
    report["checked"] = check(a.out, a.name)
    (Path(a.out) / "usd.json").write_text(json.dumps(report, indent=1, default=list))
    print(json.dumps(report["checked"], indent=1, default=list))


if __name__ == "__main__":
    main()
