"""Write a MuJoCo model (MJCF) for one aligned GLB and its VoMP rigid-body numbers (build_plan.md note 9).

    python -m harpor.physics.mjcf --glb aligned.glb --summary summary.json --out DIR [--name chair]
        [--friction 0.6] [--coacd-threshold 0.05]

Writes into DIR:
  <name>.xml   the object: one body with a free joint, the VoMP inertial (mass, COM and fullinertia
               from summary.json), visual meshes that do not collide, and convex collision meshes
  scene.xml    a floor, a light and the object, for the viewer:
               python -m mujoco.viewer --mjcf DIR/scene.xml
  assets/      visual_<k>.obj, texture_<k>.png and collision_<j>.obj
  mjcf.json    inputs, versions, counts and the starting height

The GLB must be in frame S (harpor.physics.align): glTF y up and meters, with the bottom at z = 0.
MuJoCo's z-up body frame is S, (x, y, z)_S = (x, -z, y)_glTF, and summary.json uses the same frame.
MuJoCo reads only STL, OBJ and MSH meshes, only the first object of an OBJ and only PNG textures,
so each glTF primitive becomes its own OBJ with its UVs and its base-colour texture as a PNG. Its OBJ
decoder flips v (1 - v) on load, so the OBJ keeps the usual bottom-left UV origin, as trimesh stores it.
build(..., textures=False) writes no PNG and no UVs, for a model that is only simulated (harpor.physics.supervise).
A mesh geom collides as its convex hull, so the whole object is split into convex parts with CoACD.
Every mesh asset uses inertia="shell" (MuJoCo 3.3 or newer), because MuJoCo computes each mesh's
volume even when the mesh carries no mass, and open meshes (surfaces that are not closed) fail that check.
All geoms have density 0: the body's mass and inertia come only from <inertial>.
VoMP does not predict friction or restitution. --friction sets the sliding friction of every
collision geom and of the floor in scene.xml. MuJoCo combines two geoms of equal priority by taking
the larger friction, so in another scene the effective value is max(--friction, the partner's).
Contact softness stays at MuJoCo's defaults.
"""

import argparse
import json
import re
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from harpor import frames


def shared_visual(visual):
    """A copy of a primitive's visual that keeps its material, not a copy of it: a material's copy copies its images,
    which decodes each again, once for every primitive and every node that uses it (load_s)."""
    import trimesh

    if not isinstance(visual, trimesh.visual.TextureVisuals):
        return visual.copy()
    return trimesh.visual.TextureVisuals(uv=None if visual.uv is None else visual.uv.copy(), material=visual.material,
                                         face_materials=visual.face_materials)


def load_s(glb):
    """[(name, trimesh.Trimesh)] of every glTF primitive, with node transforms applied, in frame S."""
    import trimesh

    T = np.eye(4)
    T[:3, :3] = frames.W_FROM_GLTF  # the stock glTF import: S is W's numbers
    scene = trimesh.load(glb, force="scene")
    out = []
    for node in scene.graph.nodes_geometry:
        M, g = scene.graph[node]
        src = scene.geometry[g]
        m = src.copy(include_cache=True, include_visual=False)  # keep the glTF normals; the transform rotates them
        m.visual = shared_visual(src.visual)
        m.apply_transform(T @ M)
        out.append((f"{node}" if node == g else f"{node}/{g}", m))
    return out


def write_obj(path, vertices, faces, uv=None, normals=None):
    """Minimal single-object OBJ (1-based indices; vt and vn share the vertex index)."""
    lines = [f"v {x:.7g} {y:.7g} {z:.7g}" for x, y, z in vertices]
    if uv is not None:
        lines += [f"vt {u:.7g} {v:.7g}" for u, v in uv]
    if normals is not None:
        lines += [f"vn {x:.6g} {y:.6g} {z:.6g}" for x, y, z in normals]
    f1 = np.asarray(faces) + 1
    if uv is not None and normals is not None:
        lines += [f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in f1]
    elif uv is not None:
        lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in f1]
    elif normals is not None:
        lines += [f"f {a}//{a} {b}//{b} {c}//{c}" for a, b, c in f1]
    else:
        lines += [f"f {a} {b} {c}" for a, b, c in f1]
    Path(path).write_text("\n".join(lines) + "\n")


def visual_parts(meshes, assets, textures=True):
    """Write the visual OBJs and PNGs (no PNG or UVs without textures). Returns ([{name, mesh, texture, rgba,
    faces}], [skipped]). A primitive with under 4 distinct vertices or no area cannot be a MuJoCo mesh and is
    skipped."""
    parts, skipped = [], []
    for k, (name, m) in enumerate(meshes):
        if len(np.unique(m.vertices, axis=0)) < 4 or m.area < 1e-12:
            skipped.append({"name": name, "vertices": len(m.vertices), "area_m2": float(m.area)})
            continue
        vis = getattr(m, "visual", None)
        mat = getattr(vis, "material", None)
        uv = getattr(vis, "uv", None)
        tex = getattr(mat, "baseColorTexture", None)
        factor = getattr(mat, "baseColorFactor", None)
        if factor is None and vis is not None and getattr(vis, "kind", None) == "vertex":
            factor = np.asarray(vis.vertex_colors).mean(0)
        rgba = np.ones(4) if factor is None else np.asarray(factor, float).reshape(-1)[:4]
        if rgba.max() > 1:
            rgba = rgba / 255
        has_tex = textures and tex is not None and uv is not None and len(uv) == len(m.vertices)
        write_obj(assets / f"visual_{k}.obj", m.vertices, m.faces, np.asarray(uv) if has_tex else None,
                  np.asarray(m.vertex_normals))
        if has_tex:
            tex.convert("RGB").save(assets / f"texture_{k}.png")
        parts.append({"name": name, "mesh": f"visual_{k}", "texture": f"texture_{k}" if has_tex else None,
                      "rgba": rgba.tolist(), "faces": len(m.faces)})
    return parts, skipped


def convex_parts(vertices, faces, threshold=0.05, seed=0):
    """CoACD convex decomposition: [(vertices, faces)]."""
    import coacd

    coacd.set_log_level("error")
    return [(np.asarray(v), np.asarray(f)) for v, f in
            coacd.run_coacd(coacd.Mesh(np.asarray(vertices, float), np.asarray(faces, np.int64)),
                            threshold=threshold, seed=seed)]


def _head(name, friction, visuals):
    """The model's opening lines up to its visual assets: compiler, the visual and collision classes, and
    each visual's texture, material and mesh."""
    a = quoteattr
    lines = [f'<mujoco model={a(name)}>',
             '  <compiler angle="radian" meshdir="assets" texturedir="assets"/>',
             '  <default>',
             f'    <default class={a(name + "_visual")}>',
             '      <geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/>',
             '    </default>',
             f'    <default class={a(name + "_collision")}>',
             f'      <geom type="mesh" group="3" density="0" friction="{friction:g} 0.005 0.0001" '
             'rgba="0.8 0.4 0.2 0.4"/>',
             '    </default>',
             '  </default>',
             '  <asset>']
    for v in visuals:
        if v["texture"]:
            lines.append(f'    <texture name={a(name + "_" + v["texture"])} type="2d" file="{v["texture"]}.png"/>')
        r = " ".join(f"{c:.4g}" for c in v["rgba"])
        tex = f' texture={a(name + "_" + v["texture"])}' if v["texture"] else ""
        lines.append(f'    <material name={a(name + "_" + v["mesh"])} rgba="{r}"{tex}/>')
        lines.append(f'    <mesh name={a(name + "_" + v["mesh"])} file={a(v["mesh"] + ".obj")} inertia="shell"/>')
    return lines


def object_xml(name, inertial, visuals, n_collision, friction, z0):
    a = quoteattr
    lines = _head(name, friction, visuals)
    for j in range(n_collision):
        lines.append(f'    <mesh name={a(f"{name}_collision_{j}")} file="collision_{j}.obj" inertia="shell"/>')
    pos = " ".join(f"{c:.6g}" for c in inertial["com"])
    full = " ".join(f"{c:.6g}" for c in inertial["fullinertia"])
    lines += ['  </asset>',
              '  <worldbody>',
              f'    <body name={a(name)} pos="0 0 {z0:.4g}">',
              '      <freejoint/>',
              f'      <inertial pos="{pos}" mass="{inertial["mass"]:.6g}" fullinertia="{full}"/>']
    for v in visuals:
        lines.append(f'      <geom name={a(name + "_" + v["mesh"])} class={a(name + "_visual")} '
                     f'mesh={a(name + "_" + v["mesh"])} material={a(name + "_" + v["mesh"])}/>')
    for j in range(n_collision):
        lines.append(f'      <geom name={a(f"{name}_collision_{j}")} class={a(name + "_collision")} '
                     f'mesh={a(f"{name}_collision_{j}")}/>')
    lines += ['    </body>', '  </worldbody>', '</mujoco>']
    return "\n".join(lines) + "\n"


def scene_xml(name, friction):
    return f"""<mujoco model={quoteattr(name + " scene")}>
  <include file="{name}.xml"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="135" elevation="-25"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="512" height="512" mark="edge" markrgb="0.8 0.8 0.8"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" friction="{friction:g} 0.005 0.0001" material="grid"/>
  </worldbody>
</mujoco>
"""


def build(glb, summary, out, name, friction=0.6, threshold=0.05, decompose=convex_parts, textures=True):
    """Write the model (see the module docstring) and mjcf.json."""
    import trimesh

    out = Path(out)
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    s = json.loads(Path(summary).read_text())
    rb = s["rigid_body"]
    meshes = load_s(glb)
    visuals, skipped = visual_parts(meshes, assets, textures)
    V = np.concatenate([m.vertices for _, m in meshes])
    F = np.concatenate([m.faces + o for (_, m), o in zip(meshes, np.cumsum([0] + [len(m.vertices) for _, m in meshes]))])
    # Weld the UV-seam copies so that CoACD sees one connected surface. It still remeshes (and
    # slightly inflates) a surface that is not watertight.
    welded = trimesh.Trimesh(V, F, process=False)
    welded.merge_vertices()
    hulls = decompose(np.asarray(welded.vertices), np.asarray(welded.faces), threshold)
    for j, (v, f) in enumerate(hulls):
        write_obj(assets / f"collision_{j}.obj", v, f)
    low = min(float(np.min(v[:, 2])) for v, _ in hulls)
    z0 = max(0.0, -low) + 0.005  # start just above the floor, so the hulls do not begin inside it
    (out / f"{name}.xml").write_text(object_xml(name, rb, visuals, len(hulls), friction, z0))
    (out / "scene.xml").write_text(scene_xml(name, friction))
    report = {"glb": str(Path(glb).resolve()), "summary": str(Path(summary).resolve()), "config": s.get("config"),
              "name": name, "friction": friction, "coacd_threshold": threshold, "visuals": visuals,
              "skipped_visuals": skipped, "collision_input_watertight": bool(welded.is_watertight),
              "collision_hulls": len(hulls), "collision_faces": int(sum(len(f) for _, f in hulls)),
              "lowest_collision_z": low, "start_z": z0, "inertial": {k: rb[k] for k in ("mass", "com", "fullinertia")}}
    (out / "mjcf.json").write_text(json.dumps(report, indent=1))
    return report


def check(out, name):
    """Compile scene.xml: the body's mass and inertia, to compare with the inertial it was given."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(Path(out) / "scene.xml"))
    b = m.body(name).id
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_iquat[b])
    R = R.reshape(3, 3)
    I = R @ np.diag(m.body_inertia[b]) @ R.T
    return {"mujoco": mujoco.__version__, "nbody": m.nbody, "ngeom": m.ngeom, "nmesh": m.nmesh,
            "ntex": m.ntex, "mass": float(m.body_mass[b]), "com": m.body_ipos[b].tolist(),
            "fullinertia": [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]], "njnt": int(m.njnt)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="object")
    ap.add_argument("--friction", type=float, default=0.6)
    ap.add_argument("--coacd-threshold", type=float, default=0.05)
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z_][\w-]*", a.name) or a.name == "scene":
        ap.error("--name must be a plain identifier other than 'scene' (scene.xml includes <name>.xml)")
    report = build(a.glb, a.summary, a.out, a.name, a.friction, a.coacd_threshold)
    report["compiled"] = check(a.out, a.name)
    (Path(a.out) / "mjcf.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("collision_hulls", "start_z", "inertial", "compiled")}, indent=1))


if __name__ == "__main__":
    main()
