"""Write a MuJoCo model (MJCF) for one aligned GLB and its VoMP rigid-body numbers (build_plan.md note 9).

    python -m harpor.physics.mjcf --glb aligned.glb --summary summary.json --out DIR [--name chair]
        [--friction 0.6] [--coacd-threshold 0.05] [--fixed-base]

Writes into DIR:
  <name>.xml   the object: one body with a free joint, the VoMP inertial (mass, COM and fullinertia
               from summary.json), visual meshes that do not collide, and convex collision meshes
  scene.xml    a floor, a light and the object, for the viewer:
               python -m mujoco.viewer --mjcf DIR/scene.xml
  assets/      visual_<k>.obj, texture_<k>.png and collision_<j>.obj
  soft/        <link>.npz, the binding of each soft link
  mjcf.json    inputs, versions, counts and the starting height

The GLB must be in frame S (harpor.physics.align): glTF y up and meters, with the bottom at z = 0.
MuJoCo's z-up body frame is S, (x, y, z)_S = (x, -z, y)_glTF, and summary.json uses the same frame.
MuJoCo reads only STL, OBJ and MSH meshes, only the first object of an OBJ and only PNG textures,
so each glTF primitive becomes its own OBJ with its UVs and its base-colour texture as a PNG. Its OBJ
decoder flips v (1 - v) on load, so the OBJ keeps the usual bottom-left UV origin, as trimesh stores it.
build(..., textures=False) writes no PNG and no UVs, for a model that is only simulated (harpor.physics.sim_check).
A mesh geom collides as its convex hull, so the whole object is split into convex parts with CoACD.
Every mesh asset uses inertia="shell" (MuJoCo 3.3 or newer), because MuJoCo computes each mesh's
volume even when the mesh carries no mass, and open meshes (surfaces that are not closed) fail that check.
All geoms have density 0: the body's mass and inertia come only from <inertial>.
VoMP does not predict friction or restitution. --friction sets the sliding friction of every
collision geom and of the floor in scene.xml. MuJoCo combines two geoms of equal priority by taking
the larger friction, so in another scene the effective value is max(--friction, the partner's).
Contact softness stays at MuJoCo's defaults.

A GLB with joints (harpor.parts.joints, in frame S after harpor.physics.align; build_plan.md note 11) gives
one body per link instead. The root link is the body <name> (a free joint unless the GLB has a fixed base
or --fixed-base is given); every other link is a body <name>_<link> nested under its parent at pos 0,
since all links share frame S, with a hinge or slide joint at the joint's origin and axis, limited to
[lower, upper], or no joint (fixed). A link without a joint is welded, with a warning, to the link of its
nearest glTF ancestor that holds a mesh, or else to the root (harpor.parts.joints.welds), and a soft holder
gives way to its parent.
CoACD runs per link (assets/collision_<link>_<j>.obj). Each link's inertial is its VoMP rigid body
(summary.json links, harpor.physics.vomp_run --links); a link VoMP could not voxelize gets its hulls
filled at the object's median density, with a warning.
MuJoCo does not collide bodies welded together, nor a body with its parent unless that parent is welded to
the world (a fixed base). So <contact> excludes, whatever the base, every jointed link and the links welded
to it from its parent's links, and also two links that touch at rest (convex hulls within 1 mm), such as
two doors or two stacked drawers, with a warning.
mjcf.json adds links {name: {body, inertial, source, collision_hulls, ...}}, joints, root, fixed_base,
loose, welded {loose link: link}, excluded_pairs and warnings; its inertial is then the sum of the links.
Each hinge or slide has joint damping and a position actuator of the same name, with ctrl limited to the joint's
range. Their gains scale with I, the inertia of the links that move with the joint about its axis line (parallel
axis), or their mass for a slide: kp = w^2 I, kv = 2 w I and damping 0.2 kv, with w = 2 pi 5 rad/s, so that a
joint of any size tracks alike and stays stable. At ctrl 0 the actuator holds the joint at rest. mjcf.json adds
actuators {link: {name, inertia, kp, kv, damping}}.

A soft link (build_plan.md note 11) has no body. It is a <flexcomp> <name>_<link>_flex, on the hex mesh of
harpor.physics.soft, inside its parent's body with its points in S, the parent's frame, and pinned there.
MuJoCo names its free nodes' bodies <flex>_<k>, so a number follows flex when a link's body has such a name.
Its hulls, visual OBJs and inertial stay for URDF and USD, where it is a rigid link. That inertial is the hulls
filled at the material's density (source soft_table), and its mass goes to the flexcomp, whose compiled mass
is smaller because pinned nodes drop their share (check gives soft_mass). The object's surface is one <skin>
per visual, on a bone per free node and one bone for all the pinned nodes, bound at the rest origin of the
body they hang from, and the flex stays in group 3 so that a viewer shows the textured surface.
soft/<link>.npz holds the binding, and check_binding tests it. With a soft link the model sets
integrator="discrete" and solver="CG", which a scene that includes the model must keep. <exclude> has no
effect on a flex, so contact bits do the work: the collision class is contype 3 and conaffinity 2, the flex
contype 1 and conaffinity 0. Link and link collide (2), link and flex do not (0), and links and the flex
collide with anything else in a scene (1). Then the proud hex cannot press on a neighbour at rest, and a
neighbour can pass through the soft part under load, which is accepted because the hex is a proxy.
A soft link with no usable mesh (harpor.physics.soft.SoftError) stays a rigid link fixed to its parent, with a
warning. mjcf.json adds soft {link: {file, nodes, tets, pinned, pitch, material, young, poisson, density,
mass}}, soft_rigid {link: {code, reason}} for those that stayed rigid, and soft_excluded_pairs, the pairs with
a soft link that the rigid URDF and USD still leave out.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from harpor import frames
from harpor.parts import joints as joint_spec
from harpor.physics import soft as soft_mesh
from harpor.physics.mass import combine

OMEGA = 2 * np.pi * 5  # rad/s, the natural frequency of a joint on its position actuator


def shared_visual(visual):
    """A copy of a primitive's visual that keeps its material, not a copy of it: a material's copy copies its images,
    which decodes each again, once for every primitive and every node that uses it (load_s)."""
    import trimesh

    if not isinstance(visual, trimesh.visual.TextureVisuals):
        return visual.copy()
    return trimesh.visual.TextureVisuals(uv=None if visual.uv is None else visual.uv.copy(), material=visual.material,
                                         face_materials=visual.face_materials)


def load_s(glb, by_link=False):
    """[(name, trimesh.Trimesh)] of every glTF primitive, with node transforms applied, in frame S.
    by_link=True gives [(link, name, mesh)]: link is the glTF mesh node that holds the primitive (trimesh
    gives each primitive of a node with several its own child frame)."""
    import trimesh

    T = np.eye(4)
    T[:3, :3] = frames.W_FROM_GLTF  # the stock glTF import: S is W's numbers
    scene = trimesh.load(glb, force="scene")
    if by_link:
        from harpor.physics.align import read_glb

        mesh_nodes = set(joint_spec.mesh_nodes(read_glb(glb)[0]))
        parents = scene.graph.transforms.parents
    out = []
    for node in scene.graph.nodes_geometry:
        M, g = scene.graph[node]
        src = scene.geometry[g]
        m = src.copy(include_cache=True, include_visual=False)  # keep the glTF normals; the transform rotates them
        m.visual = shared_visual(src.visual)
        m.apply_transform(T @ M)
        name = f"{node}" if node == g else f"{node}/{g}"
        if by_link:
            link = node
            while link not in mesh_nodes and link in parents:
                link = parents[link]
            out.append((link if link in mesh_nodes else node, name, m))
        else:
            out.append((name, m))
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


def _head(name, friction, visuals, soft=()):
    """The model's opening lines up to its visual assets: compiler, the visual and collision classes, and
    each visual's texture, material and mesh. soft: the links that are a flex, whose visuals are skins and need
    no mesh."""
    a = quoteattr
    bits = ' contype="3" conaffinity="2"' if soft else ""
    lines = [f'<mujoco model={a(name)}>',
             '  <compiler angle="radian" meshdir="assets" texturedir="assets"/>',
             *(['  <option integrator="discrete" solver="CG"/>'] if soft else []),
             '  <default>',
             f'    <default class={a(name + "_visual")}>',
             '      <geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/>',
             '    </default>',
             f'    <default class={a(name + "_collision")}>',
             f'      <geom type="mesh" group="3"{bits} density="0" friction="{friction:g} 0.005 0.0001" '
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
        if v.get("link") not in soft:
            lines.append(f'    <mesh name={a(name + "_" + v["mesh"])} file={a(v["mesh"] + ".obj")} inertia="shell"/>')
    return lines


def _vec(x):
    return " ".join(f"{c:.6g}" for c in x)


def _fine(x):
    """Numbers for a binding: 6 digits would move a node by microns."""
    return " ".join(f"{c:.9g}" for c in np.ravel(x))


def _ints(x):
    return " ".join(map(str, np.ravel(x)))


def flexcomp_xml(flex, rec, j, friction, ind):
    """The <flexcomp> of a soft link: rec its links record, flex its mesh, j its joint."""
    return [f'{ind}<flexcomp name={quoteattr(rec["flex"])} type="direct" dim="3" dof="full" group="3" radius="0.002" '
            f'mass="{rec["inertial"]["mass"]:.6g}" point="{_fine(flex["nodes"])}" element="{_ints(flex["tets"])}">',
            f'{ind}  <contact contype="1" conaffinity="0" selfcollide="none" friction="{friction:g} 0.005 0.0001"/>',
            f'{ind}  <edge damping="0.01"/>',
            f'{ind}  <elasticity young="{j["young"]:.6g}" poisson="{j["poisson"]:.6g}"/>',
            f'{ind}  <pin id="{_ints(flex["pinned"])}"/>',
            f'{ind}</flexcomp>']


def skin_xml(name, visual, mesh, flex, rec, pinned_body):
    """The <skin> of one visual of a soft link. mesh: that primitive; pinned_body: the parent's body. Bones are
    bound in S: a free node at its rest position, the pinned nodes at their body's rest origin, which is S's."""
    a = quoteattr
    start, stop = flex["ranges"][visual["mesh"]]
    uv = f' texcoord="{_fine(np.c_[mesh.visual.uv[:, 0], 1 - mesh.visual.uv[:, 1]])}"' if visual["texture"] else ""
    label = a(name + "_" + visual["mesh"])
    out = [f'    <skin name={label} material={label} group="2" vertex="{_fine(mesh.vertices)}"{uv} '
           f'face="{_ints(mesh.faces)}">']
    for node, ids, weights in soft_mesh.bones(flex["tets"], flex["pinned"], flex["tet_of_vertex"][start:stop],
                                              flex["weights"][start:stop]):
        body, bind = (pinned_body, np.zeros(3)) if node < 0 else (f'{rec["flex"]}_{node}', flex["nodes"][node])
        out.append(f'      <bone body={a(body)} bindpos="{_fine(bind)}" bindquat="1 0 0 0" vertid="{_ints(ids)}" '
                   f'vertweight="{_fine(weights)}"/>')
    return out + ['    </skin>']


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


def articulated_xml(name, root, children, joints, welded, links, visuals, friction, z0, free, excluded=(), soft=None,
                    primitives=None, actuators=None):
    """The object as a body tree (see the module docstring). links: {link: {body, inertial, collision_hulls}};
    visuals carry their link; welded: {loose link: link}; excluded: link pairs that do not collide; soft: {link:
    its mesh, from soft_links} of the links that are a flex; primitives: {visual name: mesh} of their skins;
    actuators: {link: gains, from actuator_gains} of the hinges and slides."""
    a = quoteattr
    soft, actuators = soft or {}, actuators or {}
    lines = _head(name, friction, visuals, soft)
    for link, rec in links.items():
        if link in soft:
            continue
        for j in range(rec["collision_hulls"]):
            lines.append(f'    <mesh name={a(f"{name}_collision_{link}_{j}")} file={a(f"collision_{link}_{j}.obj")} '
                         'inertia="shell"/>')
    lines += ['  </asset>', '  <worldbody>']

    def body(link, depth, joint):
        ind = "  " * (depth + 2)
        rec = links[link]
        pos = f"0 0 {z0:.4g}" if link == root else "0 0 0"
        out = [f'{ind}<body name={a(rec["body"])} pos="{pos}">']
        if link == root and free:
            out.append(f"{ind}  <freejoint/>")
        if joint is not None and joint["type"] not in joint_spec.STATIC:
            kind = "hinge" if joint["type"] == "revolute" else "slide"
            out.append(f'{ind}  <joint name={a(rec["body"] + "_joint")} type="{kind}" pos="{_vec(joint["origin"])}" '
                       f'axis="{_vec(joint["axis"])}" range="{joint["lower"]:.6g} {joint["upper"]:.6g}" limited="true" '
                       f'damping="{actuators[link]["damping"]:.6g}"/>')
        i = rec["inertial"]
        out.append(f'{ind}  <inertial pos="{_vec(i["com"])}" mass="{i["mass"]:.6g}" fullinertia="{_vec(i["fullinertia"])}"/>')
        for v in visuals:
            if v["link"] == link:
                out.append(f'{ind}  <geom name={a(name + "_" + v["mesh"])} class={a(name + "_visual")} '
                           f'mesh={a(name + "_" + v["mesh"])} material={a(name + "_" + v["mesh"])}/>')
        for j in range(rec["collision_hulls"]):
            out.append(f'{ind}  <geom name={a(f"{name}_collision_{link}_{j}")} class={a(name + "_collision")} '
                       f'mesh={a(f"{name}_collision_{link}_{j}")}/>')
        for c in children[link]:
            if c in soft:
                out += flexcomp_xml(soft[c], links[c], joints[c], friction, ind + "  ")
            else:
                out += body(c, depth + 1, joint_spec.full(joints[c]))
        for c in [k for k, h in welded.items() if h == link]:
            out += body(c, depth + 1, None)
        return out + [f"{ind}</body>"]

    lines += body(root, 0, None) + ['  </worldbody>']
    if excluded:
        lines.append('  <contact>')
        lines += [f'    <exclude body1={a(links[p]["body"])} body2={a(links[q]["body"])}/>' for p, q in excluded]
        lines.append('  </contact>')
    if actuators:
        lines.append('  <actuator>')
        lines += [f'    <position name={a(g["name"])} joint={a(g["name"])} kp="{g["kp"]:.6g}" kv="{g["kv"]:.6g}" '
                  f'ctrllimited="true" ctrlrange="{joints[k]["lower"]:.6g} {joints[k]["upper"]:.6g}"/>'
                  for k, g in actuators.items()]
        lines.append('  </actuator>')
    skins =[line for v in visuals if v["link"] in soft
             for line in skin_xml(name, v, primitives[v["name"]], soft[v["link"]], links[v["link"]],
                                  links[joints[v["link"]]["parent"]]["body"])]
    if skins:
        lines += ['  <deformable>', *skins, '  </deformable>']
    lines.append('</mujoco>')
    return "\n".join(lines) + "\n"


def halfspaces(hulls):
    """[(lo, hi, normals, offsets)] of convex hulls [(vertices, faces)]: the box around each and its face
    planes, n . x <= d inside. A flat hull, which has no faces of its own, stands as its box."""
    import trimesh

    out = []
    for v, f in hulls:
        v = np.asarray(v, float)
        lo, hi = v.min(0), v.max(0)
        try:
            h = trimesh.Trimesh(v, f, process=False).convex_hull
            n, d = h.face_normals, (h.face_normals * h.triangles[:, 0]).sum(1)
        except Exception:  # noqa: BLE001 - qhull refuses flat point sets
            n, d = np.vstack([np.eye(3), -np.eye(3)]), np.r_[hi, -lo]
        out.append((lo, hi, n, d))
    return out


def touching(a, b, tol=1e-3):
    """True when a hull of one set (halfspaces) overlaps one of the other or lies within about tol of it:
    some point is within tol of the inside of every face plane of both (a linear program)."""
    from scipy.optimize import linprog

    for lo_a, hi_a, na, da in a:
        for lo_b, hi_b, nb, db in b:
            if (lo_a > hi_b + tol).any() or (lo_b > hi_a + tol).any():
                continue  # the boxes around them are apart
            r = linprog(np.zeros(3), A_ub=np.vstack([na, nb]), b_ub=np.r_[da, db] + tol, bounds=[(None, None)] * 3,
                        method="highs")
            if r.status == 0:
                return True
    return False


def excluded_pairs(links, joints, welded, root, hulls):
    """[(link, link, why)] that must not collide (see the module docstring). why is "joint" for a jointed
    link (or a link welded to it) and its parent's links, "touch" for two links that touch at rest.
    hulls: {link: [(vertices, faces)]}."""
    planes = {}
    parent = {**{c: j["parent"] for c, j in joints.items() if j["parent"] != joint_spec.WORLD}, **welded}
    moving = {c for c, j in joints.items() if j["type"] not in joint_spec.STATIC}

    def group(k):  # the link whose body carries k's motion (MuJoCo's weld body)
        while k != root and k not in moving:
            k = parent[k]
        return k

    out = []
    for i, p in enumerate(links):
        for q in links[i + 1:]:
            gp, gq = group(p), group(q)
            if gp == gq:
                continue  # one rigid body: MuJoCo never collides it with itself
            if (gp in moving and group(parent[gp]) == gq) or (gq in moving and group(parent[gq]) == gp):
                out.append((p, q, "joint"))
            else:
                for k in (p, q):
                    if k not in planes:
                        planes[k] = halfspaces(hulls[k])
                if touching(planes[p], planes[q]):
                    out.append((p, q, "touch"))
    return out


def soft_links(joints, welded, loaded, visuals):
    """({link: mesh}, {link: {code, reason}}) of the soft joints: each link's hex mesh and binding (from
    harpor.physics.soft), with ranges {visual: (start, stop)} into its vertices, or why it has none.
    welded: {link: trimesh with its seams welded}; loaded and visuals: as in build_links."""
    meshes, failed = {}, {}
    for link, j in joints.items():
        if j["type"] != "soft":
            continue
        try:
            flex = soft_mesh.hex_mesh(welded[link], welded[j["parent"]])
        except soft_mesh.SoftError as e:
            failed[link] = {"code": e.code, "reason": str(e)}
            continue
        group = [(n, m) for k, n, m in loaded if k == link]
        stops = np.cumsum([len(m.vertices) for _, m in group]).tolist()
        spans = {n: (stop - len(m.vertices), stop) for (n, m), stop in zip(group, stops)}
        meshes[link] = {**flex, **soft_mesh.binding(flex, [m for _, m in group]),
                        "ranges": {v["mesh"]: spans[v["name"]] for v in visuals if v["link"] == link}}
    return meshes, failed


def flex_of(body, bodies):
    """The name of a soft link's flex: <body>_flex, or <body>_flex<n> with the smallest n for which no body of
    bodies has the name of one of its nodes, which MuJoCo names <flex>_<k> (a link <link>_flex_7)."""
    n = 0
    while any(re.fullmatch(rf"{re.escape(body)}_flex{n or ''}_\d+", b) for b in bodies):
        n += 1
    return f"{body}_flex{n or ''}"


def write_binding(path, flex, link, flex_name, start_z, glb):
    """Save soft/<link>.npz, which check_binding and the press read."""
    arrays = ("nodes", "tets", "pinned", "press", "direction", "vertices", "tet_of_vertex", "weights")
    np.savez(path, link=link, flex=flex_name, start_z=start_z, **{k: flex[k] for k in arrays}, visuals=list(flex["ranges"]),
             ranges=np.array(list(flex["ranges"].values()), int).reshape(-1, 2),
             asset_sha256=hashlib.sha256(Path(glb).read_bytes()).hexdigest())


def hull_inertial(hulls, density):
    """Mass properties of convex hulls [(vertices, faces)] filled at density (kg/m^3), about their COM."""
    import trimesh

    bodies = []
    for v, f in hulls:
        h = trimesh.Trimesh(v, f, process=False).convex_hull
        if h.volume > 0:
            h.density = density
            bodies.append({"mass": h.mass, "volume": h.volume, "com": h.center_mass.tolist(),
                           "inertia": h.moment_inertia.tolist()})
    if not bodies:
        raise ValueError("the collision hulls have no volume")
    return combine(bodies)


def moving_with(link, children, welded):
    """link and every link that moves with it: its jointed children, the links welded to these, and so on."""
    out, todo = [], [link]
    while todo:
        k = todo.pop()
        out.append(k)
        todo += children[k] + [w for w, h in welded.items() if h == k]
    return out


def axis_inertia(joint, inertials):
    """The inertia of rigid bodies [{mass, com, fullinertia}] about a hinge's axis line (parallel axis), or their
    mass for a slide."""
    whole = combine(inertials)
    if joint["type"] == "prismatic":
        return whole["mass"]
    a = np.asarray(joint["axis"], float)
    r = np.asarray(whole["com"]) - joint["origin"]
    r -= (r @ a) * a
    return float(a @ np.asarray(whole["inertia"]) @ a + whole["mass"] * r @ r)


def actuator_gains(joints, children, welded, links):
    """{link: {name, inertia, kp, kv, damping}} of every hinge and slide (see the module docstring). links: {link:
    {body, inertial}}."""
    out = {}
    for link, rec in links.items():
        j = joints.get(link)
        if j is None or j["type"] in joint_spec.STATIC:
            continue
        I = axis_inertia(j, [links[k]["inertial"] for k in moving_with(link, children, welded)])
        kv = 2 * OMEGA * I
        out[link] = {"name": rec["body"] + "_joint", "inertia": I, "kp": OMEGA ** 2 * I, "kv": kv, "damping": 0.2 * kv}
    return out


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


def build(glb, summary, out, name, friction=0.6, threshold=0.05, decompose=convex_parts, fixed_base=False,
          textures=True):
    """Write the model (see the module docstring) and mjcf.json. fixed_base applies to a GLB with joints."""
    import trimesh

    joints, links = joint_spec.load(glb)
    if joints:
        return build_links(glb, summary, out, name, joints, links, friction, threshold, decompose, fixed_base, textures)
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


def build_links(glb, summary, out, name, joints, links, friction=0.6, threshold=0.05, decompose=convex_parts,
                fixed_base=False, textures=True):
    """build() for a GLB with joints: one body per link (see the module docstring)."""
    import trimesh

    from harpor.physics.align import read_glb

    problems = joint_spec.validate(joints, links)
    if not problems and any(j["frame"] != "S" for j in joints.values()):
        problems.append("the joints must be in frame S; run harpor.physics.align first")
    if problems:
        raise ValueError(f"{glb}: bad joints: " + "; ".join(problems))
    out = Path(out)
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    s = json.loads(Path(summary).read_text())
    root, children, base = joint_spec.tree(joints, links)
    weld_to = joint_spec.welds(joints, links, root, joint_spec.holders(read_glb(glb)[0]))
    soft_holder = {k: h for k, h in weld_to.items() if joints.get(h, {}).get("type") == "soft"}
    weld_to.update({k: joints[h]["parent"] for k, h in soft_holder.items()})
    free = not (fixed_base or base)
    loaded = load_s(glb, by_link=True)
    visuals, skipped = visual_parts([(n, m) for _, n, m in loaded], assets, textures)
    link_of = {n: k for k, n, _ in loaded}
    for v in visuals:
        v["link"] = link_of[v["name"]]
    warnings = [f"{k} has no joint, so it is welded to {h}" for k, h in weld_to.items()]
    warnings += [f"{k} is held by the soft link {h}, so it is welded to its parent" for k, h in soft_holder.items()]
    rho = s["median"]["density_kg_m3"]
    recs, hulls_of, welded_of = {}, {}, {}
    for link in links:
        group = [m for k, _, m in loaded if k == link]
        if not group:
            raise ValueError(f"{glb}: the link {link} has no geometry")
        V = np.concatenate([m.vertices for m in group])
        F = np.concatenate([m.faces + o for m, o in zip(group, np.cumsum([0] + [len(m.vertices) for m in group]))])
        welded = trimesh.Trimesh(V, F, process=False)  # weld the UV-seam copies, as build() does
        welded.merge_vertices()
        hulls = decompose(np.asarray(welded.vertices), np.asarray(welded.faces), threshold)
        for j, (v, f) in enumerate(hulls):
            write_obj(assets / f"collision_{link}_{j}.obj", v, f)
        vomp = s.get("links", {}).get(link)
        if joints.get(link, {}).get("type") == "soft":  # VoMP's density would make a cushion a solid block
            inertial, source = hull_inertial(hulls, joints[link]["density"]), "soft_table"
        elif vomp:
            inertial, source = vomp["rigid_body"], "vomp"
        else:
            inertial, source = hull_inertial(hulls, rho), "hull"
            warnings.append(f"{link} has no VoMP result, so its inertial is its collision hulls filled at the "
                            f"object's median density, {rho:g} kg/m^3")
        recs[link] = {"body": name if link == root else f"{name}_{link}", "source": source,
                      "inertial": {k: inertial[k] for k in ("mass", "com", "fullinertia")},
                      "collision_hulls": len(hulls), "collision_input_watertight": bool(welded.is_watertight),
                      "visuals": [v["mesh"] for v in visuals if v["link"] == link]}
        hulls_of[link], welded_of[link] = hulls, welded
    soft, soft_rigid = soft_links(joints, welded_of, loaded, visuals)
    for link in soft:
        recs[link]["flex"] = flex_of(recs[link]["body"], [r["body"] for r in recs.values()])
    warnings += [f"{k} stays rigid: {r['reason']}" for k, r in soft_rigid.items()]
    warnings += [f"{k} is not closed, so its soft mesh is hollow" for k in soft if not recs[k]["collision_input_watertight"]]
    all_hulls = [h for link in links for h in hulls_of[link]]
    low = min(float(np.min(v[:, 2])) for v, _ in all_hulls)
    low = min([low] + [float(f["nodes"][:, 2].min()) for f in soft.values()])
    z0 = max(0.0, -low) + 0.005  # as build()
    pairs = excluded_pairs(links, joints, weld_to, root, hulls_of)
    excluded = [p for p in pairs if not soft.keys() & set(p[:2])]  # a flex has no body and misses its object anyway
    warnings += [f"{p} and {q} touch at rest, so they do not collide with each other" for p, q, why in excluded
                 if why == "touch"]
    actuators = actuator_gains(joints, children, weld_to, recs)
    (out / f"{name}.xml").write_text(articulated_xml(name, root, children, joints, weld_to, recs, visuals, friction, z0,
                                                     free, [(p, q) for p, q, _ in excluded], soft,
                                                     {n: m for _, n, m in loaded}, actuators))
    (out / "scene.xml").write_text(scene_xml(name, friction))
    soft_report, start_z = {}, float(f"{z0:.4g}")  # the root's height as the XML has it
    for link, f in soft.items():
        (out / "soft").mkdir(exist_ok=True)
        write_binding(out / "soft" / f"{link}.npz", f, link, recs[link]["flex"], start_z, glb)
        soft_report[link] = {"file": f"soft/{link}.npz", "nodes": len(f["nodes"]), "tets": len(f["tets"]),
                             "pinned": len(f["pinned"]), "pitch": f["pitch"], "material": joints[link]["material"],
                             **{k: joints[link][k] for k in ("young", "poisson", "density")},
                             "mass": recs[link]["inertial"]["mass"]}
    whole = combine([r["inertial"] for r in recs.values()])
    moving = [(c, joint_spec.full(joints[c])) for c in links if c in joints and joints[c]["parent"] != joint_spec.WORLD]
    report = {"glb": str(Path(glb).resolve()), "summary": str(Path(summary).resolve()), "config": s.get("config"),
              "name": name, "friction": friction, "coacd_threshold": threshold, "visuals": visuals,
              "skipped_visuals": skipped,
              "collision_input_watertight": all(r["collision_input_watertight"] for r in recs.values()),
              "collision_hulls": len(all_hulls), "collision_faces": int(sum(len(f) for _, f in all_hulls)),
              "lowest_collision_z": low, "start_z": z0, "inertial": {k: whole[k] for k in ("mass", "com", "fullinertia")},
              "root": root, "fixed_base": not free, "links": recs,
              "joints": [{"child": c, "parent": j["parent"], "type": j["type"], "pos": j["origin"], "axis": j["axis"],
                          "range": [j["lower"], j["upper"]]} for c, j in moving],
              "loose": list(weld_to), "welded": weld_to, "excluded_pairs": [[p, q] for p, q, _ in excluded],
              "soft_excluded_pairs": [[p, q] for p, q, _ in pairs if soft.keys() & {p, q}], "soft": soft_report,
              "soft_rigid": soft_rigid, "actuators": actuators, "warnings": warnings}
    (out / "mjcf.json").write_text(json.dumps(report, indent=1))
    return report


def _body_points(m, d, body):
    """World vertices of the mesh geoms of one body, (N, 3)."""
    import mujoco

    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == body and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
    return np.concatenate(pts)


def _flex_range(m, f):
    """The slice of flex f's nodes in flex_vertbodyid and flexvert_xpos."""
    return slice(m.flex_vertadr[f], m.flex_vertadr[f] + m.flex_vertnum[f])


def node_bodies(m):
    """Ids of the bodies that only carry a flex's nodes: they hold its slide joints and no geom. A pinned
    node lists the body it hangs from, which keeps its geoms, and m.flex_nodebodyid is empty for a direct flex."""
    return {int(b) for b in m.flex_vertbodyid if m.body_geomnum[b] == 0}


def probe(m):
    """Forward kinematics with one hinge or slide joint at a time at its upper limit: the child body's
    mesh bounds at rest and there, and its largest vertex move. A flex's node joints are left out."""
    import mujoco

    d = mujoco.MjData(m)
    nodes = node_bodies(m)
    out = []
    for i in range(m.njnt):
        if int(m.jnt_type[i]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        b = m.jnt_bodyid[i]
        if int(b) in nodes:
            continue
        d.qpos[:] = m.qpos0
        mujoco.mj_kinematics(m, d)
        rest = _body_points(m, d, b)
        d.qpos[m.jnt_qposadr[i]] = m.jnt_range[i][1]
        mujoco.mj_kinematics(m, d)
        moved = _body_points(m, d, b)
        out.append({"joint": m.joint(i).name, "body": m.body(b).name, "q": float(m.jnt_range[i][1]),
                    "max_move_m": float(np.linalg.norm(moved - rest, axis=1).max()),
                    "rest": [rest.min(0).tolist(), rest.max(0).tolist()],
                    "upper": [moved.min(0).tolist(), moved.max(0).tolist()]})
    return out


def rest_contacts(m):
    """Sorted [body, body] pairs of the object's own bodies (not the world) in contact at qpos0. A flex's
    contacts have geom -1 and are left out."""
    import mujoco

    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pairs = {tuple(sorted((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name)))
             for c in d.contact[:d.ncon]
             if min(c.geom1, c.geom2) >= 0 and m.geom_bodyid[c.geom1] and m.geom_bodyid[c.geom2]}
    return [list(p) for p in sorted(pairs)]


def check(out, name):
    """Compile scene.xml: the root body's mass and inertia (to compare with the inertial it was given), every
    body's mass, the hinge and slide joints, probe() of each, and the bodies in contact at rest. The bodies
    and joints of a flex's nodes (node_bodies) are not counted; soft_mass gives each flex's compiled mass."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(Path(out) / "scene.xml"))
    b = m.body(name).id
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_iquat[b])
    R = R.reshape(3, 3)
    I = R @ np.diag(m.body_inertia[b]) @ R.T
    nodes = node_bodies(m)
    kept = [i for i in range(m.njnt) if int(m.jnt_bodyid[i]) not in nodes]
    kinds = {int(mujoco.mjtJoint.mjJNT_HINGE): "hinge", int(mujoco.mjtJoint.mjJNT_SLIDE): "slide"}
    soft_mass = {}
    for f in range(m.nflex):
        link = re.sub(rf"^{re.escape(name)}_|_flex\d*$", "", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, f))
        soft_mass[link] = float(m.body_mass[sorted({int(b) for b in m.flex_vertbodyid[_flex_range(m, f)]} & nodes)].sum())
    joints = [{"name": m.joint(i).name, "type": kinds[int(m.jnt_type[i])], "body": m.body(m.jnt_bodyid[i]).name,
               "range": m.jnt_range[i].tolist()} for i in kept if int(m.jnt_type[i]) in kinds]
    return {"mujoco": mujoco.__version__, "nbody": m.nbody - len(nodes), "ngeom": m.ngeom, "nmesh": m.nmesh,
            "ntex": m.ntex, "mass": float(m.body_mass[b]), "com": m.body_ipos[b].tolist(),
            "fullinertia": [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]], "njnt": len(kept),
            "bodies": {m.body(i).name: float(m.body_mass[i]) for i in range(1, m.nbody) if i not in nodes},
            "soft_mass": soft_mass, "joints": joints, "probe": probe(m) if joints else [],
            "rest_contacts": rest_contacts(m)}


def check_binding(m, npz, glb):
    """Problems of a binding file (soft/<link>.npz) against the compiled model m and the GLB in S, as phrases;
    an empty list is a good binding. The nodes come from mj_forward at qpos0, because flex_vert0 is normalized
    to [0, 1] and mj_kinematics leaves flexvert_xpos at zeros."""
    import mujoco

    with np.load(npz) as z:
        b = {k: z[k] for k in z.files}
    nodes, tets, tet_of_vertex, weights, vertices = (b[k] for k in ("nodes", "tets", "tet_of_vertex", "weights", "vertices"))
    flex = str(b["flex"])
    f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, flex)
    if f < 0:
        return [f"the model has no flex {flex}"]
    r = _flex_range(m, f)
    if len(nodes) != len(m.flex_vertbodyid[r]):
        return [f"{len(nodes)} nodes against {len(m.flex_vertbodyid[r])} in the flex"]
    if not (tets.min() >= 0 and tets.max() < len(nodes) and tet_of_vertex.min() >= 0 and tet_of_vertex.max() < len(tets)):
        return ["an index is out of range"]
    if weights.shape != (len(vertices), 4) or tet_of_vertex.shape != (len(vertices),):
        return [f"{len(vertices)} vertices against weights of shape {weights.shape}"]
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = []
    if np.abs(d.flexvert_xpos[r] - [0, 0, float(b["start_z"])] - nodes).max() > 1e-6:
        out.append("the nodes are not where the flex is at rest")
    if np.abs(weights.sum(1) - 1).max() > 1e-6:
        out.append("a weight row does not sum to 1")
    if np.abs(weights).max() > 3:
        out.append("a weight is above 3")
    if np.abs((nodes[tets[tet_of_vertex]] * weights[..., None]).sum(1) - vertices).max() > 1e-5:
        out.append("the weights do not rebuild the vertices")
    group = [mesh.vertices for k, _, mesh in load_s(glb, by_link=True) if k == str(b["link"])]
    want = np.concatenate(group) if group else np.zeros((0, 3))
    if want.shape != vertices.shape or np.abs(want - vertices).max() > 1e-6:
        out.append("the vertices are not the GLB's")
    return out


def binding_warnings(out, glb, soft):
    """check_binding of each link of mjcf.json soft, as warnings: a bad skin does not stop the export."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(Path(out) / "scene.xml"))
    return [f"the binding of {link}: {why}" for link, rec in soft.items()
            for why in check_binding(m, Path(out) / rec["file"], glb)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="object")
    ap.add_argument("--friction", type=float, default=0.6)
    ap.add_argument("--coacd-threshold", type=float, default=0.05)
    ap.add_argument("--fixed-base", action="store_true", help="a GLB with joints: no free joint on the root link")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z_][\w-]*", a.name) or a.name == "scene":
        ap.error("--name must be a plain identifier other than 'scene' (scene.xml includes <name>.xml)")
    report = build(a.glb, a.summary, a.out, a.name, a.friction, a.coacd_threshold, fixed_base=a.fixed_base)
    report["compiled"] = check(a.out, a.name)
    if report.get("soft"):
        report["warnings"] += binding_warnings(a.out, a.glb, report["soft"])
    (Path(a.out) / "mjcf.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("collision_hulls", "start_z", "inertial", "compiled")}, indent=1))


if __name__ == "__main__":
    main()
