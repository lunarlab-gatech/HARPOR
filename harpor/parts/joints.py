"""Joint specs of an articulated GLB (build_plan.md note 11).

A part's joint lives on the part's glTF node as extras.gator_joint, a JSON string (Blender exports string
custom properties as extras reliably; readers accept a dict too):
    {"version": 1, "frame": "W"|"S", "parent": name|"world", "type": "revolute"|"prismatic"|"fixed"|"soft",
     "origin": [x, y, z], "axis": [unit], "lower": l, "upper": u}
The node's name is the child. origin and axis are coordinates of the frame (W: view 0's camera, which is
Blender's world; S: the gravity frame of harpor.physics.align), not of the node. The mesh as it is is the
pose q = 0, and lower <= 0 <= upper, in radians (revolute) or meters (prismatic). A fixed joint may leave
out origin, axis, lower and upper (read as 0, +x, 0, 0). parent "world" with type fixed on the root
marks a fixed base. A soft joint holds a part that dents under a press and springs back. It has no joint
coordinate either, so it may leave out the same keys, and it adds "material" (a key of SOFT_MATERIALS),
"young" (Pa), "poisson", "density" (kg/m^3) and "support": "parent". No part hangs from a soft part. The
links are the GLB's mesh nodes; one link has no parent (or the world): the root. A link other than the
root without a joint is loose. The physics export welds it to the link of its nearest glTF ancestor that
holds a mesh (holders), so a tray parented to a drawer in Blender moves with the drawer, or to the root
when it has no such ancestor (welds).

numpy, json and math only, so a Blender script can import it too.
"""

import json
import math

import numpy as np

KEY = "gator_joint"
TYPES = ("revolute", "prismatic", "fixed", "soft")
STATIC = ("fixed", "soft")  # no joint coordinate
FRAMES = ("W", "S")
WORLD = "world"
FIXED_DEFAULTS = {"origin": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0], "lower": 0.0, "upper": 0.0}
SOFT_MATERIALS = {"foam": {"young": 3e4, "poisson": 0.3, "density": 120.0},
                  "gel": {"young": 1e5, "poisson": 0.3, "density": 900.0},
                  "rubber": {"young": 1e6, "poisson": 0.3, "density": 1100.0}}
YOUNG_RANGE, POISSON_RANGE = (1e3, 1e6), (0.0, 0.45)
SOFT_AS_RIGID = "rigid here, soft in MuJoCo"  # the note of a soft link in the URDF and USD reports


def mesh_nodes(doc):
    """Names of the glTF nodes that hold a mesh, in node order (None for a node without a name)."""
    return [n.get("name") for n in doc.get("nodes", []) if "mesh" in n]


def read_joints(doc):
    """{child node name: joint dict} of every node whose extras hold gator_joint."""
    out = {}
    for i, n in enumerate(doc.get("nodes", [])):
        extras = n.get("extras")
        j = extras.get(KEY) if isinstance(extras, dict) else None
        if j is None:
            continue
        name = n.get("name")
        if name is None:
            raise ValueError(f"node {i} holds a joint but has no name")
        if isinstance(j, str):
            try:
                j = json.loads(j)
            except json.JSONDecodeError as e:
                raise ValueError(f"the joint of node {name!r} is not valid JSON: {e}") from None
        if not isinstance(j, dict):
            raise ValueError(f"the joint of node {name!r} is not a JSON object")
        if name in out:
            raise ValueError(f"two nodes named {name!r} hold a joint")
        out[name] = j
    return out


def write_joints(doc, joints):
    """Copy of a glTF dict whose nodes hold exactly these joints, as JSON strings. Every other node loses
    gator_joint. Raises ValueError for a child that names no node."""
    doc = json.loads(json.dumps(doc))
    nodes = doc.get("nodes", [])
    missing = sorted(set(joints) - {n.get("name") for n in nodes})
    if missing:
        raise ValueError(f"no node is named {', '.join(map(repr, missing))}")
    for n in nodes:
        extras = n.get("extras")
        if isinstance(extras, dict):
            extras.pop(KEY, None)
            if not extras:
                del n["extras"]
        if n.get("name") in joints:
            if not isinstance(n.get("extras"), dict):
                n["extras"] = {}
            n["extras"][KEY] = json.dumps(joints[n["name"]])
    return doc


def load(path):
    """(joints, links) of a GLB file: read_joints and mesh_nodes of its JSON chunk."""
    from harpor.physics.align import read_glb

    doc, _ = read_glb(path)
    return read_joints(doc), mesh_nodes(doc)


def full(joint):
    """Copy of a joint with the defaults of a joint without a coordinate (STATIC) filled in."""
    j = dict(joint)
    if j.get("type") in STATIC:
        for k, v in FIXED_DEFAULTS.items():
            j.setdefault(k, list(v) if isinstance(v, list) else v)
    return j


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _vec3(x):
    return isinstance(x, (list, tuple)) and len(x) == 3 and all(_number(c) for c in x)


def _within(x, lo, hi):
    return _number(x) and lo <= x <= hi


def _soft_problems(j):
    """What is wrong with the fields of a soft joint, as phrases."""
    out = []
    if j.get("material") not in tuple(SOFT_MATERIALS):
        out.append(f"material must be one of {', '.join(SOFT_MATERIALS)}")
    if not _within(j.get("young"), *YOUNG_RANGE):
        out.append("young must be within {:g} to {:g} Pa".format(*YOUNG_RANGE))
    if not _within(j.get("poisson"), *POISSON_RANGE):
        out.append("poisson must be within {:g} to {:g}".format(*POISSON_RANGE))
    if not _number(j.get("density")) or j["density"] <= 0:
        out.append("density must be a positive number")
    if j.get("support") != "parent":
        out.append('support must be "parent"')
    return out


def _top(joints, child):
    """(top link, cycle) of child's parent chain: the first link without a joint or attached to the world.
    cycle is the list of links of a loop, or None."""
    seen, node = [child], child
    while True:
        j = joints.get(node)
        parent = j.get("parent") if isinstance(j, dict) else None
        if parent is None or parent == WORLD or parent == node:
            return node, None
        if parent in seen:
            return None, seen[seen.index(parent):] + [parent]
        seen.append(parent)
        node = parent


def validate(joints, node_names):
    """Problems of a joint set as sentences; an empty list is a valid tree. node_names are the GLB's mesh
    nodes (mesh_nodes)."""
    errors = []
    names = list(node_names)
    known = {n for n in names if n is not None}
    if None in names:
        errors.append("a mesh node has no name")
    dup = sorted(n for n in known if names.count(n) > 1)
    if dup:
        errors.append(f"two mesh nodes share a name: {', '.join(dup)}")
    if WORLD in known:
        errors.append(f'a mesh node is named "{WORLD}", which is reserved for the fixed base')
    frames = set()
    for child, j in joints.items():
        p = f"joint of {child!r}"
        if not isinstance(j, dict):
            errors.append(f"the {p} is not an object")
            continue
        if child not in known:
            errors.append(f"the {p}: no mesh node is named {child!r}")
        if j.get("version") != 1:
            errors.append(f"the {p}: version must be 1")
        if j.get("frame") in FRAMES:
            frames.add(j["frame"])
        else:
            errors.append(f"the {p}: frame must be W or S")
        kind, parent = j.get("type"), j.get("parent")
        if kind not in TYPES:
            errors.append(f"the {p}: type must be revolute, prismatic, fixed or soft")
        if parent == child:
            errors.append(f"the {p}: the part is its own parent")
        elif parent == WORLD:
            if kind != "fixed":
                errors.append(f"the {p}: only a fixed joint attaches a part to the world")
        elif parent not in known:
            errors.append(f"the {p}: the parent {parent!r} is not a mesh node")
        elif isinstance(joints.get(parent), dict) and joints[parent].get("type") == "soft":
            errors.append(f"the {p}: a part cannot hang from a soft part ({parent!r})")
        if kind == "soft":
            errors += [f"the {p}: {why}" for why in _soft_problems(j)]
        moving = kind not in STATIC
        if "origin" in j or moving:
            if not _vec3(j.get("origin")):
                errors.append(f"the {p}: origin must be 3 finite numbers")
        if "axis" in j or moving:
            if not _vec3(j.get("axis")):
                errors.append(f"the {p}: axis must be 3 finite numbers")
            elif abs(np.linalg.norm(j["axis"]) - 1) > 1e-6:
                errors.append(f"the {p}: axis must be a unit vector (length {np.linalg.norm(j['axis']):.6g})")
        if "lower" in j or "upper" in j or moving:
            lo, hi = j.get("lower"), j.get("upper")
            if not (_number(lo) and _number(hi)):
                errors.append(f"the {p}: lower and upper must be finite numbers")
            elif not lo <= 0 <= hi:
                errors.append(f"the {p}: the range [{lo:g}, {hi:g}] must hold 0 (lower <= 0 <= upper)")
            elif moving and not lo < hi:
                errors.append(f"the {p}: a moving joint needs lower < upper; use type fixed for a part that "
                              "cannot move")
    if len(frames) > 1:
        errors.append("the joints mix frames W and S")
    roots, cycles = set(), []
    for child in joints:
        top, cycle = _top(joints, child)
        if cycle and sorted(cycle[:-1]) not in [sorted(c[:-1]) for c in cycles]:
            cycles.append(cycle)
        elif top in known:  # an unknown parent is reported above
            roots.add(top)
    for c in cycles:
        errors.append(f"the joints form a cycle: {' -> '.join(c)}")
    if len(roots) > 1:
        errors.append(f"the joints have {len(roots)} roots ({', '.join(sorted(roots))}); "
                      "exactly one part may have no parent")
    return errors


def transform(joints, T, frame="S"):
    """Joints moved by the rigid 4x4 T (origin as a point, axis as a direction), marked with frame."""
    T = np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    out = {}
    for child, j in joints.items():
        j = full(j)
        a = R @ np.asarray(j["axis"], float)
        out[child] = {**j, "frame": frame, "origin": (R @ np.asarray(j["origin"], float) + t).tolist(),
                      "axis": (a / np.linalg.norm(a)).tolist()}
    return out


def tree(joints, links):
    """(root, children, fixed_base) of a valid joint set over links (mesh node names, in order). children
    maps every link to its jointed children, in link order. Without joints the root is the first link.
    Raises ValueError when the joints do not have exactly one root."""
    links = list(links)
    if not joints:
        return links[0], {k: [] for k in links}, False
    roots = set()
    for child in joints:
        top, cycle = _top(joints, child)
        if cycle:
            raise ValueError(f"the joints form a cycle: {' -> '.join(cycle)}")
        roots.add(top)
    if len(roots) != 1:
        raise ValueError(f"the joints have {len(roots)} roots ({', '.join(sorted(roots))})")
    root = roots.pop()
    children = {k: [c for c in links if c in joints and joints[c].get("parent") == k] for k in links}
    return root, children, joints.get(root, {}).get("parent") == WORLD


def loose(joints, links, root):
    """Links that are neither the root nor jointed (the physics export welds them, see welds)."""
    return [k for k in links if k != root and k not in joints]


def parents(doc):
    """{node index: its parent's index} of a glTF dict. Raises ValueError when nodes form a cycle, which glTF forbids
    and which would loop every walk up the tree (holders, and harpor.agent.tools.carriers through it)."""
    nodes = doc.get("nodes", [])
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    done = set()  # nodes whose walk up ends at a root
    for i in range(len(nodes)):
        path, k = {}, i  # the walk up from node i, in order
        while k is not None and k not in done:
            if k in path:
                loop = list(path)[list(path).index(k):] + [k]
                raise ValueError(f"the glTF nodes form a cycle: {' -> '.join(map(str, loop))}")
            path[k] = None
            k = parent.get(k)
        done.update(path)
    return parent


def holders(doc):
    """{mesh node name: name of its nearest ancestor node that holds a mesh} of every mesh node that has one.
    Raises ValueError when nodes form a cycle (parents)."""
    nodes = doc.get("nodes", [])
    parent = parents(doc)
    out = {}
    for i, n in enumerate(nodes):
        if "mesh" not in n:
            continue
        p = parent.get(i)
        while p is not None and "mesh" not in nodes[p]:
            p = parent.get(p)
        if p is not None:
            out[n.get("name")] = nodes[p].get("name")
    return out


def welds(joints, links, root, held=None):
    """{loose link: the link it is welded to}: its holder (held, from holders) or else the root."""
    held = held or {}
    return {k: held.get(k, root) for k in loose(joints, links, root)}


def _rotation(axis, angle):
    """Rotation matrix of angle (rad) about a unit axis (Rodrigues)."""
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def joint_transform(joint, q):
    """4x4 motion of the child at joint value q, in the joint's frame: a turn by q rad about the axis line
    through origin (revolute), a shift of q m along the axis (prismatic), the identity (fixed, soft)."""
    j = full(joint)
    o, a = np.asarray(j["origin"], float), np.asarray(j["axis"], float)
    a = a / np.linalg.norm(a)
    T = np.eye(4)
    if j["type"] == "revolute":
        R = _rotation(a, q)
        T[:3, :3], T[:3, 3] = R, o - R @ o
    elif j["type"] == "prismatic":
        T[:3, 3] = q * a
    return T


def axis_rotation(axis):
    """Unit quaternion (w, x, y, z) of the shortest turn that takes +x onto axis (a half turn about z for
    -x). USD joints turn about or slide along their local x axis."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    c = float(a[0])
    if 1 + c < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    q = np.array([1 + c, 0.0, -a[2], a[1]])  # (1 + x.a, x cross a)
    q = q / np.linalg.norm(q)
    return tuple(float(v) for v in q)
