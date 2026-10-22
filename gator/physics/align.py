"""Gravity alignment of an agent GLB for VoMP and simulators (build_plan.md note 9).

The agent's GLB is posed in frame W (OpenCV camera of view 0), so it is not gravity-aligned. VoMP
renders with Blender +Z as image-up, and MuJoCo expects +Z up. This module writes a copy of the GLB
in frame S: z up, the footprint centred on the origin, the bottom at z = 0 and the longest horizontal
side on x. In glTF axes S is y up, which Blender's stock importer turns back into z up.

Up comes from one of two rules (--up), each signed against W's +y (the camera's down), which assumes
the camera was roughly level:
- pca: the thinnest principal axis of the mesh surface (exact second moment of its triangles). It
  holds for flat objects that rest on their broad face, such as a plate, a knife or a fork.
- normals: the area-weighted mean normal of the near-horizontal faces, refined to the dominant
  plane. It holds when one horizontal plane dominates, such as a patch of floor under the object.
--up-near X,Y,Z instead takes whichever of the object's own axes (the three principal axes and the
normals rule) lies closest to a hint in W, such as the table plane fitted to view 0's depth that the
objects of one photo set share. The object keeps its own up, so a mesh whose pose is a little off still
rests on its base, and the hint only settles which axis that is (a squat cup, a tall glass).
x comes from one of two rules (--x): pca, the major axis of the surface projected onto the
horizontal plane, or footprint, the long side of the minimum-area rectangle around the vertices
seen from above. Use footprint for near-square or open objects, where the PCA axis
can land on a diagonal, which also makes the bounding box and VoMP's voxels larger.

The GLB is not re-exported: the BIN chunk is copied byte for byte and the scene's root nodes go under
one new node that holds the transform, so textures and materials stay exactly as the agent left them.

    python -m gator.physics.align --glb IN.glb --out OUT.glb [--up pca|normals | --up-near X,Y,Z] [--x pca|footprint] [--opaque]
        [--links-out DIR]
writes OUT.glb and OUT.json (T_s_from_w, up in W, the camera tilt and the extents in S).
--opaque also removes transmission, volume and IOR extensions and makes every material opaque, because
VoMP rejects transmissive or partly transparent materials (the glass). The geometry is unchanged.

An articulated GLB (build_plan.md note 11) holds joint specs in its nodes' extras (gator.parts.joints),
in frame W. They are checked and moved to S with T_s_from_w, and OUT.json adds joints (in S), links (the
mesh nodes) and fixed_base; a rigid GLB gives {}, [] and false. --links-out DIR writes DIR/<link>.glb for
every mesh node (the old *.glb there are removed first): OUT.glb with only that node's mesh and no joint,
and the same BIN chunk, so VoMP sees each link with its textures.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from gator import frames
from gator.parts import joints as joint_spec


def _triangles(vertices_w, faces):
    tri = vertices_w[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    return tri, nrm, area


def up_from_normals(vertices_w, faces, down_w=(0.0, 1.0, 0.0), cone_deg=40.0, plane_deg=10.0):
    """Unit up in W from the near-horizontal faces (within cone_deg of camera up, either winding),
    refined to the normals within plane_deg of the dominant direction."""
    _, nrm, area = _triangles(vertices_w, faces)
    ok = area > 0
    n, a = nrm[ok] / (2 * area[ok, None]), area[ok]
    cam_up = -np.asarray(down_w, float)
    c = n @ cam_up
    sel = np.abs(c) > np.cos(np.radians(cone_deg))
    n, a = n[sel] * np.sign(c[sel])[:, None], a[sel]
    u = (a[:, None] * n).sum(0)
    for _ in range(5):
        u /= np.linalg.norm(u)
        keep = n @ u > np.cos(np.radians(plane_deg))
        u = (a[keep, None] * n[keep]).sum(0)
    return u / np.linalg.norm(u)


def footprint_yaw(xy):
    """Angle (rad) to turn about z so that the long side of the minimum-area rectangle around xy lies on x."""
    from scipy.spatial import ConvexHull

    h = xy[ConvexHull(xy).vertices]
    best = None
    for e in np.diff(np.vstack([h, h[:1]]), axis=0):
        a = np.arctan2(e[1], e[0])
        c, s = np.cos(-a), np.sin(-a)
        q = h @ np.array([[c, s], [-s, c]])
        ext = q.max(0) - q.min(0)
        if best is None or ext.prod() < best[0]:
            best = (ext.prod(), -a if ext[0] >= ext[1] else -a + np.pi / 2)
    return float((best[1] + np.pi / 2) % np.pi - np.pi / 2)


def up_near(vertices_w, faces, hint):
    """(up, name, angle_deg) of the object axis closest to hint: the three principal axes or the normals rule."""
    tri, _, area = _triangles(vertices_w, faces)
    wt = area / area.sum()
    s = tri.sum(1)
    mu = wt @ s / 3
    C = np.einsum("t,tij->ij", wt, np.einsum("tvi,tvj->tij", tri, tri) + s[:, :, None] * s[:, None, :]) / 12 - np.outer(mu, mu)
    V = np.linalg.eigh(C)[1]
    hint = np.asarray(hint, float) / np.linalg.norm(hint)
    cands = {"pca-thin": V[:, 0], "pca-mid": V[:, 1], "pca-long": V[:, 2], "normals": up_from_normals(vertices_w, faces)}
    ang = {k: float(np.degrees(np.arccos(np.clip(abs(u @ hint), 0, 1)))) for k, u in cands.items()}
    k = min(ang, key=ang.get)
    u = cands[k] * np.sign(cands[k] @ hint)
    return u, k, ang[k]


def up_frame(vertices_w, faces, up=None, down_w=(0.0, 1.0, 0.0), x_rule="pca"):
    """4x4 T_s_from_w (see the module docstring) and the surface PCA eigenvalues (ascending).
    up=None takes the thinnest principal axis (the pca rule)."""
    tri, _, area = _triangles(vertices_w, faces)
    wt = area / area.sum()
    s = tri.sum(1)
    mu = wt @ s / 3
    # Exact second moment of a uniform triangle: (aa' + bb' + cc' + ss') / 12 with s = a + b + c.
    second = np.einsum("t,tij->ij", wt, np.einsum("tvi,tvj->tij", tri, tri) + s[:, :, None] * s[:, None, :]) / 12
    C = second - np.outer(mu, mu)
    w, V = np.linalg.eigh(C)
    if up is None:
        up = V[:, 0]
    up = np.asarray(up, float) / np.linalg.norm(up)
    up = up if up @ np.asarray(down_w) < 0 else -up
    P = np.eye(3) - np.outer(up, up)
    x = np.linalg.eigh(P @ C @ P)[1][:, 2]
    R = np.stack([x, np.cross(up, x), up])
    if x_rule == "footprint":
        a = footprint_yaw(vertices_w @ R[:2].T)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ R
    p = vertices_w @ R.T
    lo, hi = p.min(0), p.max(0)
    t = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    return frames.rigid(R, t), w


def read_glb(path):
    """(json dict, BIN chunk bytes) of a GLB."""
    b = Path(path).read_bytes()
    magic, version, _ = struct.unpack("<4sII", b[:12])
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path} is not a glTF 2.0 binary")
    n, kind = struct.unpack("<II", b[12:20])
    if kind != 0x4E4F534A:
        raise ValueError(f"{path}: first chunk is not JSON")
    doc = json.loads(b[20:20 + n])
    rest = b[20 + n:]
    binary = b""
    if rest:
        m, kind = struct.unpack("<II", rest[:8])
        if kind != 0x004E4942:
            raise ValueError(f"{path}: second chunk is not BIN")
        binary = rest[8:8 + m]
    return doc, binary


def write_glb(path, doc, binary):
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    chunks = struct.pack("<II", len(js), 0x4E4F534A) + js
    if binary:
        chunks += struct.pack("<II", len(binary), 0x004E4942) + binary
    Path(path).write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)


def gltf_matrix(T_s_from_w):
    """The same transform in glTF axes, column-major as glTF node.matrix stores it."""
    G = np.eye(4)
    G[:3, :3] = frames.GLTF_FROM_W
    M = G @ T_s_from_w @ G.T
    return M.T.reshape(-1).tolist()


def wrap_scene(doc, matrix, name="gravity_align"):
    """Put the default scene's root nodes under one new node with this matrix. Returns a new dict."""
    doc = json.loads(json.dumps(doc))
    scene = doc["scenes"][doc.get("scene", 0)]
    doc["nodes"].append({"name": name, "matrix": matrix, "children": scene["nodes"]})
    scene["nodes"] = [len(doc["nodes"]) - 1]
    return doc


def link_doc(doc, node):
    """Copy of an aligned glTF dict (wrap_scene's) that holds only the mesh of node (an index), as a rigid GLB:
    the meshes list keeps that one mesh, every other node loses its mesh, no node keeps a joint, and the
    wrapper keeps only the branch that holds node. Accessors, materials and images stay, so the BIN chunk
    is unchanged."""
    doc = joint_spec.write_joints(doc, {})  # also a copy
    nodes = doc["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    wrapper = doc["scenes"][doc.get("scene", 0)]["nodes"][0]
    top = node
    while parent.get(top, wrapper) != wrapper:
        top = parent[top]
    nodes[wrapper]["children"] = [top]
    doc["meshes"] = [doc["meshes"][nodes[node]["mesh"]]]
    for i, n in enumerate(nodes):
        if i == node:
            n["mesh"] = 0
        else:
            n.pop("mesh", None)
    return doc


def write_links(doc, binary, out_dir):
    """DIR/<link>.glb (link_doc) for every mesh node of an aligned glTF dict. Returns the link names."""
    out_dir = Path(out_dir)
    names = joint_spec.mesh_nodes(doc)
    bad = [n for n in names if not n or "/" in n or "\\" in n or n.startswith(".")]
    if bad or len(set(names)) != len(names):
        raise SystemExit(f"--links-out needs unique mesh node names that are file names: {names}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.glb"):
        old.unlink()
    mesh_index = [i for i, n in enumerate(doc["nodes"]) if "mesh" in n]
    for name, i in zip(names, mesh_index):
        write_glb(out_dir / f"{name}.glb", link_doc(doc, i), binary)
    return names


def make_opaque(doc):
    """Copy of a glTF dict whose materials have no transmission, volume or IOR and are fully opaque."""
    doc = json.loads(json.dumps(doc))
    drop = {"KHR_materials_transmission", "KHR_materials_volume", "KHR_materials_ior"}
    for m in doc.get("materials", []):
        for k in drop:
            m.get("extensions", {}).pop(k, None)
        if not m.get("extensions", True):
            del m["extensions"]
        m.pop("alphaMode", None)
        m.pop("alphaCutoff", None)
        pbr = m.get("pbrMetallicRoughness", {})
        if "baseColorFactor" in pbr:
            pbr["baseColorFactor"][3] = 1.0
    for key in ("extensionsUsed", "extensionsRequired"):
        if key in doc:
            doc[key] = [e for e in doc[key] if e not in drop]
            if not doc[key]:
                del doc[key]
    return doc


def load_w(path):
    """World vertices (in frame W) and faces of every mesh in a GLB, with node transforms applied."""
    import trimesh

    s = trimesh.load(path, force="scene")
    m = s.to_geometry() if hasattr(s, "to_geometry") else s.dump(concatenate=True)
    return frames.gltf_to_w(np.asarray(m.vertices)), np.asarray(m.faces)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True, help="output .glb; the report goes next to it as .json")
    ap.add_argument("--up", choices=["pca", "normals"], default="pca")
    ap.add_argument("--up-near", help="hint X,Y,Z in W: take the object axis closest to it; overrides --up")
    ap.add_argument("--x", choices=["pca", "footprint"], default="pca")
    ap.add_argument("--opaque", action="store_true")
    ap.add_argument("--links-out", help="folder for one GLB per mesh node (the links of an articulated GLB)")
    a = ap.parse_args()
    doc, binary = read_glb(a.glb)
    joints, links = joint_spec.read_joints(doc), joint_spec.mesh_nodes(doc)
    problems = joint_spec.validate(joints, links) if joints else []  # a rigid GLB's node names do not matter
    if joints and not problems and any(j["frame"] != "W" for j in joints.values()):
        problems.append("the joints must be in frame W")
    if problems:
        raise SystemExit(f"{a.glb}: bad joints: " + "; ".join(problems))
    v, f = load_w(a.glb)
    near = None
    if a.up_near:
        up, axis, angle = up_near(v, f, [float(c) for c in a.up_near.split(",")])
        near = {"hint": a.up_near, "axis": axis, "angle_deg": angle}
    else:
        up = up_from_normals(v, f) if a.up == "normals" else None
    T, eig = up_frame(v, f, up, x_rule=a.x)
    doc = make_opaque(doc) if a.opaque else doc
    doc = wrap_scene(doc, gltf_matrix(T))
    fixed_base = False
    if joints:
        joints = joint_spec.transform(joints, T, "S")
        doc = joint_spec.write_joints(doc, joints)
        fixed_base = joint_spec.tree(joints, links)[2]
    write_glb(a.out, doc, binary)
    if a.links_out:
        write_links(doc, binary, a.links_out)
    vs, _ = load_w(a.out)  # the glTF -> W map is also Blender's import, so these are S coordinates
    up = T[2, :3]
    report = {
        "source": str(Path(a.glb).resolve()),
        "up_rule": "near" if a.up_near else a.up,
        "up_near": near,
        "x_rule": a.x,
        "opaque": a.opaque,
        "T_s_from_w": T.tolist(),
        "up_in_w": up.tolist(),
        "camera_tilt_deg": float(np.degrees(np.arccos(np.clip(-up[1], -1, 1)))),
        "pca_eigenvalues": eig.tolist(),
        "extent_s": (vs.max(0) - vs.min(0)).tolist(),
        "min_s": vs.min(0).tolist(),
        "max_s": vs.max(0).tolist(),
        "joints": joints,
        "links": links if joints else [],
        "fixed_base": fixed_base,
    }
    Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
