"""Write a URDF for one aligned GLB from its MJCF folder (build_plan.md notes 9 and 11).

    python -m gator.physics.urdf --glb aligned.glb --summary summary.json --mjcf MJCF_DIR --out DIR --name NAME

It reuses the MJCF's visual OBJs and textures, convex collision hulls and per-link inertials
(MJCF_DIR/assets and mjcf.json), so the two models agree. Writes into DIR:
  model.urdf   robot NAME. A rigid object is one link NAME with the inertial of summary.json. An
               articulated one has a link per part (the GLB's node names) and a joint per jointed part,
               <child>_joint: revolute, prismatic or fixed, with <axis> and
               <limit lower upper effort="100" velocity="1"/>; a part without a joint gets a fixed joint
               to the link the MJCF welds it to (mjcf.json welded, else the root). A fixed base adds a
               link world and the fixed joint world_joint at the MJCF's start height, as MJCF and USD.
  meshes/      link_<k>.obj per link k (one object and material per visual, the MJCF's visual_<j>.obj
               merged) with link_<k>.mtl and texture_<j>.png, and the collision OBJs copied
  urdf.json    inputs, links, joints and the check below (checked)
Link frames: every link's geometry is in S, so the root's frame is S and a child's frame sits at its
joint's origin with S's axes. A joint's <origin xyz> is its child's frame minus its parent's, and every
visual, collision and inertial <origin xyz> is minus its link's frame (the COM minus it for the inertial).
The axis, the inertia and the limits stay as they are (radians, meters). URDF names the links by part,
MJCF by body (<name> for the root, <name>_<part> for the rest). URDF has no contact filter, so the MJCF's
excluded_pairs are only listed in urdf.json; importers usually turn self-collision off (Isaac Sim does).
"""

import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np


def _shift(token, ov, ot, on):
    """An OBJ face corner (v, v/vt, v//vn or v/vt/vn) with its indices moved by the offsets."""
    return "/".join(str(int(p) + o) if p else "" for p, o in zip(token.split("/"), (ov, ot, on)))


def link_obj(path, assets, visuals):
    """path (.obj) and its .mtl from the MJCF's visual OBJs [{mesh, texture, rgba}] in assets: one object
    and material per visual, the textures copied next to it."""
    path = Path(path)
    obj, mtl = [f"mtllib {path.stem}.mtl"], []
    ov = ot = on = 0
    for v in visuals:
        obj += [f"o {v['mesh']}", f"usemtl {v['mesh']}"]
        counts = {"v": 0, "vt": 0, "vn": 0}
        for line in (Path(assets) / f"{v['mesh']}.obj").read_text().splitlines():
            key = line.split(" ", 1)[0]
            if key in counts:
                counts[key] += 1
                obj.append(line)
            elif key == "f":
                obj.append("f " + " ".join(_shift(t, ov, ot, on) for t in line.split()[1:]))
        ov, ot, on = ov + counts["v"], ot + counts["vt"], on + counts["vn"]
        mtl += [f"newmtl {v['mesh']}", "Kd " + " ".join(f"{c:.4g}" for c in v["rgba"][:3])]
        if v["texture"]:
            shutil.copyfile(Path(assets) / f"{v['texture']}.png", path.parent / f"{v['texture']}.png")
            mtl.append(f"map_Kd {v['texture']}.png")
    path.write_text("\n".join(obj) + "\n")
    path.with_suffix(".mtl").write_text("\n".join(mtl) + "\n")


def _xyz(x):
    return " ".join(f"{c:.6g}" for c in x)


def urdf_xml(name, links, joints, fixed_base):
    """links: [{name, frame, inertial, mesh, collisions}]; joints: [{name, type, parent, child, xyz, axis, range}]."""
    a = quoteattr
    lines = [f"<robot name={a(name)}>"]
    if fixed_base:
        lines.append('  <link name="world"/>')
    for link in links:
        off = -np.asarray(link["frame"], float)
        i = link["inertial"]
        xx, yy, zz, xy, xz, yz = i["fullinertia"]
        lines += [f'  <link name={a(link["name"])}>',
                  '    <inertial>',
                  f'      <origin xyz="{_xyz(np.asarray(i["com"], float) + off)}" rpy="0 0 0"/>',
                  f'      <mass value="{i["mass"]:.6g}"/>',
                  f'      <inertia ixx="{xx:.6g}" ixy="{xy:.6g}" ixz="{xz:.6g}" iyy="{yy:.6g}" iyz="{yz:.6g}" izz="{zz:.6g}"/>',
                  '    </inertial>']
        if link["mesh"]:
            lines += ['    <visual>', f'      <origin xyz="{_xyz(off)}" rpy="0 0 0"/>',
                      f'      <geometry><mesh filename={a(link["mesh"])}/></geometry>', '    </visual>']
        for c in link["collisions"]:
            lines += ['    <collision>', f'      <origin xyz="{_xyz(off)}" rpy="0 0 0"/>',
                      f'      <geometry><mesh filename={a(c)}/></geometry>', '    </collision>']
        lines.append("  </link>")
    for j in joints:
        lines += [f'  <joint name={a(j["name"])} type="{j["type"]}">',
                  f'    <parent link={a(j["parent"])}/>',
                  f'    <child link={a(j["child"])}/>',
                  f'    <origin xyz="{_xyz(j["xyz"])}" rpy="0 0 0"/>']
        if j["type"] != "fixed":
            lines += [f'    <axis xyz="{_xyz(j["axis"])}"/>',
                      f'    <limit lower="{j["range"][0]:.6g}" upper="{j["range"][1]:.6g}" effort="100" velocity="1"/>']
        lines.append("  </joint>")
    return "\n".join(lines + ["</robot>"]) + "\n"


def build(glb, summary, mjcf_dir, out, name):
    out, mjcf_dir = Path(out), Path(mjcf_dir)
    meshes = out / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    mj = json.loads((mjcf_dir / "mjcf.json").read_text())
    assets = mjcf_dir / "assets"
    if "links" in mj:  # articulated (gator.physics.mjcf with joints)
        root, fixed_base = mj["root"], mj["fixed_base"]
        recs = {k: {"inertial": r["inertial"], "hulls": [f"collision_{k}_{j}.obj" for j in range(r["collision_hulls"])],
                    "visuals": [v for v in mj["visuals"] if v["link"] == k]} for k, r in mj["links"].items()}
        holder = mj.get("welded", {})
        spec = mj["joints"] + [{"child": k, "parent": holder.get(k, root), "type": "fixed", "pos": [0.0, 0.0, 0.0],
                                "axis": [1.0, 0.0, 0.0], "range": [0.0, 0.0]} for k in mj["loose"]]
    else:
        rb = json.loads(Path(summary).read_text())["rigid_body"]
        root, fixed_base, spec = name, False, []
        recs = {name: {"inertial": {k: rb[k] for k in ("mass", "com", "fullinertia")},
                       "hulls": [f"collision_{j}.obj" for j in range(mj["collision_hulls"])], "visuals": mj["visuals"]}}
    frame = {k: [0.0, 0.0, 0.0] for k in recs}
    for j in spec:
        frame[j["child"]] = list(map(float, j["pos"]))
    links = []
    for i, (k, r) in enumerate(recs.items()):
        mesh = None
        if r["visuals"]:
            link_obj(meshes / f"link_{i}.obj", assets, r["visuals"])
            mesh = f"meshes/link_{i}.obj"
        for h in r["hulls"]:
            shutil.copyfile(assets / h, meshes / h)
        links.append({"name": k, "frame": frame[k], "inertial": r["inertial"], "mesh": mesh,
                      "collisions": [f"meshes/{h}" for h in r["hulls"]]})
    joints = [{"name": f"{j['child']}_joint", "type": j["type"], "parent": j["parent"], "child": j["child"],
               "xyz": (np.asarray(frame[j["child"]]) - frame[j["parent"]]).tolist(), "axis": j["axis"],
               "range": j["range"]} for j in spec]
    if fixed_base:
        joints.insert(0, {"name": "world_joint", "type": "fixed", "parent": "world", "child": root,
                          "xyz": [0.0, 0.0, float(mj["start_z"])], "axis": [1.0, 0.0, 0.0], "range": [0.0, 0.0]})
    (out / "model.urdf").write_text(urdf_xml(name, links, joints, fixed_base))
    report = {"glb": str(Path(glb).resolve()), "summary": str(Path(summary).resolve()), "mjcf": str(mjcf_dir.resolve()),
              "name": name, "root": root, "fixed_base": fixed_base, "links": links, "joints": joints,
              "excluded_pairs": mj.get("excluded_pairs", [])}
    (out / "urdf.json").write_text(json.dumps(report, indent=1))
    return report


def check(out):
    """Parse model.urdf: link and joint counts, joint types, and every mesh, material and texture file there.
    Loads it with yourdfpy too when that is installed."""
    out = Path(out)
    robot = ET.parse(out / "model.urdf").getroot()
    files = [m.get("filename") for m in robot.iter("mesh")]
    needed = list(files)
    for f in files:
        for line in (out / f).read_text().splitlines() if (out / f).is_file() else []:
            if line.startswith("mtllib "):
                mtl = (out / f).parent / line.split(None, 1)[1]
                needed.append(str(mtl.relative_to(out)))
                for m in mtl.read_text().splitlines() if mtl.is_file() else []:
                    if m.startswith("map_Kd "):
                        needed.append(str((mtl.parent / m.split(None, 1)[1]).relative_to(out)))
    joints = robot.findall("joint")
    result = {"links": len(robot.findall("link")), "joints": len(joints),
              "joint_types": {j.get("name"): j.get("type") for j in joints}, "meshes": len(files),
              "missing": sorted({f for f in needed if not (out / f).is_file()}), "yourdfpy": None}
    try:
        import yourdfpy
    except ImportError:
        return result
    u = yourdfpy.URDF.load(str(out / "model.urdf"))
    result["yourdfpy"] = {"version": getattr(yourdfpy, "__version__", None), "links": len(u.link_map),
                          "actuated_joints": len(u.actuated_joint_names)}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--mjcf", required=True, help="the MJCF folder from gator.physics.mjcf (meshes, hulls, inertials)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z_][\w-]*", a.name) or a.name == "world":
        ap.error("--name must be a plain identifier other than 'world'")
    out = Path(a.out)
    for old in (out / "model.urdf", out / "urdf.json"):  # only what build() writes
        old.unlink(missing_ok=True)
    shutil.rmtree(out / "meshes", ignore_errors=True)
    report = build(a.glb, a.summary, a.mjcf, a.out, a.name)
    report["checked"] = check(a.out)
    (out / "urdf.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["checked"], indent=1))


if __name__ == "__main__":
    main()
