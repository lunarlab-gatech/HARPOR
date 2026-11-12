"""Write a URDF for one aligned GLB from its MJCF folder (build_plan.md note 9).

    python -m harpor.physics.urdf --glb aligned.glb --summary summary.json --mjcf MJCF_DIR --out DIR --name NAME

It reuses the MJCF's visual OBJs and textures and convex collision hulls (MJCF_DIR/assets and mjcf.json),
so the two models agree. Writes into DIR:
  model.urdf   robot NAME: one link NAME with the inertial of summary.json, its visual mesh and its
               collision hulls. The link's frame is S.
  meshes/      link_0.obj (one object and material per visual, the MJCF's visual_<j>.obj merged) with
               link_0.mtl and texture_<j>.png, and the collision OBJs copied
  urdf.json    inputs, the link and the check below (checked)
"""

import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import quoteattr



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


def urdf_xml(name, link):
    """link: {name, inertial, mesh, collisions}, in frame S."""
    a = quoteattr
    i = link["inertial"]
    xx, yy, zz, xy, xz, yz = i["fullinertia"]
    lines = [f"<robot name={a(name)}>",
             f'  <link name={a(link["name"])}>',
             '    <inertial>',
             f'      <origin xyz="{_xyz(i["com"])}" rpy="0 0 0"/>',
             f'      <mass value="{i["mass"]:.6g}"/>',
             f'      <inertia ixx="{xx:.6g}" ixy="{xy:.6g}" ixz="{xz:.6g}" iyy="{yy:.6g}" iyz="{yz:.6g}" izz="{zz:.6g}"/>',
             '    </inertial>']
    if link["mesh"]:
        lines += ['    <visual>', '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                  f'      <geometry><mesh filename={a(link["mesh"])}/></geometry>', '    </visual>']
    for c in link["collisions"]:
        lines += ['    <collision>', '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                  f'      <geometry><mesh filename={a(c)}/></geometry>', '    </collision>']
    return "\n".join(lines + ["  </link>", "</robot>"]) + "\n"


def build(glb, summary, mjcf_dir, out, name):
    out, mjcf_dir = Path(out), Path(mjcf_dir)
    meshes = out / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    mj = json.loads((mjcf_dir / "mjcf.json").read_text())
    assets = mjcf_dir / "assets"
    rb = json.loads(Path(summary).read_text())["rigid_body"]
    hulls = [f"collision_{j}.obj" for j in range(mj["collision_hulls"])]
    mesh = None
    if mj["visuals"]:
        link_obj(meshes / "link_0.obj", assets, mj["visuals"])
        mesh = "meshes/link_0.obj"
    for h in hulls:
        shutil.copyfile(assets / h, meshes / h)
    link = {"name": name, "inertial": {k: rb[k] for k in ("mass", "com", "fullinertia")}, "mesh": mesh,
            "collisions": [f"meshes/{h}" for h in hulls]}
    (out / "model.urdf").write_text(urdf_xml(name, link))
    report = {"glb": str(Path(glb).resolve()), "summary": str(Path(summary).resolve()), "mjcf": str(mjcf_dir.resolve()),
              "name": name, "links": [link]}
    (out / "urdf.json").write_text(json.dumps(report, indent=1))
    return report


def check(out):
    """Parse model.urdf: link and joint counts, and every mesh, material and texture file there.
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
    ap.add_argument("--mjcf", required=True, help="the MJCF folder from harpor.physics.mjcf (meshes, hulls, inertials)")
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
