"""One MuJoCo scene with several objects made by gator.physics.mjcf (build_plan.md note 9).

    python -m gator.physics.combine_mjcf ROOT [--out ROOT/all_objects.xml] [--front pan,knife,...] [--gap 0.3]
    python -m gator.physics.combine_mjcf ROOT --stack-on table --stack pan,plate,coffeecup,glass,knife,fork
        [--out ROOT/table_stack.xml] [--clearance 0.04] [--tilt 8]

ROOT holds one folder per object with mjcf/<name>.xml, mjcf/assets/ and mjcf/mjcf.json, as in
results/gator_physics/. The object files cannot simply be included together: each sets its own
meshdir, and every body starts at the origin. So the scene copies each object's defaults, assets and
body, rewrites the asset paths to <object>/mjcf/assets/..., and places the bodies on one floor: the
--front objects in a front row, the rest in a back row, with --gap metres between footprints (the
footprint comes from the collision hulls). Each body keeps its own start height above the floor.

With --stack-on BASE, the scene holds BASE on the floor and the --stack objects in one column above its
top, listed bottom first, so they fall onto BASE and onto each other when the simulation starts. Each
object is turned by 40 degrees about z more than the one below, tilted by +-tilt degrees about x, and
moved 2 cm off centre on a spiral; its lowest hull point starts --clearance above the highest hull
point of the object below. The timestep is 1 ms (thin cutlery falling about half a metre moves 4 mm
per step) with the implicitfast integrator.
Open either scene with: python -m mujoco.viewer --mjcf ROOT/<scene>.xml
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

FRONT = "pan,knife,fork,plate,coffeecup,glass"


def hull_points(mjcf_dir, n):
    """All collision hull vertices of an object in its body frame, (N, 3)."""
    pts = []
    for j in range(n):
        for line in (mjcf_dir / "assets" / f"collision_{j}.obj").read_text().splitlines():
            if line.startswith("v "):
                pts.append([float(c) for c in line.split()[1:4]])
    return np.array(pts)


def load_objects(root):
    objs = []
    for d in sorted(p for p in Path(root).iterdir() if (p / "mjcf" / "mjcf.json").exists()):
        rep = json.loads((d / "mjcf" / "mjcf.json").read_text())
        pts = hull_points(d / "mjcf", rep["collision_hulls"])
        objs.append({"dir": d, "name": rep["name"], "xml": ET.parse(d / "mjcf" / f"{rep['name']}.xml").getroot(),
                     "pts": pts, "fp": (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()),
                     "friction": rep["friction"]})
    return objs


def new_scene(model, friction, azimuth=120, elevation=-25):
    """(scene, default, asset, world) with lights and a floor of the given friction."""
    scene = ET.Element("mujoco", model=model)
    ET.SubElement(scene, "compiler", angle="radian")
    vis = ET.SubElement(scene, "visual")
    ET.SubElement(vis, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
    ET.SubElement(vis, "global", azimuth=str(azimuth), elevation=str(elevation))
    default = ET.SubElement(scene, "default")
    asset = ET.SubElement(scene, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3",
                  width="512", height="512", mark="edge", markrgb="0.8 0.8 0.8")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="8 8", reflectance="0.1")
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05", material="grid",
                  friction=f"{friction:g} 0.005 0.0001")
    return scene, default, asset, world


def add_object(o, default, asset, world, pos, quat=None):
    """Copy an object's defaults, assets (paths rewritten) and body into the scene at pos (and quat w x y z)."""
    prefix = f"{o['dir'].name}/mjcf/assets/"
    for child in o["xml"].find("default"):
        default.append(child)
    for a in o["xml"].find("asset"):
        if "file" in a.attrib:
            a.set("file", prefix + a.get("file"))
        asset.append(a)
    body = o["xml"].find("worldbody/body")
    body.set("pos", " ".join(f"{c:.4f}" for c in pos))
    if quat is not None:
        body.set("quat", " ".join(f"{c:.6f}" for c in quat))
    world.append(body)


def write(scene, out):
    ET.indent(scene)
    Path(out).write_text(ET.tostring(scene, encoding="unicode") + "\n")


def combine(root, out, front=FRONT, gap=0.3):
    objs = load_objects(root)
    front = [n for n in front.split(",") if n]
    rows = [[o for o in objs if o["name"] not in front], [o for o in objs if o["name"] in front]]
    scene, default, asset, world = new_scene("gator objects", max(o["friction"] for o in objs))

    y = 0.0
    layout = {}
    for r, row in enumerate(rows):
        if not row:
            continue
        depth = max(o["fp"][3] - o["fp"][2] for o in row)
        if r:
            y -= gap + depth / 2
        width = sum(o["fp"][1] - o["fp"][0] for o in row) + gap * (len(row) - 1)
        x = -width / 2
        for o in row:
            x0, x1, y0, y1 = o["fp"]
            pos = (x - x0, y - (y0 + y1) / 2, float(o["xml"].find("worldbody/body").get("pos").split()[2]))
            add_object(o, default, asset, world, pos)
            x += x1 - x0 + gap
            layout[o["name"]] = [round(float(c), 4) for c in pos]
        y -= depth / 2
    write(scene, out)
    return layout


def stack(root, out, base, items, clearance=0.04, tilt=8.0, offset=0.02):
    """BASE on the floor and the items in one falling column above its top (see the module docstring)."""
    from scipy.spatial.transform import Rotation

    by_name = {o["name"]: o for o in load_objects(root)}
    missing = [n for n in [base, *items] if n not in by_name]
    if missing:
        raise SystemExit(f"no MJCF for {missing} under {root}")
    b = by_name[base]
    scene, default, asset, world = new_scene(f"{'+'.join(items)} on {base}", max(by_name[n]["friction"] for n in [base, *items]),
                                             azimuth=135, elevation=-20)
    zb = float(b["xml"].find("worldbody/body").get("pos").split()[2])
    add_object(b, default, asset, world, (0.0, 0.0, zb))
    top = zb + b["pts"][:, 2].max()
    layout = {base: [0.0, 0.0, round(zb, 4)]}
    for i, n in enumerate(items):
        o = by_name[n]
        R = Rotation.from_euler("zx", [40 * i, tilt if i % 2 else -tilt], degrees=True)
        p = R.apply(o["pts"])
        c = (p[:, :2].min(0) + p[:, :2].max(0)) / 2
        jitter = offset * np.array([np.cos(2.4 * i), np.sin(2.4 * i)]) if i else np.zeros(2)
        pos = (*(jitter - c), top + clearance - p[:, 2].min())
        x, y, z, w = R.as_quat()
        add_object(o, default, asset, world, pos, (w, x, y, z))
        top = pos[2] + p[:, 2].max()
        layout[n] = [round(float(v), 4) for v in pos]
    opt = ET.Element("option", timestep="0.001", integrator="implicitfast")
    scene.insert(1, opt)
    ET.SubElement(scene, "statistic", center=f"0 0 {zb + b['pts'][:, 2].max():.3f}", extent="1.2")
    write(scene, out)
    return layout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out")
    ap.add_argument("--front", default=FRONT, help="comma-separated object names for the front row")
    ap.add_argument("--gap", type=float, default=0.3)
    ap.add_argument("--stack-on", help="base object for a falling stack, e.g. table")
    ap.add_argument("--stack", default="pan,plate,coffeecup,glass,knife,fork", help="stacked objects, bottom first")
    ap.add_argument("--clearance", type=float, default=0.04)
    ap.add_argument("--tilt", type=float, default=8.0)
    a = ap.parse_args()
    if a.stack_on:
        out = Path(a.out) if a.out else Path(a.root) / f"{a.stack_on}_stack.xml"
        layout = stack(a.root, out, a.stack_on, [n for n in a.stack.split(",") if n], a.clearance, a.tilt)
    else:
        out = Path(a.out) if a.out else Path(a.root) / "all_objects.xml"
        layout = combine(a.root, out, a.front, a.gap)
    print(json.dumps(layout, indent=1))


if __name__ == "__main__":
    main()
