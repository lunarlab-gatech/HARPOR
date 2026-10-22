"""Isaac Sim scene (USD) that matches a combined MuJoCo scene from gator.physics.combine_mjcf (build_plan.md note 9).

    python -m gator.physics.usd_scene ROOT/table_stack.xml [--out ROOT/table_stack.usda]

ROOT holds one folder per object with mjcf/mjcf.json and usd/<name>.usd (gator.physics.mjcf and
gator.physics.usd). Each body of the MuJoCo scene becomes a reference to its object's USD at the body's
pos and quat, so both simulators start from the same poses. The scene adds a PhysicsScene (gravity
9.81 m/s^2 along -z), a 20 x 20 m ground box whose top is at z = 0 with the MuJoCo floor's friction,
and lights. The MuJoCo timestep becomes physxScene:timeStepsPerSecond (1 ms for the stack, MuJoCo's
default 2 ms otherwise), authored as raw PhysxSceneAPI attributes because usd-core has no PhysxSchema.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def build(mjcf_scene, out):
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

    from gator.physics.usd import _bind, _physics_material, _stage

    mjcf_scene, out = Path(mjcf_scene), Path(out)
    root = mjcf_scene.parent
    folders = {json.loads(p.read_text())["name"]: p.parent.parent.name for p in root.glob("*/mjcf/mjcf.json")}
    xml = ET.parse(mjcf_scene).getroot()
    opt = xml.find("option")
    timestep = float(opt.get("timestep", 0.002)) if opt is not None else 0.002
    floor = xml.find("worldbody/geom[@name='floor']")
    friction = float(floor.get("friction", "1").split()[0])

    stage = _stage(out)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    ps = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr(9.81)
    ps.GetPrim().AddAppliedSchema("PhysxSceneAPI")
    ps.GetPrim().CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(round(1 / timestep))
    ground = UsdGeom.Cube.Define(stage, "/World/ground")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    ground.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.1))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.4, 0.45)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    _bind(ground.GetPrim(), _physics_material(stage, "/World/groundMaterial", friction), "physics")
    UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(3000)
    UsdLux.DomeLight.Define(stage, "/World/sky").CreateIntensityAttr(500)

    placed = {}
    for body in xml.findall("worldbody/body"):
        name = body.get("name")
        usd = root / folders[name] / "usd" / f"{name}.usd"
        if not usd.exists():
            raise SystemExit(f"no USD for {name}: {usd}")
        pos = [float(c) for c in body.get("pos", "0 0 0").split()]
        w, x, y, z = [float(c) for c in body.get("quat", "1 0 0 0").split()]
        prim = UsdGeom.Xform.Define(stage, f"/World/{name}")
        prim.GetPrim().GetReferences().AddReference(f"./{folders[name]}/usd/{name}.usd")
        prim.AddTranslateOp().Set(Gf.Vec3d(*pos))
        prim.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
        placed[name] = {"usd": str(usd.relative_to(root)), "pos": pos, "quat_wxyz": [w, x, y, z]}
    stage.GetRootLayer().Save()
    return {"timesteps_per_second": round(1 / timestep), "friction": friction, "objects": placed}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mjcf_scene")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.mjcf_scene).with_suffix(".usda")
    print(json.dumps(build(a.mjcf_scene, out), indent=1))


if __name__ == "__main__":
    main()
