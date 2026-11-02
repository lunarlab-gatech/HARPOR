"""360-degree turntable of one GLB, as PNG frames (build_plan.md note 5), or the soft-part video of it (note 11).

blender -b --factory-startup --python-exit-code 1 --python turntable.py -- ARGS.json

ARGS.json: assets (a list of exactly one {path, matrix}), frames_dir, and optional frames (180), fps (30), panel (640,
pixels per side), samples (32), elevation (0: the camera's angle above the horizontal, in degrees) and soft. The GLB is
in frame W (OpenCV camera of view 0: x right, y down, z forward), which the stock glTF importer maps to Blender world.
The asset turns once about the W up axis (-y) through its own center, so the frames loop. Frame 1 looks along view
0's direction.

The asset's "matrix" (optional, 4x4, row-major) maps the GLB's Blender world into W first: for a GLB in another
frame (aligned.glb is in S), or one to be shown turned (the web app's 360 video turns its GLB upright). Frame 1
then looks along +z of the frame that the matrix maps into.

soft (harpor/webapp/softvideo.py), {motion, matrix}: the asset does not turn, and a soft part dents and springs back
instead. motion is an NPZ of rest, the part's vertices (K, 3), and disp, their displacement in each frame (frames, K,
3), both in a frame that matrix (4x4) maps into W. A vertex of the import whose matrix_world @ co lies on a rest
target, matrix @ rest, is the part's (MATCH): each frame sets its co to matrix_world^-1 @ (matrix @ (rest + disp)) and
renders, without persistent data, since the mesh changes. The part's objects lose the import's custom normals, so
that the shading follows the dent. The camera frames the part as the turntable frames the asset.

The frames go into frames_dir as PNG (0001.png, 0002.png, ...), and harpor/webapp/turntable.py encodes them.
"""
# Needs Blender 5.2, which the test environment does not have: tests/test_webapp_turntable.py runs this script on
# stand-ins for bpy and mathutils, and the fake Blender (harpor.agent.fake_agent) follows the frames_dir contract.

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_common import CV_TO_BLENDER_CAMERA, import_glb, look_at, points  # noqa: E402

A = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
assert len(A["assets"]) == 1 and A.get("frames_dir"), "turntable.py takes exactly one asset and frames_dir"
ASSET, SOFT = A["assets"][0], A.get("soft")
FRAMES, FPS, PANEL, SAMPLES = A.get("frames", 180), A.get("fps", 30), A.get("panel", 640), A.get("samples", 32)
ELEVATION = math.radians(A.get("elevation", 0))  # the camera's angle above the horizontal
FOV = math.radians(30)  # horizontal and vertical
MATCH = 1e-4  # of the soft part's diagonal: a vertex this close to a rest target is the part's


def setup():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = "GPU"
    scene.cycles.samples, scene.cycles.seed, scene.cycles.use_denoising = SAMPLES, 0, True
    scene.view_settings.view_transform = "Standard"
    scene.world = bpy.data.worlds.new("light")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.82, 0.82, 0.82, 1.0)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = Vector((0.3, 1.0, 0.6)).to_track_quat("-Z", "Y").to_euler()  # down (+y) and away (+z)
    scene.collection.objects.link(sun)
    return scene


def frames_output(sc, folder):
    """PNG frames, 8-bit RGB, as <folder>/0001.png, 0002.png, ..."""
    settings = sc.render.image_settings
    settings.media_type = "IMAGE"
    settings.file_format, settings.color_mode, settings.color_depth = "PNG", "RGB", "8"
    sc.render.filepath = str(folder) + "/"


def soft_part(objs, soft):
    """(the targets of each frame in W, (frames, K, 3); [(object, its vertices' co, the target of each, -1 for none)]
    of the objects of the soft part: those with at least half of their vertices on a rest target, so that a parent
    whose seam meets the part stays as it is). kdtree is imported here: only the soft-part video needs it."""
    from mathutils import kdtree

    S = np.asarray(soft["matrix"], float)
    with np.load(soft["motion"]) as z:
        rest, disp = z["rest"].astype(float), z["disp"]
    targets = (rest[None] + disp) @ S[:3, :3].T + S[:3, 3]
    tree = kdtree.KDTree(len(rest))
    for i, x in enumerate((rest @ S[:3, :3].T + S[:3, 3]).tolist()):
        tree.insert(x, i)
    tree.balance()
    tol, found = MATCH * float(np.linalg.norm(np.ptp(rest, 0))), []
    for o in objs:
        if o.type != "MESH" or not len(o.data.vertices):
            continue
        co = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        M = np.array(o.matrix_world)
        near = [tree.find(x) for x in (co @ M[:3, :3].T + M[:3, 3]).tolist()]
        ids = np.array([i if i is not None and dist <= tol else -1 for _, i, dist in near])
        if (ids >= 0).mean() >= 0.5:
            found.append((o, co, ids))
    assert found, "no object of the GLB lies on the soft part"
    print(f"soft part: {sum(int((ids >= 0).sum()) for *_, ids in found)} vertices of {len(found)} objects on its "
          f"{len(rest)} vertices", flush=True)
    return targets, found


def smooth(o):
    """Shading that follows the dent: the import's custom normals would keep the rest shape's."""
    if getattr(o.data, "has_custom_normals", False):
        try:
            with bpy.context.temp_override(object=o, active_object=o):
                bpy.ops.mesh.customdata_custom_splitnormals_clear()
        except Exception as e:  # the dent still shows in the outline and the shadows
            print(f"custom normals of {o.name} not cleared: {e}", flush=True)
    o.data.shade_smooth()


def move(found, targets):
    """Put the part's vertices of each object on targets (K, 3) in W."""
    for o, co, ids in found:
        inv = np.linalg.inv(np.array(o.matrix_world))
        x, on = co.copy(), ids >= 0
        x[on] = targets[ids[on]] @ inv[:3, :3].T + inv[:3, 3]
        o.data.vertices.foreach_set("co", x.astype(np.float32).ravel())
        o.data.update()


scene = setup()
objs = import_glb(ASSET["path"])
if ASSET.get("matrix") is not None:  # the GLB's Blender world into W
    M = Matrix(ASSET["matrix"])
    for o in objs:
        if o.parent is None:
            o.matrix_world = M @ o.matrix_world
    bpy.context.view_layer.update()
if SOFT is None:
    p = points(objs)
    lo, hi = p.min(0), p.max(0)
    center = (lo + hi) / 2
    pivot = bpy.data.objects.new("pivot", None)
    pivot.location = Vector(center.tolist())
    scene.collection.objects.link(pivot)
    bpy.context.view_layer.update()
    for o in objs:
        if o.parent is None:
            o.parent = pivot
            o.matrix_parent_inverse = pivot.matrix_world.inverted()
    # The asset turns inside a sphere of radius r about its center, in one square panel.
    r = float(np.linalg.norm(hi - lo)) / 2
    pivot.rotation_mode = "XYZ"
    for f, angle in ((1, 0.0), (FRAMES + 1, -2 * math.pi)):  # frame FRAMES + 1 is not rendered: a clean loop
        pivot.rotation_euler = (0.0, angle, 0.0)
        pivot.keyframe_insert("rotation_euler", frame=f)
else:  # the soft part, framed as the turning asset is
    targets, found = soft_part(objs, SOFT)
    for o, *_ in found:
        smooth(o)
    lo, hi = targets[0].min(0), targets[0].max(0)
    center, r = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2

# The panel is 2.6 r wide at the center plane: the sphere (angular radius asin(r / dist), about 12
# degrees) fits inside the 15-degree half-angle from any direction.
dist = 1.3 * r / math.tan(FOV / 2)
cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.sensor_fit, cam.data.angle = "HORIZONTAL", FOV
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = PANEL, PANEL, 100
scene.frame_start, scene.frame_end, scene.render.fps = 1, FRAMES, FPS

# The camera looks along +z of W, from ELEVATION above (W's up is -y).
eye = center + dist * np.array([0.0, -math.sin(ELEVATION), -math.cos(ELEVATION)])
cam.matrix_world = Matrix((np.linalg.inv(look_at(eye, center)) @ CV_TO_BLENDER_CAMERA).tolist())
cam.data.clip_start, cam.data.clip_end = 1e-3 * dist, 1e3 * dist

frames_output(scene, A["frames_dir"])
scene.render.use_persistent_data = SOFT is None  # keeps the scene's data from frame to frame, unless the mesh moves
if SOFT is None:
    bpy.ops.render.render(animation=True)
else:
    for f in range(FRAMES):
        move(found, targets[f])
        scene.render.filepath = f"{A['frames_dir']}/{f + 1:04d}.png"
        bpy.ops.render.render(write_still=True)
