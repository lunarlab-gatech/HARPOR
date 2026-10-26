"""360-degree turntable video of one or more GLBs, side by side (build_plan.md note 5).

blender -b --factory-startup --python-exit-code 1 --python turntable.py -- ARGS.json

ARGS.json: assets (list of {path, label, matrix}), out (.mp4), and optional frames (180), fps (30), panel (640,
pixels per asset), samples (32), reuse_panels (false: true joins existing panel videos without
rendering them again), same_scale (true: every panel is framed for the largest asset, so sizes
compare; false: each panel is framed for its own asset, for meshes whose units differ), elevation (0:
the cameras' angle above the horizontal, in degrees) and frames_dir. The GLBs are in frame W (OpenCV
camera of view 0: x right, y down, z forward), which the stock glTF importer maps to Blender world.
Each asset turns once about the W up axis (-y) through its own center, under the same lights, so the
video loops. Frame 1 looks along view 0's direction.

An asset's "matrix" (optional, 4x4, row-major) maps the GLB's Blender world into W first: for a GLB in another
frame (aligned.glb is in S), or one to be shown turned (the web app's 360 video turns its GLB upright). Frame 1
then looks along +z of the frame that the matrix maps into. "label" is optional: an asset without one has no label.
frames_dir (exactly one asset) writes the frames as PNG into that folder (0001.png, 0002.png, ...) instead of a
video, and "out" is not needed (gator/webapp/turntable.py encodes them).

Each asset is rendered alone with its own camera in the same direction, and at the same distance
unless same_scale is false, so every panel shows the same view (one camera for all would see side
panels from different angles). The panels
are then joined left to right, in the given order, with the sequencer. The per-asset videos are kept
next to the output as <out stem>_<index>.mp4.
"""
# Untested: needs Blender 5.2, which the test environment does not have. The fake Blender
# (gator.agent.fake_agent) follows the frames_dir contract.

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
FRAMES, FPS, PANEL, SAMPLES = A.get("frames", 180), A.get("fps", 30), A.get("panel", 640), A.get("samples", 32)
ELEVATION = math.radians(A.get("elevation", 0))  # the cameras' angle above the horizontal
FOV = math.radians(30)  # per panel, horizontal and vertical


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


def label(scene, text, location, size, rotation):
    curve = bpy.data.curves.new(f"label {text}", "FONT")
    curve.body, curve.align_x, curve.align_y, curve.size = text, "CENTER", "TOP", size
    mat = bpy.data.materials.new(f"label {text}")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.08, 0.08, 0.08, 1.0)
    curve.materials.append(mat)
    obj = bpy.data.objects.new(f"label {text}", curve)
    obj.matrix_world = Matrix.Translation(Vector(location)) @ rotation
    scene.collection.objects.link(obj)
    return obj


scene = setup()
assets = []
for i, a in enumerate(A["assets"]):
    objs = import_glb(a["path"])
    if a.get("matrix") is not None:  # the GLB's Blender world into W
        M = Matrix(a["matrix"])
        for o in objs:
            if o.parent is None:
                o.matrix_world = M @ o.matrix_world
        bpy.context.view_layer.update()
    p = points(objs)
    lo, hi = p.min(0), p.max(0)
    pivot = bpy.data.objects.new(f"pivot {i}", None)
    pivot.location = Vector(((lo + hi) / 2).tolist())
    scene.collection.objects.link(pivot)
    bpy.context.view_layer.update()
    for o in objs:
        if o.parent is None:
            o.parent = pivot
            o.matrix_parent_inverse = pivot.matrix_world.inverted()
    assets.append({"pivot": pivot, "radius": float(np.linalg.norm(hi - lo)) / 2, "center": (lo + hi) / 2})

# Every asset turns inside a sphere of radius r about its center; one square panel per asset.
r_max = max(a["radius"] for a in assets)
for a in assets:
    a["r"] = r_max if A.get("same_scale", True) else a["radius"]
    a["pivot"].rotation_mode = "XYZ"
    for f, angle in ((1, 0.0), (FRAMES + 1, -2 * math.pi)):  # frame FRAMES + 1 is not rendered: a clean loop
        a["pivot"].rotation_euler = (0.0, angle, 0.0)
        a["pivot"].keyframe_insert("rotation_euler", frame=f)

# The panel is 2.6 r wide at the center plane: the sphere (angular radius asin(r / dist), about 12
# degrees) fits inside the 15-degree half-angle from any direction, with room for the label below it.
for a in assets:
    a["dist"] = 1.3 * a["r"] / math.tan(FOV / 2)
cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.sensor_fit, cam.data.angle = "HORIZONTAL", FOV
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = PANEL, PANEL, 100
scene.frame_start, scene.frame_end, scene.render.fps = 1, FRAMES, FPS


def video_output(sc, path):
    settings = sc.render.image_settings
    if hasattr(settings, "media_type"):  # Blender 4.5 and later separate images from video
        settings.media_type = "VIDEO"
    settings.file_format = "FFMPEG"
    sc.render.ffmpeg.format, sc.render.ffmpeg.codec = "MPEG4", "H264"
    sc.render.ffmpeg.constant_rate_factor, sc.render.ffmpeg.ffmpeg_preset = "HIGH", "GOOD"
    sc.render.filepath, sc.render.use_file_extension = str(path), False


def frames_output(sc, folder):
    """PNG frames, 8-bit RGB, as <folder>/0001.png, 0002.png, ..."""
    settings = sc.render.image_settings
    if hasattr(settings, "media_type"):
        settings.media_type = "IMAGE"
    settings.file_format, settings.color_mode, settings.color_depth = "PNG", "RGB", "8"
    sc.render.filepath = str(folder) + "/"


def aim(i):
    """The camera of asset i; only asset i and its label render."""
    a = assets[i]
    cam.matrix_world = Matrix(a["camera"].tolist())
    cam.data.clip_start, cam.data.clip_end = 1e-3 * a["dist"], 1e3 * a["dist"]
    for j, b in enumerate(assets):
        for o in [b["label"], *b["pivot"].children_recursive]:
            if o is not None:
                o.hide_render = j != i


# The cameras all look along +z of W, from ELEVATION above (W's up is -y), so the labels share one rotation.
for i, a in enumerate(assets):
    c, text = a["center"], A["assets"][i].get("label")
    a["camera"] = np.linalg.inv(look_at(c + a["dist"] * np.array([0.0, -math.sin(ELEVATION), -math.cos(ELEVATION)]), c)) \
        @ CV_TO_BLENDER_CAMERA
    a["label"] = None if text is None else label(scene, text, (c[0], c[1] + 1.05 * a["r"], c[2]), 0.15 * a["r"],
                                                  Matrix(a["camera"].tolist()).to_3x3().to_4x4())

if A.get("frames_dir"):  # one asset, as PNG frames
    assert len(assets) == 1, "frames_dir takes exactly one asset"
    aim(0)
    frames_output(scene, A["frames_dir"])
    scene.render.use_persistent_data = True  # keeps the scene's data from frame to frame
    bpy.ops.render.render(animation=True)
else:
    out = Path(A["out"])
    panels = []
    for i in range(len(assets)):
        aim(i)
        panels.append(out.with_name(f"{out.stem}_{i}.mp4"))
        if A.get("reuse_panels") and panels[-1].exists():
            continue
        video_output(scene, panels[-1])
        bpy.ops.render.render(animation=True)

    # Join the panels left to right with the sequencer.
    comb = bpy.data.scenes.new("combine")
    comb.render.resolution_x, comb.render.resolution_y, comb.render.resolution_percentage = len(panels) * PANEL, PANEL, 100
    comb.frame_start, comb.frame_end, comb.render.fps = 1, FRAMES, FPS
    comb.view_settings.view_transform = "Standard"  # the panels already have their final colors
    editor = comb.sequence_editor_create()
    strips = getattr(editor, "strips", None)
    strips = strips if strips is not None else editor.sequences
    for i, path in enumerate(panels):
        try:
            strip = strips.new_movie(f"panel {i}", str(path), channel=i + 1, frame_start=1, fit_method="ORIGINAL")
        except TypeError:
            strip = strips.new_movie(f"panel {i}", str(path), channel=i + 1, frame_start=1)
        strip.transform.offset_x = (i - (len(panels) - 1) / 2) * PANEL
        strip.blend_type = "ALPHA_OVER"
    comb.render.use_sequencer = True
    video_output(comb, out)
    bpy.ops.render.render(animation=True, scene=comb.name)
