"""Render code of review.py (build_plan.md note 5), kept apart so that its tests can import it.

The scripts import it after putting their own folder first on sys.path, as they import crop.py. The
stock glTF importer maps glTF to W, so Blender world coordinates are W and the input cameras apply as
they are. Input views are cropped to the object (the mask's box and the projected points) and rendered
at INPUT_LONG pixels on the long side; orbit and detail views are PANEL pixels square.
"""
# Untested: needs Blender 5.2, which the test environment does not have.

import math

import bpy
import numpy as np
from crop import crop_camera, crop_window, project
from mathutils import Matrix, Vector

CV_TO_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])
SHEET_MAX_SIDE, SHEET_MAX_PIXELS, PANEL, INPUT_LONG = 2048, 2500 * 32 * 32, 512, 768


def setup(clay):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = "GPU"
    scene.cycles.samples, scene.cycles.seed, scene.cycles.use_denoising = 32, 0, True
    scene.render.film_transparent = True
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.world = bpy.data.worlds.new("light")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = Vector((0.3, 1.0, 0.6)).to_track_quat("-Z", "Y").to_euler()  # down (+y) and away (+z)
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    if clay:
        mat = bpy.data.materials.new("clay")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)
        bsdf.inputs["Alpha"].default_value = 1.0
        bpy.context.view_layer.material_override = mat
    return scene, cam


def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [o for o in bpy.data.objects if o not in before]


def points(objs):
    pts = []
    for o in objs:
        if o.type == "MESH" and len(o.data.vertices):
            co = np.empty(len(o.data.vertices) * 3)
            o.data.vertices.foreach_get("co", co)
            M = np.array(o.matrix_world)
            pts.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
    return np.concatenate(pts)


def box(p):
    lo, hi = p.min(0), p.max(0)
    return {"min": lo.tolist(), "max": hi.tolist(), "center": ((lo + hi) / 2).tolist(), "extent": (hi - lo).tolist()}


def place_camera(scene, cam, K, w2c, width, height, clip):
    """OpenCV K and world-to-camera, pixel centers at integer coordinates. Assumes fx == fy."""
    scene.render.resolution_x, scene.render.resolution_y = width, height
    cam.matrix_world = Matrix((np.linalg.inv(np.array(w2c)) @ CV_TO_BLENDER_CAMERA).tolist())
    cam.data.sensor_fit, cam.data.sensor_width = "HORIZONTAL", 36.0
    cam.data.lens = K[0][0] * 36.0 / width
    cam.data.shift_x = -(K[0][2] - (width - 1) / 2) / width
    cam.data.shift_y = (K[1][2] - (height - 1) / 2) / width
    cam.data.clip_start, cam.data.clip_end = clip


def look_at(eye, target):
    """OpenCV world-to-camera matrix from eye toward target, with image down along +y of W."""
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross([0.0, 1.0, 0.0], z)
    if np.linalg.norm(x) < 1e-6:  # looking along y: take image right along +x of W
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ eye
    return T


def orbit_views(center, diag, n=6, fov=math.radians(40), azimuths=None):
    """n views at even azimuths from the front (view 0's side), 25 degrees up; azimuths (degrees) replaces
    the even ones."""
    f = PANEL / 2 / math.tan(fov / 2)
    r = 0.55 * diag / math.sin(fov / 2)
    K = [[f, 0, (PANEL - 1) / 2], [0, f, (PANEL - 1) / 2], [0, 0, 1]]
    el = math.radians(25)
    angles = [math.radians(a) for a in azimuths] if azimuths is not None else [2 * math.pi * k / n for k in range(n)]
    views = []
    for k, a in enumerate(angles):
        eye = center + r * np.array([math.cos(el) * math.sin(a), -math.sin(el), -math.cos(el) * math.cos(a)])
        views.append({"name": f"orbit{k}", "K": K, "w2c": look_at(eye, center).tolist(), "width": PANEL, "height": PANEL})
    return views


def detail_views(rig, fov=math.radians(30)):
    """Close-up cameras from the agent: center, radius and viewing direction, all in W."""
    f = PANEL / 2 / math.tan(fov / 2)
    K = [[f, 0, (PANEL - 1) / 2], [0, f, (PANEL - 1) / 2], [0, 0, 1]]
    views = []
    for k, (cx, cy, cz, r, dx, dy, dz) in enumerate(rig):
        center, d = np.array([cx, cy, cz]), np.array([dx, dy, dz], float)
        eye = center - d / np.linalg.norm(d) * (1.1 * r / math.sin(fov / 2))
        views.append({"name": f"detail{k}", "K": K, "w2c": look_at(eye, center).tolist(), "width": PANEL, "height": PANEL})
    return views


def load(path):
    img = bpy.data.images.load(str(path))
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    bpy.data.images.remove(img)
    return a.reshape(h, w, 4)[::-1]


def save(a, path):
    img = bpy.data.images.new("sheet", a.shape[1], a.shape[0], alpha=True)
    img.pixels.foreach_set(np.ascontiguousarray(a[::-1], np.float32).ravel())
    img.filepath_raw, img.file_format = str(path), "PNG"
    img.save()


def fit(a, h, w):
    return a[np.arange(h) * a.shape[0] // h][:, np.arange(w) * a.shape[1] // w]


def on_white(a):
    out = np.ones_like(a)
    out[..., :3] = a[..., :3] * a[..., 3:] + (1 - a[..., 3:])
    return out


def input_camera(v, root, pts):
    """An input view cropped to the object: (K, width, height, photo, mask, record). The window holds the
    mask's box and the projected points pts; record has the window (crop) and its scale."""
    photo, mask = fit(load(root / v["image"]), v["height"], v["width"]), None
    if v.get("mask"):
        mask = fit(load(root / v["mask"]), v["height"], v["width"])[..., 0] > 0.5
    window = crop_window(mask, project(pts, v["K"], v["w2c"]), v["width"], v["height"])
    K, width, height, scale = crop_camera(v["K"], window, INPUT_LONG)
    u0, v0, u1, v1 = window
    photo = fit(photo[v0:v1, u0:u1], height, width)
    mask = None if mask is None else fit(mask[v0:v1, u0:u1], height, width)
    return K, width, height, photo, mask, {"crop": list(window), "scale": scale}


def render(scene, assets, show, path):
    for name, objs in assets.items():
        for o in objs:
            o.hide_render = name != show
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return load(path)


def write_sheet(rows, path):
    """rows: (panels, has_photo). Rows without a photo (orbit and detail views) are packed side by
    side, up to the width of a photo row, so their renders stay large."""
    ncols = max([len(r) for r, photo in rows if photo] or [2 * len(rows[0][0])])
    packed = []
    for r, photo in rows:
        if not photo and packed and not packed[-1][1] and len(packed[-1][0]) + len(r) <= ncols:
            packed[-1][0].extend(r)
        else:
            packed.append((list(r), photo))
    blocks = []
    for r, _ in packed:
        h = round(PANEL * r[0].shape[0] / r[0].shape[1])
        row = np.ones((h, ncols * PANEL, 4), np.float32)
        for j, panel in enumerate(r):
            row[:, j * PANEL:(j + 1) * PANEL] = fit(panel, h, PANEL)
        blocks.append(row)
    s = np.concatenate(blocks)
    k = min(1.0, SHEET_MAX_SIDE / max(s.shape[:2]), math.sqrt(SHEET_MAX_PIXELS / (s.shape[0] * s.shape[1])))
    save(fit(s, int(s.shape[0] * k), int(s.shape[1] * k)), path)
