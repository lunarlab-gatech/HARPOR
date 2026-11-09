"""harpor/webapp/turntable.py, the 360 video of a web-app job: the GLB that physics used and its upright turn, the
container's binds, the render with the fake Blender of harpor.agent.fake_agent, and the PyAV encode. Also
blender/turntable.py itself, on stand-ins for bpy and mathutils, and harpor/webapp/softvideo.py, the soft-part video:
MuJoCo's press of a synthetic soft part after the real align and MJCF, its frames and its MP4."""
import importlib.util
import json
import math
import runpy
import shlex
import signal
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("trimesh")
from test_webapp_physics_auto import TILT, box_glb, plane_view, rot, write_run  # noqa: E402

from harpor.agent import fake_agent, tools  # noqa: E402
from harpor.agent.blender import crop  # noqa: E402
from harpor.physics.align import load_w  # noqa: E402
from harpor.webapp import physics_auto  # noqa: E402
from harpor.webapp import softvideo as S  # noqa: E402
from harpor.webapp import turntable as T  # noqa: E402

FAKE_BLENDER = [sys.executable, "-m", "harpor.agent.fake_agent", "--blender"]
POINTS = np.array([[-0.1, -0.4, -0.2], [0.3, 0.0, 0.2]])  # the box of the stand-in asset, in W


def aligned_job(tmp_path):
    """A job folder after the real align: a 0.3 x 0.1 x 0.2 m box, tilted in W, as run/posed.glb."""
    J = tmp_path / "job"
    write_run(J / "run", *plane_view())
    box_glb(J / "run" / "posed.glb")
    (J / "run" / "placement.json").write_text(json.dumps({"check": {"mean_mask_iou": 0.8}}))
    physics_auto.align_job(J)
    return J


def test_source_turns_the_physics_glb_upright(tmp_path, capsys):
    """The GLB of physics/up.json, and a matrix that puts it upright in turntable.py's W: the box's height (aligned.json's
    S extent) along -y (W's up) from 0, the long side on x, the footprint centered. A missing record, a GLB that is gone,
    a symlink or a GLB outside the job folder fails with the reason."""
    J = aligned_job(tmp_path)
    glb, M, rule = T.source(J)
    assert glb == (J / "run" / "posed.glb").resolve() and rule == "no agent result"
    v = load_w(glb)[0] @ M[:3, :3].T + M[:3, 3]
    lo, hi = v.min(0), v.max(0)
    ext = json.loads((J / "physics" / "aligned.json").read_text())["extent_s"]
    assert np.allclose(hi - lo, [ext[0], ext[2], ext[1]], atol=1e-4) and np.allclose(ext, [0.3, 0.2, 0.1], atol=1e-3)
    assert np.isclose(hi[1], 0, atol=1e-4) and np.isclose(lo[1], -ext[2], atol=1e-4)  # it stands on y = 0, up is -y
    assert np.allclose((lo + hi)[[0, 2]] / 2, 0, atol=1e-4)
    assert np.allclose(M[3], [0, 0, 0, 1]) and np.isclose(np.linalg.det(M[:3, :3]), 1)

    up = json.loads((J / "physics" / "up.json").read_text())
    (tmp_path / "outside.glb").write_bytes((J / "run" / "posed.glb").read_bytes())
    (J / "run" / "link.glb").symlink_to(J / "run" / "posed.glb")
    for glb_path, why in ((J / "run" / "gone.glb", "the GLB that physics used is gone"),
                          (J / "run" / "link.glb", "the GLB that physics used is not a file of this job"),
                          (tmp_path / "outside.glb", "the GLB that physics used is not a file of this job")):
        (J / "physics" / "up.json").write_text(json.dumps({**up, "glb": str(glb_path)}))
        with pytest.raises(SystemExit, match=why):
            T.source(J)
    (J / "physics" / "up.json").write_text(json.dumps(up))
    (J / "physics" / "aligned.json").unlink()
    with pytest.raises(SystemExit, match="run align first"):
        T.source(J)
    with pytest.raises(SystemExit, match="run align first"):
        T.source(tmp_path / "nothing")


def test_blender_command_binds_only_the_folder_the_glb_and_the_scripts(tmp_path):
    from harpor.agent import controller

    J = tmp_path / "job"
    glb = J / "agent" / "s" / "work" / "exports" / "003.glb"
    cmd = T.blender_command(J, glb, "/G/sif/harpor-agent.sif")
    assert cmd[:len(controller.CONTAINER)] == controller.CONTAINER and cmd[-2:] == ["/G/sif/harpor-agent.sif", "blender"]
    binds = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--bind"]
    out = (J / "turntable").resolve()
    assert binds == [f"{out}:{out}", f"{glb}:{glb}:ro", f"{tools.BLENDER_SCRIPTS}:{tools.BLENDER_SCRIPTS}:ro"]
    with pytest.raises(ValueError, match="cannot bind"):
        T.blender_command(tmp_path / "a:b", glb, "x.sif")
    # The soft-part video's container sees its own folder instead, which J/turntable alone would hide.
    soft = T.blender_command(J, glb, "/G/sif/harpor-agent.sif", "softvideo")
    out = (J / "softvideo").resolve()
    assert [soft[i + 1] for i, a in enumerate(soft) if a == "--bind"] == [f"{out}:{out}", *binds[1:]]


def test_render_writes_the_frames_and_its_record(tmp_path):
    """turntable.py's arguments (the GLB, its matrix, the frames folder and the stage's numbers), FRAMES PNGs in a
    fresh turntable/frames, and render.json. The fake Blender draws the box's outline: on the light gray background,
    wider than tall in frame 1, since the long side faces the camera."""
    from PIL import Image

    J = aligned_job(tmp_path)
    (J / "turntable").mkdir()
    (J / "turntable" / "turntable.json").write_text("{}")  # an earlier run's marker goes
    rec = T.render(J, blender=FAKE_BLENDER)
    out = J / "turntable"
    args = json.loads((out / "args.json").read_text())
    glb, M, _ = T.source(J)
    assert args == {"assets": [{"path": str(glb), "matrix": M.tolist()}], "frames_dir": str(out.resolve() / "frames"),
                    "frames": T.FRAMES, "fps": T.FPS, "panel": T.SIDE, "samples": T.SAMPLES, "elevation": T.ELEVATION_DEG}
    assert sorted(p.name for p in (out / "frames").iterdir()) == [f"{k:04d}.png" for k in range(1, T.FRAMES + 1)]
    assert not (out / "turntable.json").exists() and (out / "blender.log").is_file()
    assert json.loads((out / "render.json").read_text()) == rec
    assert (rec["glb"], rec["glb_rule"], rec["frames"], rec["side"], rec["fps"], rec["elevation_deg"]) == \
        ("run/posed.glb", "no agent result", 240, 720, 30, 20)
    with Image.open(out / "frames" / "0001.png") as im:
        a = np.asarray(im)
    assert a[0, 0].tolist() == [209, 209, 209]
    ys, xs = np.nonzero(a[..., 0] < 100)
    assert np.ptp(xs) > 1.5 * np.ptp(ys) > 0


def run_blender_script(tmp_path, monkeypatch, args, objects=None, kdtree=None):
    """blender/turntable.py with args, on stand-ins for bpy and mathutils. render_common is the real one, except for
    its GLB import (objects, else one asset) and its points. The mathutils stand-in has kdtree only when it is given.
    Returns (bpy, the objects the script made by name, the asset, the pivot's keys)."""
    asset, pivot, keys = MagicMock(matrix_world=np.eye(4)), MagicMock(), []
    asset.parent = None
    pivot.keyframe_insert.side_effect = lambda path, frame: keys.append((path, frame, pivot.rotation_euler))
    made = {"pivot": pivot}
    bpy = MagicMock()
    bpy.data.objects.new.side_effect = lambda name, data: made.setdefault(name, MagicMock())
    bpy.context.scene.render.image_settings = SimpleNamespace()  # holds only what the script sets
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    mathutils = SimpleNamespace(Matrix=np.array, Vector=MagicMock(side_effect=lambda v: MagicMock(xyz=list(v))))
    if kdtree is not None:
        mathutils.kdtree = kdtree
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    monkeypatch.setitem(sys.modules, "crop", crop)
    spec = importlib.util.spec_from_file_location("render_common", tools.BLENDER_SCRIPTS / "render_common.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.import_glb, common.points = (lambda path: objects or [asset]), (lambda objs: POINTS)
    monkeypatch.setitem(sys.modules, "render_common", common)
    (tmp_path / "args.json").write_text(json.dumps(args))
    monkeypatch.setattr(sys, "argv", ["blender", "--", str(tmp_path / "args.json")])
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script puts its folder first
    runpy.run_path(str(tools.BLENDER_SCRIPTS / "turntable.py"))
    return bpy, made, asset, keys


def test_blender_script_turns_the_asset_once_as_png_frames(tmp_path, monkeypatch):
    """The real turntable.py on the arguments of the stage. The asset goes through its matrix and onto the pivot at
    its center, the pivot turns once about y, and the camera looks at the center along +z of W, from ELEVATION_DEG
    above, at the distance that fits the turning asset. One render of the animation writes 8-bit PNG frames."""
    J = aligned_job(tmp_path)
    T.render(J, blender=FAKE_BLENDER)
    args = json.loads((J / "turntable" / "args.json").read_text())
    bpy, made, asset, keys = run_blender_script(tmp_path, monkeypatch, args)
    scene, pivot, cam = bpy.context.scene, made["pivot"], made["camera"]
    settings = scene.render.image_settings
    assert vars(settings) == {"media_type": "IMAGE", "file_format": "PNG", "color_mode": "RGB", "color_depth": "8"}
    assert scene.render.filepath == args["frames_dir"] + "/"
    assert (scene.frame_start, scene.frame_end, scene.render.fps) == (1, T.FRAMES, T.FPS)
    assert (scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage) == (T.SIDE, T.SIDE, 100)
    assert scene.cycles.samples == T.SAMPLES and scene.render.use_persistent_data is True and scene.camera is cam
    assert np.array_equal(asset.matrix_world, np.array(args["assets"][0]["matrix"])) and asset.parent is pivot
    center, r = POINTS.mean(0), np.linalg.norm(np.ptp(POINTS, axis=0)) / 2
    assert np.allclose(pivot.location.xyz, center) and pivot.rotation_mode == "XYZ"
    assert keys == [("rotation_euler", 1, (0.0, 0.0, 0.0)), ("rotation_euler", T.FRAMES + 1, (0.0, -2 * math.pi, 0.0))]
    dist, e = 1.3 * r / math.tan(math.radians(15)), math.radians(T.ELEVATION_DEG)
    assert np.allclose(cam.matrix_world[:3, 3], center + dist * np.array([0.0, -math.sin(e), -math.cos(e)]))
    assert np.allclose(-cam.matrix_world[:3, 2], [0.0, math.sin(e), math.cos(e)])  # a Blender camera looks along its -z
    assert np.allclose(cam.matrix_world[:3, 0], [1.0, 0.0, 0.0])  # image right is +x of W
    assert (cam.data.clip_start, cam.data.clip_end) == pytest.approx((1e-3 * dist, 1e3 * dist))
    bpy.ops.render.render.assert_called_once_with(animation=True)


@pytest.mark.parametrize("args", [{"assets": [{"path": "a.glb"}, {"path": "b.glb"}], "frames_dir": "f", "out": "v.mp4"},
                                  {"assets": [{"path": "a.glb"}], "out": "v.mp4"}])
def test_blender_script_takes_one_asset_and_frames_dir(tmp_path, monkeypatch, args):
    """Anything else stops the script before it renders, in the real script and in the fake Blender."""
    with pytest.raises(AssertionError, match="exactly one asset and frames_dir"):
        run_blender_script(tmp_path, monkeypatch, args)
    with pytest.raises(AssertionError, match="exactly one asset and frames_dir"):
        fake_agent.turntable_frames(args)


def test_render_fails_with_the_reason(tmp_path):
    """A Blender that exits 1, one that writes too few frames, and one that passes the timeout each fail with the reason
    and the log's place; without a SIF or a Blender command nothing runs."""
    J = aligned_job(tmp_path)
    few = ("import json, pathlib, sys\na = json.loads(pathlib.Path(sys.argv[-1]).read_text())\n"
           "[(pathlib.Path(a['frames_dir']) / f'{k:04d}.png').write_bytes(b'') for k in range(1, 11)]")
    for blender, timeout, why in (
            ([sys.executable, "-c", "import sys; print('no GPU'); sys.exit(1)", "--"], 60, r"Blender failed \(exit 1\)"),
            ([sys.executable, "-c", few, "--"], 60, f"Blender wrote 10 of {T.FRAMES} frames"),
            ([sys.executable, "-c", "import time; time.sleep(30)", "--"], 0.5, "Blender took longer than 0.5 s")):
        with pytest.raises(SystemExit, match=why + "; see turntable/blender.log"):
            T.render(J, blender=blender, timeout=timeout)
        assert not (J / "turntable" / "render.json").exists()
    assert "no GPU" not in (J / "turntable" / "blender.log").read_text()  # each run starts a fresh log
    with pytest.raises(SystemExit, match="render needs --sif or --blender"):
        T.render(J)


def test_encode_makes_a_browser_safe_video_that_loops(tmp_path, capsys):
    """The CLI's render (the fake Blender as a shell string) and encode: H.264 in yuv420p, FRAMES frames at FPS fps
    (FRAMES / FPS s, one turn), the index (moov) before the data (mdat), so a browser plays it before it has it all.
    turntable.json, the done marker, is written last; the frames and the temporary file are gone."""
    import av

    J = aligned_job(tmp_path)
    old = signal.getsignal(signal.SIGTERM)
    try:
        T.main(["render", "--job", str(J), "--blender", shlex.join(FAKE_BLENDER)])
    finally:
        signal.signal(signal.SIGTERM, old)  # render takes SIGTERM, so Cancel kills Blender
    T.main(["encode", "--job", str(J)])
    out = J / "turntable"
    mp4, rec = out / "turntable.mp4", json.loads((out / "turntable.json").read_text())
    with av.open(str(mp4)) as c:
        s = c.streams.video[0]
        frames = list(c.decode(s))
        assert (s.codec_context.name, s.codec_context.pix_fmt, s.average_rate) == ("h264", "yuv420p", T.FPS)
        assert len(frames) == T.FRAMES and frames[0].pts == 0
        assert float(s.duration * s.time_base) == pytest.approx(T.FRAMES / T.FPS)
        keys = [f.key_frame for f in frames]
    assert sum(keys) == T.FRAMES // (2 * T.FPS) and keys[0]  # a keyframe every 2 s
    data = mp4.read_bytes()
    assert 0 < data.index(b"moov") < data.index(b"mdat")
    assert (rec["codec"], rec["pix_fmt"], rec["frames"], rec["seconds"], rec["bytes"]) == \
        ("h264", "yuv420p", T.FRAMES, T.FRAMES / T.FPS, len(data))
    assert rec["glb"] == "run/posed.glb" and rec["width"] == rec["height"] == 96  # the fake Blender's frames
    assert out.joinpath("turntable.json").stat().st_mtime_ns >= mp4.stat().st_mtime_ns
    assert not (out / "frames").exists() and not (out / ".tmp_turntable.mp4").exists()
    printed = capsys.readouterr().out
    assert "PROGRESS 1\n" in printed and "360 video: 240 frames, 96x96, 8.0 s, " in printed
    with pytest.raises(SystemExit, match="no frames to encode"):
        T.encode(J)


def test_progress_follows_the_frames_on_disk(tmp_path, capsys):
    """While Blender renders, a thread prints PROGRESS for the share of the frames on disk, up to 0.9 (encode ends at 1),
    and only when it changes."""
    import threading
    import time

    for k in range(T.FRAMES // 2):
        (tmp_path / f"{k + 1:04d}.png").write_bytes(b"")
    stop = threading.Event()
    t = threading.Thread(target=T._count, args=(tmp_path, stop, 0.02))
    t.start()
    time.sleep(0.2)
    stop.set()
    t.join()
    assert capsys.readouterr().out == "PROGRESS 0.45\n"


# ---- the soft-part video (harpor/webapp/softvideo.py) ----

class Vertices:
    """A stand-in for a Blender mesh's vertices: foreach_get reads co, and each foreach_set is kept."""

    def __init__(self, co):
        self.co, self.sets = np.asarray(co, float), []

    def __len__(self):
        return len(self.co)

    def foreach_get(self, name, out):
        out[:] = self.co.ravel()

    def foreach_set(self, name, values):
        self.sets.append(np.asarray(values, float).reshape(-1, 3))


class KDTree:
    """mathutils.kdtree.KDTree on scipy's cKDTree: find gives (co, index, distance)."""

    def __init__(self, n):
        self.points = [None] * n

    def insert(self, co, index):
        self.points[index] = co

    def balance(self):
        from scipy.spatial import cKDTree

        self.tree = cKDTree(np.array(self.points))

    def find(self, co):
        dist, i = self.tree.query(co)
        return self.points[i], int(i), float(dist)


def mesh_object(name, co):
    return SimpleNamespace(name=name, type="MESH", parent=None, matrix_world=np.eye(4), data=SimpleNamespace(
        vertices=Vertices(co), has_custom_normals=True, shade_smooth=MagicMock(), update=MagicMock()))


def test_blender_script_dents_the_soft_part(tmp_path, monkeypatch):
    """With soft the asset does not turn. The vertices whose matrix_world @ co lies on a rest target (UPRIGHT_TO_W @
    rest, found with kdtree, which the script imports only here) move to matrix_world^-1 @ (UPRIGHT_TO_W @ (rest +
    disp)) in each frame, which is one still render, without persistent data. A vertex of the part off the targets
    stays, and so does a parent whose seam meets the part, with under half of its vertices there. The part loses its
    custom normals, and the camera frames it as the turntable frames the asset."""
    rest = np.array([[0.0, 0.0, 0.1], [0.2, 0.0, 0.1], [0.0, 0.1, 0.1], [0.2, 0.1, 0.12]])  # in S
    disp = np.zeros((3, 4, 3))
    disp[1, 3, 2], disp[2, 3, 2] = -0.01, -0.004  # a corner dents and comes back
    np.savez(tmp_path / "motion.npz", rest=rest, disp=disp)
    U = T.UPRIGHT_TO_W
    M = U @ np.array([[0.0, -1.0, 0.0, 0.3], [1.0, 0.0, 0.0, -0.1], [0.0, 0.0, 1.0, 0.05], [0.0, 0.0, 0.0, 1.0]])
    inv = np.linalg.inv(M)  # the asset's matrix: the import's co are W's numbers before it

    def co(f, rows):  # the import's co of the targets rows in frame f
        w = (rest + disp[f]) @ U[:3, :3].T + U[:3, 3]
        return w[rows] @ inv[:3, :3].T + inv[:3, 3]

    rows = [2, 0, 3, 1]
    part = mesh_object("pad", np.r_[co(0, rows), [[1.0, 1.0, 1.0]]])
    parent = mesh_object("block", np.r_[co(0, [0]), [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [7.0, 0.0, 0.0]]])
    args = {"assets": [{"path": "x.glb", "matrix": M.tolist()}], "frames_dir": str(tmp_path / "frames"), "frames": 3,
            "elevation": 20, "soft": {"motion": str(tmp_path / "motion.npz"), "matrix": U.tolist()}}
    bpy, made, _, keys = run_blender_script(tmp_path, monkeypatch, args, [part, parent], SimpleNamespace(KDTree=KDTree))
    scene, cam = bpy.context.scene, made["camera"]
    assert np.array_equal(part.matrix_world, M) and keys == []
    assert [c.args[0] for c in bpy.data.objects.new.call_args_list] == ["sun", "camera"]  # no pivot
    assert len(part.data.vertices.sets) == 3 and parent.data.vertices.sets == []
    for f, x in enumerate(part.data.vertices.sets):
        assert np.allclose(x, np.r_[co(f, rows), [[1.0, 1.0, 1.0]]], atol=1e-6), f
    assert bpy.ops.render.render.call_args_list == [((), {"write_still": True})] * 3
    assert scene.render.filepath == f"{tmp_path / 'frames'}/0003.png" and scene.render.use_persistent_data is False
    bpy.ops.mesh.customdata_custom_splitnormals_clear.assert_called_once_with()
    part.data.shade_smooth.assert_called_once_with()
    parent.data.shade_smooth.assert_not_called()
    on = rest @ U[:3, :3].T + U[:3, 3]
    center, r = (on.min(0) + on.max(0)) / 2, np.linalg.norm(np.ptp(on, axis=0)) / 2
    dist, e = 1.3 * r / math.tan(math.radians(15)), math.radians(20)
    assert np.allclose(cam.matrix_world[:3, 3], center + dist * np.array([0.0, -math.sin(e), -math.cos(e)]))
    # A GLB without the part stops the script before it renders.
    with pytest.raises(AssertionError, match="no object of the GLB lies on the soft part"):
        run_blender_script(tmp_path, monkeypatch, args, [parent], SimpleNamespace(KDTree=KDTree))


def soft_job(tmp_path, lid=False):
    """A job folder after the real align and MJCF (one hull per link): a 0.6 x 0.6 x 0.2 m block with a 46 x 44 x 7 cm
    foam lens on it, soft on the block, standing tilted in W as run/posed.glb. lid: the lens lies on a 4 cm lid
    instead, hinged to the block at its back edge from -0.5 to 1.5 rad, so that a press can turn it either way."""
    import trimesh
    from test_physics import hull_parts
    from test_sim_check import lens

    from harpor import frames
    from harpor.parts.joints import SOFT_MATERIALS, write_joints
    from harpor.physics import mjcf
    from harpor.physics.align import read_glb, write_glb

    J = tmp_path / "job"
    write_run(J / "run", *plane_view())
    (J / "run" / "placement.json").write_text(json.dumps({"check": {"mean_mask_iou": 0.8}}))
    R, c = TILT @ rot(0, 90), np.array([0.0, 0.4, 2.0])  # z up onto the plane's up
    block, top = trimesh.creation.box([0.6, 0.6, 0.2]), trimesh.creation.box([0.6, 0.6, 0.04])
    block.apply_translation([0.0, 0.0, 0.1])
    top.apply_translation([0.0, 0.0, 0.22])
    parts = [("block", block.subdivide().subdivide())]
    if lid:
        parts.append(("lid", top.subdivide().subdivide()))
    scene = trimesh.Scene()
    for name, m in (*parts, ("pad", lens([0.46, 0.44, 0.07], [0.0, 0.0, 0.275 if lid else 0.235]))):
        scene.add_geometry(trimesh.Trimesh(frames.w_to_gltf(m.vertices @ R.T + c), m.faces, process=False),
                           node_name=name, geom_name=name)
    glb = J / "run" / "posed.glb"
    scene.export(glb)
    doc, binary = read_glb(glb)
    pad = {"version": 1, "frame": "W", "parent": "lid" if lid else "block", "type": "soft", "material": "foam",
           **SOFT_MATERIALS["foam"], "support": "parent"}
    hinge = {"version": 1, "frame": "W", "parent": "block", "type": "revolute", "lower": -0.5, "upper": 1.5,
             "origin": (R @ [0.0, 0.3, 0.2] + c).tolist(), "axis": (R @ [1.0, 0.0, 0.0]).tolist()}
    write_glb(glb, write_joints(doc, {"pad": pad, **({"lid": hinge} if lid else {})}), binary)
    physics_auto.align_job(J)
    (J / "physics" / "summary.json").write_text(json.dumps({"median": {"density_kg_m3": 600.0}}))
    mjcf.build(J / "physics" / "aligned.glb", J / "physics" / "summary.json", J / "physics" / "mjcf", "pad_box",
               decompose=hull_parts)
    return J


def test_motion_presses_the_soft_part_to_its_dent(tmp_path):
    """motion on the mjcf stage's model, its root fixed and no gravity: the visible dent reaches its target, 12 mm (a
    fifth of the lens's 7 cm is more), before the force cap, and the part springs back. motion.npz holds the part's
    vertices in S and FRAMES frames of their displacement, frame 0 at rest. Its rest targets, UPRIGHT_TO_W @ rest, are
    the GLB's own vertices as turntable.py sees them after the asset's matrix, so that its match finds them."""
    mujoco = pytest.importorskip("mujoco")
    from harpor.physics import mjcf

    J = soft_job(tmp_path)
    m = S.model(J / "physics" / "mjcf" / "pad_box.xml")
    assert mujoco.mjtJoint.mjJNT_FREE not in set(m.jnt_type) and not m.opt.gravity.any() and m.nflex == 1
    assert mujoco.MjModel.from_xml_path(str(J / "physics" / "mjcf" / "pad_box.xml")).jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    rec = S.motion(J)
    assert json.loads((J / "softvideo" / "motion.json").read_text()) == rec
    assert (rec["link"], rec["frames"], rec["fps"], rec["target_m"]) == ("pad", S.FRAMES, S.FPS, S.DENT_M)
    assert rec["dent_m"] >= rec["target_m"] and not rec["cap_limited"] and rec["force_n"] < rec["cap_n"]
    with np.load(J / "softvideo" / "motion.npz") as z:
        rest, disp = z["rest"], z["disp"]
    (want,) = [g.vertices for k, _, g in mjcf.load_s(J / "physics" / "aligned.glb", by_link=True) if k == "pad"]
    assert np.allclose(rest, want, atol=1e-9) and disp.shape == (S.FRAMES, len(rest), 3) and disp.dtype == np.float32
    assert np.abs(disp[0]).max() < 1e-6 and np.abs(disp[-1]).max() < 1e-4  # at rest, and back again
    assert (disp @ np.array([0.0, 0.0, -1.0], np.float32)).max() > 0.01  # the dent, toward the block
    glb, M, _ = T.source(J)
    (seen,) = [g.vertices for k, _, g in mjcf.load_s(glb, by_link=True) if k == "pad"]  # W: Blender's import
    U = T.UPRIGHT_TO_W
    assert np.abs((seen @ M[:3, :3].T + M[:3, 3]) - (rest @ U[:3, :3].T + U[:3, 3])).max() < 1e-6
    mj = json.loads((J / "physics" / "mjcf" / "mjcf.json").read_text())
    (J / "physics" / "mjcf" / "mjcf.json").write_text(json.dumps({**mj, "soft": {}}))
    with pytest.raises(SystemExit, match="has no soft part"):
        S.motion(J)


def test_motion_welds_a_hinged_parent(tmp_path):
    """A soft part on a hinged lid: model welds the lid, as every hinge and slide, since its position actuator holds it
    against the lid's own inertia and would give way under the press, while the video shows the lid at rest. So the
    press moves only the dent: the pinned nodes stay where they are, and no vertex moves much further than the dent."""
    mujoco = pytest.importorskip("mujoco")
    from harpor.physics import mjcf, sim_check

    J = soft_job(tmp_path, lid=True)
    xml = J / "physics" / "mjcf" / "pad_box.xml"
    exported = mujoco.MjModel.from_xml_path(str(xml))
    assert exported.nu == 1 and mujoco.mjtJoint.mjJNT_HINGE in set(exported.jnt_type)  # the lid's hinge and actuator
    m = S.model(xml)
    assert m.nu == 0 and set(m.jnt_bodyid) <= mjcf.node_bodies(m)  # only the flex's node slides are left
    with np.load(J / "physics" / "mjcf" / "soft" / "pad.npz") as z:
        b = {k: z[k] for k in z.files}
    flex = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, str(b["flex"]))
    rec = json.loads((J / "physics" / "mjcf" / "mjcf.json").read_text())["soft"]["pad"]
    trace = sim_check.press(m, mujoco.MjData(m), flex, b["press"], b["direction"], S.DENT_M,
                            S.CAP * rec["young"] * rec["pitch"] ** 2)
    pinned = b["pinned"]
    assert np.abs(trace["P"][:, pinned] - trace["rest"][pinned]).max() < 1e-6
    done = S.motion(J)
    with np.load(J / "softvideo" / "motion.npz") as z:
        disp = z["disp"]
    assert done["dent_m"] >= done["target_m"] and not done["cap_limited"]
    assert np.linalg.norm(disp, axis=2).max() < 1.5 * done["dent_m"]


def test_soft_part_video_renders_and_encodes(tmp_path, capsys):
    """render after the motion (here the fake stage's, without MuJoCo): turntable.py's arguments with soft (the motion
    and UPRIGHT_TO_W) on the 360 video's GLB and matrix, FRAMES frames of the fake Blender in J/softvideo/frames, and
    render.json with the motion's record; encode: an H.264 MP4 of FRAMES / FPS s, then softvideo.json last, without the
    frames. Without the motion, render stops, and a Blender that fails names the stage's log."""
    import fake_steps

    J = soft_job(tmp_path)
    with pytest.raises(SystemExit, match="the motion step did not finish"):
        S.render(J, blender=FAKE_BLENDER)
    fake_steps.soft_motion(J)
    rec = S.render(J, blender=FAKE_BLENDER)
    out = (J / "softvideo").resolve()
    glb, M, rule = T.source(J)
    assert json.loads((out / "args.json").read_text()) == {
        "assets": [{"path": str(glb), "matrix": M.tolist()}], "frames_dir": str(out / "frames"), "frames": S.FRAMES,
        "fps": S.FPS, "panel": T.SIDE, "samples": T.SAMPLES, "elevation": T.ELEVATION_DEG,
        "soft": {"motion": str(out / "motion.npz"), "matrix": T.UPRIGHT_TO_W.tolist()}}
    assert sorted(p.name for p in (out / "frames").iterdir()) == [f"{k:04d}.png" for k in range(1, S.FRAMES + 1)]
    assert json.loads((out / "render.json").read_text()) == rec
    assert (rec["link"], rec["glb"], rec["glb_rule"], rec["frames"]) == ("pad", "run/posed.glb", rule, S.FRAMES)
    S.main(["encode", "--job", str(J)])
    done = json.loads((out / "softvideo.json").read_text())
    assert (done["codec"], done["pix_fmt"], done["frames"], done["seconds"]) == ("h264", "yuv420p", S.FRAMES,
                                                                                 S.FRAMES / S.FPS)
    assert T.probe(out / "softvideo.mp4")["frames"] == S.FRAMES and done["bytes"] == (out / "softvideo.mp4").stat().st_size
    assert not (out / "frames").exists() and not (out / ".tmp_softvideo.mp4").exists()
    assert "soft-part video: 90 frames, 96x96, 3.0 s, " in capsys.readouterr().out
    with pytest.raises(SystemExit, match=r"Blender failed \(exit 1\); see softvideo/blender.log"):
        S.render(J, blender=[sys.executable, "-c", "import sys; sys.exit(1)", "--"])
