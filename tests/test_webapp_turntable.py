"""harpor/webapp/turntable.py, the 360 video of a web-app job: the GLB that physics used and its upright turn, the
container's binds, the render with the fake Blender of harpor.agent.fake_agent, and the PyAV encode. Also
blender/turntable.py itself, on stand-ins for bpy and mathutils."""
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
from test_webapp_physics_auto import box_glb, plane_view, write_run  # noqa: E402

from harpor.agent import fake_agent, tools  # noqa: E402
from harpor.agent.blender import crop  # noqa: E402
from harpor.physics.align import load_w  # noqa: E402
from harpor.webapp import physics_auto  # noqa: E402
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
