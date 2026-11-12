import errno
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("av")
pytest.importorskip("httpx")

from harpor.fileio import write_json  # noqa: E402
from harpor.webapp import jobs, run, steps  # noqa: E402
from test_webapp_server import ROOT, calibration_yaml, scene_bag, write_bag, write_video  # noqa: E402

OPUS = "sk-ant-test-" + secrets.token_hex(16)
ASTRA = "sk-test-" + secrets.token_hex(16)


class Term:
    """The command in this process, on one runs folder. A call returns (exit code, what it printed)."""

    def __init__(self, tmp_path, capsys):
        self.tmp, self.runs, self.capsys = tmp_path, tmp_path / "runs", capsys

    def exact(self, args):
        code = run.main(args)
        return code, self.capsys.readouterr().out

    def __call__(self, *args):
        return self.exact([*map(str, args), "--runs", str(self.runs)])

    def folder(self, out):
        """The folder of the job that out names."""
        return self.runs / re.search(r"^job (\S+) in ", out, re.M)[1]

    def video(self):
        return write_video(self.tmp / "cup.mp4")

    def key_file(self, text=f"ANTHROPIC_API_KEY={OPUS}\n", mode=0o600):
        d = self.tmp / "keys"
        d.mkdir(mode=0o700, exist_ok=True)
        f = d / "my.env"
        f.write_text(text)
        f.chmod(mode)
        return f


@pytest.fixture
def term(tmp_path, monkeypatch, capsys, fake_stages):
    monkeypatch.setattr(jobs, "default_key_dir", lambda: tmp_path / "shm" / "gator-webapp-1")
    monkeypatch.setattr(run, "POLL_S", 0.05)
    return Term(tmp_path, capsys)


def printed(out, mark="to go on: "):
    """The arguments of the command in the line of out that holds mark, which ends right before the command."""
    words = shlex.split(next(x for x in out.splitlines() if mark in x).split(mark, 1)[1])
    assert words[:3] == [sys.executable, "-m", "harpor.webapp.run"]
    return words[3:]


def fail_once(monkeypatch, stage, script):
    """The next run of the stage is this bash script, in the job folder."""
    real, left = steps.command, [script]

    def command(name, job, cfg, key_file=None):
        if name == stage and left:
            return ["bash", "-c", left.pop()], str(cfg.runs / job["id"]), {}
        return real(name, job, cfg, key_file)

    monkeypatch.setattr(steps, "command", command)


def stage_times(J):
    return {name: s["started"] for name, s in jobs.load(J)["stages"].items()}


def selection(J):
    return json.loads((J / "selection.json").read_text())


def job_folders(term):
    return [p for p in term.runs.glob("*") if p.is_dir()]


def test_a_job_runs_from_the_input_to_its_results(term):
    """One command, no port: every stage prints, the settings reach job.json and the input is a hard link."""
    video = write_video(term.tmp / "my cup.mp4")
    code, out = term(video, "--prompt", "Wooden box", "--n-views", 4, "--turntable")
    assert code == run.DONE, out
    J = term.folder(out)
    job, lines = jobs.load(J), out.splitlines()
    assert (job["status"], job["prompt"], job["n_views"], job["mode"], job["agent"], job["turntable"], job["auto_continue"]) \
        == ("done", "Wooden box", 4, "auto", "none", True, True)
    assert job["video"]["bytes"] == video.stat().st_size and os.path.samefile(video, J / "video" / "my_cup.mp4")
    assert "topics: skipped: video input" in lines and "agent: skipped: turned off: physics uses the Stage A mesh" in lines
    assert all(any(x.startswith(f"{name}: ") for x in lines) for name in jobs.BUILD)
    assert lines[-1] == f"results: {J / 'results.zip'}"
    with zipfile.ZipFile(J / "results.zip") as z:
        assert {"aligned.glb", "turntable.mp4", "mjcf/mjcf.json"} <= set(z.namelist())
    # Nothing of the Runner stays.
    assert not (term.runs / jobs.RUNNER_FILE).exists() and not list((term.tmp / "shm").glob("*"))

    # A done job prints its results and runs nothing.
    before = stage_times(J)
    assert term("--resume", J.name) == (run.DONE, f"results: {J / 'results.zip'}\n")
    assert stage_times(J) == before


def test_help_prints():
    with pytest.raises(SystemExit) as e:
        run.main(["--help"])
    assert e.value.code == 0


@pytest.mark.parametrize("args, reason", [
    ([], "give an input file, or --resume ID"),
    (["{video}", "--resume", "20261101-000000-abcdef"], "give an input file, or --resume ID"),
    (["{tmp}/nope.mp4"], "is not a file"),
    (["{tmp}/empty.mp4"], "is empty"),
    (["{tmp}/notes.txt"], ".txt is none of these"),
    (["{video}", "--camera-topic", "/a"], "are for a rosbag"),
    (["{video}", "--lidar-topic", "/a"], "are for a rosbag"),
    (["{video}", "--no-lidar"], "--no-lidar is for a Stray Scanner capture"),
    (["{bag}", "--calibration", "{tmp}/notes.txt"], "--calibration needs --lidar-topic"),
    (["{bag}", "--lidar-topic", "/a", "--calibration", "{tmp}/nope.yaml"], "is not a file"),
    (["{bag}", "--lidar-topic", "/a", "--calibration", "{tmp}/notes.txt"], "the calibration is .yaml, .yml, .json"),
    (["{video}", "--picks", "1,x"], "--picks takes frames"),
    (["{video}", "--view0", "x"], "--view0 takes a frame"),
    (["{video}", "--lock", "12"], "answer a view review, on a resume"),
    (["{video}", "--key-file", "{tmp}/k.env"], "--key-file needs --agent astra, opus or fable"),
])
def test_bad_arguments_stop_before_anything_runs(term, args, reason):
    (term.tmp / "empty.mp4").touch()
    (term.tmp / "notes.txt").write_text("notes")
    paths = {"tmp": term.tmp, "video": term.tmp / "cup.mp4", "bag": term.tmp / "cup.bag"}
    paths["video"].write_bytes(b"not read")
    paths["bag"].write_bytes(b"not read")
    code, out = term(*[x.format(**paths) for x in args])
    assert code == run.BAD and reason in out, out
    assert not term.runs.exists()  # no Runner started


def test_resume_of_a_job_that_is_not_there(term):
    for job_id in ("20261101-000000-abcdef", "../x"):
        code, out = term("--resume", job_id)
        assert code == run.BAD and f"no job {job_id} in" in out and "to go on" not in out


def test_without_a_prompt_the_command_stops_after_the_analysis(term):
    """The person looks at the thumbnails, and the printed command keeps the flags that job.json does not hold yet."""
    code, out = term(term.video(), "--n-views", 3, "--turntable")
    J = term.folder(out)
    assert code == run.WAITS and jobs.load(J)["status"] == "ready"
    assert f"the input has 64 frames; the thumbnails are in {J / 'frames' / 'thumbs'}, named <frame>.jpg" in out
    assert (J / "frames" / "thumbs" / "12.jpg").is_file()
    args = printed(out, "to start the build: ")
    assert args[-2:] == ["--prompt", "NOUN"] and "--n-views=3" in args and "--turntable" in args
    code, out = term.exact([*args[:-1], "box", "--picks", "8,24,40", "--view0", "24"])
    assert code == run.DONE, out
    job = jobs.load(J)
    assert (job["mode"], job["picks"], job["view0"], job["n_views"], job["turntable"]) == ("manual", [8, 24, 40], 24, 3, True)
    pool = json.loads((J / "frames" / "pool.json").read_text())
    assert selection(J)["anchor"] == f"f{pool['picks']['24']:05d}"
    assert "frames:" not in out  # the analysis stays


def test_a_refused_start_says_why_and_the_job_waits(term):
    code, out = term(term.video(), "--prompt", "box", "--picks", "1,2,9000")
    J = term.folder(out)
    assert code == run.BAD and "picks are different frames from 0 to 63" in out and jobs.load(J)["status"] == "ready"
    args = printed(out)
    assert "--picks=1,2,9000" in args
    code, out = term.exact([*(a for a in args if not a.startswith("--picks")), "--review", "--n-views", "3"])
    assert code == run.WAITS and jobs.load(J)["status"] == "review", out


def test_a_failed_stage_says_why_and_a_resume_finishes(term, monkeypatch):
    fail_once(monkeypatch, "sam3d", "echo loading; echo 'CUDA out of memory' >&2; exit 7")
    code, out = term(term.video(), "--prompt", "box", "--n-views", 3)
    J = term.folder(out)
    assert code == run.FAILED and jobs.load(J)["status"] == "failed"
    assert "the job failed: sam3d: exit 7: CUDA out of memory" in out and f"log file: {J / 'logs' / 'sam3d.log'}" in out
    assert "sam3d: failed after " in out and "place:" not in out
    assert "CUDA out of memory" in (J / "logs" / "sam3d.log").read_text()
    args, before = printed(out), stage_times(J)
    assert args[:2] == ["--resume", J.name] and not any(a.startswith("--prompt") for a in args)  # job.json holds it
    code, out = term.exact(args)
    assert code == run.DONE and "sam3d: done" in out and "select:" not in out, out
    after = stage_times(J)
    assert after["select"] == before["select"] and after["sam3d"] > before["sam3d"]


def test_a_failed_analysis_repeats_the_settings(term, monkeypatch):
    fail_once(monkeypatch, "frames", "echo 'cannot decode'; exit 1")
    code, out = term(term.video(), "--prompt", "red box", "--n-views", 3)
    J = term.folder(out)
    assert code == run.FAILED and "the job failed: frames: exit 1: cannot decode" in out
    args = printed(out)
    assert "--prompt=red box" in args and "--n-views=3" in args
    code, out = term.exact(args)
    assert code == run.DONE and jobs.load(J)["prompt"] == "red box", out


def test_a_failed_360_video_warns_and_a_resume_tries_it_again(term, monkeypatch):
    fail_once(monkeypatch, "turntable", "echo 'no GPU'; exit 3")
    code, out = term(term.video(), "--prompt", "box", "--n-views", 3, "--turntable")
    J = term.folder(out)
    assert code == run.DONE and "warning: 360 video: exit 3: no GPU" in out and "turntable: failed after " in out
    code, out = term("--resume", J.name)
    assert code == run.DONE and "turntable: done" in out and "warning" not in out, out
    with zipfile.ZipFile(J / "results.zip") as z:
        assert "turntable.mp4" in z.namelist()


def test_the_view_review_stops_and_a_resume_goes_on(term):
    code, out = term(term.video(), "--prompt", "box", "--review", "--n-views", 3)
    J = term.folder(out)
    sel = selection(J)
    assert code == run.WAITS and jobs.load(J)["status"] == "review", out
    assert "sam3d:" not in out and f"view 0: {sel['picks'][0]['name']} at " in out
    assert f"mask check images: {J / 'scene' / 'sam3_box' / 'check'}, named <view>.jpg" in out
    spare = [c["name"] for c in sel["candidates"] if c["status"] == "available"]
    assert f"left out: {spare[0]} at " in out and ": usable, not picked" in out
    assert printed(out) == ["--resume", J.name, f"--runs={term.runs}"]

    # Each answer selects again, adds to the earlier ones and stops here again.
    drop = sel["picks"][1]["name"]
    code, out = term("--resume", J.name, "--exclude", drop)
    assert code == run.WAITS and jobs.load(J)["overrides"]["excluded"] == [drop]
    assert re.search(rf"left out: {drop} at .*: excluded$", out, re.M) and "sam3d:" not in out
    code, out = term("--resume", J.name, "--lock", str(int(spare[0][1:])))  # a frame number means the same view
    over = jobs.load(J)["overrides"]
    assert code == run.WAITS and (over["locked"], over["excluded"]) == ([spare[0]], [drop])
    assert spare[0] in [p["name"] for p in selection(J)["picks"]]
    code, out = term("--resume", J.name, "--view0", spare[0])
    assert code == run.WAITS and selection(J)["anchor"] == spare[0] and f"view 0: {spare[0]} at " in out
    code, out = term("--resume", J.name, "--exclude", spare[0], "--lock", drop)  # the flags move a view across
    over = jobs.load(J)["overrides"]
    assert code == run.WAITS and (over["locked"], over["excluded"]) == ([drop], [spare[0]]), out

    # A view with no instance.
    gone = sel["picks"][0]["name"]
    code, out = term("--resume", J.name, "--instance", f"{gone}=none")
    assert code == run.WAITS and jobs.load(J)["overrides"]["instance"] == {gone: "none"}, out
    assert re.search(rf"left out: {gone} at .*: no mask of the object$", out, re.M) and "masks: done" in out

    # The route's refusals change nothing.
    times = stage_times(J)
    for flags, why in ((("--exclude", "f99999"), "lists of pool frames"), (("--instance", "f00008"), "--instance takes FRAME=K"),
                       (("--instance", f"{drop}=9"), "instances in"), (("--view0", "f99999"), "view 0 is a pool frame")):
        code, out = term("--resume", J.name, *flags)
        assert code == run.BAD and why in out and "to go on: " in out, flags
    assert stage_times(J) == times

    code, out = term("--resume", J.name)
    assert code == run.DONE and "sam3d: done" in out and "select:" not in out, out


def test_a_failed_selection_stops_for_help_without_review(term, monkeypatch):
    fail_once(monkeypatch, "select", """echo '{"ok": false, "warnings": ["5 of 9 masked views disagree"], "picks": [],
        "candidates": []}' > selection.json; exit 2""")
    code, out = term(term.video(), "--prompt", "box", "--n-views", 3)
    J = term.folder(out)
    job = jobs.load(J)
    assert code == run.WAITS and (job["status"], job["stages"]["select"]["status"]) == ("review", "failed"), out
    assert "warning: 5 of 9 masked views disagree" in out and "the automatic selection needs help" in out
    assert "start a new job with --picks" in out and not any(x.startswith("to go on") for x in out.splitlines())

    # --resume alone cannot go on. It says so again, and the job stays.
    again, same = term("--resume", J.name)
    assert again == run.WAITS and "the automatic selection needs help" in same and jobs.load(J)["status"] == "review"

    pool = json.loads((J / "frames" / "pool.json").read_text())
    code, out = term("--resume", J.name, "--exclude", next(iter(pool["names"].values())))
    assert code == run.WAITS and "to go on: " in out and selection(J)["ok"] is True, out
    code, out = term("--resume", J.name)
    assert code == run.DONE, out


def test_a_started_build_keeps_its_settings(term):
    code, out = term(term.video(), "--prompt", "box", "--review", "--n-views", 3)
    J = term.folder(out)
    times = stage_times(J)
    for flags in (("--prompt", "cup"), ("--n-views", "4"), ("--turntable",), ("--agent", "opus"),
                  ("--review",), ("--picks", "1,2"), ("--no-lidar",), ("--camera-topic", "/a")):
        code, out = term("--resume", J.name, *flags)
        assert code == run.BAD and "so it keeps its settings. Start a new job to use " + flags[0] in out, flags
        assert "to go on: " in out and jobs.load(J)["status"] == "review"
    code, out = term("--resume", J.name, "--calibration", term.video())
    assert code == run.BAD and "uses no LiDAR topic" in out, out
    assert stage_times(J) == times
    job = jobs.load(J)
    assert (job["prompt"], job["n_views"], job["turntable"], job["agent"]) == ("box", 3, False, "none")

    code, out = term("--resume", J.name, "--agent", "none")  # the job's own agent is no change
    assert code == run.DONE, out
    for flags in (("--exclude", "f00008"), ("--view0", "8"), ("--calibration", term.video())):
        code, out = term("--resume", J.name, *flags)
        assert code == run.BAD and f"job {J.name} is done. Start a new job to change it" in out, flags
    assert (J / "results.zip").is_file()


def test_a_bag_with_several_camera_topics_waits_for_the_choice(term):
    side, front = "/side/image/compressed", "/front/image/compressed"
    bag = write_bag(term.tmp / "two.bag", [side, front])
    code, out = term(bag, "--prompt", "red box", "--review", "--n-views", 3)
    J = term.folder(out)
    assert code == run.WAITS and jobs.load(J)["status"] == "choose_topic"
    assert f"camera topic: {side}, 192 x 128, 48 images" in out and f"camera topic: {front}, 192 x 128, 48 images" in out
    args = printed(out, "To choose one: ")
    assert args[-2:] == ["--camera-topic", "TOPIC"] and "--prompt=red box" in args and "--review" in args
    code, out = term.exact([*args[:-1], front])
    job = jobs.load(J)
    assert code == run.WAITS and (job["status"], job["prompt"], job["source"]["topic"]) == ("review", "red box", front), out

    code, out = term(bag, "--camera-topic", "/nope")
    assert code == run.BAD and "not one of the bag's camera topics" in out and "to go on: " in out

    code, out = term(bag, "--camera-topic", side)  # chosen at once: one analysis, on that topic
    job = jobs.load(term.folder(out))
    assert code == run.WAITS and (job["status"], job["source"]["topic"]) == ("ready", side), out


def test_a_bag_with_lidar_and_a_calibration(term):
    from test_webapp_lidar import POINTS, T_CAM_LIDAR

    bag = scene_bag(term.tmp / "no_tf.bag", tf=None)
    code, out = term(bag, "--prompt", "box", "--lidar-topic", POINTS)  # the TF links nothing: the Start route refuses
    assert code == run.BAD and "the bag's TF does not link camera_optical_frame to lidar" in out
    assert "--lidar-topic=" + POINTS in printed(out)

    cal = term.tmp / "calib.yaml"
    cal.write_text(calibration_yaml(T_CAM_LIDAR))
    code, out = term(bag, "--prompt", "box", "--n-views", 3, "--review", "--lidar-topic", POINTS, "--calibration", cal)
    J = term.folder(out)
    job = jobs.load(J)
    assert code == run.WAITS and job["lidar"] == {"kind": "rosbag", "topic": POINTS}, out
    assert (job["calibration"]["name"], job["calibration"]["lidar_topic"]) == ("calib.yaml", POINTS)
    s = json.loads((J / "pool" / "lidar.json").read_text())["s"]
    assert f"LiDAR scale: {s} from " in out and f"LiDAR check images: {J / 'lidar' / 'check'}" in out

    # A new calibration on a resume: the build starts again from the lidar stage and stops here again.
    cal.write_text(calibration_yaml(T_CAM_LIDAR, "time_offset_s: 0\n"))
    before = stage_times(J)
    code, out = term("--resume", J.name, "--calibration", cal)
    after = stage_times(J)
    assert code == run.WAITS and jobs.load(J)["calibration"]["sha256"] != job["calibration"]["sha256"], out
    assert after["lidar"] > before["lidar"] and after["masks"] == before["masks"]

    code, out = term("--resume", J.name, "--calibration", term.tmp / "nope.yaml")
    assert code == run.BAD and "cannot read the calibration" in out and "to go on: " in out


def test_a_stray_capture_keeps_its_depth_unless_it_is_turned_off(term):
    from test_webapp_stray import write_stray

    zipped = write_stray(term.tmp / "src", n=48, portrait=3)
    code, out = term(zipped, "--prompt", "box", "--review", "--n-views", 3)
    job = jobs.load(term.folder(out))
    assert code == run.WAITS and job["lidar"] == {"kind": "stray"} and "LiDAR scale: " in out, out
    code, out = term(zipped, "--prompt", "box", "--review", "--n-views", 3, "--no-lidar")
    job = jobs.load(term.folder(out))
    assert code == run.WAITS and job["lidar"] is None and job["stages"]["lidar"]["status"] == "skipped", out


def test_the_agent_line_is_one_clean_line(tmp_path, capsys):
    """The agent's reason in the job's agent result, an older one too, prints as one line without control
    characters, accents kept."""
    from test_agent_controller import CLEAN, DIRTY

    r = {"verdict": "complete", "reason": DIRTY}
    job = SimpleNamespace(J=tmp_path, call=lambda method, route: {"agent_result": r})
    assert run.Run.finished(job) == run.DONE
    assert capsys.readouterr().out == f"agent: complete: {CLEAN}\nresults: {tmp_path / 'results.zip'}\n"


def test_the_key_stays_out_of_every_output(term):
    key = term.key_file()
    code, out = term(term.video(), "--prompt", "box", "--n-views", 3, "--agent", "opus", "--key-file", key)
    J = term.folder(out)
    job = jobs.load(J)
    assert code == run.DONE and (job["agent"], job["stages"]["agent"]["status"]) == ("opus", "done"), out
    assert "agent: complete: fake agent" in out and OPUS not in out
    assert not [p for p in term.runs.rglob("*") if p.is_file() and OPUS.encode() in p.read_bytes()]
    assert not list((term.tmp / "shm").glob("*"))  # the key file of the stage is gone with its folder


@pytest.mark.parametrize("text, mode, reason", [
    (None, 0o600, "my.env is missing"),
    (f"OPENAI_API_KEY={ASTRA}\n", 0o600, "has no ANTHROPIC_API_KEY= line"),
    (f"ANTHROPIC_API_KEY={ASTRA}\n", 0o600, "Anthropic keys start with sk-ant-"),
    (f"ANTHROPIC_API_KEY={OPUS}\n", 0o644, "mode 0600"),
])
def test_a_bad_key_stops_before_the_job_starts(term, text, mode, reason):
    key = term.key_file(text or "", mode)
    if text is None:
        key.unlink()
    code, out = term(term.video(), "--prompt", "box", "--agent", "opus", "--key-file", key)
    assert code == run.BAD and reason in out and OPUS not in out and ASTRA not in out, out
    assert not job_folders(term)


def test_the_default_key_file_is_read_again_on_each_resume(term, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    conf = tmp_path / "home" / ".config" / "harpor"
    conf.mkdir(parents=True, mode=0o700)
    f = conf / "openai.env"
    f.write_text(f"OPENAI_API_KEY={ASTRA}\n")
    f.chmod(0o600)
    conf.chmod(0o700)
    code, out = term(term.video(), "--prompt", "box", "--n-views", 3, "--agent", "astra", "--review")
    J = term.folder(out)
    assert code == run.WAITS, out

    f.unlink()
    code, out = term("--resume", J.name)
    assert code == run.BAD and "cannot read the OpenAI API key" in out and "openai.env is missing" in out
    assert jobs.load(J)["status"] == "review" and "to go on: " in out

    f.write_text(f"OPENAI_API_KEY={ASTRA}\n")
    f.chmod(0o600)
    code, out = term("--resume", J.name)
    assert code == run.DONE and jobs.load(J)["stages"]["agent"]["status"] == "done", out
    assert ASTRA not in out


def test_the_input_is_copied_when_a_hard_link_cannot_be_made(term, monkeypatch):
    def no_link(src, dst):
        raise OSError(errno.EXDEV, "Cross-device link")

    video = term.video()
    monkeypatch.setattr(os, "link", no_link)
    code, out = term(video)
    J = term.folder(out)
    copy = J / "video" / "cup.mp4"
    assert code == run.WAITS and f"copying {video.resolve()} into the job folder" in out, out
    assert copy.read_bytes() == video.read_bytes() and not os.path.samefile(video, copy)
    assert stat.S_IMODE(copy.stat().st_mode) == 0o600 and [p.name for p in copy.parent.iterdir()] == ["cup.mp4"]

    # A copy that fails, on a full disk, say, leaves no job.
    def full(src, dst):
        Path(dst).write_bytes(b"half")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(shutil, "copyfile", full)
    code, out = term(video)
    assert code == run.FAILED and "cannot put the input into the job folder" in out and "to go on" not in out
    assert job_folders(term) == [J]


def test_a_symlink_keeps_its_own_name(term):
    video = term.video()
    (term.tmp / "shared").mkdir()
    blob = video.rename(term.tmp / "shared" / "abc123")
    link = term.tmp / "cup.mp4"
    link.symlink_to(blob)
    code, out = term(link)
    assert code == run.WAITS and os.path.samefile(blob, term.folder(out) / "video" / "cup.mp4"), out



def test_the_key_folder_is_not_the_servers(term):
    """A Runner deletes every *.env in its key folder. The server's key stays, and a key that a killed run left in
    this command's folder goes with the next run."""
    folder = jobs.default_key_dir()
    folder.mkdir(parents=True, mode=0o700)
    served = folder / "20261101-000000-abcdef.env"
    served.write_text("OPENAI_API_KEY=x\n")
    code, out = term(term.video())
    assert code == run.WAITS and served.is_file()
    mine = run.key_folder(steps.Config(G=term.tmp, runs=term.runs))
    assert mine != folder and not mine.exists()
    mine.mkdir(mode=0o700)
    (mine / "left.env").write_text("OPENAI_API_KEY=x\n")
    code, out = term("--resume", term.folder(out).name)
    assert code == run.WAITS and served.is_file() and not mine.exists()


def test_a_server_on_the_runs_folder_keeps_the_command_out(term):
    video = term.video()
    term.runs.mkdir()
    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "harpor.webapp"])
    try:
        write_json(term.runs / jobs.RUNNER_FILE, {"pid": server.pid, "host": socket.gethostname()})
        for args in ((video,), ("--resume", "20261101-000000-abcdef")):
            handlers = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)
            with pytest.raises(SystemExit, match="another HARPOR server runs the jobs in"):
                term(*args)
            assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)) == handlers
        assert not job_folders(term) and (term.runs / jobs.RUNNER_FILE).is_file()
    finally:
        server.kill()
        server.wait()


def command(video, runs, ignore=()):
    """The command in a process of its own (with the stand-in stages, tests/fake_launch.py), with SIGINT as a terminal
    gives it and the signals of ignore ignored."""
    def signals():
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        for sig in ignore:
            signal.signal(sig, signal.SIG_IGN)

    return subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_launch.py"), "harpor.webapp.run", str(video),
                             "--prompt", "box", "--n-views", "3", "--runs", str(runs)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, preexec_fn=signals)


def read_until(p, start):
    """The lines of p up to the first that starts with start."""
    lines = []
    for line in p.stdout:
        lines.append(line)
        if line.startswith(start):
            return "".join(lines)
    raise AssertionError(f"it ended with {p.wait()}: {''.join(lines)}")


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_a_signal_cancels_the_stage_and_the_job_can_resume(term, sig):
    p = command(term.video(), term.runs)
    try:
        out = read_until(p, "masks: ")
        p.send_signal(sig)
        if sig == signal.SIGINT:  # a second Ctrl-C does not cut the cleanup short
            out += read_until(p, "cancelling the job")
            p.send_signal(sig)
        out += p.communicate(timeout=60)[0]
    finally:
        p.kill()
    J = term.folder(out)
    assert p.returncode == 128 + sig and jobs.load(J)["status"] == "cancelled", out
    assert not (term.runs / jobs.RUNNER_FILE).exists()
    code, out = term.exact(printed(out))
    assert code == run.DONE and (J / "results.zip").is_file(), out


def test_nohup_keeps_the_job_running(term):
    p = command(term.video(), term.runs, ignore=[signal.SIGHUP])
    try:
        out = read_until(p, "masks: ")
        p.send_signal(signal.SIGHUP)
        out += p.communicate(timeout=60)[0]
    finally:
        p.kill()
    assert p.returncode == run.DONE and "results: " in out, out


def test_a_running_command_holds_its_runs_folder(term, fake_stages):
    """The Runner of a second process sees this command as a live server, by "harpor.webapp" in its command line."""
    p = command(term.video(), term.runs)
    try:
        read_until(p, "job ")
        with pytest.raises(SystemExit, match="another HARPOR server runs the jobs in"):
            jobs.Runner(steps.Config(G=term.tmp, runs=term.runs), key_dir=term.tmp / "keys")
    finally:
        p.terminate()
        p.communicate(timeout=60)


def test_a_closed_terminal_stops_no_cleanup(monkeypatch):
    def closed(*args, **kw):
        raise BrokenPipeError

    monkeypatch.setattr("builtins.print", closed)
    run.say("the job failed")
