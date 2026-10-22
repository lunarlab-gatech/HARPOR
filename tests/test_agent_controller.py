"""Scripted agent sessions through the controller and its tool server (build_plan.md note 5).

The scripted sessions run in parallel in one fixture, because each fake Blender export imports trimesh.
"""

import contextlib
import json
import os
import re
import shlex
import signal
import socket
import stat
import sys
import tempfile
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import trimesh

from gator import frames as F
from gator.agent import controller as C
from gator.agent import prompts as PR
from gator.agent import tools as T
from gator.agent.bin import gator_tool
from gator.agent.fake_agent import write_glb
from gator.agent.server import ToolServer
from gator.parts import joints as J

REPO = Path(__file__).resolve().parents[1]
FAST = {"poll_s": 0.02, "kill_after_s": 0.5, "finish_wait_s": 0.5}
FAKE_BLENDER = [sys.executable, "-m", "gator.agent.fake_agent", "--blender"]
LONG = 20.0
ORIGINAL_SIZE = 3.0
PROMPT = "Prompt 1 text, byte for byte: é\n"


def edit(size, kind="good", seconds=0.0):
    return {"do": "edit", "size": size, "kind": kind, "seconds": seconds}


def tool(*args):
    return {"do": "tool", "args": list(args)}


def finish(verdict, *limitations):
    lims = [x for lim in limitations for x in ("--limitation", lim)]
    return tool("finish", "--receipt", "{receipt}", "--verdict", verdict, "--reason", "fixed the handle", *lims)


def checkpoint(defect):
    return tool("checkpoint", "--receipt", "{receipt}", "--status", "needs_review", "--defect", defect)


REVIEW, EXPORT = tool("review"), tool("export_glb")
CRASH, SLEEP = {"do": "crash"}, {"do": "sleep", "seconds": 30}

# The fake parts GLB is a cube of edge PARTS_SIZE about W's origin (y down): body below, lid on its top
# quarter. HINGE is the lid's back edge, turning up; INSIDE turns the lid about the body's center.
PARTS_SIZE = 1.0
PARTS = {"version": 1, "frame": "W", "glb": "parts.glb", "base": "body", "seams": [],
         "parts": [{"name": "body", "parent": None, "joint_guess": "fixed", "status": "found"},
                   {"name": "lid", "parent": "body", "joint_guess": "revolute", "status": "found"}]}
HINGE = {"origin": [0.0, -0.25, 0.5], "axis": [-1.0, 0.0, 0.0], "lower": 0.0, "upper": 1.5}
INSIDE = {"origin": [0.0, 0.125, 0.0], "axis": [1.0, 0.0, 0.0], "lower": 0.0, "upper": 1.5}


def lid_joint(hinge, kind="revolute"):
    return {"version": 1, "frame": "W", "parent": "body", "type": kind, **hinge}


def set_joint(child, parent, kind, hinge=None):
    args = ["set_joint", "--child", child, "--parent", parent, "--type", kind]
    for k in ("origin", "axis", "lower", "upper") if hinge else ():
        args += [f"--{k}", *map(str, np.ravel(hinge[k]))]
    return tool(*args)


def articulated(verdict):
    return tool("finish", "--receipt", "{areceipt}", "--verdict", verdict, "--reason", "lid hinged at the back")


SESSIONS = {
    "finish_refine": ("refinement", LONG, [{"do": "read_prompt"}, edit(1.2), REVIEW,
                                           finish("complete", "seam on the back", "thin lid")]),
    "finish_unchanged": ("refinement", LONG, [edit(ORIGINAL_SIZE), REVIEW, finish("unchanged")]),
    "finish_scratch": ("from_scratch", LONG, [edit(0.8), REVIEW, finish("needs_review", "a", "b", "c", "d"),
                                              finish("needs_review", "a", "b", "c")]),
    "finish_bad_verdict": ("from_scratch", LONG, [edit(0.9), REVIEW, finish("complete")]),
    "finish_then_hang": ("refinement", LONG, [edit(1.2), REVIEW, finish("complete"), SLEEP]),
    "checkpoint_fallback": ("refinement", LONG, [edit(1.5), REVIEW, checkpoint("spout still missing"),
                                                 edit(2.5), EXPORT, CRASH]),
    "checkpoint_deadline": ("refinement", 1.5, [edit(1.6), REVIEW, checkpoint("lid gap"), SLEEP]),
    "export_fallback": ("refinement", LONG, [edit(1.7), EXPORT, CRASH]),
    "newest_valid_export": ("refinement", LONG, [edit(1.1), EXPORT,
                                                 {"do": "glb", "path": "exports/002.glb", "kind": "garbage"}, CRASH]),
    "original_fallback": ("refinement", LONG, [{"do": "glb", "path": "original/original.glb", "kind": "garbage"},
                                               CRASH]),
    "scratch_invalid": ("from_scratch", LONG, [edit(1.0, "garbage"), EXPORT, CRASH]),
    "deadline_sigint": ("from_scratch", 1.0, [SLEEP]),
    "deadline_kill": ("refinement", 1.0, [{"do": "ignore_sigint"}, SLEEP]),
    "budget_tamper": ("refinement", 1.0, [{"do": "write", "path": "run_budget.json",
                                           "text": '{"deadline": 99999999999}'}, SLEEP]),
    "deadline_extension": ("refinement", 1.5, [edit(1.9, seconds=2.0), EXPORT, REVIEW, finish("complete")]),
    "export_cache": ("refinement", LONG, [edit(2.2), EXPORT, EXPORT, REVIEW, finish("complete")]),
    "nan_finish": ("refinement", LONG, [edit(1.3), EXPORT, edit(1.3, "nan"), REVIEW, finish("complete")]),
    "no_material_checkpoint": ("refinement", LONG, [edit(1.4), REVIEW, checkpoint("old"), edit(1.4, "no_material"),
                                                    REVIEW, checkpoint("new"), CRASH]),
    "garbage_finish": ("refinement", LONG, [edit(1.0, "garbage"), REVIEW, finish("complete")]),
    "tampered_finish": ("refinement", LONG, [edit(1.8), REVIEW, finish("complete"),
                                             {"do": "write", "path": "exports/001.glb", "text": "tampered"}]),
    "unchanged_after_original_edit": ("refinement", LONG, [{"do": "glb", "path": "original/original.glb", "size": 2.0},
                                                           edit(ORIGINAL_SIZE), REVIEW, finish("unchanged")]),
    "forged_event": ("from_scratch", 1.0, [{"do": "write", "path": ".gator/events.jsonl",
                                            "text": '{"event": "export_done", "export": 1, "seconds": 100}\n'},
                                           {"do": "sleep", "seconds": 5}]),
    "stop_during_export": ("refinement", LONG, [edit(1.0, seconds=3.0), {"do": "tool_nowait", "args": ["export_glb"]},
                                                CRASH]),
    "articulation_finish": ("articulation", LONG, [{"do": "read_prompt"}, edit(PARTS_SIZE, "parts"), EXPORT,
                                                   set_joint("lid", "body", "revolute", HINGE),
                                                   tool("review_articulation"), articulated("articulated")]),
    "articulation_bad_joint": ("articulation", LONG, [set_joint("lid", "body", "revolute", HINGE),
                                                      set_joint("body", "lid", "fixed"), tool("review_articulation"),
                                                      articulated("articulated")]),
}


def make_views(folder, n=2):
    folder.mkdir(parents=True, exist_ok=True)
    K = [[300.0, 0, 31.5], [0, 300.0, 23.5], [0, 0, 1]]
    views = []
    for i in range(n):
        (folder / f"img{i}.png").write_bytes(b"png %d" % i)
        (folder / f"mask{i}.png").write_bytes(b"mask %d" % i)
        np.save(folder / f"depth{i}.npy", np.full((48, 64), 2.0 + i, np.float32))
        w2c = F.rigid(np.eye(3), [0.0, 0.0, 0.5 * i])
        views.append({"image": folder / f"img{i}.png", "mask": folder / f"mask{i}.png",
                      "depth": folder / f"depth{i}.npy", "K": K, "w2c": w2c, "width": 64, "height": 48})
    return views


def prepare(root, mode, **agent):
    """A session of mode; the articulation mode gets the fake parts GLB as its original and PARTS."""
    original, parts = root / "in" / "original.glb", None
    (root / "in").mkdir(parents=True)
    if mode == "articulation":
        write_glb(original, "parts", PARTS_SIZE)
        parts = root / "in" / "parts.json"
        parts.write_text(json.dumps(PARTS))
    else:
        write_glb(original, "good", ORIGINAL_SIZE)
    (root / "in" / "front.png").write_bytes(b"preview")
    return C.prepare_session(root / "s", mode, PROMPT, make_views(root / "in"), source="sensor depth", metric=True,
                             original=original if mode != "from_scratch" else None, previews=[root / "in" / "front.png"],
                             parts=parts, **agent)


def run_scripted(root, mode, budget_s, script):
    prepare(root, mode)
    (root / "script.json").write_text(json.dumps(script))
    with tempfile.TemporaryDirectory(prefix="gator-") as sock_dir:
        env = {**os.environ, "PYTHONPATH": str(REPO), "GATOR_SOCKET": os.path.join(sock_dir, "tools.sock")}
        cmd = [sys.executable, "-m", "gator.agent.fake_agent", str(root / "script.json")]
        return C.run_session(root / "s", cmd, env, blender=FAKE_BLENDER, sock_dir=sock_dir, budget_s=budget_s, **FAST)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    roots = {name: tmp_path_factory.mktemp(name) for name in SESSIONS}
    with ThreadPoolExecutor(len(SESSIONS)) as ex:
        results = list(ex.map(lambda name: run_scripted(roots[name], *SESSIONS[name]), SESSIONS))
    return {name: (roots[name] / "s", r) for name, r in zip(SESSIONS, results)}


def edge(path):
    """Edge length of the delivered fake box."""
    ext = trimesh.load(path, file_type="glb", force="scene", process=False).extents
    assert np.allclose(ext, ext[0])
    return round(float(ext[0]), 6)


def agent_log(root):
    return (root / "session" / "codex.jsonl").read_text()


def test_finish_refinement(runs):
    root, r = runs["finish_refine"]
    assert r["candidate_built"] is True and runs["finish_scratch"][1]["candidate_built"] is None
    assert r["previews_built"] is None  # previews were supplied
    assert (r["source"], r["verdict"], r["exit"]) == ("finish", "complete", "exited")
    assert r["reason"] == "fixed the handle" and r["limitations"] == ["seam on the back", "thin lid"]
    assert r["glb"] == "work/exports/001.glb" and edge(root / r["glb"]) == 1.2
    assert r["sha256"] == T.sha256(root / r["glb"]) and r["rejected"] == []
    assert (root / "work" / "prompt_seen.txt").read_text(encoding="utf-8") == PROMPT
    blocks = agent_log(root).split("$ ")[1:]
    assert len(blocks) == 2 and all(len(b.splitlines()) <= 20 for b in blocks)
    assert "center shift 0.000" in blocks[0] and "view00 1.00, view01 1.00" in blocks[0]
    assert json.loads((root / "result.json").read_text()) == r


def test_finish_unchanged_delivers_original_bytes(runs):
    root, r = runs["finish_unchanged"]
    assert (r["source"], r["verdict"]) == ("finish", "unchanged")
    assert r["glb"] == "original.glb" and r["sha256"] == T.sha256(root.parent / "in" / "original.glb")


def test_unchanged_ignores_agent_edits_to_work_copy(runs):
    root, r = runs["unchanged_after_original_edit"]
    assert (r["source"], r["verdict"], r["glb"]) == ("finish", "unchanged", "original.glb")
    assert edge(root / r["glb"]) == ORIGINAL_SIZE and r["sha256"] == T.sha256(root.parent / "in" / "original.glb")


def test_events_written_in_work_do_not_move_the_deadline(runs):
    root, r = runs["forged_event"]
    assert r["exit"] == "deadline_sigint" and r["timings"]["wall_s"] < 2.5
    assert r["timings"]["excluded_s"] == 0.0 and r["timings"]["deadline_s"] == 1.0
    assert T.read_events(root / "control") == []


def test_finish_from_scratch_rejects_fourth_limitation(runs):
    root, r = runs["finish_scratch"]
    assert (r["source"], r["verdict"], r["limitations"]) == ("finish", "needs_review", ["a", "b", "c"])
    assert "4 limitations given; at most 3" in agent_log(root)
    assert not (root / "original.glb").exists() and not (root / "work" / "original").exists()


def test_refinement_verdict_rejected_in_from_scratch(runs):
    root, r = runs["finish_bad_verdict"]
    assert "unknown verdict 'complete' for from_scratch" in agent_log(root)
    assert not (root / "work" / "finish.json").exists()
    assert (r["source"], r["verdict"]) == ("export", "needs_review") and edge(root / r["glb"]) == 0.9


def test_controller_ends_session_after_finish(runs):
    root, r = runs["finish_then_hang"]
    assert (r["source"], r["exit"], r["returncode"]) == ("finish", "finish_sigint", -signal.SIGINT)
    assert r["timings"]["wall_s"] < 6.0


def test_checkpoint_beats_newer_export(runs):
    root, r = runs["checkpoint_fallback"]
    assert (r["source"], r["verdict"], r["reason"]) == ("checkpoint", "needs_review", "spout still missing")
    assert edge(root / r["glb"]) == 1.5 and r["counts"]["exports"] == 2
    assert (root / r["glb"]).stat().st_mode & 0o222 == 0
    assert r["exit"] == "exited" and r["returncode"] == 3


def test_checkpoint_kept_at_deadline(runs):
    root, r = runs["checkpoint_deadline"]
    assert (r["exit"], r["source"], r["reason"]) == ("deadline_sigint", "checkpoint", "lid gap")
    assert edge(root / r["glb"]) == 1.6


def test_export_fallback(runs):
    root, r = runs["export_fallback"]
    assert (r["source"], r["verdict"]) == ("export", "needs_review") and "uninspected" in r["reason"]
    assert edge(root / r["glb"]) == 1.7


def test_glb_written_without_export_glb_is_never_delivered(runs):
    root, r = runs["newest_valid_export"]
    assert r["glb"] == "work/exports/001.glb" and edge(root / r["glb"]) == 1.1
    assert r["rejected"] == [] and (root / "work" / "exports" / "002.glb").exists()


def test_original_fallback_uses_controller_copy(runs):
    root, r = runs["original_fallback"]
    assert (r["source"], r["verdict"], r["glb"]) == ("fallback_original", "fallback_original", "original.glb")
    assert edge(root / r["glb"]) == ORIGINAL_SIZE


def test_from_scratch_without_valid_output_is_invalid(runs):
    root, r = runs["scratch_invalid"]
    assert (r["source"], r["verdict"], r["glb"], r["sha256"]) == (None, "invalid", None, None)
    assert len(r["rejected"]) == 1


def test_deadline_sigint(runs):
    root, r = runs["deadline_sigint"]
    assert (r["exit"], r["returncode"], r["verdict"]) == ("deadline_sigint", -signal.SIGINT, "invalid")
    assert 1.0 <= r["timings"]["wall_s"] < 2.5 and r["timings"]["excluded_s"] == 0.0


def test_deadline_kill_after_grace(runs):
    root, r = runs["deadline_kill"]
    assert (r["exit"], r["returncode"], r["source"]) == ("deadline_killed", -signal.SIGKILL, "fallback_original")
    assert 1.5 <= r["timings"]["wall_s"] < 3.0


def test_agent_edit_of_budget_file_is_ignored(runs):
    root, r = runs["budget_tamper"]
    assert r["exit"] == "deadline_sigint" and r["timings"]["wall_s"] < 2.5


def test_deadline_extended_by_export_time(runs):
    root, r = runs["deadline_extension"]
    assert (r["exit"], r["source"], r["verdict"]) == ("exited", "finish", "complete")
    t = r["timings"]
    assert t["wall_s"] > t["budget_s"] and t["excluded_s"] >= 2.0
    assert t["deadline_s"] == pytest.approx(t["budget_s"] + t["excluded_s"])
    done = [e for e in T.read_events(root / "control") if e["event"] == "export_done"]
    budget = json.loads((root / "work" / "run_budget.json").read_text())
    assert len(done) == 1 and budget["excluded_s"] == done[0]["seconds"] == t["excluded_s"]
    assert budget["deadline"] == pytest.approx(budget["started_at"] + t["budget_s"] + t["excluded_s"])
    assert budget["exports"] == [{"export": 1, "seconds": done[0]["seconds"]}]


def test_unchanged_scene_returns_cached_export(runs):
    root, r = runs["export_cache"]
    assert r["source"] == "finish" and r["counts"]["exports"] == 1
    assert sum(e["event"] == "export_done" for e in T.read_events(root / "control")) == 1
    assert agent_log(root).count("(cached, scene unchanged)") == 1


def test_helpers_print_time_left(runs):
    # export_cache: export, cached export, review, finish. checkpoint_fallback: review, checkpoint, export.
    # deadline_extension: a 2 s export, review, finish.
    for name, n in (("export_cache", 3), ("checkpoint_fallback", 3), ("deadline_extension", 2)):
        root, r = runs[name]
        t = r["timings"]
        lines = [x for x in agent_log(root).splitlines() if x.startswith("time left: ")]
        assert len(lines) == n and all(x.endswith(f" s of the {t['budget_s']:.0f} s budget") for x in lines)
        # Outside an export the controller's time left only drops, and an export does not use any of it, so
        # each printed value lies between what was left when the session ended and the budget.
        assert all(t["deadline_s"] - t["wall_s"] - 0.5 <= float(x.split()[2]) <= t["budget_s"] + 0.5 for x in lines)


def test_failed_helper_also_prints_time_left(tmp_path, monkeypatch):
    work = prepare(tmp_path, "refinement")
    control = work.parent / "control"
    T.start_budget(work, control, time.time() - 100.0, 600.0)
    monkeypatch.setenv("GATOR_WORK", str(work))
    monkeypatch.setenv("GATOR_CONTROL", str(control))
    with pytest.raises(SystemExit) as e:
        T.main(["checkpoint", "--receipt", "r001-0123456789ab", "--defect", "lid gap"])
    first, last = e.value.code.split("\n")
    assert first == "unknown review receipt 'r001-0123456789ab'"
    assert last in ("time left: 500 s of the 600 s budget", "time left: 499 s of the 600 s budget")
    with pytest.raises(SystemExit) as e:  # finish and argparse errors stay as they were
        T.main(["finish", "--receipt", "r001-0123456789ab", "--verdict", "complete", "--reason", "x"])
    assert e.value.code == "unknown review receipt 'r001-0123456789ab'"


def test_finish_with_nonfinite_vertices_rejected(runs):
    root, r = runs["nan_finish"]
    assert (r["source"], r["glb"]) == ("export", "work/exports/001.glb") and edge(root / r["glb"]) == 1.3
    assert [(x["from"], x["why"]) for x in r["rejected"]] == [("finish", "non-finite vertices"),
                                                              ("export", "non-finite vertices")]


def test_checkpoint_without_materials_rejected(runs):
    root, r = runs["no_material_checkpoint"]
    assert (r["source"], r["reason"]) == ("checkpoint", "old") and edge(root / r["glb"]) == 1.4
    assert r["rejected"][0]["from"] == "checkpoint" and r["rejected"][0]["why"].startswith("no materials")


def test_unparsable_finish_falls_back_to_original(runs):
    root, r = runs["garbage_finish"]
    assert r["source"] == "fallback_original"
    assert [x["why"].split(":")[0] for x in r["rejected"]] == ["unparsable", "unparsable"]


def test_finish_glb_changed_after_finish_rejected(runs):
    root, r = runs["tampered_finish"]
    assert r["rejected"] == [{"from": "finish", "glb": "work/exports/001.glb", "why": "changed after finish"},
                             {"from": "export", "glb": "work/exports/001.glb", "why": "changed after export"}]
    assert r["source"] == "fallback_original"


def test_budget_file_is_replaced_atomically(tmp_path):
    T.write_budget(tmp_path, 1000.0, 600.0)
    inode = (tmp_path / "run_budget.json").stat().st_ino
    bad, stop = [], threading.Event()

    def reader():
        while not stop.is_set():
            try:
                json.loads((tmp_path / "run_budget.json").read_text())
            except ValueError:
                bad.append(1)

    th = threading.Thread(target=reader)
    th.start()
    for i in range(300):
        T.write_budget(tmp_path, 1000.0, 600.0, float(i), [{"export": j, "seconds": 1.0} for j in range(50)])
    stop.set()
    th.join()
    b = json.loads((tmp_path / "run_budget.json").read_text())
    assert bad == [] and b["deadline"] == 1000.0 + 600.0 + 299.0 and b["excluded_s"] == 299.0
    assert (tmp_path / "run_budget.json").stat().st_ino != inode
    assert [p.name for p in tmp_path.iterdir()] == ["run_budget.json"]


def test_deadline_counts_capped_exports_and_pauses_during_one():
    ev = [{"event": "export_started", "export": 1, "time": 10.0},
          {"event": "export_done", "export": 1, "time": 12.0, "seconds": 2.0},
          {"event": "export_started", "export": 2, "time": 20.0},
          {"event": "export_done", "export": 2, "time": 520.0, "seconds": 500.0},
          {"event": "export_started", "export": 3, "time": 600.0}]
    assert C.deadline(0.0, 600.0, ev, now=610.0) == (600.0 + 192.0, 192.0)
    assert C.deadline(0.0, 600.0, ev, now=2000.0) == (600.0 + 362.0, 362.0)
    assert C.deadline(5.0, 1.0, [], now=0.0) == (6.0, 0.0)


def test_check_glb_known_answers(tmp_path):
    for kind, why in [("good", None), ("nan", "non-finite vertices"), ("no_material", "no materials"),
                      ("garbage", "unparsable")]:
        write_glb(tmp_path / f"{kind}.glb", kind)
        got = C.check_glb(tmp_path / f"{kind}.glb")
        assert got == why if why is None else got.startswith(why)
    assert C.check_glb(tmp_path / "missing.glb").startswith("unparsable")


def test_manifest_and_layout(tmp_path):
    work = prepare(tmp_path, "refinement")
    root = work.parent
    assert (root / "session" / "codex_home" / "config.toml").exists()
    assert "finish --receipt ID" in (work / "AGENTS.md").read_text()
    assert (root / "control" / "events.jsonl").exists() and not (work / ".gator").exists()
    assert json.loads((root / "control" / "manifest.json").read_text()) == json.loads((work / "input" / "manifest.json").read_text())
    m = json.loads((work / "input" / "manifest.json").read_text())
    assert m["mode"] == "refinement" and m["units"] == "meters" and m["depth_scale"] == 1.0
    assert np.allclose(m["w_from_gltf"], F.W_FROM_GLTF)
    v = m["views"][1]
    assert v["image"] == "input/images/001.png" and (work / v["image"]).read_bytes() == b"png 1"
    assert np.load(work / v["depth"])[0, 0] == 3.0 and "confidence" not in v
    assert np.allclose(v["w2c"], F.rigid(np.eye(3), [0, 0, 0.5])) and v["K"][0][2] == 31.5
    assert (work / "original" / "previews" / "front.png").exists()
    assert T.sha256(work / "original" / "original.glb") == T.sha256(work.parent / "original.glb")
    assert (work.parent / "session" / "prompt.txt").read_bytes() == PROMPT.encode("utf-8")


def test_deadline_ignores_malformed_events():
    bad = [{"event": "export_done", "export": 1, "seconds": float("nan")},
           {"event": "export_done", "export": 2, "seconds": "180"},
           {"event": "export_done", "export": 3, "seconds": -500.0},
           {"event": "export_started", "export": [4], "time": "soon"}]
    assert C.deadline(0.0, 600.0, bad, now=10.0) == (600.0, 0.0)
    assert C.deadline(0.0, 600.0, bad + [{"event": "export_done", "export": 5, "seconds": 7}], now=10.0) == (607.0, 7.0)


def test_read_events_skips_undecodable_lines(tmp_path):
    (tmp_path / "events.jsonl").write_bytes(b'\xff\xfe\n{"event": "review"}\n[1]\n')
    assert T.read_events(tmp_path) == [{"event": "review"}]


def test_watchdog_error_still_kills_agent(tmp_path, monkeypatch):
    work = prepare(tmp_path, "from_scratch")
    pid_file = tmp_path / "pid"
    cmd = [sys.executable, "-c", f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)"]

    def broken(*args):
        if pid_file.exists() and pid_file.read_text():
            raise RuntimeError("watchdog bug")
        return 1e12, 0.0

    monkeypatch.setattr(C, "deadline", broken)
    with pytest.raises(RuntimeError):
        C.run_agent(work.parent, cmd, poll_s=0.02)
    pid = int(pid_file.read_text())
    try:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)


def glb_json(src, dst, edit):
    """Copy a GLB with its JSON chunk replaced by edit(json)."""
    data = Path(src).read_bytes()
    n = int.from_bytes(data[12:16], "little")
    js = json.dumps(edit(json.loads(data[20:20 + n]))).encode()
    js += b" " * (-len(js) % 4)
    body = len(js).to_bytes(4, "little") + b"JSON" + js + data[20 + n:]
    Path(dst).write_bytes(b"glTF" + (2).to_bytes(4, "little") + (12 + len(body)).to_bytes(4, "little") + body)


def test_check_glb_malformed_json_and_posed_vertices(tmp_path):
    write_glb(tmp_path / "good.glb")
    for i, edit in enumerate([lambda g: [], lambda g: "x", lambda g: {**g, "meshes": 5},
                              lambda g: {**g, "meshes": [{"primitives": [1]}]}, lambda g: {**g, "materials": 3}]):
        glb_json(tmp_path / "good.glb", tmp_path / f"bad{i}.glb", edit)
        assert C.check_glb(tmp_path / f"bad{i}.glb").startswith("unparsable")

    def move(key, value):
        def edit(g):
            g["nodes"][0][key] = value
            return g
        return edit

    glb_json(tmp_path / "good.glb", tmp_path / "moved.glb", move("translation", [1.0, 2.0, 3.0]))
    assert C.check_glb(tmp_path / "moved.glb") is None
    for key, value in [("translation", [float("nan"), 0.0, 0.0]), ("scale", [float("inf"), 1.0, 1.0])]:
        glb_json(tmp_path / "good.glb", tmp_path / f"{key}.glb", move(key, value))
        assert C.check_glb(tmp_path / f"{key}.glb") == "non-finite vertices"


def test_atomic_json_has_plain_file_permissions(tmp_path):
    T.write_json_atomic(tmp_path / "a.json", {"x": 1})
    (tmp_path / "plain.json").write_text("{}")
    assert stat.S_IMODE((tmp_path / "a.json").stat().st_mode) == stat.S_IMODE((tmp_path / "plain.json").stat().st_mode)


def test_export_launch_failure_is_logged_and_not_excluded(tmp_path):
    work = prepare(tmp_path, "refinement")
    control = work.parent / "control"
    (work / "candidate.blend").write_text("{}")
    T.start_budget(work, control, 0.0, 600.0)
    with pytest.raises(SystemExit, match="export 001 failed"):
        T.export_glb(work, control, [str(tmp_path / "no-blender")])
    ev = T.read_events(control)
    assert [(e["event"], e.get("ok")) for e in ev] == [("export_started", None), ("export_done", False)]
    assert "no-blender" in (work / "exports" / "001.log").read_text()
    assert C.deadline(0.0, 600.0, ev, now=1e9)[1] < 1.0


def test_files_the_agent_writes_are_not_trusted(tmp_path):
    work = prepare(tmp_path, "refinement")
    root, control, ex = work.parent, work.parent / "control", work / "exports"
    write_glb(ex / "001.glb", size=1.25)
    T.log_event(control, "export_done", export=1, seconds=1.0, ok=True, glb="exports/001.glb",
                blend_sha256="b", glb_sha256=T.sha256(ex / "001.glb"))
    write_glb(ex / "009.glb", size=2.0)  # newer, but not made by export_glb
    (work / "finish.json").write_text(json.dumps({"glb": "exports/009.glb", "sha256": T.sha256(ex / "009.glb"),
                                                  "verdict": "complete", "reason": "r"}))
    os.symlink("/work/exports/001.glb", ex / "latest.glb")  # dangles on the host
    os.mkfifo(ex / "fifo.glb")
    (work / "checkpoints" / "001-x.json").write_text("not json")
    pick, rejected = C.select_deliverable(root, "refinement")
    assert (pick["source"], pick["glb"], rejected) == ("export", work / "exports" / "001.glb", [])
    T.log_event(control, "export_done", export=2, seconds=1.0, ok=True, glb="exports/fifo.glb",
                blend_sha256="b", glb_sha256="0" * 64)
    pick, rejected = C.select_deliverable(root, "refinement")
    assert pick["glb"] == ex / "001.glb" and rejected[0]["why"] == "missing or not a regular file after export"


def test_export_timeout_counts_exactly_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    work = prepare(tmp_path, "refinement")
    control = work.parent / "control"
    (work / "candidate.blend").write_text(json.dumps({"kind": "good", "size": 1.0, "seconds": 5.0}))
    T.start_budget(work, control, 0.0, 600.0)
    with pytest.raises(SystemExit, match="export 001 failed after 0.3 s"):
        T.export_glb(work, control, FAKE_BLENDER, timeout=0.3)
    done = [e for e in T.read_events(control) if e["event"] == "export_done"]
    assert [(e["ok"], e["seconds"]) for e in done] == [(False, 0.3)]
    assert json.loads((work / "run_budget.json").read_text())["deadline"] == 600.3
    assert "timed out" in (work / "exports" / "001.log").read_text()


def test_checkpoint_and_finish_require_an_intact_receipt(tmp_path):
    work = prepare(tmp_path, "refinement")
    control = work.parent / "control"
    write_glb(work / "exports" / "001.glb", size=1.25)
    rid = "r001-0123456789ab"
    rec = {"receipt": rid, "glb": "exports/001.glb", "sha256": T.sha256(work / "exports" / "001.glb")}
    (work / "reviews" / "r002-0123456789ab").mkdir()
    (work / "reviews" / "r002-0123456789ab" / "receipt.json").write_text(json.dumps(rec))  # by hand: no event
    with pytest.raises(SystemExit, match="unknown review receipt 'r002-0123456789ab'"):
        T.finish(work, control, "r002-0123456789ab", "complete", "done")
    T.log_event(control, "review", **rec)
    os.symlink(control / "events.jsonl", work / "checkpoints" / f"001-{rid}.glb")
    before = (control / "events.jsonl").read_text()
    with pytest.raises(SystemExit, match="already exists"):
        T.checkpoint(work, control, rid, "lid gap")
    assert (control / "events.jsonl").read_text() == before
    os.unlink(work / "checkpoints" / f"001-{rid}.glb")
    meta = T.checkpoint(work, control, rid, "lid gap")
    assert meta["glb"] == f"checkpoints/001-{rid}.glb" and meta["sha256"] == rec["sha256"]
    assert T.sha256(work / meta["glb"]) == rec["sha256"]
    write_glb(work / "exports" / "001.glb", size=2.5)
    with pytest.raises(SystemExit, match=f"changed after review {rid}"):
        T.checkpoint(work, control, rid, "lid gap")
    with pytest.raises(SystemExit, match=f"changed after review {rid}"):
        T.finish(work, control, rid, "complete", "done")
    assert not (work / "finish.json").exists()


def test_session_end_stops_a_running_helper(runs):
    root, r = runs["stop_during_export"]
    assert (r["exit"], r["returncode"], r["source"]) == ("exited", 3, "fallback_original")
    assert r["timings"]["wall_s"] < 2.5
    time.sleep(3.0)  # the fake export would have written its GLB by now
    assert not (root / "work" / "exports" / "001.glb").exists()
    assert [(e["event"], e.get("ok")) for e in T.read_events(root / "control")] == \
        [("export_started", None), ("export_done", False)]


def test_apptainer_command_keeps_control_out(tmp_path):
    root = tmp_path / "s"
    cmd = C.apptainer_command(root, "/sif/agent.sif", "/home/u/key.env", "/tmp/gator-x")
    binds = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--bind"]
    assert binds == [f"{root.resolve()}/work:/work", f"{root.resolve()}/session:/session",
                     f"{C.AGENT_DIR}/bin:/opt/gator/bin:ro", "/tmp/gator-x:/opt/gator/sock:ro"]
    assert cmd[:8] == C.CONTAINER and "--containall" in cmd and "control" not in " ".join(binds)
    envs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--env"]
    assert envs == ["CODEX_HOME=/session/codex_home", "GATOR_SOCKET=/opt/gator/sock/tools.sock",
                    "PREPEND_PATH=/opt/gator/bin"]
    assert cmd[cmd.index("--env-file") + 1] == "/home/u/key.env"
    assert cmd[-len(C.codex_command()) - 7:] == ["--pwd", "/work", "/sif/agent.sif", *C.CODEX_KEY_WRAPPER,
                                                 *C.codex_command()]
    assert C.apptainer_command(root, "a.sif", "k", "/tmp/x", agent=["sh", "-c", "true"])[-3:] == ["sh", "-c", "true"]
    assert "RUST_LOG=info" in C.apptainer_command(root, "a.sif", "k", "/tmp/x", extra_env={"RUST_LOG": "info"})
    with pytest.raises(ValueError, match="cannot bind"):
        C.apptainer_command(tmp_path / "a,b", "a.sif", "k", "/tmp/x")
    blender = C.blender_command(root, "/sif/agent.sif")
    work = (root / "work").resolve()
    assert blender[-2:] == ["/sif/agent.sif", "blender"] and f"{work}:{work}" in blender and f"{work}:/work" in blender


def test_note_5_command_is_the_controllers(tmp_path):
    text = (Path(__file__).resolve().parents[1] / "docs" / "build_plan.md").read_text()
    block = text.split("### 5. Stage B", 1)[1].split("```bash\n", 1)[1].split("```", 1)[0]
    root, sock, key, sif = tmp_path / "s", "/tmp/gator-x", "/home/u/.config/gator/openai.env", "/sif/gator-agent.sif"
    doc = (block.replace("\\\n", " ").replace("<root>", str(root.resolve())).replace("<socket dir>", sock)
           .replace("~/.config/gator/openai.env", key).replace(" gator/agent/bin:", f" {C.AGENT_DIR}/bin:")
           .replace(" gator-agent.sif", f" {sif}"))
    assert shlex.split(doc) == C.apptainer_command(root, sif, key, sock)


def test_cli_exit_names():
    assert [C.cli_exit(c) for c in (0, 1, 130, 143, 41, -signal.SIGINT, None)] == \
        ["success", "error_or_interrupted", "cancelled", "terminated", "exit 41", "signal SIGINT", None]


def test_tool_server_rejects_bad_requests(tmp_path):
    work = prepare(tmp_path, "refinement")
    with tempfile.TemporaryDirectory(prefix="gator-") as d:
        path = os.path.join(d, "tools.sock")
        server = ToolServer(path, work, work.parent / "control", FAKE_BLENDER).start()
        try:
            assert gator_tool.call(["rm", "-rf", "/"], path) == ("unknown command 'rm'; use one of export_glb, "
                                                                 "review, checkpoint, finish, set_joint, "
                                                                 "review_articulation\n", 2)
            assert gator_tool.call([], path)[1] == 2
            with socket.socket(socket.AF_UNIX) as s:
                s.connect(path)
                s.sendall(b"not json\n")
                assert json.loads(b"".join(iter(lambda: s.recv(4096), b"")))["code"] == 2
            output, code = gator_tool.call(["finish", "--receipt", "r9", "--verdict", "complete", "--reason", "x"], path)
            assert code == 1 and "unknown review receipt 'r9'" in output
            assert gator_tool.call(["review", "--bogus"], path)[1] == 2
        finally:
            server.stop()
        assert not os.path.exists(path)


def test_blend_path_must_stay_in_work(tmp_path):
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    os.symlink("/etc", work / "sub" / "etc")
    assert T.in_work(work, "/work/sub/a.blend") == (work / "sub" / "a.blend").resolve()
    assert T.in_work(work, "a.blend") == (work / "a.blend").resolve()
    for bad in ("/etc/passwd", "/work/../x.blend", "../x.blend", "sub/etc/passwd"):
        with pytest.raises(SystemExit, match="outside /work"):
            T.in_work(work, bad)


def test_failed_exports_cannot_stack_and_exclusion_is_capped():
    started = [{"event": "export_started", "export": k, "time": float(k)} for k in (1, 2, 3)]
    assert C.deadline(0.0, 600.0, started, now=100.0) == (697.0, 97.0)  # only the newest can be running
    done = [{"event": "export_done", "export": k, "seconds": 100.0} for k in range(10)]
    assert C.deadline(0.0, 600.0, done, now=0.0) == (600.0 + T.MAX_EXCLUDED_S, T.MAX_EXCLUDED_S)


def test_export_does_not_follow_links_and_always_logs_done(tmp_path):
    work = prepare(tmp_path, "refinement")
    control = work.parent / "control"
    (work / "candidate.blend").write_text(json.dumps({"kind": "good", "size": 1.0, "seconds": 0.0}))
    T.start_budget(work, control, 0.0, 600.0)
    target = tmp_path / "victim.txt"
    target.write_text("keep me")
    os.symlink(target, work / "exports" / "001.log")
    with pytest.raises(SystemExit, match="export 001 failed"):
        T.export_glb(work, control, FAKE_BLENDER)
    assert target.read_text() == "keep me"
    assert [(e["event"], e.get("ok")) for e in T.read_events(control)] == [("export_started", None), ("export_done", False)]
    with pytest.raises(SystemExit, match="outside /work"):
        T.export_glb(work, control, FAKE_BLENDER, "/etc/hosts")
    os.rename(work / "exports", work / "exports_real")
    os.symlink(tmp_path, work / "exports")
    with pytest.raises(SystemExit, match="must be a folder"):
        T.export_glb(work, control, FAKE_BLENDER)


def test_network_commands_are_flagged(tmp_path):
    log = tmp_path / "codex.jsonl"
    lines = [{"type": "item.completed", "item": {"id": "i1", "type": "command_execution", "command": "blender -b candidate.blend"}},
             {"type": "item.completed", "item": {"id": "i2", "type": "command_execution", "command": "curl -O https://x/y.png"}},
             {"type": "item.completed", "item": {"id": "i3", "type": "agent_message", "text": "git clone is not run here"}}]
    log.write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    assert C.network_commands(log) == ["curl -O https://x/y.png"]


def test_find_key_reports_files_but_not_the_key(tmp_path):
    key = "sk-TESTKEY0123456789abcdefghij"
    (tmp_path / "key.env").write_text(f"OPENAI_API_KEY={key}\n")
    root = tmp_path / "s"
    (root / "session").mkdir(parents=True)
    (root / "session" / "codex.jsonl").write_text(f"env ... CODEX_API_KEY={key} ...")
    (root / "session" / "clean.txt").write_text("nothing here")
    assert C.find_key(root, tmp_path / "key.env") == ["session/codex.jsonl"]


def test_paper_prompts_load_with_recorded_changes(tmp_path):
    for mode in ("refinement", "from_scratch"):
        text, orig = PR.load(mode), PR.original(mode)
        assert "work/run_budget.json" not in text.replace("/work/run_budget.json", "")
        assert len(text) - len(orig) == len(PR.CHANGES[mode])  # each change adds one "/"
    C.prepare_session(tmp_path / "s", "from_scratch", None, [], source="s", metric=False)
    assert (tmp_path / "s" / "session" / "prompt.txt").read_text(encoding="utf-8") == PR.load("from_scratch")


def test_review_passes_the_detail_rig(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    work = prepare(tmp_path, "refinement")
    control = work.parent / "control"
    (work / "candidate.blend").write_text(json.dumps({"kind": "good", "size": 1.0, "seconds": 0.0}))
    T.start_budget(work, control, 0.0, 600.0)
    rig = [[0.0, 0.0, 2.0, 0.1, 0.0, 0.0, 1.0]]
    rec = T.review(work, control, FAKE_BLENDER, detail=rig)
    assert json.loads((work / "reviews" / rec["receipt"] / "args.json").read_text())["detail"] == rig
    for bad in (rig * 5, [[0, 0, 2, 0, 0, 0, 1]], [[0, 0, 2, 0.1, 0, 0, 0]], [[0, 0, float("nan"), 0.1, 0, 0, 1]]):
        with pytest.raises(SystemExit):
            T.review(work, control, FAKE_BLENDER, detail=bad)


def test_previews_are_rendered_when_none_are_supplied(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    original = tmp_path / "original.glb"
    write_glb(original, size=2.0)
    C.prepare_session(tmp_path / "s", "refinement", None, make_views(tmp_path / "in"), source="s", metric=True,
                      original=original)
    previews = tmp_path / "s" / "work" / "original" / "previews"
    assert C.make_previews(tmp_path / "s", FAKE_BLENDER)
    assert (previews / "sheet.png").exists() and not (previews / "args.json").exists()


def test_articulation_session(runs):
    """edit, export, set_joint, review_articulation and finish articulated: the delivered GLB holds the
    joint, and every helper prints at most 20 lines."""
    root, r = runs["articulation_finish"]
    assert (r["mode"], r["source"], r["verdict"], r["exit"]) == ("articulation", "finish", "articulated", "exited")
    assert r["candidate_built"] is True and r["previews_built"] is None and r["rejected"] == []
    assert r["glb"] == "work/exports/002.glb" and edge(root / r["glb"]) == PARTS_SIZE and C.check_glb(root / r["glb"]) is None
    assert J.load(root / r["glb"]) == ({"lid": lid_joint(HINGE)}, ["body", "lid"])
    assert r["counts"] == {"exports": 2, "reviews": 0, "checkpoints": 0, "joints": 1, "articulation_reviews": 1}
    assert r["joints_delivered"] == 1
    assert (root / "work" / "prompt_seen.txt").read_text(encoding="utf-8") == PROMPT
    blocks = agent_log(root).split("$ ")[1:]
    assert len(blocks) == 4 and all(len(b.splitlines()) <= 20 for b in blocks)
    assert "nodes 2: body, lid; joints 0" in blocks[0]
    assert "joint of lid set; candidate.blend holds 1 joint (all in joints.json)\nroot body (free base)\n" in blocks[1]
    assert "  lid > body: revolute, origin (0, -0.25, 0.5), axis (-1, 0, 0), range [0, 1.5] rad\n" in blocks[1]
    assert "exports/002.glb (new export), 1 joint\n" in blocks[2] and "states s0, s1, s2: every joint at 0, 0.5, 1" in blocks[2]
    assert "  lid (revolute > body): s0 none; s1 none; s2 none\n" in blocks[2]
    assert "finished: articulated, exports/002.glb" in blocks[3]
    receipt = next(e["receipt"] for e in T.read_events(root / "control") if e["event"] == "review_articulation")
    assert receipt.startswith("a001-")
    folder = root / "work" / "reviews_articulation" / receipt
    report, args = (json.loads((folder / f).read_text()) for f in ("report.json", "args.json"))
    assert [s["q"] for s in report["states"]] == [{"lid": 0.0}, {"lid": 0.75}, {"lid": 1.5}]
    assert args["orbit"] == [-45.0, 45.0] and args["links"] == ["body", "lid"] and len(args["views"]) == 2
    assert json.loads((folder / "render.json").read_text())["views"][-1] == {
        "name": "orbit1", "s0": "orbit1_s0.png", "s1": "orbit1_s1.png", "s2": "orbit1_s2.png"}
    saved = json.loads((root / "work" / "joints.json").read_text())
    assert saved["joints"] == {"lid": lid_joint(HINGE)} and (saved["root"], saved["fixed_base"]) == ("body", False)
    assert [e["event"] for e in T.read_events(root / "control") if e["event"] not in ("export_started", "export_done")] \
        == ["joint", "review_articulation", "finish"]


def test_articulation_refuses_a_cycle_and_undoes_it(runs):
    root, r = runs["articulation_bad_joint"]
    blocks = agent_log(root).split("$ ")[1:]
    assert "refused: the joints form a cycle: lid -> body -> lid\nbody has no joint again\n" in blocks[1]
    assert "exit 1" in blocks[1] and blocks[1].splitlines()[-2].startswith("time left: ")
    joints = [(e["child"], e["ok"]) for e in T.read_events(root / "control") if e["event"] == "joint"]
    assert joints == [("lid", True), ("body", False)] and r["counts"]["joints"] == 1
    assert (root / "work" / "set_joint" / "002_undo.json").exists()
    assert json.loads((root / "work" / "joints.json").read_text())["joints"] == {"lid": lid_joint(HINGE)}
    assert (r["source"], r["verdict"]) == ("finish", "articulated") and J.load(root / r["glb"])[0] == {"lid": lid_joint(HINGE)}


def test_review_articulation_reports_penetration(tmp_path, monkeypatch):
    """A lid hinged about the body's center sweeps into the body; hinged at its back edge it does not."""
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    work = prepare(tmp_path, "articulation")
    control = work.parent / "control"
    T.start_budget(work, control, time.time(), 900.0)

    def review(hinge, **kw):
        scene = {"kind": "parts", "size": PARTS_SIZE, "seconds": 0.0, "objects": ["body", "lid"],
                 "joints": {"lid": lid_joint(hinge)}}
        (work / "candidate.blend").write_text(json.dumps(scene))
        return T.review_articulation(work, control, FAKE_BLENDER, **kw)

    rec, rep = review(INSIDE)
    states = rep["joints"]["lid"]["states"]
    assert rec["receipt"].startswith("a001-") and (work / rec["sheet"]).exists() and rep["seconds"] < 5.0
    closed = {"faces": 12, "watertight": True, "tolerance": rep["tolerance"]}  # neither part is thin
    assert rep["links"] == {"body": closed, "lid": closed} and np.isclose(rep["tolerance"], T.DEPTH_TOLERANCE * 3 ** 0.5)
    assert states[0] == {"name": "s0", "inside_share": 0.0, "max_depth": 0.0, "worst_other_link": None}
    assert all(s["inside_share"] > 0.1 and s["max_depth"] > 0.1 and s["worst_other_link"] == "body" for s in states[1:])
    assert rep["joints"]["lid"]["moving_links"] == ["lid"] and rep["joints"]["lid"]["samples"] == T.SAMPLES
    rec, rep = review(HINGE, states=2)
    assert rec["receipt"].startswith("a002-") and [s["inside_share"] for s in rep["joints"]["lid"]["states"]] == [0, 0]
    args = json.loads((work / "reviews_articulation" / rec["receipt"] / "args.json").read_text())
    M = np.array(args["states"][1]["motions"]["lid"])
    assert np.allclose(M, J.joint_transform(lid_joint(HINGE), 1.5)) and np.allclose(args["states"][1]["motions"]["body"], np.eye(4))
    assert (M @ [0.0, -0.25, -0.5, 1.0])[1] < -1.2  # the lid's front edge goes up (-y in W)
    for states in (1, 6):
        with pytest.raises(SystemExit, match="--states"):
            review(HINGE, states=states)

    def broken(*args):
        raise RuntimeError("not a mesh")

    monkeypatch.setattr(T, "penetration", broken)  # the review goes on without the check
    rec, rep = review(INSIDE, states=4)
    assert rec["receipt"].startswith("a003-") and rep["error"] == "RuntimeError: not a mesh" and rep["joints"] == {}
    scene = {"kind": "parts", "size": PARTS_SIZE, "seconds": 0.0, "objects": ["body", "lid"], "joints": {}}
    (work / "candidate.blend").write_text(json.dumps(scene))
    with pytest.raises(SystemExit, match="holds no joint; set the joints with set_joint first"):
        T.review_articulation(work, control, FAKE_BLENDER)


def test_articulation_poses_move_welded_links_and_groups_them():
    """A tray parented to a drawer in Blender has no joint: it moves with the drawer, and its samples
    count as the drawer's. Here the tray sits inside the solid body, so the drawer's joint reports it."""
    drawer = {"version": 1, "frame": "W", "parent": "body", "type": "prismatic", "origin": [0.0, 0.0, 0.0],
              "axis": [0.0, 0.0, -1.0], "lower": 0.0, "upper": 0.4}
    links = ["body", "drawer", "tray"]
    poses = T.articulation_poses({"drawer": drawer}, links, {"tray": "drawer"}, states=3)
    for p, z in zip(poses, (0.0, -0.2, -0.4)):
        assert np.allclose(p["motions"]["drawer"][:3, 3], [0, 0, z]) and np.allclose(p["motions"]["tray"], p["motions"]["drawer"])
        assert np.allclose(p["motions"]["body"], np.eye(4))
    meshes = {"body": trimesh.creation.box(bounds=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]),
              "drawer": trimesh.creation.box(bounds=[[-0.3, -0.2, -0.55], [0.3, 0.2, -0.5]]),
              "tray": trimesh.creation.box(bounds=[[-0.25, -0.15, -0.5], [0.25, 0.15, -0.1]])}
    carrier = T.carriers({"drawer": drawer}, links, {"tray": "drawer"})
    assert carrier == {"body": "body", "drawer": "drawer", "tray": "drawer"}
    rep = T.penetration(meshes, {"drawer": drawer}, poses, carrier, samples=500)
    j = rep["joints"]["drawer"]
    assert j["moving_links"] == ["drawer", "tray"] and j["samples"] == 1000
    shares = [s["inside_share"] for s in j["states"]]
    assert shares[0] > shares[1] > shares[2] == 0.0 and j["states"][0]["worst_other_link"] == "body"


def test_articulation_poses_compose_a_chain_of_joints():
    """A handle hinged on the lid turns with the lid: its motion is the lid's motion, then its own turn."""
    lid = lid_joint(HINGE)
    handle = {"version": 1, "frame": "W", "parent": "lid", "type": "revolute", "origin": [0.0, -0.5, -0.4],
              "axis": [0.0, 0.0, 1.0], "lower": -0.5, "upper": 0.5}
    joints, links = {"lid": lid, "handle": handle}, ["body", "lid", "handle"]
    assert T.carriers(joints, links, {}) == {"body": "body", "lid": "lid", "handle": "handle"}
    poses = T.articulation_poses(joints, links, {})
    assert [p["q"]["handle"] for p in poses] == [-0.5, 0.0, 0.5]
    for p in poses:
        lid_m, handle_m = p["motions"]["lid"], p["motions"]["handle"]
        assert np.allclose(lid_m, J.joint_transform(lid, p["q"]["lid"]))
        assert np.allclose(handle_m, lid_m @ J.joint_transform(handle, p["q"]["handle"]))
        assert np.allclose(handle_m @ [0.0, -0.5, -0.4, 1.0], lid_m @ [0.0, -0.5, -0.4, 1.0])  # its axis rides on the lid
    end = poses[-1]
    assert not np.allclose(end["motions"]["handle"], J.joint_transform(handle, 0.5) @ end["motions"]["lid"])


def test_penetration_groups_links_by_their_joint():
    """Two drawers slide alike, and only drawer_0 holds a tray inside the solid body. Links are grouped by
    the joint that carries them, not by their motion, so only drawer_0 reports penetration. A knob fixed
    to drawer_1 moves with drawer_1."""
    drawer = {"version": 1, "frame": "W", "parent": "body", "type": "prismatic", "origin": [0.0, 0.0, 0.0],
              "axis": [0.0, 0.0, -1.0], "lower": 0.0, "upper": 0.3}
    knob = {"version": 1, "frame": "W", "parent": "drawer_1", "type": "fixed"}
    joints = {"drawer_0": drawer, "drawer_1": dict(drawer), "knob_1": knob}
    links, held = ["body", "drawer_0", "tray_0", "drawer_1", "knob_1"], {"tray_0": "drawer_0"}
    carrier = T.carriers(joints, links, held)
    assert carrier == {"body": "body", "drawer_0": "drawer_0", "tray_0": "drawer_0", "drawer_1": "drawer_1",
                       "knob_1": "drawer_1"}
    poses = T.articulation_poses(joints, links, held)
    assert all(np.allclose(p["motions"]["drawer_0"], p["motions"]["drawer_1"]) for p in poses)
    assert all(np.allclose(p["motions"]["knob_1"], p["motions"]["drawer_1"]) for p in poses)
    meshes = {"body": trimesh.creation.box(bounds=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]),
              "drawer_0": trimesh.creation.box(bounds=[[-0.45, -0.2, -0.55], [-0.05, 0.2, -0.5]]),
              "tray_0": trimesh.creation.box(bounds=[[-0.4, -0.15, -0.5], [-0.1, 0.15, -0.1]]),
              "drawer_1": trimesh.creation.box(bounds=[[0.05, -0.2, -0.55], [0.45, 0.2, -0.5]]),
              "knob_1": trimesh.creation.box(bounds=[[0.2, -0.05, -0.6], [0.3, 0.05, -0.55]])}
    rep = T.penetration(meshes, joints, poses, carrier, samples=500)
    assert sorted(rep["joints"]) == ["drawer_0", "drawer_1"]
    j0, j1 = rep["joints"]["drawer_0"], rep["joints"]["drawer_1"]
    assert (j0["moving_links"], j1["moving_links"]) == (["drawer_0", "tray_0"], ["drawer_1", "knob_1"])
    assert all(s["inside_share"] > 0 and s["worst_other_link"] == "body" for s in j0["states"])
    assert [s["inside_share"] for s in j1["states"]] == [0, 0, 0]


def test_penetration_finds_a_part_passing_through_a_thin_part():
    """On a 2 m panel 18 mm thick, 0.5% of the diagonal is deeper than any point inside the panel can be.
    A thin link's tolerance is a quarter of its thickness, so a box pushed through the panel is found."""
    slide = {"version": 1, "frame": "W", "parent": "panel", "type": "prismatic", "origin": [0.0, 0.0, 0.0],
             "axis": [0.0, 0.0, 1.0], "lower": 0.0, "upper": 0.4}
    meshes = {"panel": trimesh.creation.box(bounds=[[-1.0, -1.0, 0.0], [1.0, 1.0, 0.018]]),
              "box": trimesh.creation.box(bounds=[[-0.2, -0.2, -0.3], [0.2, 0.2, -0.1]])}
    joints, links = {"box": slide}, ["panel", "box"]
    rep = T.penetration(meshes, joints, T.articulation_poses(joints, links, {}), T.carriers(joints, links, {}))
    assert rep["tolerance"] > 0.009 and np.isclose(rep["links"]["panel"]["tolerance"], 0.25 * 0.018)
    assert rep["links"]["box"]["tolerance"] == rep["tolerance"]
    states = rep["joints"]["box"]["states"]
    assert states[0]["inside_share"] == states[2]["inside_share"] == 0
    assert states[1]["inside_share"] > 0 and states[1]["worst_other_link"] == "panel"
    assert 0.0045 < states[1]["max_depth"] <= 0.009 + 1e-9  # at most half the panel's thickness


def test_set_joint_checks_its_arguments(tmp_path, monkeypatch):
    work = prepare(tmp_path, "articulation")
    control = work.parent / "control"
    (work / "candidate.blend").write_text(json.dumps({"kind": "parts", "size": PARTS_SIZE, "seconds": 0.0,
                                                      "objects": ["body", "lid"], "joints": {}}))
    T.start_budget(work, control, time.time(), 900.0)
    for k, v in (("GATOR_WORK", work), ("GATOR_CONTROL", control), ("GATOR_BLENDER", shlex.join(FAKE_BLENDER)),
                 ("PYTHONPATH", REPO)):
        monkeypatch.setenv(k, str(v))
    lid = ["--child", "lid", "--parent", "body"]
    hinge = ["--origin", "0", "-0.25", "0.5", "--axis", "-1", "0", "0", "--lower", "0", "--upper", "1.5"]
    for args, why in [([*lid, "--type", "revolute", *hinge[:-2]], "a revolute joint needs --upper"),
                      ([*lid, "--type", "revolute", *hinge[:5], "0", "2", "0", *hinge[8:]], "--axis must be a unit vector"),
                      ([*lid, "--type", "prismatic", *hinge[:9], "0.1", "--upper", "1"], "must hold 0 with lower < upper"),
                      ([*lid, "--type", "revolute", *hinge[:9], "0", "--upper", "0"], "must hold 0 with lower < upper"),
                      ([*lid, "--type", "revolute", *hinge[:-1], "7"], "at most 2 pi"),
                      ([*lid, "--type", "revolute", "--origin", "nan", "0", "0", *hinge[4:]], "finite"),
                      ([*lid, "--type", "fixed", *hinge[:4]], "a fixed joint takes only"),
                      (["--child", "lid", "--parent", "world", "--type", "revolute", *hinge], "only a fixed joint"),
                      (["--child", "lid", "--parent", "lid", "--type", "fixed"], "lid cannot be its own parent"),
                      (["--child", "world", "--parent", "body", "--type", "fixed"], '"world" is the fixed base'),
                      (["--child", "lid"], "give --parent and --type, or --remove"),
                      ([*lid, "--remove"], "--remove takes only --child"),
                      (["--child", "door", "--parent", "body", "--type", "fixed"], "no mesh object is named 'door'")]:
        with pytest.raises(SystemExit) as e:
            T.main(["set_joint", *args])
        assert re.search(why, e.value.code) and e.value.code.splitlines()[-1].startswith("time left: "), (args, e.value.code)
    assert json.loads((work / "candidate.blend").read_text())["joints"] == {}
    T.main(["set_joint", *lid, "--type", "revolute", *hinge[:5], "-0.70711", "0", "0.70711", *hinge[8:]])
    axis = json.loads((work / "candidate.blend").read_text())["joints"]["lid"]["axis"]
    assert np.isclose(np.linalg.norm(axis), 1, atol=1e-12) and np.allclose(axis, [-2 ** -0.5, 0, 2 ** -0.5])
    T.main(["set_joint", "--child", "lid", "--remove"])
    assert json.loads((work / "joints.json").read_text()) == {"joints": {}, "links": ["body", "lid"], "holders": {},
                                                              "problems": []}
    assert [(e["child"], e.get("action"), e["ok"]) for e in T.read_events(control) if e["event"] == "joint"] == \
        [("door", "set", False), ("lid", "set", True), ("lid", "remove", True)]
    refine = prepare(tmp_path / "r", "refinement")
    monkeypatch.setenv("GATOR_WORK", str(refine))
    monkeypatch.setenv("GATOR_CONTROL", str(refine.parent / "control"))
    T.start_budget(refine, refine.parent / "control", time.time(), 600.0)
    for cmd in (["set_joint", "--child", "lid", "--remove"], ["review_articulation"]):
        with pytest.raises(SystemExit, match="works only in the articulation mode"):
            T.main(cmd)


def test_set_joint_restores_a_refused_change_and_reports_the_problems_left(tmp_path, monkeypatch, capsys):
    """A refused change gives the child its previous joint back, and the next export holds it. A cycle
    from before a change does not block it, whichever link validate words it from, and the change that
    breaks the cycle leaves no problem and prints the root. An existing record stops set_joint before
    Blender changes the scene."""
    work = prepare(tmp_path, "articulation")
    control, blend = work.parent / "control", work / "candidate.blend"

    def scene(joints):
        blend.write_text(json.dumps({"kind": "parts", "size": PARTS_SIZE, "seconds": 0.0, "objects": ["body", "lid"],
                                     "joints": joints}))

    scene({"lid": lid_joint(HINGE)})
    T.start_budget(work, control, time.time(), 900.0)
    for k, v in (("GATOR_WORK", work), ("GATOR_CONTROL", control), ("GATOR_BLENDER", shlex.join(FAKE_BLENDER)),
                 ("PYTHONPATH", REPO)):
        monkeypatch.setenv(k, str(v))
    hinge = ["--origin", "0", "-0.25", "0.5", "--axis", "-1", "0", "0", "--lower", "0", "--upper", "1.5"]
    with pytest.raises(SystemExit) as e:
        T.main(["set_joint", "--child", "lid", "--parent", "door", "--type", "revolute", *hinge])
    assert e.value.code.startswith("refused: the joint of 'lid': the parent 'door' is not a mesh node\n"
                                   "the joint of lid is back as it was\ntime left: ")
    assert json.loads(blend.read_text())["joints"] == {"lid": lid_joint(HINGE)}
    assert json.loads((work / "set_joint" / "001_undo.json").read_text())["previous"]["parent"] == "door"
    assert J.load(T.export_glb(work, control, FAKE_BLENDER)[0])[0] == {"lid": lid_joint(HINGE)}
    scene({"lid": lid_joint(HINGE), "body": {"version": 1, "frame": "W", "parent": "lid", "type": "fixed"}})
    T.main(["set_joint", "--child", "lid", "--parent", "body", "--type", "revolute", *hinge[:-1], "1.0"])
    out = capsys.readouterr().out
    assert "problem left from before this change: the joints form a cycle: " in out and "root" not in out
    assert json.loads((work / "joints.json").read_text())["problems"] == ["the joints form a cycle: lid -> body -> lid"]
    T.main(["set_joint", "--child", "body", "--remove"])
    out = capsys.readouterr().out
    assert "root body (free base)" in out and "problem" not in out
    saved = json.loads((work / "joints.json").read_text())
    assert (saved["problems"], saved["root"]) == ([], "body")
    assert saved["joints"] == {"lid": lid_joint({**HINGE, "upper": 1.0})}
    (work / "set_joint" / "004.json").write_text("{}")
    with pytest.raises(SystemExit, match="set_joint/004.json already exists; remove it"):
        T.main(["set_joint", "--child", "lid", "--remove"])
    assert json.loads(blend.read_text())["joints"] == {"lid": lid_joint({**HINGE, "upper": 1.0})}
    assert not (work / "set_joint" / "004.log").exists()
    assert [(e["child"], e["ok"]) for e in T.read_events(control) if e["event"] == "joint"] == \
        [("lid", False), ("lid", True), ("body", True), ("lid", False)]


def test_finish_articulated_needs_reviewed_joints(tmp_path):
    work = prepare(tmp_path, "articulation")
    root, control, ex = work.parent, work.parent / "control", work / "exports"
    write_glb(ex / "001.glb", "parts", PARTS_SIZE)
    write_glb(ex / "002.glb", "parts", PARTS_SIZE, {"lid": lid_joint(HINGE)})
    sha = {k: T.sha256(ex / f"00{k}.glb") for k in (1, 2)}
    r1, r2, a = f"r001-{sha[1][:12]}", f"r002-{sha[2][:12]}", f"a001-{sha[2][:12]}"
    for rid, k in ((r1, 1), (r2, 2)):
        T.log_event(control, "review", receipt=rid, glb=f"exports/00{k}.glb", sha256=sha[k])
    with pytest.raises(SystemExit, match="exports/001.glb holds no joint"):
        T.finish(work, control, r1, "articulated", "x")
    with pytest.raises(SystemExit, match="exports/002.glb has no articulation review"):
        T.finish(work, control, r2, "articulated", "x")
    with pytest.raises(SystemExit, match="unknown verdict 'complete' for articulation"):
        T.finish(work, control, r2, "complete", "x")
    T.log_event(control, "review_articulation", receipt=a, glb="exports/002.glb", sha256=sha[2], joints=1)
    meta = T.checkpoint(work, control, a, "the drawer has no tray")
    assert meta["glb"] == f"checkpoints/001-{a}.glb" and T.sha256(work / meta["glb"]) == sha[2]
    assert T.finish(work, control, r2, "articulated", "x")["glb"] == "exports/002.glb"  # that GLB was reviewed
    assert T.finish(work, control, a, "articulated", "lid hinged")["sha256"] == sha[2]
    pick, rejected = C.select_deliverable(root, "articulation")
    assert (pick["source"], pick["verdict"], pick["glb"], rejected) == ("finish", "articulated", ex / "002.glb", [])
    T.finish(work, control, a, "unchanged", "nothing moves")
    pick, _ = C.select_deliverable(root, "articulation")
    assert (pick["verdict"], pick["glb"]) == ("unchanged", root / "original.glb")
    assert J.load(root / "original.glb") == ({}, ["body", "lid"])
    (control / "events.jsonl").write_text("")
    pick, _ = C.select_deliverable(root, "articulation")
    assert (pick["source"], pick["glb"]) == ("fallback_original", root / "original.glb")


def test_articulation_helpers_print_at_most_20_lines(tmp_path, monkeypatch, capsys):
    work = prepare(tmp_path, "articulation")
    control = work.parent / "control"
    T.start_budget(work, control, time.time(), 900.0)
    monkeypatch.setenv("GATOR_WORK", str(work))
    monkeypatch.setenv("GATOR_CONTROL", str(control))
    names = [f"door_{k}" for k in range(20)]
    rec = {"joints": {n: lid_joint(HINGE) for n in names}, "links": ["body", *names], "holders": {},
           "problems": ["a", "b"], "root": "body", "fixed_base": False, "loose": {f"tray_{k}": "body" for k in range(9)}}
    monkeypatch.setattr(T, "set_joint", lambda *a: rec)
    T.main(["set_joint", "--child", "door_0", "--remove"])
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 20 and out[-3:-1] == ["  ... and 5 more", "problem left from before this change: a (and 1 more)"]
    assert out[1].endswith("welded without a joint: tray_0 > body, tray_1 > body, tray_2 > body, tray_3 > body, "
                           "tray_4 > body, tray_5 > body ...")
    states = [{"name": f"s{k}", "t": k / 4, "inside_share": 0.25, "max_depth": 0.1, "worst_other_link": "body"}
              for k in range(5)]
    rep = {"tolerance": 0.01, "states": states, "links": {"body": {"watertight": False}},
           "joints": {n: {"type": "revolute", "parent": "body", "states": states} for n in names}}
    receipt = {"receipt": "a001-0123456789ab", "glb": "exports/001.glb", "cached": False, "joints": 20,
               "sheet": "sheet.png", "report": "report.json"}
    monkeypatch.setattr(T, "review_articulation", lambda *a: (receipt, rep))
    T.main(["review_articulation", "--states", "5"])
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 20 and out[-3:-1] == ["  ... and 9 more", "warning: not closed, so inside and outside are "
                                                                 "uncertain: body"]
    assert out[6] == "  door_0 (revolute > body): " + "; ".join(f"s{k} 25.0%, deepest 0.1, most in body" for k in range(5))


def test_check_glb_checks_joints_only_when_present(tmp_path):
    fixed_to_lid = {"version": 1, "frame": "W", "parent": "lid", "type": "fixed"}
    for name, joints, why in [("plain", None, None), ("hinged", {"lid": lid_joint(HINGE)}, None),
                              ("cycle", {"lid": lid_joint(HINGE), "body": fixed_to_lid},
                               "bad joints: the joints form a cycle: body -> lid -> body"),
                              ("closed", {"lid": lid_joint({**HINGE, "upper": 0.0})}, "bad joints: the joint of 'lid': "
                               "a moving joint needs lower < upper; use type fixed for a part that cannot move")]:
        write_glb(tmp_path / f"{name}.glb", "parts", PARTS_SIZE, joints)
        assert C.check_glb(tmp_path / f"{name}.glb") == why

    def broken(g):
        g["nodes"][1]["extras"] = {"gator_joint": "{not json"}
        return g

    glb_json(tmp_path / "plain.glb", tmp_path / "broken.glb", broken)
    assert C.check_glb(tmp_path / "broken.glb").startswith("bad joints: the joint of node 'lid' is not valid JSON")


def test_articulation_prompt_and_session_layout(tmp_path):
    assert PR.FILES["articulation"] == "articulation_prompt.md" and PR.CHANGES["articulation"] == []
    text = PR.load("articulation")
    assert text == PR.original("articulation") and "/work/run_budget.json" in text and "900 seconds" in text
    for fact in ("input/parts.json", "set_joint", "review_articulation", "articulated", "needs_review", "checkpoint",
                 "--parent world --type fixed", "lower <= 0 <= upper"):
        assert fact in text
    work = prepare(tmp_path, "articulation")
    m = json.loads((work / "input" / "manifest.json").read_text())
    assert m["mode"] == "articulation" and m["parts"] == "input/parts.json"
    assert json.loads((work / "input" / "parts.json").read_text()) == PARTS
    assert T.sha256(work.parent / "original.glb") == T.sha256(work / "original" / "original.glb")
    assert J.load(work / "original" / "original.glb")[1] == ["body", "lid"]
    assert (work / "original" / "previews" / "front.png").exists()
    with pytest.raises(ValueError, match="parts"):
        C.prepare_session(tmp_path / "x", "articulation", None, [], source="s", metric=True,
                          original=work / "original" / "original.glb")
    with pytest.raises(ValueError, match="parts"):
        C.prepare_session(tmp_path / "y", "refinement", None, [], source="s", metric=True,
                          original=work / "original" / "original.glb", parts=tmp_path / "in" / "parts.json")
    with pytest.raises(ValueError, match="mode is one of refinement, from_scratch, articulation"):
        C.prepare_session(tmp_path / "z", "rigging", None, [], source="s", metric=True)


def test_codex_session_layout_and_command(tmp_path):
    views = make_views(tmp_path / "in")
    write_glb(tmp_path / "in" / "o.glb")
    C.prepare_session(tmp_path / "s", "refinement", None, views, source="s", metric=True,
                      original=tmp_path / "in" / "o.glb")
    root = tmp_path / "s"
    assert (root / "session" / "codex_home" / "config.toml").exists()
    agents = (root / "work" / "AGENTS.md").read_text()
    assert "view_image" in agents and "read_file" not in agents and "{" not in agents.split("## Helper commands")[0]
    assert 'model_reasoning_effort="xhigh"' in C.codex_command()
    conf = tomllib.loads(C.codex_config("http://127.0.0.1:1/v1"))
    assert conf["model"] == "gpt-6-astra" and conf["model_reasoning_effort"] == "xhigh" and conf["web_search"] == "disabled"
    assert conf["model_provider"] == "proxied" and conf["model_providers"]["proxied"]["supports_websockets"] is False
    assert conf["shell_environment_policy"]["ignore_default_excludes"] is False and "model_provider" not in tomllib.loads(C.codex_config())


def test_shell_commands_counted_once(tmp_path):
    codex = tmp_path / "codex.jsonl"
    item = {"id": "i1", "type": "command_execution", "command": "bash -lc 'curl -O x'"}
    codex.write_text("\n".join(json.dumps(e) for e in [
        {"type": "item.started", "item": item}, {"type": "item.completed", "item": {**item, "exit_code": 0}},
        {"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": "pip install y"}}]) + "\n")
    assert C.shell_commands(codex) == ["bash -lc 'curl -O x'"] and C.network_commands(codex) == ["bash -lc 'curl -O x'"]


def test_claude_shell_commands_are_the_calls_that_ran(tmp_path):
    """claude_agent runs the calls of a turn only for stop reason tool_use (nothing of a refused, cut-off,
    paused or otherwise stopped turn), and not the calls before the last fallback block."""
    def call(i, command):
        return {"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {"command": command}}

    fb = {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-5"}}
    lines = [{"type": "response", "stop_reason": "tool_use", "blocks": [call(1, "ls"), call(1, "ls")]},
             {"type": "response", "stop_reason": "max_tokens", "blocks": [call(2, "pip install cut-off")]},
             {"type": "response", "stop_reason": "refusal", "blocks": [call(3, "curl -O https://x/refused")]},
             {"type": "response", "stop_reason": "pause_turn", "blocks": [call(4, "wget https://x/paused")]},
             {"type": "response", "stop_reason": "end_turn", "blocks": [call(7, "pip install not-run")]},
             {"type": "response", "stop_reason": "model_context_window_exceeded", "blocks": [call(8, "curl -O https://x/c")]},
             {"type": "response", "stop_reason": "tool_use",
              "blocks": [call(5, "curl -O https://x/declined"), fb, {"type": "text", "text": "x"}, call(6, "pip install y")]},
             {"type": "tool_result", "tool_use_id": "t1", "output": "curl -O https://x/output"}]
    log = tmp_path / "claude.jsonl"
    log.write_text("".join(json.dumps(e) + "\n" for e in lines))
    assert C.shell_commands(log) == ["ls", "pip install y"] and C.network_commands(log) == ["pip install y"]


# ---- the claude agent: Claude Opus 5.5 through gator.agent.claude_agent ----

ANTHROPIC_KEY = "sk-ant-TEST-" + "0123456789abcdef" * 3  # a stand-in; nothing here calls the API
# claude_agent's own CLI with a scripted client instead of the Anthropic API, and bash on the host:
#   python STAND_IN SCRIPT.json REQUESTS.jsonl CLAUDE_AGENT_ARGS...
# Each request's settings go to REQUESTS.jsonl; each answer is the next message of SCRIPT.json.
STAND_IN = '''
import json, sys
from types import SimpleNamespace
from anthropic.types.beta import BetaMessage
from gator.agent import claude_agent

script_file, requests_file, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
script = json.load(open(script_file))


def plain(x):  # the SDK's content blocks of earlier assistant turns through to_dict
    if hasattr(x, "to_dict"):
        return x.to_dict()
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


class Stream:
    def __init__(self, message):
        self.message = message
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def get_final_message(self):
        return self.message


class Messages:
    n = 0
    def stream(self, **kw):
        seen = {k: kw.get(k) for k in ("model", "output_config", "thinking", "betas", "fallbacks", "speed", "cache_control",
                                       "system")}
        with open(requests_file, "a") as f:
            f.write(json.dumps({**seen, "messages": len(kw["messages"]), "tools": [t["name"] for t in kw["tools"]],
                                "view_image": kw["tools"][2]["description"], "history": plain(kw["messages"])}) + "\\n")
        self.n += 1
        return Stream(BetaMessage.model_validate({"id": f"msg_{self.n}", "type": "message", "role": "assistant",
                                                  "model": "claude-opus-5-5", "stop_sequence": None, **script[self.n - 1]}))


sys.exit(claude_agent.main([*argv, "--shell", "host"], client=SimpleNamespace(beta=SimpleNamespace(messages=Messages()))))
'''


def usage(i, o, read=0, write=0):
    return {"input_tokens": i, "output_tokens": o, "cache_read_input_tokens": read, "cache_creation_input_tokens": write}


def bash(call_id, command):
    return {"type": "tool_use", "id": call_id, "name": "bash", "input": {"command": command}}


CLAUDE_SCRIPT = [
    {"content": [{"type": "thinking", "thinking": "Shrink the box, then review it.", "signature": "sig-1"},
                 {"type": "text", "text": "Editing the scene."},
                 bash("toolu_1", "env > env.txt; printf '{\"kind\": \"good\", \"size\": 1.2, \"seconds\": 0}' > candidate.blend"
                                 " && review")],
     "stop_reason": "tool_use", "usage": usage(1200, 300, write=5000)},
    {"content": [{"type": "tool_use", "id": "toolu_2", "name": "view_image", "input": {"path": "/work/original/previews/front.png"}},
                 bash("toolu_3", 'finish --receipt "$(ls reviews | tail -1)" --verdict complete --reason "fixed the handle"')],
     "stop_reason": "tool_use", "usage": usage(400, 200, read=5000, write=900)},
    {"content": [{"type": "text", "text": "Done."}], "stop_reason": "end_turn", "usage": usage(10, 5, read=5900)},
]


def all_files(root):
    return [p for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink()]


def test_claude_session_through_the_launcher(tmp_path, monkeypatch, capsys):
    """A whole claude session through run_container_session: the real claude_agent CLI with a scripted
    client and a host shell, the helpers over the socket, the deliverable, the log readers, and the
    key in no file but its own (outside the session folder)."""
    from gator.agent import loop_trace, session_report

    monkeypatch.setenv("PYTHONPATH", str(REPO))
    root = prepare(tmp_path, "refinement", agent="claude").parent
    key_file = tmp_path / "anthropic.env"
    key_file.write_text(f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}\n")
    (tmp_path / "stand_in.py").write_text(STAND_IN)
    (tmp_path / "script.json").write_text(json.dumps(CLAUDE_SCRIPT))
    real, built = C.claude_command, []

    def stand_in(*args):
        built.append(real(*args))
        return [sys.executable, str(tmp_path / "stand_in.py"), str(tmp_path / "script.json"),
                str(tmp_path / "requests.jsonl"), *built[-1][3:]]

    monkeypatch.setattr(C, "claude_command", stand_in)
    monkeypatch.setattr(C, "blender_command", lambda root, sif: FAKE_BLENDER)
    r = C.run_container_session(root, "/sif/agent.sif", key_file, **{**FAST, "finish_wait_s": 10.0})

    cmd = built[0]
    assert cmd[:4] == [sys.executable, "-m", "gator.agent.claude_agent", str(root.resolve())]
    assert cmd[cmd.index("--key-file") + 1] == str(key_file) and cmd[-2:] == ["--effort", "xhigh"]
    assert "--image-side" not in cmd and "--system" not in cmd  # claude_agent's defaults: 2576 px, SYSTEM.md
    assert (r["agent"], r["model"], r["effort"]) == ("claude", "claude-opus-5-5", "xhigh")
    assert (r["source"], r["verdict"], r["reason"], r["glb"]) == ("finish", "complete", "fixed the handle", "work/exports/001.glb")
    assert edge(root / r["glb"]) == 1.2 and r["rejected"] == [] and r["candidate_built"] is True
    # The agent ended by itself once finish ran, before the controller's stop.
    assert (r["exit"], r["returncode"], r["cli_exit"], r["end_reason"]) == ("exited", 0, "ended", "finished")
    tokens = {k: sum(s["usage"][k] for s in CLAUDE_SCRIPT[:2]) for k in C.PRICES}
    assert r["turns"] == 2 and r["tokens"] == tokens
    assert r["cost_usd"] == pytest.approx(sum(tokens[k] * C.PRICES[k] for k in tokens) / 1e6, abs=1e-4)
    assert r["key_found_in"] == [] and r["network_commands"] == []
    assert [p for p in all_files(tmp_path) if ANTHROPIC_KEY.encode() in p.read_bytes()] == [key_file]
    assert json.loads((root / "result.json").read_text()) == r

    reqs = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert [q["messages"] for q in reqs] == [1, 3]  # the prompt, then the turn and its results appended
    assert all(q["model"] == "claude-opus-5-5" and q["output_config"] == {"effort": "xhigh"} and q["speed"] is None
               and q["thinking"] == {"type": "adaptive", "display": "summarized"} and q["fallbacks"] == "default"
               and q["betas"] == ["server-side-fallback-2026-07-01"] for q in reqs)
    assert reqs[0]["tools"] == ["bash", "str_replace_based_edit_tool", "view_image"]
    assert "larger than 2576 px on the long side" in reqs[0]["view_image"]

    log = root / "session" / "claude.jsonl"
    results = {e["tool_use_id"]: e for e in C._jsonl(log) if e.get("type") == "tool_result"}
    assert results["toolu_1"]["is_error"] is False and "receipt" in results["toolu_1"]["output"]
    assert results["toolu_2"]["is_error"] is True and "not a PNG or JPEG" in results["toolu_2"]["output"]
    assert C.shell_commands(log)[1].startswith("finish --receipt")
    assert (root / "session" / "claude_stdout.log").read_text().startswith("turn 1: tool_use")
    assert (root / "session" / "system.md").read_bytes() == (C.AGENT_DIR / "claude" / "SYSTEM.md").read_bytes()
    assert all(q["system"][0]["text"] == (C.AGENT_DIR / "claude" / "SYSTEM.md").read_text(encoding="utf-8") for q in reqs)
    start = next(e for e in C._jsonl(log) if e.get("type") == "start")
    assert (start["image_side"], start["system"]) == (2576, str(C.AGENT_DIR / "claude" / "SYSTEM.md"))

    capsys.readouterr()
    loop_trace.main([str(tmp_path)])
    trace = capsys.readouterr().out
    assert "== agent (claude), in order" in trace and "views: /work/original/previews/front.png [" in trace
    assert "runs: finish --receipt" in trace and "ended: finished after 2 turns" in trace
    assert "finish complete: fixed the handle" in trace and "claude-opus-5-5: 2 turns" in trace
    session_report.main(str(tmp_path))
    report = capsys.readouterr().out
    assert "requests 2 (2 answered)" in report and "CMD finish --receipt" in report and "VIEW /work/original" in report
    assert "TOOL ERROR view_image" in report and "end: finished after 2 turns" in report


def test_claude_session_with_every_tool(tmp_path, monkeypatch):
    """A claude session at effort low in fast mode, with the baseline's image side (1568 px) and its own
    manual, through prepare_session and run_container_session: the shell keeps its directory and
    variables across turns, the text editor views a file and edits another, view_image shows a PNG
    that a command made, and finish ends the session. Every request repeats the one before it and
    appends to it, and the key the shell prints reaches neither the model nor any file."""
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    manual = tmp_path / "opus_dig" / "system_v1.md"
    manual.parent.mkdir()
    manual.write_text("The old manual: an image larger than 1568 px is shown downscaled. é\n" * 80, encoding="utf-8")
    root = prepare(tmp_path, "refinement", agent="claude", effort="low", fast=True, image_side=1568,
                   system=manual).parent
    key_file = tmp_path / "anthropic.env"
    key_file.write_text(f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}\n")
    (tmp_path / "stand_in.py").write_text(STAND_IN)
    py, editor = shlex.quote(sys.executable), "str_replace_based_edit_tool"
    script = [
        {"content": [{"type": "thinking", "thinking": "Look around first.", "signature": "sig-1"},
                     bash("t1", "cd input && export STEP=one && pwd && env | grep -cE '^(ANTHROPIC|OPENAI|CODEX)_'; "
                                "echo \"socket ${GATOR_SOCKET:+set}\"; command -v review"),
                     {"type": "tool_use", "id": "t2", "name": editor, "input": {"command": "view", "path": "/work/input/manifest.json"}},
                     bash("t3", 'echo "$STEP in $(basename "$PWD")"')],
         "stop_reason": "tool_use", "usage": usage(3000, 400, write=6000)},
        {"content": [{"type": "tool_use", "id": "t4", "name": editor,
                      "input": {"command": "create", "path": "/work/edit.py", "file_text": "size = 2.0\n"}},
                     {"type": "tool_use", "id": "t5", "name": editor,
                      "input": {"command": "str_replace", "path": "edit.py", "old_str": "2.0", "new_str": "1.2"}},
                     bash("t6", "cd .. && printf '{\"kind\": \"good\", \"size\": %s, \"seconds\": 0}' \"$(sed -n 's/^size = //p' "
                                "edit.py)\" > candidate.blend && R=$(review | sed -n 's/^receipt //p') && echo \"receipt $R\" && "
                                f"mkdir -p shots && {py} -c \"from PIL import Image; Image.new('RGB', (3000, 1500), (200, 30, 30))"
                                ".save('shots/look.png')\""),
                     bash("t7", f"cat {shlex.quote(str(key_file))}")],
         "stop_reason": "tool_use", "usage": usage(800, 600, read=6000, write=1500)},
        {"content": [{"type": "text", "text": "The box is 1.2 now."},
                     {"type": "tool_use", "id": "t8", "name": "view_image", "input": {"path": "shots/look.png"}},
                     bash("t9", 'finish --receipt "$R" --verdict complete --reason "fixed the handle"')],
         "stop_reason": "tool_use", "usage": usage(900, 300, read=7500, write=700)},
    ]  # no fourth answer: a fourth request fails the stand-in and the session
    (tmp_path / "script.json").write_text(json.dumps(script))
    real, built = C.claude_command, []

    def stand_in(*args):
        built.append(real(*args))
        return [sys.executable, str(tmp_path / "stand_in.py"), str(tmp_path / "script.json"),
                str(tmp_path / "requests.jsonl"), *built[-1][3:]]

    monkeypatch.setattr(C, "claude_command", stand_in)
    monkeypatch.setattr(C, "blender_command", lambda root, sif: FAKE_BLENDER)
    r = C.run_container_session(root, "/sif/agent.sif", key_file, **{**FAST, "finish_wait_s": 10.0})

    assert built[0][-7:] == ["--effort", "low", "--fast", "--image-side", "1568", "--system", str(manual)]
    assert C.agent_info(root) == {"agent": "claude", "model": "claude-opus-5-5", "effort": "low", "fast": True,
                                  "image_side": 1568, "system": str(manual)}
    assert (root / "session" / "system.md").read_bytes() == manual.read_bytes()
    assert (r["agent"], r["model"], r["effort"], r["source"], r["verdict"]) == ("claude", "claude-opus-5-5", "low", "finish", "complete")
    assert r["glb"] == "work/exports/001.glb" and edge(root / r["glb"]) == 1.2 and r["counts"]["reviews"] == 1
    assert (r["exit"], r["returncode"], r["cli_exit"], r["end_reason"], r["turns"]) == ("exited", 0, "ended", "finished", 3)
    tokens = {k: sum(s["usage"][k] for s in script) for k in C.PRICES}
    assert r["tokens"] == tokens  # fast mode doubles the price
    assert r["cost_usd"] == pytest.approx(2 * sum(tokens[k] * C.PRICES[k] for k in tokens) / 1e6, abs=1e-4)
    assert (root / "work" / "edit.py").read_text() == "size = 1.2\n"

    reqs = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert [q["messages"] for q in reqs] == [1, 3, 5]
    assert all(q["output_config"] == {"effort": "low"} and q["speed"] == "fast" and q["cache_control"] == {"type": "ephemeral"}
               and q["betas"] == ["server-side-fallback-2026-07-01", "fast-mode-2026-02-01"] for q in reqs)
    assert all(q["system"] == [{"type": "text", "text": manual.read_text(encoding="utf-8"),
                                "cache_control": {"type": "ephemeral"}}] for q in reqs)
    assert all("larger than 1568 px on the long side" in q["view_image"] for q in reqs)
    hist = reqs[-1]["history"]
    assert [q["history"] for q in reqs[:-1]] == [hist[:1], hist[:3]]  # append-only
    assert hist[0] == {"role": "user", "content": PROMPT}
    assert [m["role"] for m in hist] == ["user", "assistant", "user", "assistant", "user"]
    assert [b["type"] for b in hist[1]["content"]] == ["thinking", "tool_use", "tool_use", "tool_use"]
    assert hist[1]["content"][0]["signature"] == "sig-1"
    out = {b["tool_use_id"]: b for m in hist[2::2] for b in m["content"]}
    assert list(out) == [f"t{i}" for i in range(1, 8)] and not any(b.get("is_error") for b in out.values())
    lines = out["t1"]["content"].splitlines()
    assert lines[0].endswith("/work/input") and lines[1:] == ["0", "socket set", str(C.AGENT_DIR / "bin" / "review")]
    assert out["t2"]["content"].startswith("     1\t") and '"mode": "refinement"' in out["t2"]["content"]
    assert out["t3"]["content"] == "one in input\n" and out["t4"]["content"].startswith("created /work/edit.py")
    assert out["t5"]["content"].startswith("edited /work/edit.py:") and "size = 1.2" in out["t5"]["content"]
    assert out["t6"]["content"].startswith("receipt r001-")
    assert out["t7"]["content"] == "ANTHROPIC_API_KEY=[redacted]\n"

    log = root / "session" / "claude.jsonl"
    results = {e["tool_use_id"]: e for e in C._jsonl(log) if e.get("type") == "tool_result"}
    assert (results["t8"]["is_error"], results["t8"]["output"]) == (False, "[image 3000x1500]")
    start = next(e for e in C._jsonl(log) if e.get("type") == "start")
    assert (start["image_side"], start["system"]) == (1568, str(manual))
    assert results["t9"]["output"].startswith("finished: complete, exports/001.glb")
    assert r["key_found_in"] == [] and r["network_commands"] == []
    assert [p for p in all_files(tmp_path) if ANTHROPIC_KEY.encode() in p.read_bytes()] == [key_file]


def test_claude_session_layout_and_agent_record(tmp_path):
    work = prepare(tmp_path, "from_scratch", agent="claude", effort="max", fast=True)
    root = work.parent
    assert json.loads((root / "control" / "agent.json").read_text()) == {
        "agent": "claude", "model": "claude-opus-5-5", "effort": "max", "fast": True}
    assert not (work / "AGENTS.md").exists() and not (root / "session" / "codex_home").exists()
    assert (root / "session" / "system.md").read_text() == (C.AGENT_DIR / "claude" / "SYSTEM.md").read_text()
    assert (root / "session" / "prompt.txt").read_text(encoding="utf-8") == PROMPT
    assert C.agent_info(root)["effort"] == "max"
    codex = prepare(tmp_path / "codex", "from_scratch").parent
    assert C.agent_info(codex) == {"agent": "codex", "model": "gpt-6-astra", "effort": "xhigh", "fast": False}
    (codex / "control" / "agent.json").unlink()  # a session from before the choice
    assert C.agent_info(codex)["agent"] == "codex"
    manual = tmp_path / "system_v1.md"
    manual.write_text("old manual\n")
    for bad, why in [({"agent": "gpt"}, "agent is codex or claude"), ({"agent": "claude", "effort": "extreme"}, "effort is one of"),
                     ({"effort": "high"}, "settings of the claude agent"), ({"fast": True}, "settings of the claude agent"),
                     ({"image_side": 1568}, "settings of the claude agent"), ({"system": manual}, "settings of the claude agent"),
                     *[({"agent": "claude", "image_side": side}, "image_side is a positive number of pixels")
                       for side in (0, -1, 1.5, True, "1568")],
                     ({"agent": "claude", "system": tmp_path / "missing.md"}, "is not a file"),
                     ({"agent": "claude", "system": tmp_path}, "is not a file"),
                     ({"agent": "claude", "codex_base_url": "http://127.0.0.1:1/v1"}, "setting of the codex agent")]:
        with pytest.raises(ValueError, match=why):
            C.prepare_session(tmp_path / "bad", "from_scratch", "p", [], source="s", metric=False, **bad)
        assert not (tmp_path / "bad").exists()


def test_claude_image_side_and_system_go_to_the_record(tmp_path, monkeypatch):
    """image_side and system are written to control/agent.json when given (system as an absolute path,
    copied to session/system.md), and run_container_session passes them to claude_command; without them
    the record and the command are as before."""
    (tmp_path / "runs" / "opus_dig").mkdir(parents=True)
    (tmp_path / "runs" / "opus_dig" / "system_v1.md").write_text("old manual, 1568 px\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    work = C.prepare_session(tmp_path / "s", "from_scratch", "p", [], source="s", metric=False, agent="claude",
                             image_side=1568, system="runs/opus_dig/system_v1.md")
    root, manual = work.parent, str(tmp_path / "runs" / "opus_dig" / "system_v1.md")
    assert json.loads((root / "control" / "agent.json").read_text()) == {
        "agent": "claude", "model": "claude-opus-5-5", "effort": "xhigh", "fast": False, "image_side": 1568, "system": manual}
    assert (root / "session" / "system.md").read_text(encoding="utf-8") == "old manual, 1568 px\n"
    assert C.agent_record("claude", image_side=2576) == {"agent": "claude", "model": "claude-opus-5-5", "effort": "xhigh",
                                                         "fast": False, "image_side": 2576}
    plain = C.prepare_session(tmp_path / "p", "from_scratch", "p", [], source="s", metric=False, agent="claude").parent
    assert C.agent_info(plain) == {"agent": "claude", "model": "claude-opus-5-5", "effort": "xhigh", "fast": False}
    assert (plain / "session" / "system.md").read_bytes() == (C.AGENT_DIR / "claude" / "SYSTEM.md").read_bytes()
    built = []
    monkeypatch.setattr(C, "run_session", lambda root, cmd, env, **kw: built.append(cmd) or {})
    (tmp_path / "key.env").write_text(f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}\n")
    for r in (root, plain):
        C.run_container_session(r, "/sif/agent.sif", tmp_path / "key.env")
    assert built[0][-6:] == ["--effort", "xhigh", "--image-side", "1568", "--system", manual]
    assert built[1][-2:] == ["--effort", "xhigh"] and "--image-side" not in built[1] and "--system" not in built[1]


def test_claude_shell_and_agent_commands(tmp_path):
    root = tmp_path / "s"
    shell = C.shell_command(root, "/sif/agent.sif", "/tmp/gator-x")
    codex = C.apptainer_command(root, "/sif/agent.sif", "/home/u/key.env", "/tmp/gator-x")
    assert shell[:8] == C.CONTAINER and shell[8:11] == ["--net", "--network", "none"]
    binds = [shell[i + 1] for i, a in enumerate(shell) if a == "--bind"]
    # Codex's binds without /session: claude_agent's log is there, and result.json trusts it.
    assert binds == [b for i, a in enumerate(codex) if a == "--bind" for b in [codex[i + 1]] if ":/session" not in b]
    assert binds == [f"{root.resolve()}/work:/work", f"{C.AGENT_DIR}/bin:/opt/gator/bin:ro", "/tmp/gator-x:/opt/gator/sock:ro"]
    assert "control" not in " ".join(binds) and "session" not in " ".join(shell)
    envs = [shell[i + 1] for i, a in enumerate(shell) if a == "--env"]
    assert envs == ["GATOR_SOCKET=/opt/gator/sock/tools.sock", "PREPEND_PATH=/opt/gator/bin"]
    assert "--env-file" not in shell and "/home/u/key.env" not in shell
    assert shell[-6:] == ["--pwd", "/work", "/sif/agent.sif", "bash", "--noprofile", "--norc"]
    cmd = C.claude_command("s", "agent.sif", "/tmp/gator-x", "key.env")
    assert cmd == [sys.executable, "-m", "gator.agent.claude_agent", os.path.abspath("s"), "--sif", os.path.abspath("agent.sif"),
                   "--sock-dir", "/tmp/gator-x", "--key-file", os.path.abspath("key.env"), "--effort", "xhigh"]
    assert C.claude_command(root, "a.sif", "/tmp/x", "/k.env", "low", fast=True)[-3:] == ["--effort", "low", "--fast"]
    assert C.claude_command(root, "a.sif", "/tmp/x", "/k.env", "low", True, 1568, "v1.md")[-7:] == \
        ["--effort", "low", "--fast", "--image-side", "1568", "--system", os.path.abspath("v1.md")]
    assert C.claude_command(root, "a.sif", "/tmp/x", "/k.env", image_side=2576)[-4:] == \
        ["--effort", "xhigh", "--image-side", "2576"]
    assert [C.cli_exit(c, "claude") for c in (0, 2, 3, 130, 1, -signal.SIGKILL)] == \
        ["ended", "refusal", "api_error", "interrupted", "exit 1", "signal SIGKILL"]


def test_claude_agent_override_runs_from_the_repo(tmp_path):
    """agent= replaces claude_agent's whole host command: it runs from the repository with no stdin, its
    stdout goes to claude_stdout.log, and extra_env reaches it. Without an end line the usage is summed."""
    root = prepare(tmp_path, "from_scratch", agent="claude").parent
    (tmp_path / "key.env").write_text(f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}\n")
    script = ("import json, os, sys, time\n"
              "print('cwd', os.getcwd(), 'stdin', repr(sys.stdin.read()), 'seen', os.environ['SEEN'])\n"
              "line = {'t': time.time(), 'type': 'response', 'usage': {'input_tokens': 1000, 'output_tokens': 100, "
              "'cache_read_input_tokens': 2000, 'cache_creation_input_tokens': None}, 'blocks': [{'type': 'tool_use', "
              "'id': 't1', 'name': 'bash', 'input': {'command': 'curl -O https://x/y.png'}}]}\n"
              "open(sys.argv[1], 'a').write(json.dumps(line) + '\\n')\n"
              "sys.exit(2)\n")
    r = C.run_container_session(root, "/sif/agent.sif", tmp_path / "key.env", extra_env={"SEEN": "yes"},
                                agent=[sys.executable, "-c", script, str(root / "session" / "claude.jsonl")], **FAST)
    assert (r["exit"], r["returncode"], r["cli_exit"], r["verdict"]) == ("exited", 2, "refusal", "invalid")
    assert (root / "session" / "claude_stdout.log").read_text() == f"cwd {C.REPO} stdin '' seen yes\n"
    assert (r["turns"], r["end_reason"], r["network_commands"]) == (1, None, ["curl -O https://x/y.png"])
    assert r["tokens"] == usage(1000, 100, read=2000) and r["cost_usd"] == pytest.approx(0.0064)


def test_claude_usage_prefers_the_end_line(tmp_path):
    log = tmp_path / "claude.jsonl"
    response = {"type": "response", "usage": usage(1_000_000, 0)}
    log.write_text(json.dumps(response) + "\nnot json\n[1]\n" + json.dumps({**response, "usage": "bad"}) + "\n")
    assert C.claude_usage(log)["cost_usd"] == 4.0 and C.claude_usage(log, fast=True)["cost_usd"] == 8.0
    end = {"type": "end", "reason": "end_turn", "turns": 7, "totals": usage(5, 6, 7, 8), "cost_usd": 1.5}
    with open(log, "a") as f:
        f.write(json.dumps(end) + "\n")
    assert C.claude_usage(log) == {"turns": 7, "tokens": usage(5, 6, 7, 8), "cost_usd": 1.5, "end_reason": "end_turn"}
    assert C.claude_usage(tmp_path / "missing.jsonl")["tokens"] == usage(0, 0)


def test_claude_usage_without_an_end_line_takes_each_lines_cost(tmp_path):
    """claude_agent prices each response by the models that ran its attempts, and logs every attempt's
    tokens as billed when a refusal fallback ran; the controller sums those, and prices only a line
    without its own cost at PRICES."""
    log = tmp_path / "claude.jsonl"
    lines = [{"type": "response", "usage": usage(100, 40), "billed": usage(200, 540), "cost_usd": 0.0126},
             {"type": "response", "usage": usage(10, 20), "cost_usd": 0.003175},
             {"type": "response", "usage": usage(1_000_000, 0)}]
    log.write_text("".join(json.dumps(e) + "\n" for e in lines))
    r = C.claude_usage(log)
    assert r["tokens"] == usage(1_000_210, 560) and r["turns"] == 3
    assert r["cost_usd"] == round(0.0126 + 0.003175 + 4.0, 4)


def test_find_key_reads_anthropic_keys(tmp_path):
    (tmp_path / "key.env").write_text(f"# for the claude agent\nANTHROPIC_API_KEY={ANTHROPIC_KEY}\n")
    root = tmp_path / "s"
    (root / "work").mkdir(parents=True)
    (root / "work" / "env.txt").write_text(f"X={ANTHROPIC_KEY}\n")
    (root / "work" / "other.txt").write_text("sk-ant-something-else")
    assert C.find_key(root, tmp_path / "key.env") == ["work/env.txt"]
    (tmp_path / "quoted.env").write_text(f'ANTHROPIC_API_KEY="{ANTHROPIC_KEY}"\n')  # the shell and claude_agent unquote it
    assert C.read_key(tmp_path / "quoted.env") == ANTHROPIC_KEY and C.find_key(root, tmp_path / "quoted.env") == ["work/env.txt"]
    (tmp_path / "none.env").write_text("OTHER=1\n")
    assert C.read_key(tmp_path / "none.env") == "" and C.find_key(root, tmp_path / "none.env") == []


def test_find_key_checks_every_key_of_a_shared_env_file(tmp_path):
    """An env file can hold both keys, and the Codex container gets both through --env-file: a file
    with either key is reported, whichever line comes first."""
    openai_key = "sk-proj-TEST-" + "fedcba9876543210" * 3
    root = tmp_path / "s"
    (root / "work").mkdir(parents=True)
    (root / "work" / "openai.txt").write_text(f"token {openai_key}\n")
    (root / "work" / "anthropic.txt").write_text(f"token {ANTHROPIC_KEY}\n")
    (root / "work" / "clean.txt").write_text("nothing\n")
    for order in ((ANTHROPIC_KEY, openai_key), (openai_key, ANTHROPIC_KEY)):
        lines = {ANTHROPIC_KEY: f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}", openai_key: f'OPENAI_API_KEY="{openai_key}"'}
        (tmp_path / "both.env").write_text("".join(lines[k] + "\n" for k in order))
        assert C.read_keys(tmp_path / "both.env") == list(order) and C.read_key(tmp_path / "both.env") == order[0]
        assert C.find_key(root, tmp_path / "both.env") == ["work/anthropic.txt", "work/openai.txt"]


def test_tool_server_does_not_import_from_the_work_folder(tmp_path):
    """The model can write any file into /work. The helper runs on the host, outside the agent's
    container, so a json.py or a gator/ package there must never be imported."""
    work = prepare(tmp_path, "refinement")
    marker = tmp_path / "HOST_RAN.txt"
    plant = f"open({str(marker)!r}, 'a').write(__name__ + chr(10))\n"
    (work / "json.py").write_text(plant + "from json_real_is_gone import x\n")
    (work / "gator" / "agent").mkdir(parents=True)
    for f in ("gator/__init__.py", "gator/agent/__init__.py", "gator/agent/tools.py"):
        (work / f).write_text(plant)
    with tempfile.TemporaryDirectory(prefix="gator-") as d:
        server = ToolServer(os.path.join(d, "tools.sock"), work, work.parent / "control", FAKE_BLENDER)
        try:
            output, code = server.run(["finish", "--receipt", "r9", "--verdict", "complete", "--reason", "x"])
        finally:
            server.stop()
    assert code == 1 and "unknown review receipt 'r9'" in output
    assert not marker.exists()
