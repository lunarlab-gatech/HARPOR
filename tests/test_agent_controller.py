"""Scripted agent sessions through the controller and its tool server (build_plan.md note 5).

The scripted sessions run in parallel in one fixture, because each fake Blender export imports trimesh.
"""

import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from harpor import frames as F
from harpor.agent import controller as C
from harpor.agent import prompts as PR
from harpor.agent import tools as T
from harpor.agent import validate as V
from harpor.agent.bin import gator_tool
from harpor.agent.fake_agent import write_glb
from harpor.agent.server import ToolServer

REPO = Path(__file__).resolve().parents[1]
FAST = {"poll_s": 0.02, "kill_after_s": 0.5, "finish_wait_s": 0.5}
FAKE_BLENDER = [sys.executable, "-m", "harpor.agent.fake_agent", "--blender"]
LONG = 20.0
# The fake original's edge. Every GLB a session below delivers keeps its size (C.SIZE_RATIO: 0.725 to 2.9).
ORIGINAL_SIZE = 1.45
PROMPT = "Prompt 1 text, byte for byte: é\n"
# Agent text with OSC 52 (a clipboard write), OSC 0 (a window title), CSI, a bidi override and line breaks, and the
# one line that T.printable keeps of it. The webapp tests use them too.
DIRTY = "Café\x1b]52;c;aGk=\x07 fixed\x1b]0;title\x07 the\x1b[2J lid‮ gap\nsecond line\r\n\tthird"
CLEAN = "Café]52;c;aGk= fixed]0;title the[2J lid gap second line third"


def edit(size, kind="good", seconds=0.0):
    return {"do": "edit", "size": size, "kind": kind, "seconds": seconds}


def tool(*args):
    return {"do": "tool", "args": list(args)}


def finish(verdict, *limitations):
    lims = [x for lim in limitations for x in ("--limitation", lim)]
    return tool("finish", "--receipt", "{receipt}", "--verdict", verdict, "--reason", "fixed the handle", *lims)


def checkpoint(defect):
    return tool("checkpoint", "--receipt", "{receipt}", "--defect", defect)


REVIEW, EXPORT = tool("review"), tool("export_glb")
CRASH, SLEEP = {"do": "crash"}, {"do": "sleep", "seconds": 30}

SESSIONS = {
    "finish_refine": (LONG, [{"do": "read_prompt"}, edit(1.2), REVIEW,
                                           finish("complete", "seam on the back", "thin lid")]),
    "finish_unchanged": (LONG, [edit(ORIGINAL_SIZE), REVIEW, finish("unchanged")]),
    "finish_limitations": (LONG, [edit(1.35), REVIEW, finish("needs_review", "a", "b", "c", "d"),
                                                finish("needs_review", "a", "b", "c")]),
    "finish_bad_verdict": (LONG, [edit(1.55), REVIEW, finish("articulated")]),  # a verdict of the old articulation
    "finish_then_hang": (LONG, [edit(1.2), REVIEW, finish("complete"), SLEEP]),
    "checkpoint_fallback": (LONG, [edit(1.5), REVIEW, checkpoint("spout still missing"),
                                                 edit(1.75), EXPORT, CRASH]),
    "checkpoint_deadline": (1.5, [edit(1.6), REVIEW, checkpoint("lid gap"), SLEEP]),
    "export_fallback": (LONG, [edit(1.7), EXPORT, CRASH]),
    "newest_valid_export": (LONG, [edit(1.65), EXPORT,
                                                 {"do": "glb", "path": "exports/002.glb", "kind": "garbage"}, CRASH]),
    "original_fallback": (LONG, [{"do": "glb", "path": "original/original.glb", "kind": "garbage"},
                                               CRASH]),
    "invalid_export": (LONG, [edit(1.0, "garbage"), EXPORT, CRASH]),
    "deadline_sigint": (1.0, [SLEEP]),
    "deadline_kill": (1.0, [{"do": "ignore_sigint"}, SLEEP]),
    "budget_tamper": (1.0, [{"do": "write", "path": "run_budget.json",
                                           "text": '{"deadline": 99999999999}'}, SLEEP]),
    "deadline_extension": (5.0, [edit(1.75, seconds=5.5), EXPORT, REVIEW, finish("complete")]),
    "export_cache": (LONG, [edit(1.25), EXPORT, EXPORT, REVIEW, finish("complete")]),
    "nan_finish": (LONG, [edit(1.3), EXPORT, edit(1.3, "nan"), REVIEW, finish("complete")]),
    "no_material_checkpoint": (LONG, [edit(1.4), REVIEW, checkpoint("old"), edit(1.4, "no_material"),
                                                    REVIEW, checkpoint("new"), CRASH]),
    "garbage_finish": (LONG, [edit(1.0, "garbage"), REVIEW, finish("complete")]),
    "tampered_finish": (LONG, [edit(1.8), REVIEW, finish("complete"),
                                             {"do": "write", "path": "exports/001.glb", "text": "tampered"}]),
    "unchanged_after_original_edit": (LONG, [{"do": "glb", "path": "original/original.glb", "size": 2.0},
                                                           edit(ORIGINAL_SIZE), REVIEW, finish("unchanged")]),
    "forged_event": (1.0, [{"do": "write", "path": ".gator/events.jsonl",
                                          "text": '{"event": "export_done", "export": 1, "seconds": 100}\n'},
                                         {"do": "sleep", "seconds": 5}]),
    "stop_during_export": (LONG, [edit(1.0, seconds=3.0), {"do": "tool_nowait", "args": ["export_glb"]},
                                                CRASH]),
    "dirty_text": (LONG, [edit(1.2), REVIEW, checkpoint(DIRTY),
                                        tool("finish", "--receipt", "{receipt}", "--verdict", "complete", "--reason", DIRTY,
                                             "--limitation", DIRTY, "--limitation", "plain text, é and ü")]),
    "dirty_checkpoint": (LONG, [edit(1.3), REVIEW, checkpoint(DIRTY), CRASH]),
}


def make_views(folder, n=2):
    folder.mkdir(parents=True, exist_ok=True)
    K = [[300.0, 0, 31.5], [0, 300.0, 23.5], [0, 0, 1]]
    views = []
    for i in range(n):
        (folder / f"img{i}.png").write_bytes(b"png %d" % i)
        (folder / f"mask{i}.png").write_bytes(b"mask %d" % i)
        w2c = F.rigid(np.eye(3), [0.0, 0.0, 0.5 * i])
        views.append({"image": folder / f"img{i}.png", "mask": folder / f"mask{i}.png", "K": K, "w2c": w2c,
                      "width": 64, "height": 48})
    return views


def prepare(root, **agent):
    """A refinement session of the fake box of edge ORIGINAL_SIZE."""
    original = root / "in" / "original.glb"
    (root / "in").mkdir(parents=True)
    write_glb(original, "good", ORIGINAL_SIZE)
    return C.prepare_session(root / "s", PROMPT, make_views(root / "in"), source="sensor depth", metric=True,
                             original=original, **agent)


def run_scripted(root, budget_s, script):
    prepare(root)
    (root / "script.json").write_text(json.dumps(script))
    with tempfile.TemporaryDirectory(prefix="gator-") as sock_dir:
        env = {**os.environ, "PYTHONPATH": str(REPO), "GATOR_SOCKET": os.path.join(sock_dir, "tools.sock")}
        cmd = [sys.executable, "-m", "harpor.agent.fake_agent", str(root / "script.json")]
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
    assert r["candidate_built"] is True
    assert r["previews_built"] is True and (root / "work" / "original" / "previews" / "sheet.png").is_file()
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


def test_finish_rejects_fourth_limitation(runs):
    root, r = runs["finish_limitations"]
    assert (r["source"], r["verdict"], r["limitations"]) == ("finish", "needs_review", ["a", "b", "c"])
    assert "4 limitations given; at most 3" in agent_log(root)


def test_an_unknown_verdict_is_rejected(runs):
    root, r = runs["finish_bad_verdict"]
    assert "unknown verdict 'articulated'; use one of complete, unchanged, needs_review" in agent_log(root)
    assert not (root / "work" / "finish.json").exists()
    assert (r["source"], r["verdict"]) == ("export", "needs_review") and edge(root / r["glb"]) == 1.55


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
    assert r["glb"] == "work/exports/001.glb" and edge(root / r["glb"]) == 1.65
    assert r["rejected"] == [] and (root / "work" / "exports" / "002.glb").exists()


def test_original_fallback_uses_controller_copy(runs):
    root, r = runs["original_fallback"]
    assert (r["source"], r["verdict"], r["glb"]) == ("fallback_original", "fallback_original", "original.glb")
    assert edge(root / r["glb"]) == ORIGINAL_SIZE


def test_invalid_export_is_rejected_for_the_original(runs):
    root, r = runs["invalid_export"]
    assert (r["source"], r["verdict"], r["glb"]) == ("fallback_original", "fallback_original", "original.glb")
    assert [x["from"] for x in r["rejected"]] == ["export"]


def test_deadline_sigint(runs):
    root, r = runs["deadline_sigint"]
    assert (r["exit"], r["returncode"], r["verdict"]) == ("deadline_sigint", -signal.SIGINT, "fallback_original")
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
    assert t["wall_s"] > t["budget_s"] and t["excluded_s"] >= 5.5
    assert t["deadline_s"] == pytest.approx(t["budget_s"] + t["excluded_s"])
    done = [e for e in T.read_events(root / "control") if e["event"] == "export_done"]
    budget = json.loads((root / "work" / "run_budget.json").read_text())
    assert len(done) == 1 and budget["excluded_s"] == done[0]["seconds"] == t["excluded_s"]
    assert budget["deadline"] == pytest.approx(budget["started_at"] + t["budget_s"] + t["excluded_s"])
    assert budget["exports"] == [{"export": 1, "seconds": done[0]["seconds"]}]


def test_a_group_of_finished_processes_is_no_error(tmp_path, monkeypatch):
    """macOS answers a signal to a group of finished processes with EPERM, where Linux sends it to nobody, and that
    is the normal end of a session whose agent has just exited."""
    def refused(pgid, sig):
        raise PermissionError(1, "Operation not permitted")

    p = SimpleNamespace(pid=12345, send_signal=lambda sig: None, wait=lambda timeout=None: 0)
    work = prepare(tmp_path)
    with tempfile.TemporaryDirectory(prefix="gator-") as d:
        server = ToolServer(os.path.join(d, "tools.sock"), work, work.parent / "control", FAKE_BLENDER)
        monkeypatch.setattr(os, "killpg", refused)
        C._signal_group(p, signal.SIGINT)
        T._kill_group(p)
        server._proc = p
        server.stop()
    monkeypatch.setattr(T, "_children", [p])
    with pytest.raises(SystemExit) as e:
        T._on_sigterm(signal.SIGTERM, None)
    assert e.value.code == 128 + signal.SIGTERM


def test_unchanged_scene_returns_cached_export(runs):
    root, r = runs["export_cache"]
    assert r["source"] == "finish" and r["counts"]["exports"] == 1
    assert sum(e["event"] == "export_done" for e in T.read_events(root / "control")) == 1
    assert agent_log(root).count("(cached, scene unchanged)") == 1


def test_helpers_print_time_left(runs):
    # export_cache: export, cached export, review, finish. checkpoint_fallback: review, checkpoint, export.
    # deadline_extension: a 5.5 s export, review, finish.
    for name, n in (("export_cache", 3), ("checkpoint_fallback", 3), ("deadline_extension", 2)):
        root, r = runs[name]
        t = r["timings"]
        lines = [x for x in agent_log(root).splitlines() if x.startswith("time left: ")]
        assert len(lines) == n and all(x.endswith(f" s of the {t['budget_s']:.0f} s budget") for x in lines)
        # Outside an export the controller's time left only drops, and an export does not use any of it, so
        # each printed value lies between what was left when the session ended and the budget.
        assert all(t["deadline_s"] - t["wall_s"] - 0.5 <= float(x.split()[2]) <= t["budget_s"] + 0.5 for x in lines)


def test_failed_helper_also_prints_time_left(tmp_path, monkeypatch):
    work = prepare(tmp_path)
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


def test_printable_keeps_one_line_of_printable_characters():
    for text, clean in [("plain text", "plain text"), ("Café crème, naïve, Ærø, 日本", "Café crème, naïve, Ærø, 日本"),
                        ("é", "é"),  # a combining accent
                        ("a\tb\nc\r\nd e\x85f\x0bg", "a b c d e f g"),
                        ("\x1b]52;c;aGk=\x07x", "]52;c;aGk=x"), ("\x1b]0;title\x1b\\y", "]0;title\\y"),  # OSC 52, OSC 0
                        ("\x1b[2J\x1b[31mred\x1b[0m", "[2J[31mred[0m"), ("\x9b31m", "31m"),  # CSI, and the C1 CSI
                        ("‮evil‬ ​", "evil"),  # a bidi override and a zero-width space (Cf)
                        ("a\x7fb\x00c", "abc"), ("x\ud800͸y", "xy"),  # a surrogate, unassigned, private use
                        ("  spaced \x1b  out  ", "spaced out"), (DIRTY, CLEAN)]:
        assert T.printable(text) == clean, repr(text)


def test_agent_text_is_stored_as_one_printable_line(runs):
    """OSC 52, OSC 0, CSI, a bidi override and line breaks in finish's reason and limitations and in checkpoint's
    defect, sent through the tool server as any agent sends them, are stored as one line without them (T.printable):
    in finish.json, the checkpoint's record, the events and result.json. Plain text and accented letters stay."""
    root, r = runs["dirty_text"]
    events = {e["event"]: e for e in T.read_events(root / "control")}
    fin = json.loads((root / "work" / "finish.json").read_text())
    assert fin["reason"] == events["finish"]["reason"] == r["reason"] == CLEAN
    assert fin["limitations"] == events["finish"]["limitations"] == r["limitations"] == [CLEAN, "plain text, é and ü"]
    meta = json.loads((root / "work" / events["checkpoint"]["glb"]).with_suffix(".json").read_text())
    assert meta["defect"] == events["checkpoint"]["defect"] == CLEAN and f"defect: {CLEAN}\n" in agent_log(root)
    root, r = runs["dirty_checkpoint"]
    assert (r["source"], r["reason"]) == ("checkpoint", CLEAN)


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
    work = prepare(tmp_path)
    root = work.parent
    assert (root / "session" / "codex_home" / "config.toml").exists()
    assert "finish --receipt ID" in (work / "AGENTS.md").read_text()
    assert (root / "control" / "events.jsonl").exists() and not (work / ".gator").exists()
    for d in ("exports", "staging"):  # the helpers' Blender container binds them, so they must exist
        assert (root / "control" / d).is_dir()
    assert json.loads((root / "control" / "manifest.json").read_text()) == json.loads((work / "input" / "manifest.json").read_text())
    m = json.loads((work / "input" / "manifest.json").read_text())
    assert m["units"] == "meters"
    assert set(m) == {"frame", "glb_axes", "w_from_gltf", "cameras", "source", "units", "views"}
    assert np.allclose(m["w_from_gltf"], F.W_FROM_GLTF)
    v = m["views"][1]
    assert v["image"] == "input/images/001.png" and (work / v["image"]).read_bytes() == b"png 1"
    assert set(v) == {"index", "K", "w2c", "width", "height", "image", "mask"}
    assert sorted(p.name for p in (work / "input").iterdir()) == ["images", "manifest.json", "masks"]
    assert np.allclose(v["w2c"], F.rigid(np.eye(3), [0, 0, 0.5])) and v["K"][0][2] == 31.5
    assert not any((work / "original" / "previews").iterdir())  # run_session renders them
    for manual in (work / "AGENTS.md", C.AGENT_DIR / "claude" / "SYSTEM.md"):  # what each input view holds
        assert "under `input/` (`image` and `mask`)" in " ".join(manual.read_text().split())
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
    assert T.read_events(tmp_path / "none") == [] and list(T.read_jsonl(tmp_path / "none.jsonl")) == []
    (tmp_path / "bad" / "events.jsonl").mkdir(parents=True)  # there, but not readable as a file
    with pytest.raises(OSError):
        T.read_events(tmp_path / "bad")
    assert list(T.read_jsonl(tmp_path / "bad" / "events.jsonl")) == []


def test_watchdog_error_still_kills_agent(tmp_path, monkeypatch):
    work = prepare(tmp_path)
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


def test_export_launch_failure_is_logged_and_not_excluded(tmp_path):
    work = prepare(tmp_path)
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
    work = prepare(tmp_path)
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
    pick, rejected = C.select_deliverable(root)
    assert (pick["source"], pick["glb"], rejected) == ("export", work / "exports" / "001.glb", [])
    T.log_event(control, "export_done", export=2, seconds=1.0, ok=True, glb="exports/fifo.glb",
                blend_sha256="b", glb_sha256="0" * 64)
    pick, rejected = C.select_deliverable(root)
    assert pick["glb"] == ex / "001.glb" and rejected[0]["why"] == "missing or not a regular file after export"


def test_a_glb_of_another_size_is_not_delivered(tmp_path, monkeypatch):
    """A GLB whose box, after the node transforms, grew past SIZE_RATIO of root/original.glb's on every side,
    or shrank below it on every side, is not delivered (a unit slip in Blender): finish refuses it, so the
    agent sees why, and select_deliverable rejects it, and the fallback order goes on. A repair that
    lengthens only some sides, such as legs under a bare top, keeps the size."""
    work = prepare(tmp_path)
    root, control, ex = work.parent, work.parent / "control", work / "exports"
    for n, size in ((1, 1.2 * ORIGINAL_SIZE), (2, 10 * ORIGINAL_SIZE)):
        write_glb(ex / f"00{n}.glb", size=size)
        sha = T.sha256(ex / f"00{n}.glb")
        T.log_event(control, "export_done", export=n, seconds=1.0, ok=True, glb=f"exports/00{n}.glb",
                    blend_sha256="b", glb_sha256=sha)
        T.log_event(control, "review", receipt=f"r00{n}-{sha[:12]}", glb=f"exports/00{n}.glb", sha256=sha)
    why = "wrong size: every side of its box is at least 10 times the original's (over 2.0)"
    with pytest.raises(SystemExit, match=re.escape(f"exports/002.glb cannot be delivered: {why}")):
        T.finish(work, control, f"r002-{sha[:12]}", "complete", "fixed the handle")
    with monkeypatch.context() as m:  # a finish that skipped the check: the controller still rejects it
        m.setattr(C, "check_size", lambda *a: None)
        T.finish(work, control, f"r002-{sha[:12]}", "complete", "fixed the handle")
    pick, rejected = C.select_deliverable(root)
    assert (pick["source"], pick["glb"]) == ("export", ex / "001.glb")
    assert rejected == [{"from": "finish", "glb": "work/exports/002.glb", "why": why},
                        {"from": "export", "glb": "work/exports/002.glb", "why": why}]

    original = root / "original.glb"
    for ratio, ok in ((0.49, False), (0.51, True), (1.99, True), (2.01, False)):
        write_glb(tmp_path / "x.glb", size=ratio * ORIGINAL_SIZE)
        assert (C.check_size(tmp_path / "x.glb", original) is None) == ok, ratio

    def scaled(*k):  # the original's mesh, scaled through its node
        def edit(g):
            g["nodes"][0]["scale"] = list(k)
            return g
        return edit

    glb_json(original, tmp_path / "node.glb", scaled(10.0, 10.0, 10.0))
    assert C.check_glb(tmp_path / "node.glb") is None and C.check_size(tmp_path / "node.glb", original) == why
    for k, ok in (((1.0, 3.0, 5.0), True), ((2.54, 2.54, 2.54), False), ((1.4, 3.0, 4.0), True),
                  ((0.3, 0.3, 0.3), False), ((0.3, 1.0, 0.3), True)):
        glb_json(original, tmp_path / "node.glb", scaled(*k))
        assert (C.check_size(tmp_path / "node.glb", original) is None) == ok, k
    glb_json(original, tmp_path / "node.glb", scaled(0.3, 0.4, 0.45))
    assert C.check_size(tmp_path / "node.glb", original) == (
        "wrong size: every side of its box is at most 0.45 times the original's (under 0.5)")
    assert C.check_size(tmp_path / "missing.glb", original).startswith("size not checked")



def test_export_timeout_counts_exactly_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    work = prepare(tmp_path)
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
    work = prepare(tmp_path)
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
                     f"{C.AGENT_DIR}/bin:/opt/gator/bin:ro", "/tmp/gator-x:/opt/gator/sock:ro",
                     "/etc/resolv.conf:/etc/resolv.conf:ro", "/etc/hosts:/etc/hosts:ro"]
    assert cmd[:len(C.CONTAINER)] == C.CONTAINER and "--containall" in cmd and "control" not in " ".join(binds)
    envs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--env"]
    assert envs == ["CODEX_HOME=/session/codex_home", "GATOR_SOCKET=/opt/gator/sock/tools.sock",
                    "PREPEND_PATH=/opt/gator/bin"]
    assert cmd[cmd.index("--env-file") + 1] == "/home/u/key.env"
    assert cmd[-len(C.codex_command()) - 7:] == ["--pwd", "/work", "/sif/agent.sif", *C.CODEX_KEY_WRAPPER,
                                                 *C.codex_command()]
    assert C.apptainer_command(root, "a.sif", "k", "/tmp/x", agent=["sh", "-c", "true"])[-3:] == ["sh", "-c", "true"]
    with pytest.raises(ValueError, match="cannot bind"):
        C.apptainer_command(tmp_path / "a,b", "a.sif", "k", "/tmp/x")
    blender = C.blender_command(root, "/sif/agent.sif")
    work, control = (root / "work").resolve(), (root / "control").resolve()
    assert blender[-2:] == ["/sif/agent.sif", "blender"] and f"{work}:{work}" in blender and f"{work}:/work" in blender
    # Of control/, the helpers' Blender writes only staging/, where export_glb.py writes, and sees exports/, the
    # host's copies, read-only, and the tool server hands the helpers this command as GATOR_BLENDER.
    assert [b for b in (blender[i + 1] for i, a in enumerate(blender) if a == "--bind") if str(control) in b] == \
        [f"{control}/staging:{control}/staging", f"{control}/exports:{control}/exports:ro"]
    with tempfile.TemporaryDirectory(prefix="gator-") as d:
        server = ToolServer(os.path.join(d, "tools.sock"), root / "work", root / "control", blender)
        server.stop()
    assert shlex.split(server.env["GATOR_BLENDER"]) == blender


def test_note_5_command_is_the_controllers(tmp_path):
    text = (Path(__file__).resolve().parents[1] / "docs" / "build_plan.md").read_text()
    block = text.split("### 5. Stage B", 1)[1].split("```bash\n", 1)[1].split("```", 1)[0]
    root, sock, key, sif = tmp_path / "s", "/tmp/gator-x", "/home/u/.config/harpor/openai.env", "/sif/harpor-agent.sif"
    doc = (block.replace("\\\n", " ").replace("<root>", str(root.resolve())).replace("<socket dir>", sock)
           .replace("~/.config/harpor/openai.env", key).replace(" harpor/agent/bin:", f" {C.AGENT_DIR}/bin:")
           .replace(" harpor-agent.sif", f" {sif}"))
    assert shlex.split(doc) == C.apptainer_command(root, sif, key, sock)
    assert f'AG="{shlex.join(C.CONTAINER)} $G/sif/harpor-agent.sif"' in text  # the sandbox check runs the same flags


def test_only_the_codex_container_binds_the_dns_files(tmp_path):
    """--no-mount bind-paths turns off the site's binds in every container. The Codex CLI needs the network, so its
    container binds the two files itself. The claude shell and the Blender containers get CONTAINER alone."""
    root = tmp_path / "s"

    def binds(cmd):
        return [cmd[i + 1] for i, a in enumerate(cmd) if a == "--bind"]

    assert C.CONTAINER == ["apptainer", "exec", "--nv", "--containall", "--no-mount", "bind-paths"]
    codex = C.apptainer_command(root, "/sif/agent.sif", "/home/u/key.env", "/tmp/gator-x")
    assert binds(codex)[-2:] == ["/etc/resolv.conf:/etc/resolv.conf:ro", "/etc/hosts:/etc/hosts:ro"]
    others = (C.shell_command(root, "/sif/agent.sif", "/tmp/gator-x"), C.blender_command(root, "/sif/agent.sif"),
              C.sif_blender("/sif/agent.sif", rw=["/data"], ro=["/in.glb"]))
    for cmd in others:
        assert cmd[:len(C.CONTAINER)] == C.CONTAINER and not [b for b in binds(cmd) if b.startswith("/etc/")]


def test_cli_exit_names():
    assert [C.cli_exit(c) for c in (0, 1, 130, 143, 41, -signal.SIGINT, None)] == \
        ["success", "error_or_interrupted", "cancelled", "terminated", "exit 41", "signal SIGINT", None]


def test_tool_server_rejects_bad_requests(tmp_path):
    work = prepare(tmp_path)
    with tempfile.TemporaryDirectory(prefix="gator-") as d:
        path = os.path.join(d, "tools.sock")
        server = ToolServer(path, work, work.parent / "control", FAKE_BLENDER).start()
        try:
            assert gator_tool.call(["rm", "-rf", "/"], path) == ("unknown command 'rm'; use one of export_glb, "
                                                                 "review, checkpoint, finish\n", 2)
            assert gator_tool.call([], path)[1] == 2
            # review with --clay and its 4 detail views is the longest command line: 34 arguments
            longest = ["review", "--clay", *["--detail", *"0001001"] * 4]
            assert len(longest) == 34 and "candidate.blend is missing" in gator_tool.call(longest, path)[0]
            assert "argv must be a list of 1 to 64 strings" in gator_tool.call(["review"] * 65, path)[0]
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


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_every_helper_command_has_its_client():
    from harpor.agent.server import COMMANDS

    for name in COMMANDS:
        path = C.AGENT_DIR / "bin" / name
        assert path.read_text() == f'#!/bin/sh\nexec python3 "$(dirname "$0")/gator_tool.py" {name} "$@"\n'
        assert path.stat().st_mode & 0o777 == 0o755, name


def test_export_does_not_follow_links_and_always_logs_done(tmp_path):
    work = prepare(tmp_path)
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


def after_blender_wrote(control, ex, name, then):
    """A thread that runs then() once the fake Blender has written the GLB name (control/staging/, or exports/ as
    before), and before it writes the report: the scene's pause, as the real script re-imports the GLB meanwhile."""
    def wait():
        end = time.time() + LONG
        while not ((control / "staging" / name).exists() or (ex / name).exists()) and time.time() < end:
            time.sleep(0.01)
        then()

    th = threading.Thread(target=wait)
    th.start()
    return th


def test_an_export_is_hashed_as_blender_wrote_it(tmp_path, monkeypatch):
    """An agent that renames its own files over exports/001.glb and 001.json while Blender runs gets neither logged
    nor delivered: Blender writes into control/staging/, out of the agent's reach, and the agent's copies, Blender's
    bytes, replace its files only after the host has hashed them."""
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    work = prepare(tmp_path)
    root, control, ex = work.parent, work.parent / "control", work / "exports"
    (work / "candidate.blend").write_text(json.dumps({"kind": "good", "size": 1.2, "seconds": 0.0, "pause": 1.0}))
    T.start_budget(work, control, time.time(), 600.0)
    write_glb(tmp_path / "agent.glb", size=1.3)
    planted = {"001.glb": (tmp_path / "agent.glb").read_bytes(), "001.json": b'{"meshes": 9, "warnings": []}'}

    def swap():
        for name, data in planted.items():
            (tmp_path / name).write_bytes(data)
            os.replace(tmp_path / name, ex / name)

    th = after_blender_wrote(control, ex, "001.glb", swap)
    glb, rep, cached = T.export_glb(work, control, FAKE_BLENDER)
    th.join()
    done = [(e["ok"], e["glb"], e["glb_sha256"]) for e in T.read_events(control) if e["event"] == "export_done"]
    assert len(done) == 1 and done[0][2] != hashlib.sha256(planted["001.glb"]).hexdigest()
    blender = (control / "exports" / "001.glb").read_bytes()
    assert done == [(True, "exports/001.glb", hashlib.sha256(blender).hexdigest())] and not cached
    assert glb.read_bytes() == blender != planted["001.glb"] and edge(glb) == 1.2
    assert rep["meshes"] == 1 and (ex / "001.json").read_bytes() == (control / "exports" / "001.json").read_bytes()
    pick, rejected = C.select_deliverable(root)
    assert (pick["source"], rejected) == ("export", []) and pick["glb"].read_bytes() == blender


def test_an_export_replaces_links_at_its_names(tmp_path, monkeypatch):
    """Links that the agent put at exports/001.glb and 001.json are replaced, never written through. A folder that the
    agent swaps for a link while Blender runs gets nothing, and that export fails."""
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    work = prepare(tmp_path)
    control, ex = work.parent / "control", work / "exports"
    T.start_budget(work, control, time.time(), 600.0)
    victim, outside = tmp_path / "victim.txt", tmp_path / "outside"
    victim.write_text("keep me")
    outside.mkdir()
    os.symlink(victim, ex / "001.glb")
    os.symlink(outside / "001.json", ex / "001.json")  # dangling: a write through it would make the file
    (work / "candidate.blend").write_text(json.dumps({"kind": "good", "size": 1.2, "seconds": 0.0}))
    T.export_glb(work, control, FAKE_BLENDER)
    assert victim.read_text() == "keep me" and not any(outside.iterdir())
    for name in ("001.glb", "001.json"):
        assert not (ex / name).is_symlink() and (ex / name).read_bytes() == (control / "exports" / name).read_bytes()

    (work / "candidate.blend").write_text(json.dumps({"kind": "good", "size": 1.3, "seconds": 0.0, "pause": 1.0}))

    def swap():
        os.rename(ex, work / "exports_real")
        os.symlink(outside, ex)

    th = after_blender_wrote(control, ex, "002.glb", swap)
    with pytest.raises(SystemExit, match="export 002 failed"):
        T.export_glb(work, control, FAKE_BLENDER)
    th.join()
    assert not any(outside.iterdir()) and not (work / "exports_real" / "002.glb").exists()
    assert [e["ok"] for e in T.read_events(control) if e["event"] == "export_done"] == [True, False]


def test_an_export_log_is_not_written_through_a_swapped_folder(tmp_path, monkeypatch):
    """exports/ is checked when the export starts, and the cache check may then hash a large GLB of the agent's for a
    second. A link that the agent swaps in for the folder meanwhile gets no log: the log is opened through a
    descriptor of the folder (T.folder), and that export fails."""
    work = prepare(tmp_path)
    control, outside = work.parent / "control", tmp_path / "outside"
    outside.mkdir()
    (work / "candidate.blend").write_text(json.dumps({"kind": "good", "size": 1.2, "seconds": 0.0}))
    T.start_budget(work, control, time.time(), 600.0)
    real = T.sha256

    def swap_then_hash(path):  # the first hash, of the .blend, comes after the check of exports/
        if not (work / "exports").is_symlink():
            os.rename(work / "exports", work / "exports_real")
            os.symlink(outside, work / "exports")
        return real(path)

    monkeypatch.setattr(T, "sha256", swap_then_hash)
    with pytest.raises(SystemExit, match="export 001 failed"):
        T.export_glb(work, control, FAKE_BLENDER)
    assert not any(outside.iterdir()) and not any((work / "exports_real").iterdir())
    assert [(e["event"], e.get("ok")) for e in T.read_events(control)] == [("export_started", None), ("export_done", False)]


def test_a_blender_run_cannot_change_the_hosts_copies(tmp_path, monkeypatch):
    """Blender writes through the links that the agent puts in work/, such as a review's sheet.png, so its container
    writes only control/staging/ and sees the host's copies in control/exports/ read-only (blender_command). Each
    export moves out of staging/, and what another run left there is removed before the next export: an export
    whose Blender writes nothing then fails and does not take those bytes."""
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    work = prepare(tmp_path)
    control = work.parent / "control"
    (work / "candidate.blend").write_text(json.dumps({"kind": "good", "size": 1.2, "seconds": 0.0}))
    T.start_budget(work, control, time.time(), 600.0)
    T.export_glb(work, control, FAKE_BLENDER)
    assert sorted(p.name for p in (control / "exports").iterdir()) == ["001.glb", "001.json"]
    assert not any((control / "staging").iterdir())
    write_glb(control / "staging" / "002.glb", size=1.3)  # as a run's write through a link in work/ would leave
    (control / "staging" / "002.json").write_text(json.dumps({"meshes": 1, "warnings": []}))
    (work / "candidate.blend").write_text(json.dumps({"kind": "good", "size": 1.4, "seconds": 0.0}))
    with pytest.raises(SystemExit, match="export 002 failed"):
        T.export_glb(work, control, [sys.executable, "-c", "pass"])  # exits 0 and writes nothing
    assert not any((control / "staging").iterdir()) and not (work / "exports" / "002.glb").exists()
    assert [e["ok"] for e in T.read_events(control) if e["event"] == "export_done"] == [True, False]


def test_a_review_writes_nothing_through_a_swapped_folder(tmp_path, monkeypatch, command="review"):
    """The export and the render take minutes, in which the agent can swap the review folder for a link. The host
    makes and writes the receipt's folder through descriptors (T.new_folder): a link there when the export ends
    stops the review, and one swapped in during the render gets nothing, so the receipt stays in the folder that the
    host made."""
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    folder = "reviews"
    work = prepare(tmp_path)
    control, outside = work.parent / "control", tmp_path / "outside"
    outside.mkdir()
    (work / "candidate.blend").write_text(json.dumps({"kind": "good", "size": 1.2, "seconds": 0.0, "pause": 1.0}))
    T.start_budget(work, control, time.time(), 900.0)

    def swap():
        os.rename(work / folder, work / f"{folder}_real")
        os.symlink(outside, work / folder)

    th = after_blender_wrote(control, work / "exports", "001.glb", swap)
    with pytest.raises(SystemExit, match=re.escape(f"{folder}/ must be a folder, not a link or a file")):
        getattr(T, command)(work, control, FAKE_BLENDER)
    th.join()
    assert not any(outside.iterdir())
    os.unlink(work / folder)
    os.rename(work / f"{folder}_real", work / folder)
    run_ok = T.run_ok

    def render(cmd, log, timeout):  # the agent swaps the folder while Blender renders
        if Path(cmd[cmd.index("--python") + 1]).name == f"{command}.py":
            swap()
            return True
        return run_ok(cmd, log, timeout)

    monkeypatch.setattr(T, "run_ok", render)
    rec = getattr(T, command)(work, control, FAKE_BLENDER)
    assert not any(outside.iterdir()) and rec["cached"]
    got = work / f"{folder}_real" / rec["receipt"]
    assert json.loads((got / "receipt.json").read_text()) == rec and (got / "args.json").is_file()
    assert [e["receipt"] for e in T.read_events(control) if e["event"] == command] == [rec["receipt"]]


def test_helpers_give_blender_resolved_paths(tmp_path, monkeypatch):
    """blender_command binds the resolved work and control folders, and Blender's container has no other path. A
    session root reached through a symlink (a HARPOR_ROOT under ~/scratch on ICE) still reaches Blender resolved:
    the export's files and every path of the review's args."""
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    real = prepare(tmp_path / "real").parent
    (tmp_path / "link").symlink_to(real)
    work, control = tmp_path / "link" / "work", tmp_path / "link" / "control"
    (work / "candidate.blend").write_text(json.dumps({"kind": "good", "size": 1.2, "seconds": 0.0}))
    T.start_budget(work, control, time.time(), 600.0)
    blender = C.blender_command(tmp_path / "link", "/sif/agent.sif")
    bound = [Path(blender[i + 1].split(":")[0]) for i, a in enumerate(blender) if a == "--bind"]
    seen, run = [], T._run
    monkeypatch.setattr(T, "_run", lambda cmd, log, timeout: seen.append(cmd) or run(cmd, log, timeout))
    for k, v in (("GATOR_WORK", work), ("GATOR_CONTROL", control), ("GATOR_BLENDER", shlex.join(FAKE_BLENDER))):
        monkeypatch.setenv(k, str(v))
    T.main(["review"])
    paths = [Path(a) for cmd in seen for a in cmd[cmd.index("--") + 1:]]
    args = json.loads(paths[-1].read_text())
    paths += [Path(args[k]) for k in ("root", "out", "candidate", "original")]
    assert len(seen) == 2 and all(any(p.is_relative_to(b) for b in bound) for p in paths), paths


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


def test_paper_prompt_loads_with_recorded_changes(tmp_path):
    text, orig = PR.load(), PR.original()
    assert "work/run_budget.json" not in text.replace("/work/run_budget.json", "")
    assert len(text) - len(orig) == len(PR.CHANGES)  # each change adds one "/"
    write_glb(tmp_path / "original.glb")
    C.prepare_session(tmp_path / "s", None, [], source="s", metric=False, original=tmp_path / "original.glb")
    assert (tmp_path / "s" / "session" / "prompt.txt").read_text(encoding="utf-8") == PR.load()


def test_review_passes_the_detail_rig(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    work = prepare(tmp_path)
    control = work.parent / "control"
    (work / "candidate.blend").write_text(json.dumps({"kind": "good", "size": 1.0, "seconds": 0.0}))
    T.start_budget(work, control, 0.0, 600.0)
    rig = [[0.0, 0.0, 2.0, 0.1, 0.0, 0.0, 1.0]]
    rec = T.review(work, control, FAKE_BLENDER, detail=rig)
    assert json.loads((work / "reviews" / rec["receipt"] / "args.json").read_text())["detail"] == rig
    for bad in (rig * 5, [[0, 0, 2, 0, 0, 0, 1]], [[0, 0, 2, 0.1, 0, 0, 0]], [[0, 0, float("nan"), 0.1, 0, 0, 1]]):
        with pytest.raises(SystemExit):
            T.review(work, control, FAKE_BLENDER, detail=bad)


def test_make_previews_writes_the_sheet(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    original = tmp_path / "original.glb"
    write_glb(original, size=2.0)
    C.prepare_session(tmp_path / "s", None, make_views(tmp_path / "in"), source="s", metric=True,
                      original=original)
    previews = tmp_path / "s" / "work" / "original" / "previews"
    assert C.make_previews(tmp_path / "s", FAKE_BLENDER)
    assert (previews / "sheet.png").exists() and not (previews / "args.json").exists()


def test_a_session_always_renders_its_previews(tmp_path, monkeypatch):
    """Nothing supplies previews any more, so a PNG that is already in the folder does not stop the render."""
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    work = prepare(tmp_path)
    (work / "original" / "previews" / "old.png").write_bytes(b"")
    with tempfile.TemporaryDirectory(prefix="gator-") as sock_dir:
        r = C.run_session(work.parent, [sys.executable, "-c", "pass"], blender=FAKE_BLENDER, sock_dir=sock_dir,
                          budget_s=LONG, **FAST)
    assert r["previews_built"] is True and (work / "original" / "previews" / "sheet.png").is_file()


@pytest.mark.parametrize("gone", [{"depth_scale": 2.0}, {"confidence": "per pixel"}, {"previews": ()}, {"image_side": 1568},
                                  {"system": "manual.md"}])
def test_prepare_session_takes_no_comparison_inputs(tmp_path, gone):
    original = tmp_path / "original.glb"
    write_glb(original)
    with pytest.raises(TypeError, match="unexpected keyword"):
        C.prepare_session(tmp_path / "s", "p", [], source="s", metric=False, original=original, **gone)
    assert not (tmp_path / "s").exists()
    with pytest.raises(TypeError, match="positional"):
        C.claude_command("s", "a.sif", "/tmp/x", "/k.env", "xhigh", 1568)


@pytest.mark.parametrize("key", ["image", "mask"])
def test_a_view_without_its_image_or_mask_stops_the_session(tmp_path, key):
    """SYSTEM.md and AGENTS.md tell the agent that every view has both files."""
    views = make_views(tmp_path / "in")
    del views[1][key]
    write_glb(tmp_path / "in" / "o.glb")
    with pytest.raises(KeyError, match=key):
        C.prepare_session(tmp_path / "s", "p", views, source="s", metric=True,
                          original=tmp_path / "in" / "o.glb")


def test_check_glb_refuses_a_cycle_of_nodes(tmp_path):
    write_glb(tmp_path / "plain.glb")

    def cycle(g):
        g["nodes"].append({"name": "loop", "children": [len(g["nodes"])]})
        return g

    glb_json(tmp_path / "plain.glb", tmp_path / "cycle.glb", cycle)
    assert C.check_glb(tmp_path / "plain.glb") is None
    assert C.check_glb(tmp_path / "cycle.glb").startswith("bad nodes: the glTF nodes form a cycle")


def test_codex_session_layout_and_command(tmp_path):
    views = make_views(tmp_path / "in")
    write_glb(tmp_path / "in" / "o.glb")
    C.prepare_session(tmp_path / "s", None, views, source="s", metric=True,
                      original=tmp_path / "in" / "o.glb")
    root = tmp_path / "s"
    assert (root / "session" / "codex_home" / "config.toml").exists()
    agents = (root / "work" / "AGENTS.md").read_text()
    assert "view_image" in agents and "read_file" not in agents and "{" not in agents.split("## Helper commands")[0]
    assert 'model_reasoning_effort="xhigh"' in C.codex_command()
    conf = tomllib.loads(C.codex_config())
    assert conf["model"] == "gpt-6-astra" and conf["model_reasoning_effort"] == "xhigh" and conf["web_search"] == "disabled"
    assert conf["shell_environment_policy"]["ignore_default_excludes"] is False


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


# ---- the claude agent: Claude Opus 5.5 through harpor.agent.claude_agent ----

ANTHROPIC_KEY = "sk-ant-TEST-" + "0123456789abcdef" * 3  # a stand-in; nothing here calls the API
# claude_agent's own CLI with a scripted client instead of the Anthropic API, and bash on the host:
#   python STAND_IN SCRIPT.json REQUESTS.jsonl CLAUDE_AGENT_ARGS...
# Each request's settings go to REQUESTS.jsonl; each answer is the next message of SCRIPT.json.
STAND_IN = '''
import json, sys
from types import SimpleNamespace
from anthropic.types.beta import BetaMessage
from harpor.agent import claude_agent

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
                                                  "model": kw["model"], "stop_sequence": None, **script[self.n - 1]}))


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
    {"content": [{"type": "tool_use", "id": "toolu_2", "name": "view_image", "input": {"path": "/work/original/previews/sheet.png"}},
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
    pytest.importorskip("anthropic")  # the Claude client
    from harpor.agent import loop_trace

    monkeypatch.setenv("PYTHONPATH", str(REPO))
    root = prepare(tmp_path, agent="claude").parent
    key_file = tmp_path / "anthropic.env"
    key_file.write_text(f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}\n")
    (tmp_path / "stand_in.py").write_text(STAND_IN)
    (tmp_path / "script.json").write_text(json.dumps(CLAUDE_SCRIPT))
    real, built = C.claude_command, []

    def stand_in(*args, **kw):
        built.append(real(*args, **kw))
        return [sys.executable, str(tmp_path / "stand_in.py"), str(tmp_path / "script.json"),
                str(tmp_path / "requests.jsonl"), *built[-1][3:]]

    monkeypatch.setattr(C, "claude_command", stand_in)
    monkeypatch.setattr(C, "blender_command", lambda root, sif: FAKE_BLENDER)
    r = C.run_container_session(root, "/sif/agent.sif", key_file, **{**FAST, "finish_wait_s": 10.0})

    cmd = built[0]
    assert cmd[:4] == [sys.executable, "-m", "harpor.agent.claude_agent", str(root.resolve())]
    assert cmd[cmd.index("--key-file") + 1] == str(key_file) and cmd[-4:] == ["--model", "claude-opus-5-5", "--effort", "xhigh"]
    assert (r["agent"], r["model"], r["effort"]) == ("claude", "claude-opus-5-5", "xhigh")
    assert (r["source"], r["verdict"], r["reason"], r["glb"]) == ("finish", "complete", "fixed the handle", "work/exports/001.glb")
    assert edge(root / r["glb"]) == 1.2 and r["rejected"] == [] and r["candidate_built"] is True
    # The agent ended by itself once finish ran, before the controller's stop.
    assert (r["exit"], r["returncode"], r["cli_exit"], r["end_reason"]) == ("exited", 0, "ended", "finished")
    tokens = {k: sum(s["usage"][k] for s in CLAUDE_SCRIPT[:2]) for k in T.TOKENS}
    assert r["turns"] == 2 and r["tokens"] == tokens
    assert r["cost_usd"] == pytest.approx(sum(tokens[k] * T.PRICES[T.CLAUDE_MODEL][k] for k in tokens) / 1e6, abs=1e-4)
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
    results = {e["tool_use_id"]: e for e in T.read_jsonl(log) if e.get("type") == "tool_result"}
    assert results["toolu_1"]["is_error"] is False and "receipt" in results["toolu_1"]["output"]
    assert results["toolu_2"]["is_error"] is True and "not a PNG or JPEG" in results["toolu_2"]["output"]
    assert C.shell_commands(log)[1].startswith("finish --receipt")
    assert (root / "session" / "claude_stdout.log").read_text().startswith("turn 1: tool_use")
    assert (root / "session" / "system.md").read_bytes() == (C.AGENT_DIR / "claude" / "SYSTEM.md").read_bytes()
    assert all(q["system"][0]["text"] == (C.AGENT_DIR / "claude" / "SYSTEM.md").read_text(encoding="utf-8") for q in reqs)
    start = next(e for e in T.read_jsonl(log) if e.get("type") == "start")
    assert (start["image_side"], start["system"]) == (2576, str(C.AGENT_DIR / "claude" / "SYSTEM.md"))

    capsys.readouterr()
    loop_trace.main([str(tmp_path)])
    trace = capsys.readouterr().out
    assert "== agent (claude), in order" in trace and "views: /work/original/previews/sheet.png [" in trace
    assert "runs: finish --receipt" in trace and "ended: finished after 2 turns" in trace
    assert "finish complete: fixed the handle" in trace and "claude-opus-5-5: 2 turns" in trace


def test_loop_trace_of_a_codex_session(runs, tmp_path, capsys):
    """The trace reads the agent from control/agent.json and takes no turntable arguments. A session folder without
    that file is an error: it is not taken for a Codex session."""
    from harpor.agent import loop_trace

    root = runs["finish_refine"][0]
    loop_trace.main([str(root.parent)])
    trace = capsys.readouterr().out
    assert "== agent (codex), in order" in trace and "finish complete: fixed the handle" in trace
    with pytest.raises(SystemExit) as e:
        loop_trace.main([str(root.parent), "--turntable-args", str(tmp_path / "args.json")])
    assert e.value.code == 2 and not (tmp_path / "args.json").exists()
    shutil.copytree(root, tmp_path / "s")
    (tmp_path / "s" / "control" / "agent.json").unlink()
    with pytest.raises(FileNotFoundError, match="agent.json"):
        loop_trace.main([str(tmp_path)])


def test_loop_trace_prints_the_agents_text_on_clean_lines(runs, tmp_path, capsys):
    """A session from before the helpers cleaned the agent's text, with raw escapes in a finish event and in the
    Codex log, which the Codex agent can write, still traces as lines without control characters."""
    from harpor.agent import loop_trace

    s = tmp_path / "s"
    shutil.copytree(runs["dirty_text"][0], s)
    with open(s / "control" / "events.jsonl", "a") as f:
        f.write(json.dumps({"time": time.time(), "event": "finish", "verdict": "complete", "reason": DIRTY}) + "\n")
    with open(s / "session" / "codex.jsonl", "a") as f:
        for item in ({"type": "agent_message", "text": DIRTY}, {"type": "command_execution", "command": DIRTY,
                                                                "exit_code": DIRTY}):
            f.write(json.dumps({"type": "item.completed", "item": {"id": item["type"], **item}}) + "\n")
    loop_trace.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert f"  says: {CLEAN}\n" in out and f"  runs [exit {CLEAN}]: {CLEAN}\n" in out
    assert out.count(f" finish complete: {CLEAN}\n") == 2  # the session's own, stored clean, and the raw one
    assert next(x for x in out.splitlines() if " checkpoint checkpoints/001-" in x).endswith(f": {CLEAN}")
    assert [c for c in out if not c.isprintable() and c != "\n"] == []


def test_loop_trace_skips_what_the_agent_wrote_in_another_shape(runs, tmp_path):
    """codex.jsonl and the review files in work/ are the agent's to write. A line that is not an object, an item that
    is not one, an id that is a list, a review.json that is a list and a FIFO, which open() would wait on forever,
    are skipped, and the host's records still print. The trace runs as the script of note 5, with a timeout."""
    s = tmp_path / "s"
    shutil.copytree(runs["finish_refine"][0], s)
    with open(s / "session" / "codex.jsonl", "a") as f:
        for line in ([1, 2], {"type": "item.completed", "item": "x"},
                     {"type": "item.completed", "item": {"id": [1], "type": "agent_message", "text": "a list id"}}):
            f.write(json.dumps(line) + "\n")
    report = next((s / "work" / "reviews").glob("r001-*/review.json"))
    report.unlink()
    os.mkfifo(report)

    def trace():
        p = subprocess.run([sys.executable, str(REPO / "harpor" / "agent" / "loop_trace.py"), str(tmp_path)],
                           capture_output=True, text=True, timeout=30)
        assert p.returncode == 0, p.stderr
        return p.stdout

    out = trace()
    assert "  says: a list id\n" in out and "finish complete: fixed the handle" in out
    assert "review r001-" in out and "mask IoU" not in out
    (s / "session" / "codex.jsonl").unlink()
    os.mkfifo(s / "session" / "codex.jsonl")
    with open(s / "control" / "events.jsonl", "a") as f:  # a review whose review.json is a list
        f.write(json.dumps({"time": time.time(), "event": "review", "receipt": "r009-0123456789ab", "glb": "x"}) + "\n")
    (s / "work" / "reviews" / "r009-0123456789ab").mkdir(parents=True)
    (s / "work" / "reviews" / "r009-0123456789ab" / "review.json").write_text("[]")
    out = trace()
    assert "review r009-0123456789ab of x" in out and "mask IoU" not in out and "says:" not in out


def test_claude_session_with_every_tool(tmp_path, monkeypatch):
    """A claude session of Claude Fable 5.2 at effort low, through prepare_session and run_container_session: the shell
    keeps its directory and variables across turns, the text editor views a file and edits another,
    view_image shows a PNG that a command made, and finish ends the session. Every request repeats the
    one before it and appends to it, and the key the shell prints reaches neither the model nor any
    file."""
    pytest.importorskip("anthropic")  # the Claude client
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    manual = C.AGENT_DIR / "claude" / "SYSTEM.md"
    root = prepare(tmp_path, agent="claude", effort="low", model=T.FABLE_MODEL).parent
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

    def stand_in(*args, **kw):
        built.append(real(*args, **kw))
        return [sys.executable, str(tmp_path / "stand_in.py"), str(tmp_path / "script.json"),
                str(tmp_path / "requests.jsonl"), *built[-1][3:]]

    monkeypatch.setattr(C, "claude_command", stand_in)
    monkeypatch.setattr(C, "blender_command", lambda root, sif: FAKE_BLENDER)
    r = C.run_container_session(root, "/sif/agent.sif", key_file, **{**FAST, "finish_wait_s": 10.0})

    assert built[0][-4:] == ["--model", "claude-fable-5-2", "--effort", "low"]
    assert C.agent_info(root) == {"agent": "claude", "model": "claude-fable-5-2", "effort": "low"}
    assert (root / "session" / "system.md").read_bytes() == manual.read_bytes()
    assert (r["agent"], r["model"], r["effort"], r["source"], r["verdict"]) == ("claude", "claude-fable-5-2", "low", "finish", "complete")
    assert r["glb"] == "work/exports/001.glb" and edge(root / r["glb"]) == 1.2 and r["counts"]["reviews"] == 1
    assert (r["exit"], r["returncode"], r["cli_exit"], r["end_reason"], r["turns"]) == ("exited", 0, "ended", "finished", 3)
    tokens = {k: sum(s["usage"][k] for s in script) for k in T.TOKENS}
    assert r["tokens"] == tokens
    assert r["cost_usd"] == pytest.approx(sum(tokens[k] * T.PRICES[T.FABLE_MODEL][k] for k in tokens) / 1e6, abs=1e-4)
    assert (root / "work" / "edit.py").read_text() == "size = 1.2\n"

    reqs = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert [q["messages"] for q in reqs] == [1, 3, 5]
    assert all(q["model"] == "claude-fable-5-2" and q["output_config"] == {"effort": "low"} and q["speed"] is None
               and q["cache_control"] == {"type": "ephemeral"}
               and q["betas"] == ["server-side-fallback-2026-07-01"] for q in reqs)
    assert all(q["system"] == [{"type": "text", "text": manual.read_text(encoding="utf-8"),
                                "cache_control": {"type": "ephemeral"}}] for q in reqs)
    assert all("larger than 2576 px on the long side" in q["view_image"] for q in reqs)
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
    assert out["t2"]["content"].startswith("     1\t") and '"units": "meters"' in out["t2"]["content"]
    assert out["t3"]["content"] == "one in input\n" and out["t4"]["content"].startswith("created /work/edit.py")
    assert out["t5"]["content"].startswith("edited /work/edit.py:") and "size = 1.2" in out["t5"]["content"]
    assert out["t6"]["content"].startswith("receipt r001-")
    assert out["t7"]["content"] == "ANTHROPIC_API_KEY=[redacted]\n"

    log = root / "session" / "claude.jsonl"
    results = {e["tool_use_id"]: e for e in T.read_jsonl(log) if e.get("type") == "tool_result"}
    assert (results["t8"]["is_error"], results["t8"]["output"]) == (False, "[image 3000x1500]")
    start = next(e for e in T.read_jsonl(log) if e.get("type") == "start")
    assert (start["image_side"], start["system"]) == (2576, str(manual))
    assert results["t9"]["output"].startswith("finished: complete, exports/001.glb")
    assert r["key_found_in"] == [] and r["network_commands"] == []
    assert [p for p in all_files(tmp_path) if ANTHROPIC_KEY.encode() in p.read_bytes()] == [key_file]


def test_claude_session_layout_and_agent_record(tmp_path):
    work = prepare(tmp_path, agent="claude", effort="max")
    root = work.parent
    assert json.loads((root / "control" / "agent.json").read_text()) == {
        "agent": "claude", "model": "claude-opus-5-5", "effort": "max"}
    assert not (work / "AGENTS.md").exists() and not (root / "session" / "codex_home").exists()
    assert (root / "session" / "system.md").read_text() == (C.AGENT_DIR / "claude" / "SYSTEM.md").read_text()
    assert (root / "session" / "prompt.txt").read_text(encoding="utf-8") == PROMPT
    assert C.agent_info(root)["effort"] == "max"
    codex = prepare(tmp_path / "codex").parent
    assert C.agent_info(codex) == {"agent": "codex", "model": "gpt-6-astra", "effort": "xhigh"}
    for bad, why in [({"agent": "gpt"}, "agent is codex or claude"), ({"agent": "claude", "effort": "extreme"}, "effort is one of"),
                     ({"effort": "high"}, "effort is a setting of the claude agent"),
                     ({"agent": "claude", "model": "claude-opus-5"}, "model is one of claude-opus-5-5, claude-fable-5-2"),
                     ({"model": "claude-fable-5-2"}, "model is a setting of the claude agent")]:
        with pytest.raises(ValueError, match=why):
            C.prepare_session(tmp_path / "bad", "p", [], source="s", metric=False, **bad)
        assert not (tmp_path / "bad").exists()


def test_claude_shell_and_agent_commands(tmp_path):
    root = tmp_path / "s"
    shell = C.shell_command(root, "/sif/agent.sif", "/tmp/gator-x")
    codex = C.apptainer_command(root, "/sif/agent.sif", "/home/u/key.env", "/tmp/gator-x")
    n = len(C.CONTAINER)
    assert shell[:n] == C.CONTAINER and shell[n:n + 3] == ["--net", "--network", "none"]
    binds = [shell[i + 1] for i, a in enumerate(shell) if a == "--bind"]
    # Codex's binds without /session (claude_agent's log is there, and result.json trusts it) and without the DNS files.
    assert binds == [b for i, a in enumerate(codex) if a == "--bind" for b in [codex[i + 1]]
                     if ":/session" not in b and not b.startswith("/etc/")]
    assert binds == [f"{root.resolve()}/work:/work", f"{C.AGENT_DIR}/bin:/opt/gator/bin:ro", "/tmp/gator-x:/opt/gator/sock:ro"]
    assert "control" not in " ".join(binds) and "session" not in " ".join(shell)
    envs = [shell[i + 1] for i, a in enumerate(shell) if a == "--env"]
    assert envs == ["GATOR_SOCKET=/opt/gator/sock/tools.sock", "PREPEND_PATH=/opt/gator/bin"]
    assert "--env-file" not in shell and "/home/u/key.env" not in shell
    assert shell[-6:] == ["--pwd", "/work", "/sif/agent.sif", "bash", "--noprofile", "--norc"]
    cmd = C.claude_command("s", "agent.sif", "/tmp/gator-x", "key.env")
    assert cmd == [sys.executable, "-m", "harpor.agent.claude_agent", os.path.abspath("s"), "--sif", os.path.abspath("agent.sif"),
                   "--sock-dir", "/tmp/gator-x", "--key-file", os.path.abspath("key.env"), "--model", "claude-opus-5-5",
                   "--effort", "xhigh"]
    assert C.claude_command(root, "a.sif", "/tmp/x", "/k.env", "low")[-2:] == ["--effort", "low"]
    assert C.claude_command(root, "a.sif", "/tmp/x", "/k.env", "high", model="claude-fable-5-2")[-4:] == \
        ["--model", "claude-fable-5-2", "--effort", "high"]
    assert [C.cli_exit(c, "claude") for c in (0, 2, 3, 130, 1, -signal.SIGKILL)] == \
        ["ended", "refusal", "api_error", "interrupted", "exit 1", "signal SIGKILL"]


def test_claude_agent_override_runs_from_the_repo(tmp_path, monkeypatch):
    """agent= replaces claude_agent's whole host command: it runs from the repository with no stdin and its
    stdout goes to claude_stdout.log. Without an end line the usage is summed."""
    root = prepare(tmp_path, agent="claude").parent
    monkeypatch.setattr(C, "blender_command", lambda root, sif: FAKE_BLENDER)
    (tmp_path / "key.env").write_text(f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}\n")
    script = ("import json, os, sys, time\n"
              "print('cwd', os.getcwd(), 'stdin', repr(sys.stdin.read()))\n"
              "line = {'t': time.time(), 'type': 'response', 'stop_reason': 'tool_use', 'cost_usd': 0.0064, "
              "'usage': {'input_tokens': 1000, 'output_tokens': 100, "
              "'cache_read_input_tokens': 2000, 'cache_creation_input_tokens': None}, 'blocks': [{'type': 'tool_use', "
              "'id': 't1', 'name': 'bash', 'input': {'command': 'curl -O https://x/y.png'}}]}\n"
              "open(sys.argv[1], 'a').write(json.dumps(line) + '\\n')\n"
              "sys.exit(2)\n")
    r = C.run_container_session(root, "/sif/agent.sif", tmp_path / "key.env",
                                agent=[sys.executable, "-c", script, str(root / "session" / "claude.jsonl")], **FAST)
    assert (r["exit"], r["returncode"], r["cli_exit"], r["verdict"]) == ("exited", 2, "refusal", "fallback_original")
    assert (root / "session" / "claude_stdout.log").read_text() == f"cwd {C.REPO} stdin ''\n"
    assert (r["turns"], r["end_reason"], r["network_commands"]) == (1, None, ["curl -O https://x/y.png"])
    assert r["tokens"] == usage(1000, 100, read=2000) and r["cost_usd"] == pytest.approx(0.0064)


def test_claude_usage_prefers_the_end_line(tmp_path):
    log = tmp_path / "claude.jsonl"
    response = {"type": "response", "stop_reason": "end_turn", "usage": usage(1_000_000, 0), "cost_usd": 4.0}
    log.write_text(json.dumps(response) + "\nnot json\n[1]\n" + json.dumps({**response, "usage": "bad", "cost_usd": "bad"}) + "\n")
    assert C.claude_usage(log)["cost_usd"] == 4.0
    end = {"type": "end", "reason": "end_turn", "turns": 7, "totals": usage(5, 6, 7, 8), "cost_usd": 1.5}
    with open(log, "a") as f:
        f.write(json.dumps(end) + "\n")
    assert C.claude_usage(log) == {"turns": 7, "tokens": usage(5, 6, 7, 8), "cost_usd": 1.5, "end_reason": "end_turn"}
    assert C.claude_usage(tmp_path / "missing.jsonl")["tokens"] == usage(0, 0)


def test_claude_usage_without_an_end_line_takes_each_lines_cost(tmp_path):
    """claude_agent prices each response by the models that ran its attempts, and logs every attempt's
    tokens as billed when a refusal fallback ran; the controller sums those, and a line without a cost
    adds 0."""
    log = tmp_path / "claude.jsonl"
    lines = [{"type": "response", "usage": usage(100, 40), "billed": usage(200, 540), "cost_usd": 0.0126},
             {"type": "response", "usage": usage(10, 20), "cost_usd": 0.003175},
             {"type": "response", "usage": usage(1_000_000, 0)}]
    log.write_text("".join(json.dumps(e) + "\n" for e in lines))
    r = C.claude_usage(log)
    assert r["tokens"] == usage(1_000_210, 560) and r["turns"] == 3
    assert r["cost_usd"] == round(0.0126 + 0.003175, 4)


def test_find_key_reads_anthropic_keys(tmp_path):
    (tmp_path / "key.env").write_text(f"# for the claude agent\nANTHROPIC_API_KEY={ANTHROPIC_KEY}\n")
    root = tmp_path / "s"
    (root / "work").mkdir(parents=True)
    (root / "work" / "env.txt").write_text(f"X={ANTHROPIC_KEY}\n")
    (root / "work" / "other.txt").write_text("sk-ant-something-else")
    assert C.find_key(root, tmp_path / "key.env") == ["work/env.txt"]
    (tmp_path / "quoted.env").write_text(f'ANTHROPIC_API_KEY="{ANTHROPIC_KEY}"\n')  # the shell and claude_agent unquote it
    assert C.read_keys(tmp_path / "quoted.env") == [ANTHROPIC_KEY]
    assert C.find_key(root, tmp_path / "quoted.env") == ["work/env.txt"]
    (tmp_path / "none.env").write_text("OTHER=1\n")
    assert C.read_keys(tmp_path / "none.env") == [] and C.find_key(root, tmp_path / "none.env") == []


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
        assert C.read_keys(tmp_path / "both.env") == list(order)
        assert C.find_key(root, tmp_path / "both.env") == ["work/anthropic.txt", "work/openai.txt"]


def test_find_key_scans_big_files_in_chunks(tmp_path):
    """find_key reads chunk bytes at a time with an overlap, so a key across two chunks is found, and a
    file over 50 MB is scanned too (it was skipped before). Symlinks and FIFOs are still not opened."""
    key = "sk-TESTKEY0123456789abcdefghij"
    (tmp_path / "key.env").write_text(f"OPENAI_API_KEY={key}\n")
    root = tmp_path / "s"
    root.mkdir()
    for i, at in enumerate((0, 50, 64 - 10, 300 - len(key))):  # at the start, inside, across 64, at the end
        data = bytearray(b"x" * 300)
        data[at:at + len(key)] = key.encode()
        (root / f"f{i}.bin").write_bytes(bytes(data))
    (root / "clean.bin").write_bytes(b"x" * 300)
    (root / "cut.bin").write_bytes(b"x" * 60 + key.encode()[:-1])  # a key without its last character
    hits = ["f0.bin", "f1.bin", "f2.bin", "f3.bin"]
    for chunk in (7, 64, 50 << 20):  # 7: shorter than the key
        assert C.find_key(root, tmp_path / "key.env", chunk=chunk) == hits, chunk
    (tmp_path / "outside.txt").write_text(key)
    (root / "link.txt").symlink_to(tmp_path / "outside.txt")
    os.mkfifo(root / "pipe")
    for at in ((50 << 20) - 10, (50 << 20) + 5):  # across the default chunk's edge, and after it
        with open(root / "big.glb", "wb") as f:
            f.seek(at)
            f.write(key.encode())
        assert C.find_key(root, tmp_path / "key.env") == ["big.glb", *hits], at
    two = root / "two.bin"
    two.write_bytes(b"y" * 100 + b"sk-ant-" + b"z" * 30 + b"y" * 100)
    assert C.has_key(two, [b"short-key-0123456", b"sk-ant-" + b"z" * 30], chunk=16)
    assert not C.has_key(two, [b"short-key-0123456"], chunk=16)
    with pytest.raises(OSError):
        C.has_key(root / "link.txt", [key.encode()])


def test_tool_server_does_not_import_from_the_work_folder(tmp_path):
    """The model can write any file into /work. The helper runs on the host, outside the agent's
    container, so a json.py or a harpor/ package there must never be imported."""
    work = prepare(tmp_path)
    marker = tmp_path / "HOST_RAN.txt"
    plant = f"open({str(marker)!r}, 'a').write(__name__ + chr(10))\n"
    (work / "json.py").write_text(plant + "from json_real_is_gone import x\n")
    (work / "harpor" / "agent").mkdir(parents=True)
    for f in ("harpor/__init__.py", "harpor/agent/__init__.py", "harpor/agent/tools.py"):
        (work / f).write_text(plant)
    with tempfile.TemporaryDirectory(prefix="gator-") as d:
        server = ToolServer(os.path.join(d, "tools.sock"), work, work.parent / "control", FAKE_BLENDER)
        try:
            output, code = server.run(["finish", "--receipt", "r9", "--verdict", "complete", "--reason", "x"])
        finally:
            server.stop()
    assert code == 1 and "unknown review receipt 'r9'" in output
    assert not marker.exists()


# ---- rounds: a refinement with physics supervision (controller.run_rounds) ----

READ_PROMPT = {"do": "read_prompt"}
# The physics judge's answers (physics_review.apply's form), by outcome. fake_physics gives them in order.
JUDGMENTS = {
    "pass": {"outcome": "pass", "reason": "it rests on its base", "findings": []},
    "fail": {"outcome": "fail", "reason": "it tips over", "findings": [{"why": "it tips onto its side",
                                                                       "repair": "widen the base"}]},
    "inconclusive": {"outcome": "inconclusive", "reason": "Physics supervision unavailable: OSError", "findings": []},
}


def exported(k):
    """A round that saves a box, exports it and crashes: its export is taken as needs_review. k changes the box, so
    that each round exports."""
    return [edit(1.2 + 0.01 * k), EXPORT, CRASH]


# Every box that the rounds below hand off is the original scaled, so none of its faces is the original's, and the
# surface check finds all of the original gone. Its findings are report only while surface.ENFORCE is off: they come
# first in result.json's rounds and in the evidence, but decide no outcome or scope.
LOST = [{"code": "surface_identity", "measured": 0.0, "expected": "at least 0.5"},
        {"code": "surface_deleted", "measured": 1.0, "expected": "at most 0.02"}]
SURFACE = [f["code"] for f in LOST]


ROUND_SESSIONS = {  # name: (the agent's rounds, the judge's outcomes in order, the last one repeated)
    "rounds_pass": ([[READ_PROMPT, edit(1.2), REVIEW, finish("complete")]], ["pass"]),
    "rounds_repair": ([exported(0), [READ_PROMPT, edit(1.25), REVIEW, finish("complete")]], ["pass"]),
    "rounds_physics": ([[edit(1.2), REVIEW, finish("complete")], [READ_PROMPT, edit(1.3), REVIEW, finish("complete")]],
                       ["fail", "pass"]),
    "rounds_cap": ([exported(0), exported(1), exported(2), [CRASH]], ["pass"]),
    # Round 1 finishes, but the judge fails it; round 2 only exports; round 3 hands off nothing.
    "rounds_regress": ([[edit(1.2), REVIEW, finish("complete")], exported(1), [READ_PROMPT, CRASH]], ["fail"]),
    "rounds_unchanged": ([[REVIEW, finish("unchanged")]] * 3, ["pass"]),
    "rounds_inconclusive": ([[edit(1.2), REVIEW, finish("complete")], [CRASH]], ["inconclusive"]),
    # Round 2 reviews round 1's scene again, unchanged, and finishes it complete: the same bytes.
    "rounds_review_then_pass": ([[edit(1.2), REVIEW, finish("needs_review")], [REVIEW, finish("complete")]], ["pass"]),
}


def fake_physics(outcomes):
    """A stand-in for physics_review.check(root, glb): the next of outcomes (the last one again past the end), with
    the candidate's sha256, as the real check gives it. It logs each glb in root/control/judged.txt."""
    def check(root, glb, timeout=600):
        log = Path(root) / "control" / "judged.txt"
        seen = log.read_text().splitlines() if log.is_file() else []
        log.write_text("".join(f"{x}\n" for x in [*seen, os.path.relpath(glb, root)]))
        return {**JUDGMENTS[outcomes[min(len(seen), len(outcomes) - 1)]], "asset_sha256": T.sha256(glb), "seconds": 0.0}
    return check


def run_in_rounds(root, rounds, budget_s=LONG):
    """A refinement session with physics supervision on (control/physics/config.json), with a fake agent script of
    rounds. The caller replaces physics_review.check."""
    work = prepare(root)
    (work.parent / "control" / "physics").mkdir()
    (work.parent / "control" / "physics" / "config.json").write_text("{}")
    (root / "script.json").write_text(json.dumps({"rounds": rounds}))
    with tempfile.TemporaryDirectory(prefix="gator-") as sock_dir:
        env = {**os.environ, "PYTHONPATH": str(REPO), "GATOR_SOCKET": os.path.join(sock_dir, "tools.sock")}
        cmd = [sys.executable, "-m", "harpor.agent.fake_agent", str(root / "script.json")]
        return C.run_session(work.parent, cmd, env, blender=FAKE_BLENDER, sock_dir=sock_dir, budget_s=budget_s, **FAST)


@pytest.fixture(scope="module")
def round_runs(tmp_path_factory):
    from harpor.agent import physics_review

    roots = {name: tmp_path_factory.mktemp(name) for name in ROUND_SESSIONS}
    by_root = {str(roots[name] / "s"): fake_physics(outcomes) for name, (_, outcomes) in ROUND_SESSIONS.items()}
    with pytest.MonkeyPatch.context() as m, ThreadPoolExecutor(len(ROUND_SESSIONS)) as ex:
        m.setattr(physics_review, "check", lambda root, glb, timeout=600: by_root[str(root)](root, glb, timeout))
        results = list(ex.map(lambda name: run_in_rounds(roots[name], ROUND_SESSIONS[name][0]), ROUND_SESSIONS))
    return {name: (roots[name] / "s", r) for name, r in zip(ROUND_SESSIONS, results)}


def codes(rec):
    return [f["code"] for f in rec["findings"]]


def test_a_session_with_physics_supervision_runs_in_rounds(round_runs):
    """A build that passes is the only round. The round is logged, run_budget.json names it, its check (the surface
    and the physics judgment) goes to work/evidence/, and the result adds rounds and candidate. The deliverable is
    the round's stored candidate, a read-only copy in control/candidates, with the agent's verdict and reason, since
    its round passed. The session folder keeps the layout of a session without rounds."""
    root, r = round_runs["rounds_pass"]
    control, work = root / "control", root / "work"
    glb = "control/candidates/v01/candidate.glb"
    assert (r["source"], r["verdict"], r["glb"], r["exit"], r["reason"]) == (
        "finish", "complete", glb, "exited", "fixed the handle")
    sha = T.sha256(root / r["glb"])
    assert sha == r["sha256"] == T.sha256(work / "exports" / "001.glb") and edge(root / r["glb"]) == 1.2
    assert (root / glb).stat().st_mode & 0o777 == 0o444
    assert r["candidate"] == {"version": 1, "round": 1, "kind": "build", "from": "work/exports/001.glb",
                              "agent_verdict": "complete", "outcome": "pass", "why": "the only eligible candidate"}
    assert r["rounds"] == [{"round": 1, "kind": "build", "budget_s": LONG, "exit": "exited",
                            "wall_s": r["timings"]["wall_s"], "outcome": "pass", "scope": "none", "findings": LOST,
                            "candidate": {"version": 1, "source": "finish", "verdict": "complete",
                                          "glb": "work/exports/001.glb", "sha256": sha}}]
    assert (control / "judged.txt").read_text() == f"{glb}\n"  # the judge sees the stored copy
    stored = json.loads((control / "candidates" / "v01" / "report.json").read_text())
    assert stored["report"] == json.loads((work / "evidence" / "round_01.json").read_text())
    assert stored["report"]["physics_review"]["outcome"] == "pass"
    ev = json.loads((control / "evidence.json").read_text())
    assert (ev["object"], ev["diagonal_m"], ev["rounds"]) == ("sensor depth", pytest.approx(ORIGINAL_SIZE * 3 ** 0.5),
                                                              r["rounds"])
    assert ev["selected"] == {"version": 1, "verdict": "complete", "why": "the only eligible candidate",
                              "rank": [{"version": 1, "rank": [True, True, True, 1]}], "ineligible": []}
    (c,) = ev["candidates"]
    assert (c["glb"], c["from"], c["sha256"], c["source"], c["agent_verdict"], c["outcome"]) == (
        glb, "work/exports/001.glb", sha, "finish", "complete", "pass")
    assert c["physics_review"]["reason"] == "it rests on its base"
    assert (r["surface"]["shares"]["present"], r["surface"]["shares"]["deleted"]) == (0.0, pytest.approx(1.0))
    assert [x["code"] for x in r["surface"]["findings"]] == SURFACE and r["surface"]["blocks"] is False
    assert r["timings"]["budget_s"] == LONG
    assert r["counts"] == {"exports": 1, "reviews": 1, "checkpoints": 0}
    events = T.read_events(control)
    assert {k: events[0][k] for k in ("event", "round", "kind")} == {"event": "round_start", "round": 1, "kind": "build"}
    budget = json.loads((work / "run_budget.json").read_text())
    assert (budget["round"], budget["kind"], budget["budget_s"]) == (1, "build", LONG)
    assert not (root / "session" / "rounds").exists()
    assert (work / "prompt_seen.txt").read_text(encoding="utf-8") == PROMPT
    assert json.loads((root / "result.json").read_text()) == r


def test_a_local_failure_gets_a_repair_round(round_runs):
    """Round 1 leaves only an export: not_finished, a local finding, so a repair of 450 s follows. Its agent is new,
    and its prompt is the refinement prompt, then the round's budget, the evidence file, the best candidate so far,
    the check's feedback and the physics guidance. Round 1's logs and prompt move to rounds/01/, and the result's
    exit is the last round's, with the timings summed."""
    from harpor.agent.physics_review import GUIDANCE

    root, r = round_runs["rounds_repair"]
    session, work = root / "session", root / "work"
    assert [(x["round"], x["kind"], x["budget_s"], x["outcome"], x["scope"]) for x in r["rounds"]] == [
        (1, "build", LONG, "fail", "local"), (2, "repair", C.ROUND_BUDGET_S["repair"], "pass", "none")]
    first = r["rounds"][0]
    assert first["findings"] == [*LOST, {"code": "not_finished", "measured": "needs_review", "expected": "complete"}]
    assert (first["candidate"]["source"], first["candidate"]["glb"]) == ("export", "work/exports/001.glb")
    assert (r["source"], r["verdict"], r["glb"], r["exit"]) == (
        "finish", "complete", "control/candidates/v02/candidate.glb", "exited")
    assert (r["candidate"]["from"], r["candidate"]["why"]) == (
        "work/exports/002.glb", "first of 2 eligible candidates; it beats v01 on round outcome pass")
    t = r["timings"]
    assert t["budget_s"] == LONG + 450.0 and t["wall_s"] == pytest.approx(sum(x["wall_s"] for x in r["rounds"]))
    evidence = json.loads((work / "evidence" / "round_01.json").read_text())
    best = {"glb": "exports/001.glb", "round": 1, "passed": False}
    prompt = PR.round_prompt("repair", 450.0, V.feedback(evidence), best, "evidence/round_01.json") + GUIDANCE
    assert (work / "prompt_seen.txt").read_text(encoding="utf-8") == prompt
    assert (session / "prompt.txt").read_text(encoding="utf-8") == prompt
    assert prompt.startswith(PR.load()) and "450 second budget" in prompt and "not_finished" in prompt
    assert "/work/exports/001.glb, from round 1; its check did not pass" in prompt
    assert sorted(p.name for p in (session / "rounds" / "01").iterdir()) == ["codex.jsonl", "prompt.txt", "stderr.log"]
    assert (session / "rounds" / "01" / "prompt.txt").read_text(encoding="utf-8") == PROMPT
    assert "$ export_glb" in (session / "rounds" / "01" / "codex.jsonl").read_text()
    assert "$ review" in agent_log(root) and "$ export_glb" not in agent_log(root)
    assert (work / "AGENTS.md").read_text() == (C.AGENT_DIR / "codex" / "AGENTS.md").read_text()
    budget = json.loads((work / "run_budget.json").read_text())
    assert (budget["round"], budget["kind"], budget["budget_s"]) == (2, "repair", 450.0)
    starts = [e for e in T.read_events(root / "control") if e["event"] == "round_start"]
    assert [(e["round"], e["kind"]) for e in starts] == [(1, "build"), (2, "repair")]
    assert json.loads((root / "result.json").read_text()) == r


def test_a_physics_failure_gets_a_repair_round(round_runs):
    """The judge fails round 1's finished candidate: physics_implausible, with its repair advice in the next prompt.
    The repair passes and is delivered."""
    root, r = round_runs["rounds_physics"]
    work = root / "work"
    assert [(x["kind"], x["outcome"], x["scope"]) for x in r["rounds"]] == [("build", "fail", "local"),
                                                                            ("repair", "pass", "none")]
    assert r["rounds"][0]["findings"] == [*LOST, {"code": "physics_implausible", "measured": None, "expected": None}]
    evidence = json.loads((work / "evidence" / "round_01.json").read_text())
    assert evidence["physics_review"]["outcome"] == "fail"
    assert [f["why"] for f in evidence["findings"] if f["code"] == "physics_implausible"] == [
        "it tips onto its side Repair: widen the base"]
    assert "widen the base" in (work / "prompt_seen.txt").read_text(encoding="utf-8")
    assert (r["verdict"], r["glb"], edge(root / r["glb"])) == ("complete", "control/candidates/v02/candidate.glb", 1.3)


def test_the_rounds_stop_after_three_builds(round_runs):
    """Three local failures are the most a session builds (MAX_BUILDS); round 4's script never runs. The three
    stored candidates rank alike but for their version, so the session delivers the newest, as needs_review with
    the host's reason and limitations."""
    root, r = round_runs["rounds_cap"]
    assert [(x["kind"], x["outcome"], x["scope"]) for x in r["rounds"]] == [
        ("build", "fail", "local"), ("repair", "fail", "local"), ("repair", "fail", "local")]
    assert all(codes(x) == [*SURFACE, "not_finished"] for x in r["rounds"]) and C.next_round(r["rounds"]) is None
    assert (r["source"], r["verdict"], r["glb"]) == ("export", "needs_review", "control/candidates/v03/candidate.glb")
    assert r["reason"] == "the host's check of candidate v03 (round 3) did not pass; open: not_finished"
    assert r["limitations"] == ["not_finished"]
    assert r["candidate"]["why"] == "first of 3 eligible candidates; it beats v02 on version"
    assert edge(root / r["glb"]) == 1.22 and r["counts"]["exports"] == 3
    assert sorted(p.name for p in (root / "session" / "rounds").iterdir()) == ["01", "02"]


def test_loop_trace_shows_the_last_round(round_runs, capsys):
    """budget.json is the last round's, so the trace shows that round's events on its clock, and the rounds of the
    result."""
    from harpor.agent import loop_trace

    root, r = round_runs["rounds_repair"]
    loop_trace.main([str(root.parent)])
    trace = capsys.readouterr().out
    assert "== round 2 of 2 (repair); earlier rounds' logs are in session/rounds/\n" in trace
    host = trace.split("== host records")[1].split("== result")[0]
    assert "export 1 started" not in host and "export 2 started" in host and "finish complete" in host
    assert not re.search(r"^\s+-\d+ s ", host, re.M)
    assert "   round 1 (build): exited, outcome fail, scope local\n" in trace
    assert "   round 2 (repair): exited, outcome pass, scope none\n" in trace


def test_an_earlier_candidate_wins_when_a_repair_regresses(round_runs, tmp_path):
    """Round 1 finishes, but the judge fails it; round 2 only exports, and round 3 hands off nothing, so it stores no
    candidate. No round passed, so the session delivers round 1's finished candidate over round 2's newer export, as
    needs_review with the host's reason. harpor.webapp.steps.agent_glb finds it in a job folder as before."""
    from harpor.agent.physics_review import GUIDANCE
    from harpor.webapp import steps

    root, r = round_runs["rounds_regress"]
    control, work = root / "control", root / "work"
    assert [(x["kind"], x["outcome"], x["candidate"]["version"], codes(x)) for x in r["rounds"]] == [
        ("build", "fail", 1, [*SURFACE, "physics_implausible"]),
        ("repair", "fail", 2, [*SURFACE, "physics_implausible", "not_finished"]),
        ("repair", "fail", None, ["no_candidate"])]
    assert (r["source"], r["verdict"], r["glb"]) == ("finish", "needs_review", "control/candidates/v01/candidate.glb")
    assert r["reason"] == "the host's check of candidate v01 (round 1) did not pass; open: physics_implausible"
    assert r["limitations"] == ["physics_implausible"]
    assert r["candidate"] == {"version": 1, "round": 1, "kind": "build", "from": "work/exports/001.glb",
                              "agent_verdict": "complete", "outcome": "fail",
                              "why": "first of 2 eligible candidates; it beats v02 on source not an export"}
    assert sorted(p.name for p in (control / "candidates").iterdir()) == ["v01", "v02"]
    # Round 3's prompt points the agent at the best candidate, round 1's, not at round 2's newer export.
    evidence = json.loads((work / "evidence" / "round_02.json").read_text())
    best = {"glb": "exports/001.glb", "round": 1, "passed": False}
    assert (work / "prompt_seen.txt").read_text(encoding="utf-8") == PR.round_prompt(
        "repair", 450.0, V.feedback(evidence), best, "evidence/round_02.json") + GUIDANCE
    ev = json.loads((control / "evidence.json").read_text())
    assert (ev["selected"]["version"], [x["version"] for x in ev["selected"]["rank"]]) == (1, [1, 2])
    job = tmp_path / "job"
    shutil.copytree(root, job / "agent" / "s")
    (job / "agent" / "result.json").write_text(json.dumps(r))
    assert steps.agent_glb(job) == job / "agent" / "s" / "control" / "candidates" / "v01" / "candidate.glb"


def test_a_pass_beats_an_earlier_reviewed_failure(round_runs):
    """Round 2 hands off round 1's reviewed GLB again, now complete. Each round stores its own candidate, though the
    bytes are the same, so each keeps its own verdict, and round 2's host check reuses round 1's cached report. The
    passing round wins and is delivered with the agent's reason."""
    root, r = round_runs["rounds_review_then_pass"]
    assert [(x["kind"], x["outcome"], x["candidate"]["version"]) for x in r["rounds"]] == [
        ("build", "fail", 1), ("repair", "pass", 2)]
    assert codes(r["rounds"][0]) == [*SURFACE, "not_finished"]
    one, two = (json.loads((root / "control" / "candidates" / v / "report.json").read_text()) for v in ("v01", "v02"))
    assert one["sha256"] == two["sha256"] and (one["agent_verdict"], two["agent_verdict"]) == ("needs_review", "complete")
    assert (one["report"]["cached"], two["report"]["cached"]) == (False, True)
    assert C.rank(one)[1:3] == C.rank(two)[1:3]
    assert (r["source"], r["verdict"], r["glb"], r["reason"]) == (
        "finish", "complete", "control/candidates/v02/candidate.glb", "fixed the handle")
    assert r["candidate"]["why"] == "first of 2 eligible candidates; it beats v01 on round outcome pass"


def test_a_structural_failure_gets_one_planning_round(tmp_path, monkeypatch):
    """With surface.ENFORCE on, a build that loses the original's surface (surface_identity) fails as structural: a
    planning round of 300 s follows, in which every helper is refused, so it logs no event and stores no candidate;
    the build's finish does not end it, and it writes plan.md. The repair after it is told to follow the plan, and its
    box keeps the original's surface, so it passes."""
    from harpor.agent import physics_review, surface
    from harpor.agent.physics_review import GUIDANCE

    monkeypatch.setattr(surface, "ENFORCE", True)
    monkeypatch.setattr(physics_review, "check", fake_physics(["pass"]))
    plan = [READ_PROMPT, EXPORT, finish("needs_review"), {"do": "sleep", "seconds": 1.0},
            {"do": "write", "path": "plan.md", "text": "keep the original's faces\n"}]
    r = run_in_rounds(tmp_path, [[edit(1.2), REVIEW, finish("complete")], plan,
                                 [READ_PROMPT, edit(ORIGINAL_SIZE), REVIEW, finish("complete")]])
    root = tmp_path / "s"
    session, work, control = root / "session", root / "work", root / "control"
    assert [(x["round"], x["kind"], x["budget_s"], x["outcome"], x["scope"]) for x in r["rounds"]] == [
        (1, "build", LONG, "fail", "structural"), (2, "replan", 300.0, None, None), (3, "repair", 450.0, "pass", "none")]
    assert "surface_identity" in codes(r["rounds"][0])
    planning = r["rounds"][1]
    assert (planning["exit"], planning["candidate"], planning["findings"]) == ("exited", None, [])
    assert planning["wall_s"] > 1.0 and (work / "plan.md").read_text() == "keep the original's faces\n"
    log = (session / "rounds" / "02" / "codex.jsonl").read_text()
    for cmd in ("export_glb", "finish"):
        assert f"{cmd} refused: this is a planning round" in log
    starts = [e["kind"] for e in T.read_events(control) if e["event"] == "round_start"]
    assert starts == ["build", "replan", "repair"] and not (work / "evidence" / "round_02.json").exists()
    evidence = json.loads((work / "evidence" / "round_01.json").read_text())
    feedback = V.feedback(evidence)  # v01 lost the surface, so it is not eligible and no best candidate is named
    assert (session / "rounds" / "02" / "prompt.txt").read_text(encoding="utf-8") == PR.round_prompt(
        "replan", 300.0, feedback, None, "evidence/round_01.json") + GUIDANCE
    seen = (work / "prompt_seen.txt").read_text(encoding="utf-8")  # round 3's
    assert seen == PR.round_prompt("repair", 450.0, feedback, None, "evidence/round_01.json", planned=True) + GUIDANCE
    assert PR.PLAN in seen and r["timings"]["budget_s"] == LONG + 300.0 + 450.0
    assert (r["source"], r["verdict"], r["glb"]) == ("finish", "complete", "control/candidates/v02/candidate.glb")
    assert edge(root / r["glb"]) == ORIGINAL_SIZE


def test_an_unchanged_verdict_is_stored_and_can_pass(round_runs):
    """unchanged hands off root/original.glb. It is stored and checked like any candidate, and when the judge passes
    it the session ends after round 1 with the verdict unchanged."""
    root, r = round_runs["rounds_unchanged"]
    sha = T.sha256(root / "original.glb")
    assert [(x["candidate"]["version"], x["candidate"]["sha256"], x["outcome"]) for x in r["rounds"]] == [(1, sha, "pass")]
    assert (r["source"], r["verdict"], r["glb"]) == ("finish", "unchanged", "control/candidates/v01/candidate.glb")
    assert r["sha256"] == sha


def test_an_inconclusive_judgment_ends_the_rounds(round_runs):
    """A judge that cannot run (no MuJoCo, no frames, no answer) is inconclusive: physics_unassessed, scope none, so
    no repair can be checked and the session ends after its build, as needs_review."""
    root, r = round_runs["rounds_inconclusive"]
    assert [(x["kind"], x["outcome"], x["scope"], codes(x)) for x in r["rounds"]] == [
        ("build", "fail", "none", [*SURFACE, "physics_unassessed"])]
    assert (r["verdict"], r["glb"]) == ("needs_review", "control/candidates/v01/candidate.glb")
    assert r["reason"] == "the host's check of candidate v01 (round 1) did not pass; open: physics_unassessed"


def stored(version, outcome="fail", sha="a", blocks=False, source="finish"):
    """A stored candidate's record of sha256 "a" whose report is of the bytes sha."""
    return {"version": version, "round": version, "sha256": "a", "source": source,
            "report": {"sha256": sha, "outcome": outcome, "scope": "none", "surface": {"blocks": blocks},
                       "findings": []}}


def test_a_passing_round_ranks_first():
    """rank: a passing round beats a newer failure, which paper A.6's order would not ensure; then a surface that
    does not block, a source other than an uninspected export, and the newer version. A pass counts only with a
    report of the candidate's own bytes."""
    assert C.RANK == ("round outcome pass", "surface not blocking", "source not an export", "version")
    assert C.rank(stored(1, "pass")) > C.rank(stored(2)) > C.rank(stored(1))
    assert C.rank(stored(1)) > C.rank(stored(2, blocks=True))
    assert C.rank(stored(1, source="checkpoint")) > C.rank(stored(2, source="export"))
    assert C.passed(stored(1, "pass")) and not C.passed(stored(1, "pass", sha="b"))
    ranked = sorted([stored(1), stored(2, "pass"), stored(3, source="export")], key=C.rank, reverse=True)
    assert [c["version"] for c in ranked] == [2, 1, 3]
    assert C.why_first(ranked) == "first of 3 eligible candidates; it beats v01 on round outcome pass"
    assert C.why_first(ranked[1:]) == "first of 2 eligible candidates; it beats v03 on source not an export"
    assert C.why_first(ranked[:1]) == "the only eligible candidate"


def store(root, n, report, kind="good", size=ORIGINAL_SIZE, verdict="needs_review"):
    """Store a box as round n's candidate, its record holding report (stored's) as a report of the copy's bytes."""
    glb = root / "work" / "exports" / f"{n:03d}.glb"
    write_glb(glb, kind, size)
    c = C.store_candidate(root, n, "build", {"source": "finish", "verdict": verdict, "glb": glb,
                                            "reason": "the agent's reason", "limitations": ["the agent's limitation"]})
    c["report"] = {**report, "sha256": c["sha256"]}
    (root / c["glb"]).with_name("report.json").write_text(json.dumps(c))
    return c


def test_select_candidate_delivers_the_best_eligible_candidate(tmp_path, monkeypatch):
    """A stored candidate that is not a valid GLB, has another size, changed after it was stored or (only while
    surface.ENFORCE is on) lost most of the original is not eligible (paper A.6). The best of the others is
    delivered from its read-only copy: with the agent's words when its round passed, else needs_review with the
    host's. With none eligible the session falls back to the original, unless the last round's verdict was
    unchanged."""
    root = prepare(tmp_path).parent
    gone = {"code": "surface_identity", "measured": 0.3, "expected": "at least 0.5",
            "why": "most of the native area is gone"}
    open_ = {"code": "not_finished", "measured": "needs_review", "expected": "complete", "why": "the agent's own words"}
    first = store(root, 1, stored(1, "pass")["report"], verdict="complete")
    store(root, 2, stored(2)["report"], kind="garbage")
    store(root, 3, stored(3)["report"], size=3.0 * ORIGINAL_SIZE)
    store(root, 4, {**stored(4)["report"], "outcome": "fail", "findings": [gone, open_]})
    pick, chosen, selected = C.select_candidate(root)
    assert pick == {"source": "finish", "glb": root / "control" / "candidates" / "v01" / "candidate.glb",
                    "verdict": "complete", "reason": "the agent's reason", "limitations": ["the agent's limitation"]}
    assert chosen == first and C.check_glb(pick["glb"]) is None
    assert selected["rank"] == [{"version": 1, "rank": [True, True, True, 1]},
                                {"version": 4, "rank": [False, True, True, 4]}]
    assert [(x["from"], x["glb"], x["why"].split(":")[0]) for x in selected["ineligible"]] == [
        ("candidate v02", "control/candidates/v02/candidate.glb", "unparsable"),
        ("candidate v03", "control/candidates/v03/candidate.glb", "wrong size")]
    v01 = root / first["glb"]
    v01.chmod(0o644)
    v01.write_bytes(v01.read_bytes() + b"\0")
    pick, chosen, selected = C.select_candidate(root)
    assert (pick["verdict"], pick["glb"], chosen["version"]) == (
        "needs_review", root / "control" / "candidates" / "v04" / "candidate.glb", 4)
    assert pick["reason"] == "the host's check of candidate v04 (round 4) did not pass; open: not_finished"
    assert pick["limitations"] == ["not_finished"] and selected["ineligible"][0]["why"] == "changed after it was stored"
    monkeypatch.setattr("harpor.agent.surface.ENFORCE", True)
    pick, chosen, selected = C.select_candidate(root)
    assert (pick, chosen) == (C.original_pick(root), None)
    assert (selected["why"], selected["ineligible"][-1]["why"]) == (
        "no stored candidate is eligible", "surface_identity: most of the native area is gone")
    last = {"source": "finish", "verdict": "unchanged", "glb": root / "original.glb", "reason": "nothing to do",
            "limitations": []}
    assert C.select_candidate(root, last)[0] == last
    assert C.select_candidate(tmp_path / "none")[2]["why"] == "no build or repair handed off a GLB"


def test_the_evidence_and_the_candidates_outlive_a_failed_loop(tmp_path, monkeypatch, capsys):
    """control/evidence.json is written after every round, so it holds what the rounds found when the loop itself
    fails, here at the start of round 2; the session still selects from the stored candidates and writes
    result.json, and the evidence then names the choice."""
    from harpor.agent import physics_review

    monkeypatch.setattr(physics_review, "check", fake_physics(["pass"]))
    real = C.run_round

    def run_round(root, cmd, env, blender, sock_dir, n, *args):
        if n == 2:
            raise RuntimeError("the socket cannot be bound")
        return real(root, cmd, env, blender, sock_dir, n, *args)

    monkeypatch.setattr(C, "run_round", run_round)
    r = run_in_rounds(tmp_path, [exported(0), [CRASH]])
    assert "rounds stopped after round 1: RuntimeError: the socket cannot be bound\n" in capsys.readouterr().err
    assert (r["source"], r["verdict"], r["glb"]) == ("export", "needs_review", "control/candidates/v01/candidate.glb")
    assert r["reason"] == "the host's check of candidate v01 (round 1) did not pass; open: not_finished"
    ev = json.loads((tmp_path / "s" / "control" / "evidence.json").read_text())
    assert ev["rounds"] == r["rounds"] and [c["version"] for c in ev["candidates"]] == [1]
    assert ev["selected"]["version"] == 1


def test_a_failed_check_is_a_finding_but_cancel_ends_the_session(tmp_path, monkeypatch):
    """The steps after a round read what the agent made, so an exception or a string SystemExit there is a
    validation_error of the round, and the rounds go on. Cancel's integer SystemExit (tools._on_sigterm) ends the
    session, and no result.json is written."""
    from harpor.agent import physics_review

    monkeypatch.setattr(physics_review, "check", fake_physics(["pass"]))
    real, calls = V.check, []

    def check(*args, **kw):
        calls.append(args[0])
        if len(calls) == 1:
            raise RuntimeError("a bug in a check")
        if len(calls) == 2:
            sys.exit("reviews/ must be a folder, not a link or a file")
        return real(*args, **kw)

    monkeypatch.setattr(V, "check", check)
    r = run_in_rounds(tmp_path / "a", [exported(0), exported(1), exported(2)])
    assert [codes(x) for x in r["rounds"]] == [["validation_error", "not_finished"], ["validation_error", "not_finished"],
                                              [*SURFACE, "not_finished"]]
    stored = tmp_path / "a" / "s" / "control" / "candidates"
    assert calls == [stored / f"v0{n}" / "candidate.glb" for n in (1, 2, 3)]  # each check reads the stored copy
    assert r["rounds"][0]["scope"] == "local"
    # The judge is not asked about a round whose host check failed.
    assert (tmp_path / "a" / "s" / "control" / "judged.txt").read_text() == "control/candidates/v03/candidate.glb\n"
    evidence = [json.loads((tmp_path / "a" / "s" / "work" / "evidence" / f"round_0{n}.json").read_text()) for n in (1, 2)]
    assert [e["findings"][0]["why"] for e in evidence] == [
        "RuntimeError: a bug in a check", "reviews/ must be a folder, not a link or a file"]
    assert V.feedback(evidence[0]).startswith("no candidate checked: outcome fail, scope local\n")

    monkeypatch.setattr(V, "check", lambda *a, **kw: sys.exit(143))
    with pytest.raises(SystemExit) as e:
        run_in_rounds(tmp_path / "b", [exported(0), [CRASH]])
    assert e.value.code == 143 and not (tmp_path / "b" / "s" / "result.json").exists()
    assert sum(x["event"] == "round_start" for x in T.read_events(tmp_path / "b" / "s" / "control")) == 1


CANCEL = '''
import os, signal, sys
from harpor.agent import controller, physics_review, tools
signal.signal(signal.SIGTERM, tools._on_sigterm)  # as harpor.webapp.agent_step runs the stage
physics_review.check = lambda root, glb, timeout=600: {"outcome": "pass", "reason": "it rests", "findings": [],
                                                       "asset_sha256": tools.sha256(glb), "seconds": 0.0}
root, script, sock_dir = sys.argv[1:]
env = {**os.environ, "GATOR_SOCKET": os.path.join(sock_dir, "tools.sock")}
controller.run_session(root, [sys.executable, "-m", "harpor.agent.fake_agent", script], env,
                       blender=[sys.executable, "-m", "harpor.agent.fake_agent", "--blender"], sock_dir=sock_dir,
                       budget_s=20.0, poll_s=0.02, kill_after_s=0.5, finish_wait_s=0.5)
'''


def test_cancel_ends_a_session_in_its_second_round(tmp_path):
    """SIGTERM (Cancel) in round 2 ends the session as it ends one without rounds: the agent's group is killed, the
    step exits with 143, and no result.json is written. control/evidence.json keeps round 1 and its candidate, with
    no choice made."""
    root = prepare(tmp_path).parent
    (root / "control" / "physics").mkdir()
    (root / "control" / "physics" / "config.json").write_text("{}")
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"rounds": [exported(0), [{"do": "sleep", "seconds": 60}]]}))
    (tmp_path / "cancel.py").write_text(CANCEL)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    with tempfile.TemporaryDirectory(prefix="gator-") as sock_dir:
        p = subprocess.Popen([sys.executable, str(tmp_path / "cancel.py"), str(root), str(script), sock_dir], env=env)
        try:
            end = time.time() + 60.0
            while time.time() < end and not any(e.get("round") == 2 for e in T.read_events(root / "control")):
                time.sleep(0.05)
            time.sleep(1.0)  # the round's agent is asleep
            p.send_signal(signal.SIGTERM)
            assert p.wait(30) == 143
        finally:
            if p.poll() is None:
                p.kill()
    assert not (root / "result.json").exists()
    assert str(script) not in subprocess.run(["ps", "-ax", "-o", "command"], capture_output=True, text=True).stdout
    ev = json.loads((root / "control" / "evidence.json").read_text())
    assert [x["round"] for x in ev["rounds"]] == [1] and ev["selected"] is None
    assert [(c["version"], c["glb"], c["outcome"]) for c in ev["candidates"]] == [
        (1, "control/candidates/v01/candidate.glb", "fail")]


def test_round_events_and_round_info(tmp_path):
    """Without a round_start every event is the round's. With rounds, the round's readers see only the events after
    its round_start, while file numbers stay global."""
    T.log_event(tmp_path, "export_started", export=1)
    assert [e["event"] for e in T.round_events(tmp_path)] == ["export_started"] and T.round_info(tmp_path) == {}
    T.log_event(tmp_path, "round_start", round=1, kind="build")
    T.log_event(tmp_path, "finish", verdict="complete")
    T.log_event(tmp_path, "round_start", round=2, kind="repair")
    assert T.round_events(tmp_path) == [] and T.round_info(tmp_path) == {"round": 2, "kind": "repair"}
    T.log_event(tmp_path, "export_started", export=2)
    assert [e["export"] for e in T.round_events(tmp_path)] == [2] and T._count(tmp_path, "export_started") == 2


def test_the_budget_view_keeps_the_round_and_counts_only_its_time(tmp_path):
    """start_budget and _extend_budget keep round and kind in run_budget.json, and both they and the controller's
    deadline count only this round's off-clock time. A session without rounds has neither key."""
    work = prepare(tmp_path)
    control = work.parent / "control"
    T.write_budget(work, 0.0, 600.0)
    assert not {"round", "kind"} & set(json.loads((work / "run_budget.json").read_text()))
    T.log_event(control, "round_start", round=1, kind="build")
    T.log_event(control, "export_done", export=1, seconds=100.0, ok=True)
    T.log_event(control, "round_start", round=2, kind="repair")
    T.start_budget(work, control, 1000.0, 450.0)
    view = json.loads((work / "run_budget.json").read_text())
    assert (view["round"], view["kind"], view["excluded_s"], view["exports"]) == (2, "repair", 0.0, [])
    T.log_event(control, "export_done", export=2, seconds=30.0, ok=True)
    assert T._extend_budget(work, control) == 1000.0 + 450.0 + 30.0
    view = json.loads((work / "run_budget.json").read_text())
    assert (view["round"], view["kind"], view["exports"]) == (2, "repair", [{"export": 2, "seconds": 30.0}])
    assert C.deadline(1000.0, 450.0, T.round_events(control), now=1001.0) == (1480.0, 30.0)


def test_next_round_follows_the_papers_budget():
    """At most 3 builds (the build and 2 repairs) and 1 replan, which a build always follows; a pass or scope none
    ends the rounds. The worst case is 4 rounds."""
    def r(kind, outcome=None, scope=None):
        return {"kind": kind, "outcome": outcome, "scope": scope}

    local, structural = r("build", "fail", "local"), r("build", "fail", "structural")
    assert (C.MAX_BUILDS, C.ROUND_BUDGET_S) == (3, {"repair": 450.0, "replan": 300.0})
    assert C.next_round([r("build", "pass", "none")]) is None and C.next_round([r("build", "fail", "none")]) is None
    assert C.next_round([local]) == "repair" and C.next_round([structural]) == "replan"
    assert C.next_round([structural, r("replan")]) == "repair"
    assert C.next_round([structural, r("replan"), r("repair", "fail", "structural")]) == "repair"  # one replan only
    assert C.next_round([local, r("repair", "fail", "structural")]) == "replan"
    assert C.next_round([local, r("repair", "fail", "structural"), r("replan")]) == "repair"
    assert C.next_round([local, r("repair", "fail", "structural"), r("replan"), r("repair", "fail", "local")]) is None
    assert C.next_round([local, r("repair", "fail", "local"), r("repair", "fail", "structural")]) is None


def test_round_prompts_carry_the_whole_task():
    """A new agent knows only its prompt: a repair is the refinement prompt, then the round's budget, the evidence,
    the best candidate and the check's feedback; a replan is a short planning prompt that names the evidence and
    /work/plan.md, with the feedback."""
    feedback = 'candidate sha256 0123456789ab: outcome fail, scope local\n- local physics_implausible: {"x": 1}\n'
    best = {"glb": "exports/002.glb", "round": 1, "passed": False}
    repair = PR.round_prompt("repair", 450.0, feedback, best, "evidence/round_01.json")
    assert repair.startswith(PR.load()) and repair.endswith(feedback)
    flat = " ".join(repair.split())
    for fact in ("physics repair round with a 450 second budget", "evidence/round_01.json",
                 "/work/exports/002.glb, from round 1; its check did not pass", "finish complete"):
        assert fact in flat
    # Only the repair right after a planning round follows its plan, which came from the same check.
    planned = PR.round_prompt("repair", 450.0, feedback, best, "evidence/round_01.json", planned=True)
    assert "plan.md" not in repair and PR.PLAN in planned and planned.replace(PR.PLAN, "") == repair
    replan = PR.round_prompt("replan", 300.0, feedback, None, "evidence/round_02.json")
    assert replan.startswith("This is a planning round with 300 seconds.") and PR.load()[:200] not in replan
    assert "evidence/round_02.json" in replan and "/work/plan.md" in replan and replan.endswith(feedback)
    assert "No round has handed off a valid candidate yet" in PR._best(None)


def test_a_planning_round_refuses_every_helper(tmp_path, monkeypatch):
    """One check in tools.main: a replan makes no export, receipt or off-clock time, so its budget holds and no
    fourth build can be handed off through a later finish. All but finish end with the time left."""
    from harpor.agent.server import COMMANDS

    work = prepare(tmp_path)
    control = work.parent / "control"
    T.log_event(control, "round_start", round=2, kind="replan")
    T.start_budget(work, control, time.time(), 300.0)
    monkeypatch.setenv("GATOR_WORK", str(work))
    monkeypatch.setenv("GATOR_CONTROL", str(control))
    receipt = "r001-0123456789ab"
    args = {"checkpoint": ["--receipt", receipt, "--defect", "x"],
            "finish": ["--receipt", receipt, "--verdict", "complete", "--reason", "x"]}
    before = T.read_events(control)
    for cmd in COMMANDS:
        with pytest.raises(SystemExit) as e:
            T.main([cmd, *args.get(cmd, [])])
        first, *rest = e.value.code.split("\n")
        assert first == (f"{cmd} refused: this is a planning round, in which no helper runs; write your plan to "
                         "/work/plan.md and stop")
        assert rest == [] if cmd == "finish" else (len(rest) == 1 and rest[0].startswith("time left: "))
    assert T.read_events(control) == before and not any((work / "exports").iterdir())


def test_start_round_moves_links_and_does_not_follow_them(tmp_path):
    """The agent of a round may leave links at the names that start_round moves or writes. They are moved or
    replaced, never followed, so the next round's prompt, AGENTS.md and Codex settings are the host's own and the
    file they point to stays as it was. session/rounds and codex_home must be real folders."""
    work = prepare(tmp_path)
    root, session = work.parent, work.parent / "session"
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    links = [session / "prompt.txt", session / "codex.jsonl", session / "stderr.log", work / "AGENTS.md",
             session / "codex_home" / "config.toml"]
    for link in links:
        link.unlink(missing_ok=True)
        link.symlink_to(victim)
    C.start_round(root, 2, "round 2: é\n")
    assert victim.read_text() == "keep me"
    done = session / "rounds" / "01"
    assert sorted(p.name for p in done.iterdir()) == ["codex.jsonl", "prompt.txt", "stderr.log"]
    assert all((done / p.name).is_symlink() for p in links[:3])
    for path, text in ((session / "prompt.txt", "round 2: é\n"), (work / "AGENTS.md", (C.AGENT_DIR / "codex" / "AGENTS.md")
                       .read_text(encoding="utf-8")), (session / "codex_home" / "config.toml", C.codex_config())):
        assert not path.is_symlink() and path.read_text(encoding="utf-8") == text
    assert not [p for d in (session, work, session / "codex_home") for p in d.iterdir() if p.name.startswith(".")]
    (session / "codex_home").rename(session / "codex_home_real")
    (session / "codex_home").symlink_to(tmp_path)
    with pytest.raises(SystemExit, match="codex_home/ must be a folder"):
        C.start_round(root, 3, "round 3\n")
    assert not (tmp_path / "config.toml").exists()
    shutil.rmtree(session / "rounds")
    (session / "rounds").symlink_to(tmp_path)
    with pytest.raises(SystemExit, match="rounds/ must be a folder"):
        C.start_round(root, 3, "round 3\n")


def test_usage_and_network_commands_are_summed_per_round(tmp_path):
    """Each round's agent writes its own log. Its usage is read alone and summed, and so are its commands, since a
    new Codex process may use the same item ids again; the limit applies to all of them."""
    session = tmp_path / "session"
    (session / "rounds" / "01").mkdir(parents=True)
    end = {"type": "end", "reason": "finished", "turns": 3, "totals": usage(10, 20, 30, 40), "cost_usd": 0.25}
    (session / "rounds" / "01" / "claude.jsonl").write_text(json.dumps(end) + "\n")
    (session / "claude.jsonl").write_text(json.dumps({"type": "response", "usage": usage(1, 2), "cost_usd": 0.01}) + "\n")
    logs = C.round_logs(tmp_path, "claude")
    assert logs == [session / "rounds" / "01" / "claude.jsonl", session / "claude.jsonl"]
    assert C.claude_usage(*logs) == {"turns": 4, "tokens": usage(11, 22, 30, 40), "cost_usd": 0.26, "end_reason": None}
    for k, folder in enumerate((session / "rounds" / "01", session)):
        items = [{"type": "item.completed", "item": {"id": f"item_{i}", "type": "command_execution",
                                                     "command": f"pip install r{k}_{i}"}} for i in range(3)]
        (folder / "codex.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    logs = C.round_logs(tmp_path, "codex")
    assert C.network_commands(*logs) == [f"pip install r{k}_{i}" for k, i in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))]
    assert C.round_logs(tmp_path / "one", "codex") == [tmp_path / "one" / "session" / "codex.jsonl"]


def test_round_logs_keep_the_log_of_a_round_whose_next_failed_to_start(tmp_path):
    """start_round moves round 1's log before anything of round 2 can fail, so the session then has 1 run and no
    log in session/. The logs are listed from the folders, so round 1's commands and cost still count."""
    session = tmp_path / "session"
    (session / "rounds" / "01").mkdir(parents=True)
    end = {"type": "end", "reason": "finished", "turns": 3, "totals": usage(10, 20, 30, 40), "cost_usd": 0.25}
    (session / "rounds" / "01" / "claude.jsonl").write_text(json.dumps(end) + "\n")
    logs = C.round_logs(tmp_path, "claude")
    assert logs == [session / "rounds" / "01" / "claude.jsonl", session / "claude.jsonl"]
    assert (C.claude_usage(*logs)["turns"], C.claude_usage(*logs)["cost_usd"]) == (3, 0.25)


def test_a_planted_log_does_not_hang_or_lead_the_controller(tmp_path):
    """The Codex agent can write /session, and start_round moves what is there into session/rounds/NN/. A FIFO there
    would block a plain read until something writes to it, and a link would lead the host out of the session: both
    give nothing, as tools.read_file reads."""
    session = tmp_path / "session"
    (session / "rounds" / "01").mkdir(parents=True)
    os.mkfifo(session / "rounds" / "01" / "codex.jsonl")
    elsewhere = tmp_path / "elsewhere.jsonl"
    elsewhere.write_text(json.dumps({"type": "item.completed", "item": {"id": "x", "type": "command_execution",
                                                                        "command": "pip install x"}}) + "\n")
    (session / "codex.jsonl").symlink_to(elsewhere)
    got = []
    reader = threading.Thread(target=lambda: got.append(C.network_commands(*C.round_logs(tmp_path, "codex"))),
                              daemon=True)
    reader.start()
    reader.join(10.0)
    assert not reader.is_alive() and got == [[]]
    assert C.network_commands(elsewhere) == ["pip install x"]


def test_the_best_candidate_of_an_unchanged_round_is_the_agents_copy(tmp_path):
    """An unchanged pick is stored from root/original.glb, which the agent's container does not see; the next round's
    prompt names the agent's copy of the same bytes, work/original/original.glb, never ../original.glb."""
    root = prepare(tmp_path).parent
    c = C.store_candidate(root, 1, "build", {"source": "finish", "verdict": "unchanged", "glb": root / "original.glb",
                                            "reason": "nothing to fix", "limitations": []})
    assert c["from"] == "original.glb"
    best = C.best_candidate(root)
    assert best == {"glb": "original/original.glb", "round": 1, "passed": False}
    assert T.sha256(root / "work" / best["glb"]) == c["sha256"]
    assert "/work/original/original.glb, from round 1" in PR._best(best)
