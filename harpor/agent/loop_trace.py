"""Trace of one agent session's refinement loop (build_plan.md note 5).

    python3 harpor/agent/loop_trace.py RUN_DIR

Prints the loop in order: the agent's messages and tool calls from its log (Codex: codex.jsonl;
claude: claude.jsonl, as control/agent.json says), merged with the host's records of exports,
reviews, joints, articulation reviews, simulate runs, checkpoints and finish (with seconds since the
clock started), and the review numbers. A session with rounds (controller.run_rounds) shows its last
round: the clock and the agent's log are that round's, and earlier rounds' logs are in
session/rounds/NN/. Text that the agent wrote is printed as harpor.agent.tools.printable cleans it, one line
without control characters, also from a session older than that cleaning. Files that the agent can write,
codex.jsonl and the review files in work/, are read as tools.read_file reads (a link or a FIFO gives
nothing), and a line or file of another shape is skipped. Standard library only, with harpor.agent.tools,
which needs no more.
"""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run as a script: harpor/ is in the repository
from harpor.agent.tools import printable, read_file, read_jsonl  # noqa: E402


def work_json(s, *parts):
    """A JSON file in work/, which the agent can write: read as tools.read_file reads."""
    return json.loads(read_file(os.path.join(s, "work", *parts)))


def agent(s):
    """"codex" or "claude", from control/agent.json."""
    return json.load(open(os.path.join(s, "control", "agent.json")))["agent"]


def codex_trace(s):
    seen = set()
    for e in read_jsonl(os.path.join(s, "session", "codex.jsonl")):
        item = e.get("item")
        if e.get("type") != "item.completed" or not isinstance(item, dict) or repr(item.get("id")) in seen:
            continue
        seen.add(repr(item.get("id")))  # repr: an id the agent wrote may be a list
        if item.get("type") == "agent_message":
            print("  says:", printable(item.get("text"))[:220])
        elif item.get("type") == "command_execution":  # the Codex agent can write codex.jsonl, exit_code too
            print("  runs [exit %s]: %s" % (printable(item.get("exit_code")), printable(item.get("command"))[:160]))


def claude_trace(s, t0):
    """Each response's text and tool calls, with the seconds on the clock when it arrived, and each
    call's time and whether it failed (from its tool_result line)."""
    log = os.path.join(s, "session", "claude.jsonl")
    done = {e.get("tool_use_id"): e for e in read_jsonl(log) if e.get("type") == "tool_result"}
    for e in read_jsonl(log):
        if e.get("type") == "end":
            print("  ended: %s after %s turns" % (e.get("reason"), e.get("turns")))
        if e.get("type") != "response":
            continue
        print("  %6.0f s  turn %s, %s (%.1f s)" % (e.get("t", t0) - t0, e.get("turn"), e.get("stop_reason"),
                                                   e.get("seconds") or 0))
        for b in e.get("blocks") or []:
            if b.get("type") == "text":
                print("  says:", printable(b.get("text"))[:220])
            elif b.get("type") == "fallback":
                print("  fallback:", json.dumps(b)[:160])
            elif b.get("type") == "tool_use":
                inp, r = b.get("input") or {}, done.get(b.get("id"), {})
                if b.get("name") == "bash":
                    what = "runs: " + str(inp.get("command") or "restart")
                elif b.get("name") == "view_image":
                    what = "views: %s" % inp.get("path")
                elif b.get("name") == "str_replace_based_edit_tool":
                    what = "edits (%s): %s" % (inp.get("command"), inp.get("path"))
                else:
                    what = "calls %s: %s" % (b.get("name"), json.dumps(inp))
                state = "no result" if not r else "%.1f s%s" % (r.get("seconds") or 0, ", error" if r.get("is_error") else "")
                print("  %s [%s]" % (printable(what)[:160], state))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args(argv)
    s = os.path.join(os.path.abspath(a.run.rstrip("/")), "s")
    t0 = json.load(open(os.path.join(s, "control", "budget.json")))["started_at"]
    events = list(read_jsonl(os.path.join(s, "control", "events.jsonl")))
    starts = [i for i, e in enumerate(events) if e.get("event") == "round_start"]
    if starts:  # budget.json is the last round's, so only its events have a time on that clock
        e = events[starts[-1]]
        print("== round %s of %s (%s); earlier rounds' logs are in session/rounds/" % (e.get("round"), len(starts),
                                                                                       e.get("kind")))
        events = events[starts[-1] + 1:]

    print("== agent (%s), in order" % agent(s))
    if agent(s) == "claude":
        claude_trace(s, t0)
    else:
        codex_trace(s)

    print("== host records (seconds on the clock)")
    for e in events:
        t = e["time"] - t0
        what = {"export_started": "export %s started" % e.get("export"),
                "export_done": "export %s done in %.1f s, ok=%s" % (e.get("export"), e.get("seconds", 0), e.get("ok")),
                "review": "review %s of %s" % (e.get("receipt"), e.get("glb")),
                "joint": "joint of %s %s%s" % (e.get("child"), e.get("action"), "" if e.get("ok") else ", refused: %s"
                                                % e.get("why")),
                "review_articulation": "articulation review %s of %s, %s joints" % (e.get("receipt"), e.get("glb"),
                                                                                    e.get("joints")),
                "simulate_started": "simulate %s of %s started" % (e.get("simulate"), e.get("glb")),
                "simulate_done": "simulate %s done in %.1f s, ok=%s, outcome %s, scope %s" % (
                    e.get("simulate"), e.get("seconds", 0), e.get("ok"), e.get("outcome"), e.get("scope")),
                "checkpoint": "checkpoint %s: %s" % (e.get("glb"), e.get("defect")),
                "finish": "finish %s: %s" % (e.get("verdict"), e.get("reason"))}.get(e["event"], e["event"])
        print("  %6.0f s  %s" % (t, printable(what)))  # a defect, a reason and a joint's child are the agent's
        # review.json and report.json are in work/, which the agent can write: one of another shape is skipped
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError, AttributeError):
            if e["event"] == "review":
                rv = work_json(s, "reviews", e["receipt"], "review.json")
                ious = ", ".join("%s %.3f" % (printable(v["name"]), v["iou_candidate"]) for v in rv["views"]
                                 if "iou_candidate" in v)
                print("           mask IoU %s; center shift %.4f; extent ratio %s" % (
                    ious, rv.get("center_shift", 0), " ".join("%.3f" % x for x in rv.get("extent_ratio", []))))
            if e["event"] == "review_articulation":
                rp = work_json(s, "reviews_articulation", e["receipt"], "report.json")
                for child, j in rp["joints"].items():
                    print("           %s: penetration share %s, deepest %s" % (
                        printable(child), " ".join("%.3f" % x["inside_share"] for x in j["states"]),
                        " ".join("%.3g" % x["max_depth"] for x in j["states"])))
    r = json.load(open(os.path.join(s, "result.json")))
    print("== result: %s from %s (%s), %.0f s on the clock" % (r["verdict"], r["source"], r["glb"], r["timings"]["wall_s"]))
    for x in r.get("rounds") or []:
        print("   round %s (%s): %s, outcome %s, scope %s" % (x["round"], x["kind"], x["exit"], x["outcome"], x["scope"]))
    if r.get("tokens"):
        print("   %s: %s turns, tokens %s, about $%.2f" % (r.get("model"), r.get("turns"), r["tokens"], r.get("cost_usd") or 0))


if __name__ == "__main__":
    sys.exit(main())
