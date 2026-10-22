"""Trace of one agent session's refinement loop (build_plan.md note 5).

    python3 gator/agent/loop_trace.py RUN_DIR [--turntable-args OUT.json]

Prints the loop in order: the agent's messages and tool calls from its log (Codex: codex.jsonl;
claude: claude.jsonl, as control/agent.json says), merged with the host's records of exports,
reviews, joints, articulation reviews, checkpoints and finish (with seconds since the clock started),
and the review numbers.
With --turntable-args, writes arguments for blender/turntable.py that show the original and every
successful export side by side. Standard library only.
"""

import argparse
import json
import os
import sys


def lines(path):
    for line in open(path, errors="replace"):
        try:
            yield json.loads(line)
        except ValueError:
            continue


def agent(s):
    """"codex" or "claude", from control/agent.json; sessions from before it ran Codex."""
    path = os.path.join(s, "control", "agent.json")
    return json.load(open(path)).get("agent", "codex") if os.path.exists(path) else "codex"


def codex_trace(s):
    seen = set()
    for e in lines(os.path.join(s, "session", "codex.jsonl")):
        item = e.get("item") or {}
        if e.get("type") != "item.completed" or item.get("id") in seen:
            continue
        seen.add(item.get("id"))
        if item.get("type") == "agent_message":
            print("  says:", " ".join(str(item.get("text")).split())[:220])
        elif item.get("type") == "command_execution":
            cmd = " ".join(str(item.get("command")).split())
            print("  runs [exit %s]: %s" % (item.get("exit_code"), cmd[:160]))


def claude_trace(s, t0):
    """Each response's text and tool calls, with the seconds on the clock when it arrived, and each
    call's time and whether it failed (from its tool_result line)."""
    log = os.path.join(s, "session", "claude.jsonl")
    done = {e.get("tool_use_id"): e for e in lines(log) if e.get("type") == "tool_result"}
    for e in lines(log):
        if e.get("type") == "end":
            print("  ended: %s after %s turns" % (e.get("reason"), e.get("turns")))
        if e.get("type") != "response":
            continue
        print("  %6.0f s  turn %s, %s (%.1f s)" % (e.get("t", t0) - t0, e.get("turn"), e.get("stop_reason"),
                                                   e.get("seconds") or 0))
        for b in e.get("blocks") or []:
            if b.get("type") == "text":
                print("  says:", " ".join(str(b.get("text")).split())[:220])
            elif b.get("type") == "fallback":
                print("  fallback:", json.dumps(b)[:160])
            elif b.get("type") == "tool_use":
                inp, r = b.get("input") or {}, done.get(b.get("id"), {})
                if b.get("name") == "bash":
                    what = "runs: " + " ".join(str(inp.get("command") or "restart").split())
                elif b.get("name") == "view_image":
                    what = "views: %s" % inp.get("path")
                elif b.get("name") == "str_replace_based_edit_tool":
                    what = "edits (%s): %s" % (inp.get("command"), inp.get("path"))
                else:
                    what = "calls %s: %s" % (b.get("name"), json.dumps(inp))
                state = "no result" if not r else "%.1f s%s" % (r.get("seconds") or 0, ", error" if r.get("is_error") else "")
                print("  %s [%s]" % (what[:160], state))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--turntable-args")
    a = ap.parse_args(argv)
    s = os.path.join(os.path.abspath(a.run.rstrip("/")), "s")
    t0 = json.load(open(os.path.join(s, "control", "budget.json")))["started_at"]
    events = list(lines(os.path.join(s, "control", "events.jsonl")))

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
                "checkpoint": "checkpoint %s: %s" % (e.get("glb"), e.get("defect")),
                "finish": "finish %s: %s" % (e.get("verdict"), e.get("reason"))}.get(e["event"], e["event"])
        print("  %6.0f s  %s" % (t, what))
        if e["event"] == "review":
            rv = json.load(open(os.path.join(s, "work", "reviews", e["receipt"], "review.json")))
            ious = ", ".join("%s %.3f" % (v["name"], v["iou_candidate"]) for v in rv["views"] if "iou_candidate" in v)
            print("           mask IoU %s; center shift %.4f; extent ratio %s" % (
                ious, rv.get("center_shift", 0), " ".join("%.3f" % x for x in rv.get("extent_ratio", []))))
        if e["event"] == "review_articulation":
            rp = json.load(open(os.path.join(s, "work", "reviews_articulation", e["receipt"], "report.json")))
            for child, j in rp["joints"].items():
                print("           %s: penetration share %s, deepest %s" % (
                    child, " ".join("%.3f" % x["inside_share"] for x in j["states"]),
                    " ".join("%.3g" % x["max_depth"] for x in j["states"])))
    r = json.load(open(os.path.join(s, "result.json")))
    print("== result: %s from %s (%s), %.0f s on the clock" % (r["verdict"], r["source"], r["glb"], r["timings"]["wall_s"]))
    if r.get("tokens"):
        print("   %s: %s turns, tokens %s, about $%.2f" % (r.get("model"), r.get("turns"), r["tokens"], r.get("cost_usd") or 0))

    if a.turntable_args:
        assets = [{"path": os.path.join(s, "original.glb"), "label": "SAM 3D"}]
        for e in events:
            if e["event"] == "export_done" and e.get("ok"):
                assets.append({"path": os.path.join(s, "work", e["glb"]), "label": "Export %s" % e["export"]})
        json.dump({"assets": assets, "out": os.path.splitext(os.path.abspath(a.turntable_args))[0] + ".mp4",
                   "frames": 180, "fps": 30, "panel": 512, "samples": 32}, open(a.turntable_args, "w"), indent=1)
        print("turntable args:", a.turntable_args, len(assets), "panels")


if __name__ == "__main__":
    sys.exit(main())
