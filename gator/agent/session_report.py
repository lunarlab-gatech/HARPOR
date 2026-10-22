"""Summarize one agent session (build_plan.md note 5): commands, errors, model time and tokens.

    python3 gator/agent/session_report.py RUN_DIR

RUN_DIR holds s/ (the session root). For a Codex session it may also hold requests.jsonl from
pilot/logproxy.py: the Codex CLI's --json log has no timestamps, so times come from the proxy log.
For a claude session (control/agent.json) everything comes from session/claude.jsonl, which times
every request. Standard library only, so it runs on the login node.
"""

import json
import os
import sys

USAGE = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def lines(path):
    for line in open(path, errors="replace"):
        try:
            yield json.loads(line)
        except ValueError:
            continue


def codex(run, t0):
    usage, items = None, {}
    for e in lines(run + "/s/session/codex.jsonl"):
        item = e.get("item") or {}
        if e.get("type") == "item.completed" and item.get("id") not in items:
            items[item.get("id")] = item
            kind = item.get("type")
            if kind == "command_execution":
                print("  CMD [exit %s] %s" % (item.get("exit_code"), str(item.get("command"))[:120].replace("\n", " ")))
            elif kind == "file_change":
                print("  FILE", [c.get("path") for c in item.get("changes", [])][:4])
            elif kind == "agent_message":
                print("  SAY", str(item.get("text"))[:140].replace("\n", " "))
            else:
                print("  ITEM", kind)
        elif e.get("type") in ("turn.failed", "error"):
            print("  ERROR", str(e)[:200])
        elif e.get("type") == "turn.completed":
            usage = e.get("usage")
    print("usage (cumulative)", usage)
    if os.path.exists(run + "/requests.jsonl"):
        reqs = list(lines(run + "/requests.jsonl"))
        sg = [e for e in reqs if e["path"].endswith("/responses")]
        total = sum(e.get("total_s") or 0 for e in sg)
        print("requests %d, model time %.0f s, mean %.1f s, max first byte %.1f s, statuses %s, first at %.0f s, last ends %.0f s" % (
            len(sg), total, total / max(len(sg), 1), max([e.get("ttfb_s") or 0 for e in sg] or [0]),
            sorted(set(e["status"] for e in sg)), (sg[0]["t"] - t0) if sg else 0,
            (sg[-1]["t"] + (sg[-1].get("total_s") or 0) - t0) if sg else 0))


def claude(run, t0):
    """Tool calls, messages, failed tool calls and stops other than tool_use and end_turn, then the
    request count, model time, token totals and the agent's end line."""
    requests, responses, end = [], [], None
    for e in lines(run + "/s/session/claude.jsonl"):
        kind = e.get("type")
        if kind == "request":
            requests.append(e)
        elif kind == "response":
            responses.append(e)
            for b in e.get("blocks") or []:
                inp = b.get("input") or {}
                if b.get("type") == "text":
                    print("  SAY", str(b.get("text"))[:140].replace("\n", " "))
                elif b.get("type") == "fallback":
                    print("  FALLBACK", json.dumps(b)[:160])
                elif b.get("name") == "bash":
                    print("  CMD", str(inp.get("command") or "restart")[:120].replace("\n", " "))
                elif b.get("name") == "str_replace_based_edit_tool":
                    print("  FILE", inp.get("command"), inp.get("path"))
                elif b.get("name") == "view_image":
                    print("  VIEW", inp.get("path"))
                elif b.get("type") == "tool_use":
                    print("  TOOL", b.get("name"))
            if e.get("stop_reason") not in ("tool_use", "end_turn"):
                print("  STOP", e.get("stop_reason"), e.get("stop_details") or "")
        elif kind == "tool_result" and e.get("is_error"):
            print("  TOOL ERROR", e.get("name"), str(e.get("output"))[:120].replace("\n", " "))
        elif kind == "end":
            end = e
        elif kind != "tool_result":
            print("  %s %s" % (str(kind).upper(), json.dumps(e)[:200]))
    secs = [e.get("seconds") or 0 for e in responses]
    print("requests %d (%d answered), model time %.0f s, mean %.1f s, max %.1f s, first at %.0f s, last ends %.0f s, "
          "models %s" % (len(requests), len(responses), sum(secs), sum(secs) / max(len(secs), 1), max(secs or [0]),
                         (requests[0]["t"] - t0) if requests else 0, (responses[-1]["t"] - t0) if responses else 0,
                         sorted(set(str(e.get("model")) for e in responses))))
    totals = {k: sum((e.get("billed") or e.get("usage") or {}).get(k) or 0 for e in responses) for k in USAGE}
    print("tokens", totals)
    if end:
        print("end: %s after %s turns, about $%.2f" % (end.get("reason"), end.get("turns"), end.get("cost_usd") or 0))
    else:
        print("end: no end line (the agent was killed)")


def main(run):
    t0 = json.load(open(run + "/s/control/budget.json"))["started_at"]
    path = run + "/s/control/agent.json"
    kind = json.load(open(path)).get("agent") if os.path.exists(path) else "codex"
    (claude if kind == "claude" else codex)(run, t0)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
