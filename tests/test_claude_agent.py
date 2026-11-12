"""The Claude Opus 5.5 agent loop (harpor/agent/claude_agent.py) with fake clients (build_plan.md note 5).

No test reaches the network. The scripted responses are the SDK's own beta types
(BetaMessage.model_validate), so their content blocks behave like real ones; one test runs the real
anthropic client over an httpx2 MockTransport that serves server-sent events.
"""

import base64
import contextlib
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from PIL import Image

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")
from anthropic.types.beta import BetaMessage  # noqa: E402

from harpor.agent import claude_agent as A  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROMPT = "Prompt 1, byte for byte: é\n"
KEY = "sk-ant-api03-" + "Kq7Zt2" * 12


# ------------------------------------------------------------------------------------- fake client


def msg(*blocks, stop="tool_use", model=A.MODEL, stop_details=None, usage=None):
    msg.n = getattr(msg, "n", 0) + 1
    return BetaMessage.model_validate({
        "id": f"msg_{msg.n:04d}", "type": "message", "role": "assistant", "model": model, "content": list(blocks),
        "stop_reason": stop, "stop_sequence": None, "stop_details": stop_details,
        "usage": usage or {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 1000,
                           "cache_creation_input_tokens": 200}})


def thinking(text="weighing it"):
    return {"type": "thinking", "thinking": text, "signature": "c2lnbmF0dXJl"}


def text(t):
    return {"type": "text", "text": t}


def use(name, inp, id):
    return {"type": "tool_use", "id": id, "name": name, "input": inp}


def bash(command, id):
    return use("bash", {"command": command}, id)


def edit(id, **inp):
    return use("str_replace_based_edit_tool", inp, id)


def fallback(src=A.MODEL, dst="claude-opus-5"):
    return {"type": "fallback", "from": {"model": src}, "to": {"model": dst},
            "trigger": {"type": "refusal", "category": "cyber"}}


def plain(x):
    """JSON-ready copy of request parameters (SDK models through to_dict)."""
    if hasattr(x, "to_dict"):
        return x.to_dict()
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


class MidStream:
    """A scripted error raised after the stream opened (from get_final_message), as the real SDK raises
    an error event or a dropped connection."""

    def __init__(self, error):
        self.error = error


class FakeStream:
    def __init__(self, item):
        self.item = item

    def __enter__(self):
        if isinstance(self.item, anthropic.APIError):  # the real SDK raises these when the request is sent
            raise self.item
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        if isinstance(self.item, MidStream):
            raise self.item.error
        if isinstance(self.item, BaseException):
            raise self.item
        return self.item


class FakeClient:
    """beta.messages.stream(**kw) returns the next scripted item: a BetaMessage, or an exception to raise.
    Every call's parameters are kept (raw, and as JSON taken at call time)."""

    def __init__(self, script):
        self.script, self.raw, self.calls = list(script), [], []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))

    def stream(self, **kw):
        self.raw.append(kw)
        self.calls.append(json.loads(json.dumps(plain(kw))))
        return FakeStream(self.script.pop(0))


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "s"
    for d in ("work/input", "session", "control"):
        (r / d).mkdir(parents=True)
    (r / "session" / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    (r / "control" / "events.jsonl").touch()
    return r


@pytest.fixture
def key_file(tmp_path):
    f = tmp_path / "keys" / "anthropic.env"  # outside the session root, as on ICE (/dev/shm)
    f.parent.mkdir()
    f.write_text(f"ANTHROPIC_API_KEY={KEY}\n")
    return f


def run(root, script, key_file=None, **kw):
    client, out = FakeClient(script), io.StringIO()
    code = A.run(root, sock_dir=str(root.parent), key_file=key_file, shell="host", client=client, out=out, **kw)
    return code, client, log_lines(root), out.getvalue()


def log_lines(root):
    return [json.loads(line) for line in (root / "session" / "claude.jsonl").read_text().splitlines()]


def of(lines, kind):
    return [e for e in lines if e["type"] == kind]


def results(call):
    """{tool_use_id: tool_result} of the user message that ends a request."""
    last = call["messages"][-1]
    assert last["role"] == "user"
    return {r["tool_use_id"]: r for r in last["content"]}


def assert_append_only(client):
    calls = client.calls
    for prev, cur in zip(calls, calls[1:]):
        assert cur["messages"][:len(prev["messages"])] == prev["messages"]
        assert len(cur["messages"]) > len(prev["messages"])
        assert json.dumps(cur["system"]) == json.dumps(prev["system"])
        assert json.dumps(cur["tools"]) == json.dumps(prev["tools"])
        assert {k: v for k, v in cur.items() if k != "messages"} == {k: v for k, v in prev.items() if k != "messages"}


# ------------------------------------------------------------------------------------------- loop


def test_multi_turn_session_with_persistent_shell(root):
    script = [
        msg(thinking("look first"), text("Setting up."), bash("cd input && export FOO=bar && echo ok", "toolu_a"),
            edit("toolu_b", command="create", path="/work/notes/plan.txt", file_text="one\ntwo\n")),
        msg(bash("pwd; echo $FOO; cat ../notes/plan.txt", "toolu_c")),
        msg(text("All done."), stop="end_turn"),
    ]
    code, client, lines, out = run(root, script)
    assert code == 0
    assert [e["type"] for e in lines] == ["start", "request", "response", "tool_result", "tool_result", "request",
                                          "response", "tool_result", "request", "response", "end"]
    r1, r2 = results(client.calls[1]), results(client.calls[2])
    assert list(r1) == ["toolu_a", "toolu_b"]  # one user message, in tool_use order
    assert r1["toolu_a"]["content"] == "ok\n" and "is_error" not in r1["toolu_a"]
    assert r1["toolu_b"]["content"].startswith("created /work/notes/plan.txt")
    pwd, foo, one, two = r2["toolu_c"]["content"].splitlines()
    assert pwd.endswith("/s/work/input") and foo == "bar" and (one, two) == ("one", "two")
    assert (root / "work" / "notes" / "plan.txt").read_text() == "one\ntwo\n"
    # The first user message is the prompt, unchanged; each response goes back as the very same content.
    assert client.calls[0]["messages"] == [{"role": "user", "content": PROMPT}]
    for i, m in enumerate(script[:2]):
        assert client.raw[i + 1]["messages"][2 * i + 1] == {"role": "assistant", "content": m.content}
        assert client.raw[i + 1]["messages"][2 * i + 1]["content"] is m.content
    assert_append_only(client)
    end = of(lines, "end")[0]
    assert (end["reason"], end["exit_code"], end["turns"]) == ("end_turn", 0, 3)
    assert end["totals"] == {"input_tokens": 300, "output_tokens": 150, "cache_read_input_tokens": 3000,
                             "cache_creation_input_tokens": 600}
    assert end["cost_usd"] == round((300 * 4 + 150 * 20 + 3000 * 0.2 + 600 * 5) / 1e6, 4)
    resp = of(lines, "response")[0]
    assert resp["model"] == A.MODEL and resp["stop_reason"] == "tool_use" and resp["id"] == script[0].id
    assert resp["blocks"][:2] == [{"type": "thinking", "text": "look first"}, {"type": "text", "text": "Setting up."}]
    assert resp["blocks"][2] == {"type": "tool_use", "id": "toolu_a", "name": "bash",
                                 "input": {"command": "cd input && export FOO=bar && echo ok"}}
    tr = of(lines, "tool_result")[0]
    assert (tr["turn"], tr["tool_use_id"], tr["name"], tr["is_error"], tr["output"]) == (1, "toolu_a", "bash", False, "ok\n")
    assert "turn 1: tool_use" in out and "end: end_turn (exit 0)" in out


def test_request_parameters(root):
    code, client, lines, _ = run(root, [msg(text("hi"), stop="end_turn")])
    call = client.calls[0]
    assert code == 0 and set(call) == {"model", "max_tokens", "system", "tools", "thinking", "output_config",
                                       "cache_control", "betas", "fallbacks", "messages"}
    assert call["model"] == "claude-opus-5-5" and call["max_tokens"] == 64000
    assert call["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert call["output_config"] == {"effort": "high"}
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert call["cache_control"] == {"type": "ephemeral"}
    assert call["system"] == [{"type": "text", "text": A.SYSTEM_FILE.read_text(encoding="utf-8"),
                               "cache_control": {"type": "ephemeral"}}]
    assert call["tools"] == [
        {"type": "bash_20250124", "name": "bash"},
        {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters": 30000},
        {"name": "view_image", "description": A.TOOLS[2]["description"], "eager_input_streaming": True,
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
                          "additionalProperties": False}}]
    assert call["tools"] == json.loads(json.dumps(A.TOOLS))
    desc = A.TOOLS[2]["description"]
    for fact in ("larger than 2576 px on the long side", "lossless when it fits 2.5 MB as base64 (about 1.9 MB of file)",
                 "as JPEG", "at most 100 images and 24 MB of them"):
        assert fact in desc
    start = of(lines, "start")[0]
    assert (start["model"], start["effort"]) == ("claude-opus-5-5", "high")
    assert (start["image_side"], start["system"]) == (2576, str(A.SYSTEM_FILE))
    assert start["system_sha256"] == hashlib.sha256(A.SYSTEM_FILE.read_bytes()).hexdigest()


def test_image_limits_fit_the_request_limit():
    """Opus 5.5 takes 2576 px on the long side. Every image stays in the history and goes with every
    request, which the API limits to 32 MB: 24 MB of base64 images leave room for the text."""
    assert (A.MAX_IMAGE_SIDE, A.MAX_IMAGE_B64, A.JPEG_QUALITIES) == (2576, 2_500_000, (92, 80, 70))
    assert (A.IMAGE_BUDGET_BYTES, A.MAX_IMAGES) == (24_000_000, 100)
    assert A.IMAGE_BUDGET_BYTES + 4_000_000 <= 32_000_000 and A.b64_size(A.API_IMAGE_BYTES) <= 5_000_000
    budget = A.ImageBudget()
    assert (budget.limit, budget.count) == (24_000_000, 100)
    for n in (0, 1, 2, 3, 4, 1000, 1_875_000, 1_875_001):
        assert A.b64_size(n) == len(base64.b64encode(bytes(n)))
    assert A.b64_size(1_875_000) == A.MAX_IMAGE_B64 < A.b64_size(1_875_001)


def test_effort(root):
    code, client, lines, _ = run(root, [msg(text("hi"), stop="end_turn")], effort="max")
    assert client.calls[0]["output_config"] == {"effort": "max"}
    with pytest.raises(ValueError):
        A.run(root, client=FakeClient([]), effort="disabled")


def test_model(root):
    """--model claude-fable-5-2: the requests name Claude Fable 5.2 and are otherwise those of Opus 5.5, and its
    turns bill at its prices ($8 input, $40 output, $0.20 cache read, $10 cache write per million tokens). A model outside
    tools.CLAUDE_MODELS is refused before any request."""
    fable = A.tools.FABLE_MODEL
    code, client, lines, _ = run(root, [msg(text("hi"), stop="end_turn", model=fable)], model=fable, effort="high")
    assert code == 0 and client.calls[0]["model"] == "claude-fable-5-2"
    opus = json.loads(json.dumps(A.request_params(A.MODEL, "high", A.SYSTEM_FILE.read_text(encoding="utf-8"))))
    assert {k: v for k, v in client.calls[0].items() if k != "messages"} == {**opus, "model": fable}
    assert of(lines, "start")[0]["model"] == fable
    assert of(lines, "response")[0]["cost_usd"] == round((100 * 8 + 50 * 40 + 1000 * 0.2 + 200 * 10) / 1e6, 6)
    with pytest.raises(ValueError, match="model must be one of"):
        A.run(root, client=FakeClient([]), model="claude-opus-5")
    with pytest.raises(SystemExit):
        A.main([str(root), "--key-file", "k.env", "--model", "claude-opus-5"])


def test_system_prompt_is_the_work_folder_manual():
    s = A.SYSTEM_FILE.read_text(encoding="utf-8")
    for fact in ("no network", "300 s", "run_budget.json", "view_image", "export_glb", "review", "checkpoint",
                 "finish", "--python-exit-code", "str_replace_based_edit_tool"):
        assert fact in s
    from harpor.agent import tools

    for verdict in tools.VERDICTS:  # every verdict that finish takes is named for the model
        assert f"`{verdict}`" in s, verdict
    assert len(s) > 3000  # above the 512-token minimum for a cache entry
    assert "An image larger than 2576 px on its long side is shown downscaled." in s and "1568" not in s
    # The deadline facts are those of Codex's AGENTS.md, without time pressure on top.
    assert "up to 600 s in total. Read the file again after an export.\n" in s
    assert "The session ends at the" not in s and "thinking time" not in s
    agents = (A.AGENT_DIR / "codex" / "AGENTS.md").read_text(encoding="utf-8")
    assert "The session ends at the" not in agents and "thinking time" not in agents
    # Both manuals describe the shared time-left line; only this one has the measured step times.
    line = "Every helper but `finish` ends its output with the seconds left before the deadline, also when it"
    assert line in s and line in agents
    assert "## Measured times" in s and "median 29 s" in s and "Measured times" not in agents


def test_stdout_shows_the_models_text_and_output_as_printable_lines(root):
    """The model's text and its commands' output reach stdout (claude_stdout.log) as one line each, without control
    characters (tools.printable)."""
    script = [msg(text("look\x1b]0;title\x07\n‮done"), bash(r"printf 'a\033]52;c;aGk=\007\nb'", "toolu_1")),
              msg(text("ok"), stop="end_turn")]
    code, client, lines, out = run(root, script)
    assert code == 0 and "  text: look]0;title done\n" in out and "  -> bash: a]52;c;aGk= b" in out
    assert [c for c in out if not c.isprintable() and c != "\n"] == []


def test_the_log_is_ascii(root):
    """An OSC made of C1 controls alone (U+009D ... U+009C) needs no ESC, and a terminal that takes C1 controls in
    UTF-8 runs it on cat or tail -f of the log. JSON escapes it, and DEL, bidi and zero-width characters, in the
    model's text and in command output; json.loads gives the same strings back."""
    dirty = "csi\x9b2J osc\x9d52;c;aGk=\x9c del\x7f rlo‮abc zwj‍ é"
    code, client, lines, _ = run(root, [msg(text(dirty), bash(f"printf '%s' '{dirty}'", "toolu_1")),
                                        msg(text("ok"), stop="end_turn")])
    data = (root / "session" / "claude.jsonl").read_bytes()
    assert code == 0 and data.isascii()
    assert of(lines, "response")[0]["blocks"][0]["text"] == dirty and dirty in of(lines, "tool_result")[0]["output"]


def test_end_turn_and_max_tokens_without_tool_use_end_the_session(root):
    code, client, lines, _ = run(root, [msg(text("cut off mid-sent"), stop="max_tokens")])
    assert code == 0 and len(client.calls) == 1 and of(lines, "end")[0]["reason"] == "max_tokens"


def test_max_tokens_with_tool_use_runs_nothing(root):
    script = [msg(bash("touch made.txt", "toolu_1"), edit("toolu_2", command="create", path="x.txt", file_text="ab"),
                  stop="max_tokens"),
              msg(text("ok"), stop="end_turn")]
    code, client, lines, _ = run(root, script)
    assert code == 0 and not (root / "work" / "made.txt").exists() and not (root / "work" / "x.txt").exists()
    r = results(client.calls[1])
    assert list(r) == ["toolu_1", "toolu_2"] and all(x["is_error"] is True for x in r.values())
    assert all("cut off at the output limit" in x["content"] for x in r.values())
    assert [e["is_error"] for e in of(lines, "tool_result")] == [True, True]
    assert_append_only(client)


def test_refusal_stops_before_any_tool(root):
    details = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    code, client, lines, out = run(root, [msg(text("Sure, "), bash("touch made.txt", "toolu_1"), stop="refusal",
                                              stop_details=details)])
    assert code == 2 and len(client.calls) == 1 and not (root / "work" / "made.txt").exists()
    assert of(lines, "tool_result") == []
    assert of(lines, "response")[0]["stop_details"]["category"] == "cyber"
    end = of(lines, "end")[0]
    assert (end["reason"], end["exit_code"], end["stop_details"]["explanation"]) == ("refusal", 2, "declined")


@pytest.mark.parametrize("stop", ["end_turn", "stop_sequence", "compaction"])
def test_tools_run_only_for_stop_reason_tool_use(root, stop):
    """A tool_use of a turn that did not stop for it may be cut off: it gets a not-run error result."""
    script = [msg(bash("touch made.txt", "toolu_1"), stop=stop), msg(text("ok"), stop="end_turn")]
    code, client, lines, _ = run(root, script)
    assert code == 0 and len(client.calls) == 2 and not (root / "work" / "made.txt").exists()
    r = results(client.calls[1])["toolu_1"]
    assert r["is_error"] is True and r["content"].startswith(f"not run: the turn stopped with stop reason '{stop}'")
    assert_append_only(client)


def test_context_window_exceeded_runs_nothing_and_ends(root):
    """The tolerant parser drops an unfinished new_str; str_replace would then delete old_str."""
    (root / "work" / "a.py").write_text("x = 1\n")
    cut = edit("toolu_1", command="str_replace", path="a.py", old_str="x = 1")
    code, client, lines, _ = run(root, [msg(cut, stop="model_context_window_exceeded")])
    assert code == 0 and len(client.calls) == 1 and (root / "work" / "a.py").read_text() == "x = 1\n"
    assert of(lines, "tool_result") == []
    end = of(lines, "end")[0]
    assert (end["reason"], end["message"]) == ("model_context_window_exceeded", "the conversation filled the context window")


def test_pause_turn_is_sent_back_unchanged(root):
    paused = msg(thinking(), text("half way"), stop="pause_turn")
    code, client, lines, _ = run(root, [paused, msg(text("done"), stop="end_turn")])
    assert code == 0 and len(client.calls) == 2
    assert client.raw[1]["messages"][-1] == {"role": "assistant", "content": paused.content}
    assert_append_only(client)


def test_pause_turn_continuations_are_capped(root):
    def paused():
        return msg(text("still working"), stop="pause_turn")

    code, client, _, _ = run(root, [paused() for _ in range(5)] + [msg(text("done"), stop="end_turn")])
    assert code == 0 and len(client.calls) == 6
    code, client, lines, _ = run(root, [paused() for _ in range(6)])
    end = of(lines, "end")[-1]
    assert code == 3 and len(client.calls) == 6
    assert (end["reason"], end["message"]) == ("pause_turn_limit", "the API paused the turn 6 times in a row")


def iteration(kind, model, i, o, read=0, write=0):
    return {"type": kind, "model": model, "input_tokens": i, "output_tokens": o, "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write}


def test_cost_counts_every_attempt_at_its_models_prices(root):
    """With a refusal fallback the top-level usage covers only the fallback's attempt; usage.iterations
    lists the declined attempt too, which bills at Opus 5.5's prices, and the fallback at Opus 5's. A
    sticky-routed turn without iterations is priced by msg.model; 1-hour cache writes at 2x input."""
    fell_back = msg(fallback(), text("Continuing."), bash("echo hi", "toolu_1"), model="claude-opus-5", usage={
        "input_tokens": 100, "output_tokens": 40, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0,
        "iterations": [iteration("message", None, 100, 500, read=1000),
                       iteration("fallback_message", "claude-opus-5", 100, 40, read=1000)]})
    sticky = msg(bash("echo again", "toolu_2"), model="claude-opus-5", usage={
        "input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 300,
        "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200}})
    last = msg(text("done"), stop="end_turn", usage={"input_tokens": 1, "output_tokens": 2})
    code, client, lines, out = run(root, [fell_back, sticky, last])
    want = [(100 * 4 + 500 * 20 + 1000 * 0.2 + 100 * 5 + 40 * 25 + 1000 * 0.5) / 1e6,
            (10 * 5 + 20 * 25 + 100 * 6.25 + 200 * 10) / 1e6,
            (1 * 4 + 2 * 20) / 1e6]
    responses = of(lines, "response")
    assert code == 0 and [r["cost_usd"] for r in responses] == [round(c, 6) for c in want]
    assert responses[0]["iterations"] == [iteration("message", A.MODEL, 100, 500, read=1000),
                                          iteration("fallback_message", "claude-opus-5", 100, 40, read=1000)]
    assert responses[0]["billed"] == {"input_tokens": 200, "output_tokens": 540, "cache_read_input_tokens": 2000,
                                      "cache_creation_input_tokens": 0}
    assert responses[0]["usage"]["output_tokens"] == 40 and "iterations" not in responses[1]
    end = of(lines, "end")[0]
    assert end["cost_usd"] == round(sum(want), 4) and end["reissued"] == 0
    assert end["totals"] == {"input_tokens": 211, "output_tokens": 562, "cache_read_input_tokens": 2000,
                             "cache_creation_input_tokens": 300}
    assert "fallback_message claude-opus-5 out 40" in out
    assert A._attempt_cost(SimpleNamespace(input_tokens=1_000_000), "claude-unknown-9") == 8.0  # the dearest prices


def test_fallback_mid_output(root):
    """Before the last fallback block only text and fallback blocks go back; the declined model's tool
    call never runs. The model that served the turn is logged."""
    m = msg(thinking("declined part"), bash("touch a.txt", "toolu_a"), fallback(), text("Continuing."),
            bash("touch b.txt", "toolu_b"), model="claude-opus-5")
    code, client, lines, _ = run(root, [m, msg(text("done"), stop="end_turn")])
    assert code == 0 and not (root / "work" / "a.txt").exists() and (root / "work" / "b.txt").exists()
    sent = client.calls[1]["messages"][1]["content"]
    assert [b["type"] for b in sent] == ["fallback", "text", "tool_use"] and sent[2]["id"] == "toolu_b"
    assert list(results(client.calls[1])) == ["toolu_b"]
    resp = of(lines, "response")[0]
    assert resp["model"] == "claude-opus-5"
    assert resp["blocks"][2] == {"type": "fallback", "from": {"model": A.MODEL}, "to": {"model": "claude-opus-5"},
                                 "trigger": {"type": "refusal", "category": "cyber"}}
    assert_append_only(client)


def test_pre_output_fallback_content_is_unchanged(root):
    m = msg(fallback(), thinking(), bash("echo hi", "toolu_1"), model="claude-opus-5")
    code, client, _, _ = run(root, [m, msg(text("done"), stop="end_turn")])
    assert code == 0 and client.raw[1]["messages"][1]["content"] is m.content


def test_unparseable_tool_json_is_reissued(root):
    code, client, lines, _ = run(root, [ValueError("Unable to parse tool parameter JSON"), msg(text("ok"), stop="end_turn")])
    assert code == 0 and len(client.calls) == 2 and client.calls[0] == client.calls[1]
    assert [e["type"] for e in lines] == ["start", "request", "retry", "request", "response", "end"]
    assert of(lines, "request")[1] == {**of(lines, "request")[1], "turn": 1, "attempt": 2}


def test_unparseable_tool_json_gives_up_after_two_reissues(root):
    code, client, lines, _ = run(root, [ValueError("bad json")] * 3)
    assert code == 3 and len(client.calls) == 3
    end = of(lines, "end")[0]
    assert (end["reason"], end["exit_code"], len(of(lines, "retry"))) == ("invalid_tool_json", 3, 3)


def api_error(cls, status, etype):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    if cls is anthropic.APIConnectionError:
        return cls(request=request)
    response = httpx2.Response(status, request=request, headers={"request-id": "req_test"})
    return cls(f"Error code: {status}", response=response,
               body={"type": "error", "error": {"type": etype, "message": "nope"}})


@pytest.mark.parametrize("cls,status,etype,label", [
    (anthropic.AuthenticationError, 401, "authentication_error", "the API key was rejected"),
    (anthropic.PermissionDeniedError, 403, "permission_error", "the key may not use this model or feature"),
    (anthropic.NotFoundError, 404, "not_found_error", "model or endpoint not found"),
    (anthropic.BadRequestError, 400, "invalid_request_error", "the API rejected the request"),
    (anthropic.RequestTooLargeError, 413, "request_too_large", "the request is too large (the images in the history)"),
    (anthropic.RateLimitError, 429, "rate_limit_error", "rate limited, also after the SDK's retries"),
    (anthropic.InternalServerError, 500, "api_error", "API server error, also after the SDK's retries"),
    (anthropic.OverloadedError, 529, "overloaded_error", "API overloaded, also after the SDK's retries"),
    (anthropic.UnprocessableEntityError, 422, "invalid_request_error", "the API answered with an error status"),
    (anthropic.APIConnectionError, None, None, "could not reach the API, also after the SDK's retries"),
])
def test_api_error_ends_with_exit_3(root, cls, status, etype, label):
    """An error raised when the request is sent is not sent again: the SDK has retried it if it could."""
    code, client, lines, out = run(root, [msg(bash("echo one", "toolu_1")), api_error(cls, status, etype)])
    assert code == 3 and len(client.calls) == 2 and of(lines, "retry") == []
    end = of(lines, "end")[0]
    assert (end["reason"], end["exit_code"], end["message"], end["turns"]) == ("api_error", 3, label, 1)
    assert (end["error"]["class"], end["error"]["status"], end["error"]["type"]) == (cls.__name__, status, etype)
    assert "end: api_error (exit 3), 1 turns" in out


def error_event(etype):
    """The SDK's exception for an SSE error event: an APIStatusError with the stream's status, 200."""
    return api_error(anthropic.APIStatusError, 200, etype)


def test_stream_broken_after_it_opened_is_sent_again(root, monkeypatch):
    """The SDK retries only the request; an error event, a dropped connection or a stream without a stop
    reason after it opened sends the same request again, after a doubling backoff."""
    sleeps, real_sleep = [], time.sleep  # the backoffs are recorded, not slept; subprocess polling still sleeps
    monkeypatch.setattr(A.time, "sleep", lambda s: sleeps.append(s) if s >= A.RETRY_BACKOFF_S else real_sleep(s))
    dropped = httpx2.ReadError("connection reset", request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    incomplete = msg(text("half an answ"), stop=None)
    script = [msg(bash("echo once >> ran.txt", "toolu_1")),
              MidStream(error_event("overloaded_error")), MidStream(dropped), incomplete,
              msg(text("done"), stop="end_turn")]
    code, client, lines, out = run(root, script)
    assert code == 0 and len(client.calls) == 5 and (root / "work" / "ran.txt").read_text() == "once\n"
    assert client.calls[1] == client.calls[2] == client.calls[3] == client.calls[4]  # the history stays append-only
    retries = of(lines, "retry")
    assert [r["attempt"] for r in retries] == [1, 2, 3] and sleeps == [2.0, 4.0, 8.0]
    assert "overloaded_error" in retries[0]["error"] and "ReadError" in retries[1]["error"]
    assert "without a stop reason" in retries[2]["error"]
    assert [(e["turn"], e.get("attempt")) for e in of(lines, "request")] == [(1, None), (2, None), (2, 2), (2, 3), (2, 4)]
    end = of(lines, "end")[0]
    assert (end["reason"], end["exit_code"], end["turns"], end["reissued"]) == ("end_turn", 0, 2, 3)
    assert "(at least; 3 failed attempts not counted)" in out


def test_stream_breaking_off_gives_up_after_three_reissues(root, monkeypatch):
    monkeypatch.setattr(A, "RETRY_BACKOFF_S", 0.0)
    code, client, lines, _ = run(root, [MidStream(error_event("api_error"))] * 4)
    assert code == 3 and len(client.calls) == 4 and len(of(lines, "retry")) == 4
    end = of(lines, "end")[0]
    assert (end["reason"], end["exit_code"], end["reissued"]) == ("api_error", 3, 4)
    assert end["message"] == "the response stream broke off, also after 3 re-issues" and "api_error" in end["error"]


def test_error_event_that_cannot_be_retried_ends_the_session(root):
    code, client, lines, _ = run(root, [MidStream(error_event("invalid_request_error"))])
    end = of(lines, "end")[0]
    assert code == 3 and len(client.calls) == 1 and of(lines, "retry") == []
    assert (end["message"], end["error"]["status"], end["error"]["type"]) == \
        ("the API answered with an error status", 200, "invalid_request_error")


def test_finish_helper_ends_the_session(root):
    """Once the finish helper logged its event, no further request is sent."""
    events = root / "control" / "events.jsonl"
    cmd = f"""echo '{{"time": 1, "event": "finish", "verdict": "complete"}}' >> {events}"""
    code, client, lines, _ = run(root, [msg(bash(cmd, "toolu_1"))])
    assert code == 0 and len(client.calls) == 1 and of(lines, "end")[0]["reason"] == "finished"


def test_a_finish_of_an_earlier_round_does_not_end_this_one(root):
    """In a session with rounds (controller.run_rounds), each round is a new agent, and only a finish after the last
    round_start ends it (tools.round_events)."""
    events = root / "control" / "events.jsonl"
    events.write_text("".join(json.dumps(e) + "\n" for e in [
        {"time": 1, "event": "round_start", "round": 1, "kind": "build"},
        {"time": 2, "event": "finish", "verdict": "needs_review"},
        {"time": 3, "event": "round_start", "round": 2, "kind": "repair"}]))
    cmd = f"""echo '{{"time": 4, "event": "finish", "verdict": "articulated"}}' >> {events}"""
    code, client, lines, _ = run(root, [msg(bash("true", "toolu_1")), msg(bash(cmd, "toolu_2"))])
    assert code == 0 and len(client.calls) == 2 and of(lines, "end")[0]["reason"] == "finished"


def test_unknown_tool_and_bad_inputs_get_error_results(root):
    script = [msg(use("web_search", {"q": "x"}, "toolu_1"), use("bash", {"cmd": "ls"}, "toolu_2"),
                  use("view_image", {"path": "a.png", "extra": 1}, "toolu_3"),
                  edit("toolu_4", command="undo_edit", path="a.txt")),
              msg(text("ok"), stop="end_turn")]
    code, client, lines, _ = run(root, script)
    r = results(client.calls[1])
    assert code == 0 and list(r) == ["toolu_1", "toolu_2", "toolu_3", "toolu_4"]
    assert all(x["is_error"] for x in r.values())
    assert "unknown tool 'web_search'" in r["toolu_1"]["content"] and "unknown command 'undo_edit'" in r["toolu_4"]["content"]


def test_unexpected_error_is_logged_then_raised(root):
    (root / "session" / "prompt.txt").unlink()
    with pytest.raises(FileNotFoundError):
        run(root, [])
    end = of(log_lines(root), "end")[0]
    assert (end["reason"], end["exit_code"]) == ("error", 1) and "prompt.txt" in end["message"]


def test_cli_needs_sif_without_host_shell(root, key_file):
    with pytest.raises(SystemExit) as e:
        A.main([str(root), "--sock-dir", "/tmp", "--key-file", str(key_file)], client=FakeClient([]))
    assert e.value.code == 2


def cli(root, key_file, script, *args):
    client = FakeClient(script)
    with contextlib.redirect_stdout(io.StringIO()):
        code = A.main([str(root), "--sock-dir", str(root.parent), "--key-file", str(key_file), "--shell", "host", *args],
                      client=client)
    return code, client, log_lines(root)


def test_cli_sends_the_manual_and_downscales_to_2576_px(root, key_file):
    png(root / "work" / "sheet.png", (3000, 1500))
    code, client, lines = cli(root, key_file, [msg(use("view_image", {"path": "sheet.png"}, "toolu_1")),
                                               msg(text("seen"), stop="end_turn")])
    assert code == 0 and client.calls[0]["system"][0]["text"] == A.SYSTEM_FILE.read_text(encoding="utf-8")
    assert client.calls[0]["tools"] == json.loads(json.dumps(A.TOOLS))
    assert results(client.calls[1])["toolu_1"]["content"][1]["text"] == "/work/sheet.png 3000x1500, shown at 2576x1288"
    start = of(lines, "start")[0]
    assert (start["image_side"], start["system"]) == (2576, str(A.SYSTEM_FILE))


@pytest.mark.parametrize("option,value", [("--image-side", "1568"), ("--system", "manual.md")])
def test_the_image_side_and_the_manual_are_fixed(root, key_file, capsys, option, value):
    """The command line takes neither: every session sends SYSTEM_FILE and downscales to MAX_IMAGE_SIDE."""
    with pytest.raises(SystemExit) as e:
        cli(root, key_file, [], option, value)
    assert e.value.code == 2 and f"unrecognized arguments: {option}" in capsys.readouterr().err
    with pytest.raises(TypeError, match="unexpected keyword"):
        A.run(root, client=FakeClient([]), shell="host", **{option[2:].replace("-", "_"): value})


def test_missing_system_file_is_logged_then_raised(root, tmp_path, monkeypatch):
    monkeypatch.setattr(A, "SYSTEM_FILE", tmp_path / "missing.md")
    with pytest.raises(FileNotFoundError):
        run(root, [])
    end = of(log_lines(root), "end")[0]
    assert (end["reason"], end["exit_code"]) == ("error", 1) and "missing.md" in end["message"]


def test_missing_key_ends_with_exit_3(root, tmp_path):
    (tmp_path / "empty.env").write_text("OPENAI_API_KEY=sk-other\n")
    code = A.run(root, sock_dir=str(tmp_path), key_file=tmp_path / "empty.env", shell="host", out=io.StringIO())
    end = of(log_lines(root), "end")[0]
    assert code == 3 and end["reason"] == "no_key" and "sk-other" not in json.dumps(end)


# ------------------------------------------------------------------------------------------- shell


@pytest.fixture
def shell(tmp_path):
    (tmp_path / "work").mkdir(exist_ok=True)
    sh = A.ContainerShell(tmp_path, sock_dir=str(tmp_path), host=True, timeout=2)
    yield sh
    sh.stop()


def gone(pid, wait_s=5.0):
    t = time.time() + wait_s
    while time.time() < t:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_shell_state_errors_stdin_and_exit(shell, tmp_path):
    assert shell.run("cd /tmp && export FOO=bar && echo -n hi") == ("hi", False)
    out, err = shell.run("pwd; echo $FOO; false")
    assert out.endswith("tmp\nbar\n[exit code 1]") and not err
    out, err = shell.run("if then")  # a syntax error cannot leave the shell waiting for more input
    assert "syntax error" in out and "[exit code" in out and not err
    assert shell.run("cat") == ("(no output)", False)  # stdin is empty
    assert shell.run("echo 'it''s' \"$((6 * 7))\" >&2") == ("its 42\n", False)  # stderr merged
    assert shell.run("cat <<'EOF'\nline $HOME\nEOF") == ("line $HOME\n", False)
    out, err = shell.run("exit 4")
    assert "the shell ended with code 4" in out and not err
    out, _ = shell.run("pwd; echo ${FOO:-unset}")
    assert out == f"{tmp_path.resolve() / 'work'}\nunset\n"


@pytest.mark.parametrize("command,code", [("exit 3", 3), ("set -euo pipefail; echo before; false", 1)])
def test_shell_exit_while_a_background_job_holds_its_output(shell, tmp_path, command, code):
    """The background job keeps the output pipe open, so no end of file comes; the exit is seen from
    the process itself, well before the timeout (2 s here)."""
    assert shell.run("sleep 30 & echo $! > bg.pid; echo started") == ("started\n", False)
    try:
        t = time.time()
        out, err = shell.run(command)
        assert time.time() - t < 1.5 and not err
        assert f"[the shell ended with code {code};" in out and ("before" in out) == (code == 1)
        assert shell.run("echo new shell") == ("new shell\n", False)
    finally:
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.kill(int((tmp_path / "work" / "bg.pid").read_text()), signal.SIGKILL)


def test_shell_long_command(shell, tmp_path):
    """A command far larger than a pipe buffer, with every kind of quote, arrives whole."""
    body = "".join(f"line {i} 'single' \"double\" $HOME `tick` \\\n" for i in range(5000))
    out, err = shell.run(f"cat > big.txt <<'EOF'\n{body}EOF\nwc -c < big.txt")
    assert not err and int(out) == len(body.encode()) and (tmp_path / "work" / "big.txt").read_text() == body
    assert shell.run("echo still here") == ("still here\n", False)


def test_children_from_proc(tmp_path):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    for pid, comm, ppid in ((10, "bash", 1), (11, "sleep (x) 2", 10), (12, "sh", 10), (13, "python3", 12)):
        (proc / str(pid)).mkdir()
        (proc / str(pid) / "stat").write_text(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1\n")
    (proc / "99").mkdir()  # a process that ended while being read
    assert {k: sorted(v) for k, v in A._children(str(proc)).items()} == {1: [10], 10: [11, 12], 12: [13]}


def test_shell_env_has_helpers_and_no_keys(shell, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9/")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-test")
    monkeypatch.setenv("BASH_ENV", str(tmp_path / "rc.sh"))
    (tmp_path / "rc.sh").write_text("echo sourced\n")
    out, _ = shell.run("env | grep -c -e ^ANTHROPIC_ -e ^OPENAI_ -e ^CODEX_ -e BASH_ENV; command -v finish; echo $GATOR_SOCKET")
    count, finish, sock = out.splitlines()
    assert count == "0" and finish == str(A.AGENT_DIR / "bin" / "finish") and sock == str(tmp_path / "tools.sock")


def test_shell_timeout_kills_and_restarts(shell, tmp_path):
    w = tmp_path / "work"
    shell.run("export X=1")
    t = time.time()
    out, err = shell.run("echo $$ > shell.pid; sleep 300 & echo $! > child.pid; echo started; sleep 300")
    assert err and out == "started\ntimed out after 2 s; the shell was restarted" and time.time() - t < 5
    assert gone(int((w / "shell.pid").read_text())) and gone(int((w / "child.pid").read_text()))
    assert shell.run("echo ${X:-unset}") == ("unset\n", False)


def test_shell_restart_input(shell):
    shell.run("export X=1")
    out, err = A.bash_tool(shell, {"restart": True})
    assert "restarted" in out and not err
    assert shell.run("echo ${X:-unset}") == ("unset\n", False)
    for bad in ({}, {"command": ""}, {"command": 3}, {"restart": "yes"}, ["ls"]):
        with pytest.raises(A.ToolError):
            A.bash_tool(shell, bad)


@pytest.mark.parametrize("n", [20000, 400000])  # 400000 also passes the in-memory trim (2.7 MB)
def test_shell_output_cap(shell, n):
    full = "".join(f"{i}\n" for i in range(1, n + 1))
    out, err = shell.run(f"seq 1 {n}")
    assert not err and len(out) <= 30000
    head, rest = out[:2000], out[2000:]
    note, tail = rest.split("\n", 2)[1], rest.split("\n", 2)[2]
    omitted = int(note.split()[1])
    assert head == full[:2000] and full.endswith(tail) and 2000 + omitted + len(tail) == len(full)
    assert note == f"[... {omitted} characters omitted ...]"


def test_cap_output():
    assert A.cap_output("abc") == "abc"
    out = A.cap_output("x" * 50 + "y" * 50, limit=60, head=10)
    assert len(out) <= 60 and out.startswith("x" * 10) and out.endswith("y" * 10)
    n = int(out.split("[... ")[1].split()[0])
    assert 10 + n + len(out.rsplit("\n", 1)[1]) == 100


def test_container_mode_uses_controller_shell_command(tmp_path, monkeypatch):
    """Without --shell host the argv comes from controller.shell_command(root, sif, sock_dir), imported
    when the shell starts; PATH is left to the container."""
    import harpor.agent

    seen = []

    def shell_command(root, sif, sock_dir):
        seen.append((root, sif, sock_dir))
        return ["bash", "--noprofile", "--norc"]

    fake = SimpleNamespace(shell_command=shell_command)
    monkeypatch.setitem(sys.modules, "harpor.agent.controller", fake)
    monkeypatch.setattr(harpor.agent, "controller", fake, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    (tmp_path / "work").mkdir()
    sh = A.ContainerShell(tmp_path, "agent.sif", "/tmp/sock")
    try:
        out, _ = sh.run("echo $PATH; env | grep -c ANTHROPIC_API_KEY || true")
    finally:
        sh.stop()
    assert seen == [(tmp_path.resolve(), "agent.sif", "/tmp/sock")]
    path, count = out.splitlines()
    assert str(A.AGENT_DIR / "bin") not in path and count == "0"


# ------------------------------------------------------------------------------------------- files


@pytest.fixture
def work(tmp_path):
    w = tmp_path / "s" / "work"
    w.mkdir(parents=True)
    (tmp_path / "s" / "control").mkdir()
    (tmp_path / "s" / "control" / "budget.json").write_text("{}")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("host file\n")
    return w


@pytest.fixture
def editor(work):
    return A.TextEditor(A.WorkFiles(work))


def test_editor_create_view_and_ranges(editor, work):
    assert editor({"command": "create", "path": "/work/a/b/c.txt", "file_text": "one\ntwo\nthree\n"}) == \
        "created /work/a/b/c.txt (3 lines)"
    assert (work / "a" / "b" / "c.txt").read_text() == "one\ntwo\nthree\n"
    assert editor({"command": "view", "path": "a/b/c.txt"}) == "     1\tone\n     2\ttwo\n     3\tthree"
    assert editor({"command": "view", "path": "a/b/c.txt", "view_range": [2, -1]}) == "     2\ttwo\n     3\tthree"
    assert editor({"command": "view", "path": "/work/a/b/c.txt", "view_range": [1, 1]}) == "     1\tone"
    for bad in ([0, 2], [2, 9], [3, 2], [1], "1-2", [True, 2]):
        with pytest.raises(A.ToolError):
            editor({"command": "view", "path": "a/b/c.txt", "view_range": bad})
    assert editor({"command": "create", "path": "a/b/c.txt", "file_text": ""}).startswith("overwrote")
    assert editor({"command": "view", "path": "a/b/c.txt"}) == "/work/a/b/c.txt is empty"
    with pytest.raises(A.ToolError, match="does not exist"):
        editor({"command": "view", "path": "nope.txt"})
    with pytest.raises(A.ToolError, match="is a folder"):
        editor({"command": "create", "path": "/work/a", "file_text": "x"})


def test_editor_view_cuts_long_output(work):
    ed = A.TextEditor(A.WorkFiles(work), max_chars=100)
    (work / "long.txt").write_text("x" * 500 + "\n")
    out = ed({"command": "view", "path": "long.txt"})
    assert out.startswith("     1\t" + "x" * 93) and "[cut at 100 characters; use view_range" in out


def test_editor_listing(editor, work):
    (work / "input" / "images").mkdir(parents=True)
    (work / "input" / "images" / "000.png").write_bytes(b"png")
    (work / "input" / "images" / "deep").mkdir()
    (work / "input" / "images" / "deep" / "hidden_by_depth.txt").write_text("x")
    (work / ".hidden").write_text("x")
    (work / "run_budget.json").write_text("{}")
    out = editor({"command": "view", "path": "/work"})
    assert out.splitlines() == ["/work/, 2 levels deep, hidden files left out:", "/work/input/",
                                "/work/input/images/", "/work/run_budget.json (2 bytes)"]
    out = editor({"command": "view", "path": "input"}).splitlines()
    assert out[1:] == ["/work/input/images/", "/work/input/images/000.png (3 bytes)", "/work/input/images/deep/"]


def test_editor_str_replace(editor, work):
    (work / "e.py").write_text("a = 1\nb = 2\nb = 2\n")
    with pytest.raises(A.ToolError, match="occurs 2 times"):
        editor({"command": "str_replace", "path": "e.py", "old_str": "b = 2", "new_str": "b = 3"})
    with pytest.raises(A.ToolError, match="was not found"):
        editor({"command": "str_replace", "path": "e.py", "old_str": "c = 1", "new_str": ""})
    out = editor({"command": "str_replace", "path": "e.py", "old_str": "a = 1\n", "new_str": "a = 10\nz = 0\n"})
    assert out.startswith("edited /work/e.py:\n     1\ta = 10\n     2\tz = 0")
    assert (work / "e.py").read_text() == "a = 10\nz = 0\nb = 2\nb = 2\n"
    editor({"command": "str_replace", "path": "e.py", "old_str": "z = 0\n"})  # new_str defaults to ""
    assert (work / "e.py").read_text() == "a = 10\nb = 2\nb = 2\n"


def test_editor_insert(editor, work):
    (work / "i.txt").write_text("a\nb\n")
    editor({"command": "insert", "path": "i.txt", "insert_line": 0, "insert_text": "top\n"})
    editor({"command": "insert", "path": "i.txt", "insert_line": 3, "insert_text": "end1\nend2"})
    editor({"command": "insert", "path": "i.txt", "insert_line": 2, "insert_text": "mid"})
    assert (work / "i.txt").read_text() == "top\na\nmid\nb\nend1\nend2\n"
    (work / "n.txt").write_text("no newline")
    editor({"command": "insert", "path": "n.txt", "insert_line": 1, "insert_text": "x"})
    assert (work / "n.txt").read_text() == "no newline\nx"
    for bad in (-1, 9, "1", None):
        with pytest.raises(A.ToolError):
            editor({"command": "insert", "path": "i.txt", "insert_line": bad, "insert_text": "x"})


def test_editor_refuses_binary_files(editor, work):
    (work / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n\0\0")
    (work / "latin.txt").write_bytes("caf\xe9".encode("latin-1"))
    for name in ("img.png", "latin.txt"):
        with pytest.raises(A.ToolError, match="binary file; use view_image for PNG or JPEG"):
            editor({"command": "view", "path": name})
        with pytest.raises(A.ToolError, match="binary file"):
            editor({"command": "str_replace", "path": name, "old_str": "a", "new_str": "b"})


def test_paths_are_confined_to_work(editor, work, tmp_path):
    outside = tmp_path / "outside"
    (work / "link_out").symlink_to(outside)
    (work / "file_out").symlink_to(outside / "secret.txt")
    (work / "dangling_out").symlink_to(outside / "new.txt")
    (work / "sub").mkdir()
    (work / "sub" / "in.txt").write_text("inside\n")
    (work / "link_in").symlink_to(work / "sub")
    bad = ["../control/budget.json", "/work/../control/budget.json", "/etc/passwd", str(outside / "secret.txt"),
           "link_out/secret.txt", "/work/file_out", "dangling_out", "sub/../../control/budget.json", "", "a\0b",
           "/workspace/x"]
    for path in bad:
        for inp in ({"command": "view", "path": path},
                    {"command": "create", "path": path, "file_text": "pwned\n"},
                    {"command": "str_replace", "path": path, "old_str": "host", "new_str": "pwned"},
                    {"command": "insert", "path": path, "insert_line": 0, "insert_text": "pwned"}):
            with pytest.raises(A.ToolError):
                editor(inp)
        with pytest.raises(A.ToolError):
            A.view_image(editor.files, {"path": path})
    assert (outside / "secret.txt").read_text() == "host file\n" and not (outside / "new.txt").exists()
    assert sorted(p.name for p in outside.iterdir()) == ["secret.txt"]
    # Symlinks that stay inside work are fine, and so is the host path of a file in work.
    assert editor({"command": "view", "path": "link_in/in.txt"}) == "     1\tinside"
    assert editor({"command": "view", "path": str(work / "sub" / "in.txt")}) == "     1\tinside"
    assert editor({"command": "view", "path": "./sub/in.txt"}) == "     1\tinside"


def test_open_does_not_follow_a_symlink_swapped_in_after_the_check(work, tmp_path):
    """Between resolve() and the open, the agent's shell could replace a folder with a symlink."""
    files = A.WorkFiles(work)
    (work / "sub").symlink_to(tmp_path / "outside")
    for flags in (os.O_RDONLY, os.O_WRONLY | os.O_CREAT):
        with pytest.raises(OSError):
            files.open(PurePosixPath("sub/secret.txt"), flags)
    (work / "f").symlink_to(tmp_path / "outside" / "secret.txt")
    with pytest.raises(OSError):
        files.write(PurePosixPath("f"), b"pwned")
    assert (tmp_path / "outside" / "secret.txt").read_text() == "host file\n"


def png(path, size, mode="RGB", **kw):
    im = Image.new(mode, size, **({"color": 128} if mode in ("L", "P") else {"color": (200, 100, 50)}))
    im.save(path, **kw)
    return path


def test_view_image_downscales_large_png(work):
    png(work / "sheet.png", (3000, 1500))
    content, (W, H), (w, h) = A.view_image(A.WorkFiles(work), {"path": "/work/sheet.png"})
    assert (W, H, w, h) == (3000, 1500, 2576, 1288)
    img, label = content
    assert img["type"] == "image" and img["source"]["type"] == "base64" and img["source"]["media_type"] == "image/png"
    assert Image.open(io.BytesIO(base64.b64decode(img["source"]["data"]))).size == (2576, 1288)
    assert label == {"type": "text", "text": "/work/sheet.png 3000x1500, shown at 2576x1288"}


def test_view_image_small_jpeg_is_sent_as_is(work):
    png(work / "photo.jpg", (100, 50), format="JPEG")
    content, size, shown = A.view_image(A.WorkFiles(work), {"path": "photo.jpg"})
    assert content[0]["source"]["media_type"] == "image/jpeg" and size == shown == (100, 50)
    assert base64.b64decode(content[0]["source"]["data"]) == (work / "photo.jpg").read_bytes()
    assert content[1]["text"] == "/work/photo.jpg 100x50, shown at 100x50"


def test_view_image_tall_palette_png(work):
    png(work / "mask.png", (400, 2000), mode="P")
    content, size, shown = A.view_image(A.WorkFiles(work), {"path": "mask.png"})
    assert size == shown == (400, 2000)  # under 2576 px: as it is, byte for byte
    assert base64.b64decode(content[0]["source"]["data"]) == (work / "mask.png").read_bytes()
    png(work / "tall.png", (400, 3000), mode="P")
    assert A.view_image(A.WorkFiles(work), {"path": "tall.png"})[2] == (343, 2576)


def test_view_image_rejects_other_files_and_inputs(work):
    files = A.WorkFiles(work)
    (work / "fake.png").write_text("not an image")
    (work / "pic.gif").write_bytes(b"GIF89a....")
    (work / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 20)
    for inp, why in (({"path": "fake.png"}, "not a PNG or JPEG"), ({"path": "pic.gif"}, "not a PNG or JPEG"),
                     ({"path": "broken.png"}, "could not be read"), ({"path": "missing.png"}, "does not exist"),
                     ({"path": "/work"}, "not a regular file"), ({"path": 3}, "takes"), ({}, "takes"),
                     ({"path": "fake.png", "detail": "high"}, "takes"), ("fake.png", "takes")):
        with pytest.raises(A.ToolError, match=why):
            A.view_image(files, inp)


def test_view_image_in_the_loop(root):
    png(root / "work" / "sheet.png", (3000, 1500))
    code, client, lines, _ = run(root, [msg(use("view_image", {"path": "sheet.png"}, "toolu_1")),
                                        msg(text("seen"), stop="end_turn")])
    r = results(client.calls[1])["toolu_1"]
    assert code == 0 and [b["type"] for b in r["content"]] == ["image", "text"] and "is_error" not in r
    assert r["content"][1]["text"] == "/work/sheet.png 3000x1500, shown at 2576x1288"
    assert of(lines, "tool_result")[0]["output"] == "[image 3000x1500]"
    assert "larger than 2576 px" in client.calls[0]["tools"][2]["description"]


def noise(path, size, mode="RGB"):
    """A PNG that does not compress, as a rendered review panel: several MB at 1568 px."""
    Image.frombytes(mode, size, os.urandom(size[0] * size[1] * len(mode))).save(path)
    return path


def jpeg_quality(data):
    """The quality, 1 to 100, whose quantization tables are those of the JPEG data (they depend on the
    quality and the mode only)."""
    im = Image.open(io.BytesIO(data))
    for q in range(100, 0, -1):
        buf = io.BytesIO()
        Image.new(im.mode, (16, 16)).save(buf, "JPEG", quality=q)
        if Image.open(buf).quantization == im.quantization:
            return q
    return None


def test_png_stays_lossless_up_to_2_5_mb_of_base64(work):
    """A PNG goes as it is while its base64 fits 2.5 MB: a 1.4 MB render that does not compress is no
    longer sent as JPEG, and neither is a small mask."""
    noise(work / "render.png", (1200, 1200), "L")
    size = (work / "render.png").stat().st_size
    assert 1_000_000 < size and A.b64_size(size) <= A.MAX_IMAGE_B64
    for name in ("render.png", "mask.png"):
        if name == "mask.png":
            png(work / "mask.png", (1000, 800), mode="L")
        content, _, _ = A.view_image(A.WorkFiles(work), {"path": name})
        assert content[0]["source"]["media_type"] == "image/png" and "JPEG" not in content[1]["text"]
        assert base64.b64decode(content[0]["source"]["data"]) == (work / name).read_bytes()
    # Over 2.5 MB of base64 by a little: JPEG.
    noise(work / "over.png", (800, 800))
    assert A.b64_size((work / "over.png").stat().st_size) > A.MAX_IMAGE_B64
    content, _, _ = A.view_image(A.WorkFiles(work), {"path": "over.png"})
    assert content[0]["source"]["media_type"] == "image/jpeg" and content[1]["text"].endswith(", sent as JPEG")


def test_large_png_goes_as_jpeg_at_quality_92(work):
    """Every image is sent again with each later request, so a large render must not go as a multi-MB
    PNG. A PNG whose base64 is over 2.5 MB goes as JPEG (transparent areas white), at quality 92 when
    that fits."""
    noise(work / "render.png", (1568, 1176), "RGBA")
    assert (work / "render.png").stat().st_size > 5_000_000
    im = Image.open(work / "render.png").convert("RGBA")
    im.putpixel((0, 0), (0, 0, 0, 0))
    im.save(work / "render.png")
    content, size, shown = A.view_image(A.WorkFiles(work), {"path": "render.png"})
    img, label = content
    data = base64.b64decode(img["source"]["data"])
    assert img["source"]["media_type"] == "image/jpeg" and data.startswith(A.JPEG_MAGIC)
    assert len(img["source"]["data"]) <= A.MAX_IMAGE_B64
    assert jpeg_quality(data) == 92
    assert size == shown == (1568, 1176)
    assert label["text"] == "/work/render.png 1568x1176, shown at 1568x1176, sent as JPEG with transparent areas white"
    assert min(Image.open(io.BytesIO(data)).convert("RGB").getpixel((0, 0))) > 200  # transparent -> white


def test_jpeg_quality_steps_down_only_to_fit(work):
    """A JPEG whose base64 is over 2.5 MB is sent again at 92, else 80, else 70 (even if that does not
    fit); one that fits goes as it is."""
    Image.frombytes("RGB", (1500, 1500), os.urandom(1500 * 1500 * 3)).save(work / "photo.jpg", quality=100)
    assert A.b64_size((work / "photo.jpg").stat().st_size) > A.MAX_IMAGE_B64
    content, _, _ = A.view_image(A.WorkFiles(work), {"path": "photo.jpg"})
    data = base64.b64decode(content[0]["source"]["data"])
    assert content[0]["source"]["media_type"] == "image/jpeg" and len(content[0]["source"]["data"]) <= A.MAX_IMAGE_B64
    assert content[1]["text"] == "/work/photo.jpg 1500x1500, shown at 1500x1500"
    assert jpeg_quality(data) == 80  # 92 does not fit 2.5 MB at 1500 x 1500 of noise
    big = Image.frombytes("RGB", (2400, 1500), os.urandom(2400 * 1500 * 3))
    lowest = A._jpeg(big)
    assert A.b64_size(len(lowest)) > A.MAX_IMAGE_B64 and jpeg_quality(lowest) == 70
    small = Image.frombytes("RGB", (600, 600), os.urandom(600 * 600 * 3))
    assert jpeg_quality(A._jpeg(small)) == 92
    small.save(work / "small.jpg", quality=100)
    content, _, _ = A.view_image(A.WorkFiles(work), {"path": "small.jpg"})
    assert base64.b64decode(content[0]["source"]["data"]) == (work / "small.jpg").read_bytes()


def test_image_budget_of_the_session(root, monkeypatch):
    """Images stay in the history, so their number and size per session are capped; the image that does
    not fit gets an error result and is not sent."""
    monkeypatch.setattr(A, "MAX_IMAGES", 2)
    for name in ("a", "b", "c"):
        png(root / "work" / f"{name}.png", (40, 30))
    views = [use("view_image", {"path": f"{n}.png"}, f"toolu_{n}") for n in "abc"]
    code, client, lines, _ = run(root, [msg(*views), msg(text("seen"), stop="end_turn")])
    r = results(client.calls[1])
    assert code == 0 and [x.get("is_error", False) for x in r.values()] == [False, False, True]
    assert r["toolu_c"]["content"].startswith("not shown: the conversation already holds 2 images")
    assert json.dumps(client.calls[1]).count('"type": "image"') == 2
    budget = A.ImageBudget(limit=100)
    budget.take([{"source": {"data": "x" * 60}}])
    with pytest.raises(A.ToolError, match=r"at most 2 images and 0 MB of them \(this one needs 0.00 MB\)"):
        budget.take([{"source": {"data": "x" * 41}}])
    assert (budget.n, budget.used) == (1, 60)


# -------------------------------------------------------------------------------------------- key


def test_key_never_in_log_stdout_or_tool_results(root, key_file, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)  # the agent's environment; the shell must not get it
    script = [msg(text(f"My key might be {KEY}"), bash(f"cat {key_file}; env | grep -c ANTHROPIC_API_KEY", "toolu_1")),
              msg(text("done"), stop="end_turn")]
    code, client, lines, out = run(root, script, key_file=key_file)
    assert code == 0
    r = results(client.calls[1])["toolu_1"]["content"]
    assert r == "ANTHROPIC_API_KEY=[redacted]\n0\n[exit code 1]"  # grep -c found nothing
    assert KEY not in json.dumps(client.calls[1]["messages"][2])
    assert KEY not in (root / "session" / "claude.jsonl").read_text() and KEY not in out
    assert "[redacted]" in (root / "session" / "claude.jsonl").read_text()
    for p in root.rglob("*"):
        assert not p.is_file() or KEY.encode() not in p.read_bytes(), p


def test_read_key(tmp_path):
    f = tmp_path / "k.env"
    f.write_text('OPENAI_API_KEY=sk-x\nANTHROPIC_API_KEY="sk-ant-quoted"\n')
    assert A.read_key(f) == "sk-ant-quoted"
    f.write_text("ANTHROPIC_API_KEY= sk-ant-plain \n")
    assert A.read_key(f) == "sk-ant-plain"
    assert A.read_key(tmp_path / "missing.env") is None


# ------------------------------------------------------------------------------ real SDK, no network


def sse(blocks, stop, model=A.MODEL):
    """The server-sent events of one streamed message. Tool input is sent in two input_json_delta parts."""
    ev = [{"type": "message_start", "message": {
        "id": f"msg_sse_{len(blocks)}_{stop}", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 1, "cache_read_input_tokens": 70,
                  "cache_creation_input_tokens": 30}}}]
    for i, b in enumerate(blocks):
        if b["type"] == "thinking":
            ev += [{"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": "",
                                                                                  "signature": ""}},
                   {"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": b["thinking"]}},
                   {"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta",
                                                                          "signature": b["signature"]}}]
        elif b["type"] == "text":
            ev += [{"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                   {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}}]
        else:
            raw = b["raw_json"] if "raw_json" in b else json.dumps(b["input"])
            ev += [{"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": b["id"],
                                                                                  "name": b["name"], "input": {}}}]
            ev += [{"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": p}}
                   for p in (raw[:len(raw) // 2], raw[len(raw) // 2:])]
        ev.append({"type": "content_block_stop", "index": i})
    ev += [{"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 42}},
           {"type": "message_stop"}]
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in ev)


def mock_client(replies, key=KEY):
    seen = []

    def handler(request):
        seen.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        status, body = replies.pop(0)
        if status != 200:
            return httpx2.Response(status, json=body, headers={"request-id": "req_mock"})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    return anthropic.Anthropic(api_key=key, max_retries=0, http_client=http), seen


def test_real_sdk_session_over_mock_transport(root, key_file):
    png(root / "work" / "sheet.png", (64, 32))
    bad_json = {"type": "tool_use", "id": "toolu_bad", "name": "view_image", "raw_json": 'xx{"path'}
    replies = [
        (200, sse([thinking("plan"), {"type": "tool_use", "id": "toolu_1", "name": "bash",
                                      "input": {"command": "echo $((40 + 2))"}},
                   {"type": "tool_use", "id": "toolu_2", "name": "view_image", "input": {"path": "/work/sheet.png"}}],
                  "tool_use")),
        (200, sse([text("Looking."), bad_json], "tool_use")),  # unparseable eager JSON: the SDK raises ValueError
        (200, sse([text("All good.")], "end_turn")),
    ]
    client, seen = mock_client(replies)
    out = io.StringIO()
    code = A.run(root, sock_dir=str(root.parent), key_file=key_file, shell="host", client=client, out=out)
    lines = log_lines(root)
    assert code == 0 and len(seen) == 3 and [e["type"] for e in lines].count("retry") == 1
    first, second, third = (s["body"] for s in seen)
    assert seen[0]["headers"]["anthropic-beta"] == "server-side-fallback-2026-07-01"
    assert seen[0]["headers"]["x-api-key"] == KEY  # the key goes to the API and nowhere else
    assert first["stream"] is True and first["model"] == "claude-opus-5-5" and first["max_tokens"] == 64000
    assert first["fallbacks"] == "default" and first["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert first["output_config"] == {"effort": "high"} and first["cache_control"] == {"type": "ephemeral"}
    assert first["tools"] == json.loads(json.dumps(A.TOOLS)) and "tool_choice" not in first and "speed" not in first
    assert first["messages"] == [{"role": "user", "content": PROMPT}]
    assert second == third  # the re-issued turn is the same request
    assistant, user = second["messages"][1:]
    assert assistant["content"][0] == {"type": "thinking", "thinking": "plan", "signature": "c2lnbmF0dXJl"}
    assert [b["type"] for b in assistant["content"]] == ["thinking", "tool_use", "tool_use"]
    assert [(r["tool_use_id"], type(r["content"]).__name__) for r in user["content"]] == [("toolu_1", "str"),
                                                                                          ("toolu_2", "list")]
    assert user["content"][0]["content"] == "42\n" and user["content"][1]["content"][0]["type"] == "image"
    assert second["messages"][:1] == first["messages"]
    end = of(lines, "end")[0]
    assert end["totals"] == {"input_tokens": 22, "output_tokens": 84, "cache_read_input_tokens": 140,
                             "cache_creation_input_tokens": 60}
    assert KEY not in (root / "session" / "claude.jsonl").read_text() and KEY not in out.getvalue()


def test_real_sdk_authentication_error(root, key_file):
    client, seen = mock_client([(401, {"type": "error", "error": {"type": "authentication_error",
                                                                  "message": "invalid x-api-key"}})])
    code = A.run(root, sock_dir=str(root.parent), key_file=key_file, shell="host", client=client, out=io.StringIO())
    end = of(log_lines(root), "end")[0]
    assert code == 3 and len(seen) == 1
    assert (end["error"]["class"], end["error"]["status"], end["error"]["type"], end["error"]["request_id"]) == \
        ("AuthenticationError", 401, "authentication_error", "req_mock")


def test_real_sdk_stream_failures_after_it_opened(root, key_file, monkeypatch):
    """With the real SDK (max_retries 4): an overloaded_error event after some text raises a bare
    APIStatusError with status 200, a dropped connection raises httpx2.ReadError, and a stream without
    message_delta gives stop_reason None. The SDK retries none of them; the loop sends the turn again."""
    monkeypatch.setattr(A, "RETRY_BACKOFF_S", 0.0)
    full = sse([text("Partial ans")], "end_turn").split(b"\n\n")
    head = b"\n\n".join(full[:3]) + b"\n\n"  # message_start, content_block_start, the text delta

    def overloaded():
        yield head
        yield b'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}\n\n'

    def dropped():
        yield head
        raise httpx2.ReadError("connection reset")

    def no_stop():
        yield head

    bodies, seen = [overloaded, dropped, no_stop, lambda: iter([sse([text("All good.")], "end_turn")])], []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=bodies.pop(0)())

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    client = anthropic.Anthropic(api_key=KEY, max_retries=4, http_client=http)
    code = A.run(root, sock_dir=str(root.parent), key_file=key_file, shell="host", client=client, out=io.StringIO())
    lines = log_lines(root)
    assert code == 0 and len(seen) == 4 and seen[0] == seen[1] == seen[2] == seen[3]
    errors = [e["error"] for e in of(lines, "retry")]
    assert "APIStatusError" in errors[0] and "overloaded_error" in errors[0] and "ReadError" in errors[1]
    assert "without a stop reason" in errors[2]
    assert [b["text"] for b in of(lines, "response")[0]["blocks"]] == ["All good."]
    assert of(lines, "end")[0]["reissued"] == 3


# --------------------------------------------------------------------------------------- signals


AGENT_SCRIPT = r'''
import sys, time
from types import SimpleNamespace
from anthropic.types.beta import BetaMessage
from harpor.agent import claude_agent as A

root, key_file, command = sys.argv[1:4]


class Stream:
    def __init__(self, n):
        self.n = n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        if self.n > 1:
            time.sleep(120)  # the second request hangs until the signal
        return BetaMessage.model_validate({
            "id": "msg_1", "type": "message", "role": "assistant", "model": A.MODEL, "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": command}}],
            "usage": {"input_tokens": 5, "output_tokens": 5}})


calls = []
client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(
    stream=lambda **kw: calls.append(1) or Stream(len(calls)))))
sys.exit(A.main([root, "--sock-dir", root, "--key-file", key_file, "--shell", "host"], client=client))
'''


def start_agent(root, key_file, command, tmp_path):
    script = tmp_path / "agent_script.py"
    script.write_text(AGENT_SCRIPT)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    # Its own process group, as the controller's run_agent starts it.
    return subprocess.Popen([sys.executable, str(script), str(root), str(key_file), command], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)


def wait_for(cond, proc, timeout=30.0):
    t = time.time() + timeout
    while time.time() < t:
        if cond():
            return
        assert proc.poll() is None, proc.stdout.read().decode()
        time.sleep(0.05)
    raise AssertionError("timed out")


def requests_logged(root):
    f = root / "session" / "claude.jsonl"
    return f.exists() and f.read_text().count('"type": "request"')


def test_sigint_while_waiting_for_the_api(root, key_file, tmp_path):
    """SIGINT to the agent alone: it stops the shell with what it started, logs, and exits 130."""
    w = root / "work"
    proc = start_agent(root, key_file, "echo $$ > shell.pid; sleep 300 & echo $! > child.pid", tmp_path)
    try:
        wait_for(lambda: requests_logged(root) == 2, proc)
        os.kill(proc.pid, signal.SIGINT)
        assert proc.wait(15) == 130
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    end = of(log_lines(root), "end")[0]
    assert (end["reason"], end["exit_code"], end["signal"], end["turns"]) == ("interrupted", 130, "SIGINT", 1)
    assert gone(int((w / "shell.pid").read_text())) and gone(int((w / "child.pid").read_text()))
    assert b"end: interrupted (exit 130)" in proc.stdout.read()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_group_signal_during_a_command(root, key_file, tmp_path, sig):
    """The controller's watchdog signals the whole group while a command runs."""
    w = root / "work"
    proc = start_agent(root, key_file, "echo $$ > shell.pid; sh -c 'echo $$ > child.pid; exec sleep 300'", tmp_path)
    try:
        wait_for(lambda: (w / "child.pid").exists() and (w / "child.pid").read_text().strip(), proc)
        t = time.time()
        os.killpg(proc.pid, sig)
        assert proc.wait(15) == 130 and time.time() - t < 10
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    end = of(log_lines(root), "end")[0]
    assert (end["reason"], end["exit_code"], end["signal"]) == ("interrupted", 130, signal.Signals(sig).name)
    assert gone(int((w / "shell.pid").read_text())) and gone(int((w / "child.pid").read_text()))
