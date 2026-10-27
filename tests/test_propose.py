import base64
import io
import json
import os
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
from PIL import Image

anthropic = pytest.importorskip("anthropic")
from gator.parts import propose as P  # noqa: E402

OPUS_KEY = "sk-ant-api03-" + "q" * 40
OPENAI_KEY = "sk-proj-" + "z" * 40
GOOD = json.dumps({"base": "cabinet body", "parts": [
    {"name": "drawer", "parent": "cabinet body", "joint": "prismatic", "count": 2, "axis_hint": "slides out toward the camera"},
    {"name": "door", "parent": "", "joint": "Revolute", "count": 1, "axis_hint": "hinge on the left edge"}]})


class FakeClient:
    """Stands in for anthropic.Anthropic: beta.messages.create returns the answers in turn."""

    def __init__(self, answers, stop="end_turn"):
        self.answers, self.calls, self.stop = list(answers), [], stop
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason=self.stop, content=[SimpleNamespace(type="thinking", thinking=""),
                                                               SimpleNamespace(type="text", text=self.answers.pop(0))])


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def job(tmp_path, agent="opus", mode="vlm", parts=()):
    J = tmp_path / "job"
    (J / "run" / "input" / "images").mkdir(parents=True)
    for n in ("1", "0", "10"):
        Image.new("RGB", (1600, 1200), (100, 120, 140)).save(J / "run" / "input" / "images" / f"{n}.png")
    (J / "job.json").write_text(json.dumps({"id": "x", "prompt": "cabinet", "agent": agent,
                                            "parts": {"mode": mode, "list": list(parts), "confirmed": False}}))
    return J


def key_file(tmp_path, line):
    f = tmp_path / "keys" / "k.env"
    f.parent.mkdir(exist_ok=True)
    f.write_text(line + "\n")
    os.chmod(f, 0o600)
    return f


def no_key_in(folder, key):
    return [p for p in folder.rglob("*") if p.is_file() and key.encode() in p.read_bytes()]


def test_validate_normalizes_a_good_list():
    parts = P.validate([{"name": "Lid", "parent": " box   body", "joint": "Revolute"}, "box body"], noun="box")
    assert parts == [{"name": "box body", "parent": None, "joint": "fixed", "count": 1, "axis_hint": None},
                     {"name": "Lid", "parent": "box body", "joint": "revolute", "count": 1, "axis_hint": None}]
    proposal = {"base": "box body", "parts": parts}
    assert P.validate(proposal, noun="box") == parts


@pytest.mark.parametrize("parts,why", [
    ([], "1 to 12"),
    ([{"name": f"p{i}", "parent": None if i == 0 else "p0"} for i in range(13)], "1 to 12"),
    ([{"name": "body"}, {"name": "Body", "parent": "body"}], "same name"),
    ([{"name": "cabinet"}, {"name": "door", "parent": "cabinet"}], "whole object"),
    ([{"name": "body"}, {"name": "door"}], "exactly one part"),
    ([{"name": "body", "parent": "door"}, {"name": "door", "parent": "body"}], "exactly one part"),
    ([{"name": "body"}, {"name": "door", "parent": "wall"}], "not in the list"),
    ([{"name": "body"}, {"name": "door", "parent": "door"}], "its own parent"),
    ([{"name": "body"}, {"name": "a", "parent": "b"}, {"name": "b", "parent": "a"}], "loop"),
    ([{"name": "body"}, {"name": "door", "parent": "body", "joint": "hinge"}], "joint"),
    ([{"name": "body"}, {"name": "door", "parent": "body", "count": 0}], "count"),
    ([{"name": "body"}, {"name": "door", "parent": "body", "count": True}], "count"),
    ([{"name": "body"}, {"name": "?!", "parent": "body"}], "letters or digits"),
    ([{"name": "body"}, {"name": "x" * 41, "parent": "body"}], "longer"),
    ([{"name": "body"}, {"parent": "body"}], "needs a name"),
    ([{"name": "globe stand"}, {"name": "World", "parent": "globe stand"}], "reserved"),
    ([{"name": "world"}, {"name": "globe", "parent": "world"}], "reserved"),
])
def test_validate_catches_each_fault(parts, why):
    with pytest.raises(ValueError, match=why):
        P.validate(parts, noun="cabinet")


def test_validate_checks_the_named_base():
    with pytest.raises(ValueError, match="the base is"):
        P.validate({"base": "frame", "parts": [{"name": "body"}, {"name": "door", "parent": "body"}]})


def test_parse_takes_the_first_json_object():
    assert P.parse('Sure! {"base": "x", "parts": []} and {"other": 1}') == {"base": "x", "parts": []}
    assert P.parse('{not json} then {"a": [1, {"b": 2}]}') == {"a": [1, {"b": 2}]}
    with pytest.raises(ValueError, match="no JSON"):
        P.parse("[1, 2] no object here")


def test_from_answer_hangs_parts_on_the_base_and_renames_a_base_named_like_the_noun():
    parts, warnings = P.from_answer(json.dumps({"base": "Cabinet", "parts": [
        {"name": "door", "parent": "cabinet", "joint": "revolute"}, {"name": "cabinet"}, {"name": "knob", "parent": "door"}]}),
        "cabinet")
    assert [(p["name"], p["parent"]) for p in parts] == [("cabinet body", None), ("door", "cabinet body"), ("knob", "door")]
    assert len(warnings) == 2 and "cabinet body" in warnings[0] and "repeats the base" in warnings[1]
    parts, warnings = P.from_answer('{"base": "cabinet body", "parts": []}', "cabinet")
    assert len(parts) == 1 and warnings == ["no moving part was listed"]


def test_opus_proposes_from_view_0(tmp_path, monkeypatch):
    J = job(tmp_path)
    fake, made = FakeClient([GOOD]), []
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: made.append(kw) or fake)
    proposal = P.run(J, key_file(tmp_path, f'ANTHROPIC_API_KEY="{OPUS_KEY}"'))
    assert made[0]["api_key"] == OPUS_KEY
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["output_config"] == {"effort": "high"}
    assert "temperature" not in call and "thinking" not in call and call["fallbacks"] == "default"
    image, text = call["messages"][0]["content"]
    im = Image.open(io.BytesIO(base64.b64decode(image["source"]["data"])))
    assert image["source"]["media_type"] == "image/jpeg" and im.format == "JPEG" and im.size == (1024, 768)
    assert "The photo shows a cabinet." in text["text"] and "At most 11 parts" in text["text"]
    assert proposal["source"] == "vlm" and proposal["agent"] == "opus" and proposal["model"] == "claude-opus-5-5"
    assert proposal["base"] == "cabinet body" and proposal["mode"] == "vlm"
    assert [(p["name"], p["parent"], p["joint"], p["count"]) for p in proposal["parts"]] == [
        ("cabinet body", None, "fixed", 1), ("drawer", "cabinet body", "prismatic", 2), ("door", "cabinet body", "revolute", 1)]
    assert json.loads((J / "run" / "proposal.json").read_text()) == proposal
    assert no_key_in(J, OPUS_KEY) == []


def test_a_bad_answer_is_asked_once_more(tmp_path):
    J = job(tmp_path)
    fake = FakeClient(["I think it has a door.", GOOD])
    proposal = P.propose_vlm("cabinet", J / "run" / "input" / "images" / "0.png", "opus", OPUS_KEY, client=fake)
    assert proposal["attempts"] == 2 and len(fake.calls) == 2
    assert "could not be used: the answer holds no JSON object" in fake.calls[1]["messages"][0]["content"][1]["text"]
    fake = FakeClient(["no", '{"base": "cabinet"'])
    with pytest.raises(SystemExit, match="twice"):
        P.propose_vlm("cabinet", J / "run" / "input" / "images" / "0.png", "opus", OPUS_KEY, client=fake)
    with pytest.raises(SystemExit, match="declined"):
        P.propose_vlm("cabinet", J / "run" / "input" / "images" / "0.png", "opus", OPUS_KEY,
                      client=FakeClient([GOOD], stop="refusal"))


def test_astra_proposes_through_the_responses_api(tmp_path, monkeypatch):
    J = job(tmp_path, agent="astra")
    sent = []

    def urlopen(req, timeout=None):
        sent.append(req)
        answer = {"status": "completed", "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": "Here: " + GOOD.replace("left", OPENAI_KEY)}]}]}
        return FakeResponse(json.dumps(answer).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    keys = key_file(tmp_path, f"OPENAI_API_KEY={OPENAI_KEY}\nANTHROPIC_API_KEY={OPUS_KEY}")
    proposal = P.run(J, keys)
    req = sent[0]
    body = json.loads(req.data)
    assert req.full_url == "https://api.openai.com/v1/responses" and req.get_method() == "POST"
    assert req.get_header("Authorization") == f"Bearer {OPENAI_KEY}"
    assert body["model"] == "gpt-6-astra" and body["reasoning"] == {"effort": "high"}
    text, image = body["input"][0]["content"]
    assert text["type"] == "input_text" and image["type"] == "input_image"
    assert image["image_url"].startswith("data:image/jpeg;base64,")
    assert proposal["model"] == "gpt-6-astra" and len(proposal["parts"]) == 3
    assert "[redacted API key]" in proposal["parts"][2]["axis_hint"] and "[redacted API key]" in proposal["answer"]
    assert no_key_in(J, OPENAI_KEY) == [] and no_key_in(J, OPUS_KEY) == []


def test_a_key_is_scrubbed_before_the_answer_is_cut(tmp_path):
    # The key crosses both cuts: an axis_hint's 120 characters and the stored answer's 4000.
    answer = json.dumps({"base": "cabinet body", "parts": [{"name": "door", "parent": "cabinet body",
                                                            "axis_hint": "a" * 100 + " " + OPUS_KEY}]})
    answer += " " + "y" * (3990 - len(answer) - 1) + OPUS_KEY
    assert answer.index(OPUS_KEY, 3000) == 3990
    J = job(tmp_path)
    proposal = P.run(J, key_file(tmp_path, f"ANTHROPIC_API_KEY={OPUS_KEY}"), client=FakeClient([answer]))
    assert proposal["parts"][1]["axis_hint"] == "a" * 100 + " [redacted API key]"
    assert proposal["answer"].count("[redacted API key]") == 2 and OPUS_KEY[:10] not in proposal["answer"]
    assert no_key_in(J, OPUS_KEY[:10]) == []


def test_api_errors_report_only_their_kind(tmp_path, monkeypatch, capsys):
    J = job(tmp_path, agent="astra")

    def urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {},
                                     io.BytesIO(f'{{"error": "Incorrect API key provided: {OPENAI_KEY}"}}'.encode()))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(SystemExit, match="answered HTTP 401") as e:
        P.run(J, key_file(tmp_path, f"OPENAI_API_KEY={OPENAI_KEY}"))
    assert OPENAI_KEY not in str(e.value.code) and OPENAI_KEY not in capsys.readouterr().out
    assert not (J / "run" / "proposal.json").exists()

    def denied(**kw):
        raise anthropic.AuthenticationError(f"bad key {OPUS_KEY}", response=SimpleNamespace(
            status_code=401, headers={}, request=None), body=None)

    J2 = job(tmp_path / "b")
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=denied))))
    with pytest.raises(SystemExit, match="the Anthropic key was rejected") as e:
        P.run(J2, key_file(tmp_path, f"ANTHROPIC_API_KEY={OPUS_KEY}"))
    assert OPUS_KEY not in str(e.value.code)
    with pytest.raises(SystemExit, match="no ANTHROPIC_API_KEY line"):
        P.run(J2, key_file(tmp_path, f"OPENAI_API_KEY={OPENAI_KEY}"))
    with pytest.raises(SystemExit, match="needs --key-file"):
        P.run(J2)


def test_the_user_list_needs_no_key(tmp_path):
    J = job(tmp_path, agent="none", mode="vlm", parts=[{"name": "body"}, {"name": "lid", "parent": "body", "joint": "revolute"}])
    proposal = P.run(J)
    assert proposal["source"] == "user" and proposal["agent"] == "none" and proposal["model"] is None
    assert [p["name"] for p in proposal["parts"]] == ["body", "lid"] and proposal["base"] == "body"
    J2 = job(tmp_path / "b", agent="opus", mode="user")
    with pytest.raises(SystemExit, match="no part list"):
        P.run(J2)
    J3 = job(tmp_path / "c", agent="opus", mode="user", parts=[{"name": "cabinet"}])
    with pytest.raises(SystemExit, match="whole object"):
        P.run(J3)


def test_confirm_writes_the_edited_list(tmp_path):
    J = job(tmp_path)
    P.run(J, key_file(tmp_path, f"ANTHROPIC_API_KEY={OPUS_KEY}"), client=FakeClient([GOOD]))
    edited = P.confirm(J, [{"name": "cabinet body"}, {"name": "drawer", "parent": "cabinet body", "joint": "prismatic",
                                                      "count": 3}])
    saved = json.loads((J / "run" / "proposal.json").read_text())
    assert saved == edited and saved["confirmed"] and saved["edited"] and saved["source"] == "vlm"
    assert [p["count"] for p in saved["parts"]] == [1, 3]
    with pytest.raises(ValueError, match="exactly one part"):
        P.confirm(J, [{"name": "a"}, {"name": "b"}])
