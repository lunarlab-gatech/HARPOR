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
from harpor.parts import propose as P  # noqa: E402

OPUS_KEY = "sk-ant-api03-" + "q" * 40
OPENAI_KEY = "sk-proj-" + "z" * 40
GOOD = json.dumps({"base": "cabinet body", "parts": [
    {"name": "drawer", "parent": "cabinet body", "joint": "prismatic", "count": 2, "axis_hint": "slides out toward the camera"},
    {"name": "door", "parent": "", "joint": "Revolute", "count": 1, "axis_hint": "hinge on the left edge"}]})
CHAIR = json.dumps({"base": "chair base", "parts": [
    {"name": "seat and backrest", "parent": "chair base", "joint": "revolute", "axis_hint": "the gas lift",
     "motion": "turns  all the way\n round on the base", "range": 360, "material": "rubber"},
    {"name": "seat  cushion", "parent": "seat and backrest", "joint": "soft", "motion": "x" * 200, "range": 5,
     "material": " Gel "},
    {"name": "armrest pad", "parent": "seat and backrest", "joint": "soft", "material": "memory foam"},
    {"name": "footrest", "parent": "chair base", "joint": "prismatic", "motion": "slides out", "range": 30},
    {"name": "lever", "parent": "chair base", "joint": "fixed", "motion": "stays put", "range": float("nan")},
    {"name": "knob", "parent": "chair base", "joint": "unknown", "range": "a quarter turn"}]})


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
    parts = P.validate([{"name": "Lid", "parent": " box   body", "joint": "Revolute"}, {"name": "box body"}], noun="box")
    assert parts == [{"name": "box body", "parent": None, "joint": "fixed", "count": 1, "axis_hint": None},
                     {"name": "Lid", "parent": "box body", "joint": "revolute", "count": 1, "axis_hint": None}]


@pytest.mark.parametrize("parts,why", [
    ([], "1 to 12"),
    (["body"], "an object with a name"),
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
    ([{"name": "body"}, {"name": "pad", "parent": "body", "joint": "soft"}, {"name": "cover", "parent": "pad"}],
     "'cover' cannot hang from the soft part 'pad'"),
])
def test_validate_catches_each_fault(parts, why):
    with pytest.raises(ValueError, match=why):
        P.validate(parts, noun="cabinet")


def test_validate_takes_a_soft_part():
    parts = P.validate([{"name": "chair base"}, {"name": "seat", "parent": "chair base", "joint": "revolute"},
                        {"name": "Seat cushion", "parent": "seat", "joint": "Soft"}], noun="office chair")
    assert [p["joint"] for p in parts] == ["fixed", "revolute", "soft"]
    # The base is always fixed, whatever the list says, so a part may hang from it.
    parts = P.validate([{"name": "body", "joint": "soft"}, {"name": "door", "parent": "body"}], noun="cabinet")
    assert parts[0]["joint"] == "fixed"


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
    assert "the seat cushion of an office chair is a soft part" in text["text"] and '"material"' in text["text"]
    assert proposal["source"] == "vlm" and proposal["agent"] == "opus" and proposal["model"] == "claude-opus-5-5"
    assert proposal["base"] == "cabinet body" and proposal["mode"] == "vlm"
    assert [(p["name"], p["parent"], p["joint"], p["count"]) for p in proposal["parts"]] == [
        ("cabinet body", None, "fixed", 1), ("drawer", "cabinet body", "prismatic", 2), ("door", "cabinet body", "revolute", 1)]
    # An answer without motion, range and material: the default text, still the VLM's requirement.
    assert proposal["requirements"] == [
        {"part": "drawer", "joint": "prismatic", "text": "slides along its track", "range": None, "source": "vlm"},
        {"part": "door", "joint": "revolute", "text": "turns about its hinge or axle", "range": None, "source": "vlm"}]
    assert json.loads((J / "run" / "proposal.json").read_text()) == proposal
    assert no_key_in(J, OPUS_KEY) == []


def test_the_answer_fields_become_requirements(tmp_path, capsys):
    """motion is cut as axis_hint is, range kept only as a number within the limits, material only from the
    table (else foam); a fixed part and the base get no requirement."""
    J = job(tmp_path)
    proposal = P.run(J, key_file(tmp_path, f"ANTHROPIC_API_KEY={OPUS_KEY}"), client=FakeClient([CHAIR]))
    assert proposal["requirements"] == [
        {"part": "seat and backrest", "joint": "revolute", "text": "turns all the way round on the base", "range": 360.0,
         "source": "vlm"},
        {"part": "seat cushion", "joint": "soft", "text": "x" * 120, "material": "gel", "source": "vlm"},
        {"part": "armrest pad", "joint": "soft", "text": "dents under a press and springs back", "material": "foam",
         "source": "vlm"},
        {"part": "footrest", "joint": "prismatic", "text": "slides out", "range": 30.0, "source": "vlm"},
        {"part": "knob", "joint": "unknown", "text": "moves relative to its parent", "range": None, "source": "vlm"}]
    assert json.loads((J / "run" / "proposal.json").read_text()) == proposal  # the NaN range was not kept
    assert "requirement of seat and backrest: turns all the way round on the base, range 360 (vlm)" in capsys.readouterr().out


@pytest.mark.parametrize("given,kept", [(90, 90.0), (0.5, 0.5), (720, 720.0), (0, None), (-90, None), (721, None),
                                        ("90", None), (True, None), (None, None), (float("nan"), None),
                                        (float("inf"), None), ([90], None)])
def test_a_range_is_a_number_above_0_and_at_most_720(given, kept):
    parts = P.validate([{"name": "body"}, {"name": "lid", "parent": "body", "joint": "revolute"}])
    (req,) = P.requirements_from([{"name": "lid", "range": given}], parts)
    assert req["range"] == kept


@pytest.mark.parametrize("given,kept", [("foam", "foam"), (" Gel ", "gel"), ("RUBBER", "rubber"), ("memory foam", "foam"),
                                        ("steel", "foam"), (3, "foam"), (None, "foam")])
def test_a_material_comes_from_the_table_else_foam(given, kept):
    parts = P.validate([{"name": "chair"}, {"name": "cushion", "parent": "chair", "joint": "soft"}])
    (req,) = P.requirements_from([{"name": "cushion", "material": given, "range": 10}], parts)
    assert req == {"part": "cushion", "joint": "soft", "text": "dents under a press and springs back", "material": kept,
                   "source": "vlm"}


def test_a_user_list_gets_the_default_requirements():
    parts = P.validate([{"name": "body"}, {"name": "lid", "parent": "body", "joint": "revolute"},
                        {"name": "drawer", "parent": "body", "joint": "prismatic", "count": 2},
                        {"name": "pad", "parent": "lid", "joint": "soft"}, {"name": "foot", "parent": "body", "joint": "fixed"},
                        {"name": "knob", "parent": "body"}])
    assert P.default_requirements(parts) == [
        {"part": "lid", "joint": "revolute", "text": "turns about its hinge or axle", "range": None, "source": "default"},
        {"part": "drawer", "joint": "prismatic", "text": "slides along its track", "range": None, "source": "default"},
        {"part": "pad", "joint": "soft", "text": "dents under a press and springs back", "material": "foam",
         "source": "default"},
        {"part": "knob", "joint": "unknown", "text": "moves relative to its parent", "range": None, "source": "default"}]


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
    # The key crosses all three cuts: an axis_hint's and a requirement text's 120 characters and the stored
    # answer's 4000.
    answer = json.dumps({"base": "cabinet body", "parts": [{"name": "door", "parent": "cabinet body",
                                                            "axis_hint": "a" * 100 + " " + OPUS_KEY,
                                                            "motion": "m" * 100 + " " + OPUS_KEY}]})
    answer += " " + "y" * (3990 - len(answer) - 1) + OPUS_KEY
    assert answer.index(OPUS_KEY, 3000) == 3990
    J = job(tmp_path)
    proposal = P.run(J, key_file(tmp_path, f"ANTHROPIC_API_KEY={OPUS_KEY}"), client=FakeClient([answer]))
    assert proposal["parts"][1]["axis_hint"] == "a" * 100 + " [redacted API key]"
    assert proposal["requirements"][0]["text"] == "m" * 100 + " [redacted API key]"
    assert proposal["answer"].count("[redacted API key]") == 3 and OPUS_KEY[:10] not in proposal["answer"]
    assert no_key_in(J, OPUS_KEY[:10]) == []


def test_a_key_spelt_with_json_escapes_is_scrubbed(tmp_path):
    """The answer writes the key's first letter as an escape, which json.loads turns back into the key; the motion
    cut at 120 characters would keep a part of it."""
    escaped = "\\u0073" + OPUS_KEY[1:]
    answer = json.dumps({"base": "cabinet body", "parts": [{"name": "door", "parent": "cabinet body",
                                                            "motion": "m" * 100 + " KEY", "axis_hint": "KEY"}]})
    J = job(tmp_path)
    proposal = P.run(J, key_file(tmp_path, f"ANTHROPIC_API_KEY={OPUS_KEY}"),
                     client=FakeClient([answer.replace("KEY", escaped)]))
    assert proposal["requirements"][0]["text"] == "m" * 100 + " [redacted API key]"
    assert proposal["parts"][1]["axis_hint"] == "[redacted API key]" and OPUS_KEY[1:10] not in proposal["answer"]
    assert no_key_in(J, OPUS_KEY[1:10]) == []
    # An escaped backslash, a quote and a control character stay escaped, so the object decodes to the same.
    text = json.dumps({"a": "\\u0073 \" \x1b \\ s"})
    assert P.unescape(text) == text and P.unescape('{"a": "\\u0041\\u0022"}') == '{"a": "A\\u0022"}'


def test_control_characters_leave_the_answer_text(tmp_path, capsys):
    """A terminal escape or a right-to-left mark in a name, a hint or a motion would reach the stage log."""
    answer = json.dumps({"base": "cabinet body", "parts": [{"name": "door‮", "parent": "cabinet body",
                                                            "joint": "revolute", "axis_hint": "left\x1b[2J edge",
                                                            "motion": "swings‮\x07 open"}]})
    proposal = P.run(job(tmp_path), key_file(tmp_path, f"ANTHROPIC_API_KEY={OPUS_KEY}"), client=FakeClient([answer]))
    assert proposal["parts"][1]["name"] == "door" and proposal["parts"][1]["axis_hint"] == "left[2J edge"
    assert proposal["requirements"][0]["text"] == "swings open"
    out = capsys.readouterr().out
    assert "door > cabinet body : revolute (left[2J edge)" in out and not any(c in out for c in "\x1b\x07‮")


def test_an_answer_that_is_not_json_says_so(tmp_path):
    """A range with units breaks the outer object: the reason is the decode error, not the base that parse() missed
    when it went on to the object of the second part (or found no object at all with one part)."""
    two = json.loads(GOOD)
    two["parts"][1]["range"] = "RANGE"
    for answer in (json.dumps(two), json.dumps({**two, "parts": two["parts"][1:]})):
        answer = answer.replace('"RANGE"', "90 degrees")
        with pytest.raises(ValueError, match=r"not valid JSON \(Expecting ',' delimiter: .*\); .* no units"):
            P.from_answer(answer, "cabinet")
    with pytest.raises(ValueError, match='no "base"'):
        P.from_answer('{"parts": [{"name": "door"}]}', "cabinet")
    J = job(tmp_path)
    fake = FakeClient([json.dumps(two).replace('"RANGE"', "90 degrees"), GOOD])
    P.propose_vlm("cabinet", J / "run" / "input" / "images" / "0.png", "opus", OPUS_KEY, client=fake)
    assert "could not be used: the answer is not valid JSON" in fake.calls[1]["messages"][0]["content"][1]["text"]


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
    assert proposal["mode"] == "user"  # the mode the job runs in, not the one it asked for
    assert [p["name"] for p in proposal["parts"]] == ["body", "lid"] and proposal["base"] == "body"
    assert proposal["requirements"] == [{"part": "lid", "joint": "revolute", "text": "turns about its hinge or axle",
                                         "range": None, "source": "default"}]
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


def test_confirm_keeps_a_requirement_only_for_the_same_name_and_joint(tmp_path):
    J = job(tmp_path)
    vlm = P.run(J, key_file(tmp_path, f"ANTHROPIC_API_KEY={OPUS_KEY}"), client=FakeClient([CHAIR]))["requirements"]
    edited = [{"name": "chair base"}, {"name": "seat and backrest", "parent": "chair base", "joint": "revolute", "count": 2},
              {"name": "seat cushion", "parent": "seat and backrest", "joint": "soft"},
              {"name": "footrest", "parent": "chair base", "joint": "revolute"},
              {"name": "headrest", "parent": "seat and backrest", "joint": "prismatic"}]
    saved = P.confirm(J, edited)
    assert saved["requirements"] == [
        vlm[0], vlm[1],  # the count does not matter, nor does the armrest pad that was left out
        {"part": "footrest", "joint": "revolute", "text": "turns about its hinge or axle", "range": None, "source": "default"},
        {"part": "headrest", "joint": "prismatic", "text": "slides along its track", "range": None, "source": "default"}]
    assert json.loads((J / "run" / "proposal.json").read_text())["requirements"] == saved["requirements"]
    # The VLM's list confirmed as it is keeps every requirement.
    assert P.confirm(J, saved["parts"])["requirements"] == saved["requirements"]
    # A change of case keeps the requirement, as the page keeps the axis hint, under the confirmed name.
    renamed = [{**p, "name": "Seat And Backrest"} if p["name"] == "seat and backrest" else
               {**p, "parent": "Seat And Backrest"} if p["parent"] == "seat and backrest" else p for p in saved["parts"]]
    assert P.confirm(J, renamed)["requirements"] == [{**vlm[0], "part": "Seat And Backrest"}, *saved["requirements"][1:]]
    # A proposal written before requirements (a job that waited at its part list across the update) gets the defaults.
    old = {k: v for k, v in saved.items() if k != "requirements"}
    (J / "run" / "proposal.json").write_text(json.dumps(old))
    assert P.confirm(J, saved["parts"])["requirements"] == P.default_requirements(saved["parts"])
