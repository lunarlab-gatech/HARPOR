"""The part list of an articulated object: from the job's VLM or from the user (build_plan.md note 11).

Runs in envs/sam3d through `python -m harpor.parts.stage propose` and writes run/proposal.json:

  {source: vlm|user, agent, model, noun, base, parts: [{name, parent, joint, count, axis_hint}],
   requirements: [{part, joint, text, range or material, source}], warnings, seconds}

The VLM is the job's agent model. Opus is Claude Opus 5.5 through the anthropic SDK (beta messages
with the server-side refusal fallback, as claude_agent; effort high; Opus 5.5 always thinks and takes
no temperature). Astra is GPT-6-Astra through the OpenAI Responses API, called with urllib (no openai
SDK is installed). Both get view 0's photo (run/input/images, as JPEG, at most 1024 px) and a prompt
that asks for one JSON object. parse() takes the first JSON object of the answer; an answer that does
not parse or validate is asked again once, with the reason.

validate() is the rule for every list, the VLM's and the user's: 1 to 12 parts, unique slugs
(sam3_masks.slug) that differ from the noun's and from "world" (joints.WORLD, the parent of a fixed
base), every parent in the list, exactly one root, which is the base, no loops, and no part that hangs
from a soft part. The prompt asks for whole rigid pieces (a chair's seat and backrest are one part), for
each soft part on its own even when it is fixed to a moving part (a chair's seat cushion, whose parent
is the seat and backrest), and for a base name that SAM 3 can find apart from the parts: part_masks
segments the base too, and the face vote uses its masks as evidence for the faces that do not move.
joint is revolute, prismatic, fixed, soft or unknown; count is how many identical parts the name covers
(the face vote numbers them); axis_hint is a few words on where the hinge or slide is. validate()
returns only these fields, since the web page's round trip carries no others. Names and texts lose
their control and format characters (a terminal escape, a right-to-left mark), since the stage log
prints them.

requirements say what the built model must do (paper Eq. 2; harpor.parts.stage finish expands them to
the found links). Each part that is not fixed has one, so the base has none: text, a few words on the
motion; range (not for a soft part), the full travel in degrees (revolute) or cm (prismatic), or None;
material (a soft part only), a key of joints.SOFT_MATERIALS; and source, vlm or default. The VLM answers
motion, range and material per part, and requirements_from() keeps them only within the limits. A
user's list gets default_requirements(): the default text, no range, foam. confirm() keeps the
requirement of a part with the same name (by slug, so a change of case keeps it, as the page keeps the
axis hint) and joint, under the confirmed name, and gives any other part the default (reconcile).
A requirement of joint unknown takes either moving joint.

The API key comes from the key file (the line of the agent's provider) into memory only. It is never
printed or stored: the VLM's answer passes through agent_step.scrub with every key of the file
(controller.read_keys) as soon as it arrives, before it is parsed or cut, with its JSON escapes of
plain characters written out first (unescape), so that a key spelt with escapes is found too. So do
every output and every error message; the API errors report only their kind or HTTP status, because
an error body can quote the key.
"""

import base64
import io
import json
import re
import time
from pathlib import Path

from harpor.agent.tools import CLAUDE_MODEL, CODEX_MODEL, FALLBACK_BETA, env_values
from harpor.fileio import write_json
from harpor.parts.joints import SOFT_MATERIALS, WORLD
from harpor.stage_a.sam3_masks import slug
from harpor.webapp.steps import KEY_VAR

JOINTS = ("revolute", "prismatic", "fixed", "soft", "unknown")
MAX_PARTS, MAX_COUNT, MAX_NAME, MAX_HINT = 12, 8, 40, 120
MAX_RANGE = 720  # degrees, or cm for a slide
SOFT_DEFAULT = "foam"
TEXT = {"revolute": "turns about its hinge or axle", "prismatic": "slides along its track",
        "soft": "dents under a press and springs back", "unknown": "moves relative to its parent"}
OPENAI_URL = "https://api.openai.com/v1/responses"
OPUS_BETAS = [FALLBACK_BETA]
EFFORT = "high"
MAX_TOKENS = 16000
IMAGE_SIDE = 1024
TIMEOUT_S = 300

PROMPT = """The photo shows a {noun}. We are building a simulation model of this {noun} in which its moving parts can move and its soft parts can dent.

List the parts of the {noun} that move relative to its main body, such as doors, drawers, lids, flaps, turning knobs or wheels, and its soft parts. Use what you can see in the photo. A rigid part is the whole rigid piece that moves as one, with every rigid piece fixed to it: handles, hinges, feet and trim move with their part and are not parts of their own, and the seat, backrest and armrests of an office chair turn together as one part, "seat and backrest". A soft part dents under a press and springs back, such as a cushion or a pad. List it as its own part with joint soft, even when it is fixed to a moving part, and give as its parent the part it sits on: the seat cushion of an office chair is a soft part whose parent is "seat and backrest". Leave out parts that look small in the photo (casters, levers, small buttons) unless the {noun} has no larger moving part. If nothing moves and nothing is soft, give an empty list.

Answer with one JSON object and nothing else:
{{"base": "<the main body, which does not move, as a short noun phrase>",
 "parts": [{{"name": "<short noun phrase>", "parent": "<the base, or the other part that this part is attached to>",
            "joint": "revolute or prismatic or fixed or soft or unknown", "count": <how many such parts the {noun} has>,
            "axis_hint": "<where the hinge or slide is and which way the part moves, in a few words>",
            "motion": "<what the part does when someone uses it, in a few words>",
            "range": <the full travel of the part from one end to the other, in degrees when it turns and in cm when it slides, or null>,
            "material": "<foam or gel or rubber for a soft part, else null>"}}]}}

Rules:
- A name is 1 to 4 words that an image segmentation model can find in the photo, such as "drawer", "glass door" or "lid", and it covers the whole part. Names differ from each other and from "{noun}".
- Name the base the same way, as what stays still when the parts move, so that the model can find it apart from them: "chair base" for an office chair whose seat turns.
- Identical parts get one entry with a count ("drawer", count 3). Give parts separate names only when they differ ("top drawer" and "bottom door").
- revolute: the part turns about a hinge or axle (door, lid, flap, knob, a turning seat). prismatic: the part slides (drawer, sliding door). soft: the part dents under a press and springs back (seat cushion, ear pad). unknown: you cannot tell.
- A part has one joint. When it both turns and slides, give its main motion.
- range is a number without units: 90 for a door that opens a quarter turn, 360 for a seat that turns all the way round, 40 for a drawer that pulls out 40 cm. Give null for a soft or fixed part, and when you cannot tell.
- material: foam for a cushion or pad, gel for a gel pad, rubber for a rubber grip or bumper.
- At most {most} parts."""


def _slug(name):
    try:
        return slug(name)
    except SystemExit as e:  # slug() is the CLI's check
        raise ValueError(str(e)) from None


def _words(s):
    """s on one line, its words one space apart, without control and format characters."""
    return " ".join("".join(c for c in s if c.isprintable() or c.isspace()).split())


def validate(parts, noun=None):
    """The part list in its normal form, the base first: [{name, parent, joint, count, axis_hint}].
    parts: a list of dicts. Missing joints are "unknown", missing counts 1; the base gets joint "fixed"
    and count 1, and every parent is written as its part's name. Raises ValueError with the reason when
    the list breaks a rule (see the module docstring)."""
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_PARTS:
        raise ValueError(f"a part list has 1 to {MAX_PARTS} parts")
    noun_slug = _slug(noun) if noun else None
    out, by_slug = [], {}
    for p in parts:
        if not isinstance(p, dict):
            raise ValueError(f"a part is an object with a name, not {p!r}")
        name = p.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("every part needs a name")
        name = _words(name)
        if len(name) > MAX_NAME:
            raise ValueError(f"part name {name!r} is longer than {MAX_NAME} characters")
        s = _slug(name)
        if s in by_slug:
            raise ValueError(f"parts {by_slug[s]['name']!r} and {name!r} give the same name {s!r}")
        if s == noun_slug:
            raise ValueError(f"part {name!r} has the name of the whole object; call its main body '{noun} body'")
        if s == WORLD:
            raise ValueError(f'part {name!r}: "{WORLD}" is reserved for the world that a fixed base hangs from')
        parent = p.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise ValueError(f"part {name!r}: the parent is a part name")
        joint = p.get("joint") or "unknown"
        if not isinstance(joint, str) or joint.strip().lower() not in JOINTS:
            raise ValueError(f"part {name!r}: joint is one of {', '.join(JOINTS)}, not {joint!r}")
        count = p.get("count")
        count = 1 if count is None else count
        if isinstance(count, float) and count.is_integer():
            count = int(count)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_COUNT:
            raise ValueError(f"part {name!r}: count is a whole number from 1 to {MAX_COUNT}")
        hint = p.get("axis_hint")
        hint = _words(hint)[:MAX_HINT] if isinstance(hint, str) else None
        parent = _words(parent) if parent else ""
        rec = {"name": name, "parent": parent or None, "joint": joint.strip().lower(), "count": count,
               "axis_hint": hint or None}
        by_slug[s] = rec
        out.append(rec)
    roots = [p["name"] for p in out if p["parent"] is None]
    if len(roots) != 1:
        raise ValueError(f"exactly one part, the base, has no parent; here {len(roots)} have none: {roots}")
    for p in out:
        if p["parent"] is not None:
            ps = _slug(p["parent"])
            if ps not in by_slug:
                raise ValueError(f"part {p['name']!r}: its parent {p['parent']!r} is not in the list")
            if ps == _slug(p["name"]):
                raise ValueError(f"part {p['name']!r} is its own parent")
            p["parent"] = by_slug[ps]["name"]
    for p in out:
        seen, q = set(), p
        while q["parent"] is not None:
            if q["name"] in seen:
                raise ValueError(f"the parents of {p['name']!r} form a loop")
            seen.add(q["name"])
            q = by_slug[_slug(q["parent"])]
    root = by_slug[_slug(roots[0])]
    root.update(joint="fixed", count=1)
    for p in out:
        if p["parent"] is not None and by_slug[_slug(p["parent"])]["joint"] == "soft":
            raise ValueError(f"part {p['name']!r} cannot hang from the soft part {p['parent']!r}")
    return [root] + [p for p in out if p is not root]


def _range(r):
    """r as a float when it is a finite number above 0 and at most MAX_RANGE, else None."""
    if isinstance(r, bool) or not isinstance(r, (int, float)) or not 0 < r <= MAX_RANGE:
        return None  # NaN and inf fail the comparison too
    return float(r)


def _requirement(part, source, motion=None, span=None, material=None):
    """The requirement of a part of validate()'s list (see the module docstring). motion, span and
    material: the VLM's answer fields, kept only within the limits."""
    text = _words(motion)[:MAX_HINT] if isinstance(motion, str) else ""
    rec = {"part": part["name"], "joint": part["joint"], "text": text or TEXT[part["joint"]]}
    if part["joint"] == "soft":
        material = material.strip().lower() if isinstance(material, str) else None
        rec["material"] = material if material in SOFT_MATERIALS else SOFT_DEFAULT
    else:
        rec["range"] = _range(span)
    rec["source"] = source
    return rec


def requirements_from(items, parts):
    """The requirements of a VLM answer: items, its "parts" objects; parts, validate()'s list of them. A
    part takes the motion, range and material of the item of its name; the fixed parts get none."""
    said = {_words(p["name"]): p for p in items if isinstance(p, dict) and isinstance(p.get("name"), str)}
    out = []
    for p in parts:
        if p["joint"] != "fixed":
            item = said.get(p["name"], {})
            out.append(_requirement(p, "vlm", item.get("motion"), item.get("range"), item.get("material")))
    return out


def default_requirements(parts):
    """The requirements of a list without the VLM's answer fields (the user's): the default text, no range,
    foam."""
    return [_requirement(p, "default") for p in parts if p["joint"] != "fixed"]


def reconcile(parts, old):
    """The requirements of a confirmed list parts: a part keeps the one of old with its name's slug and
    its joint, under its confirmed name; any other part gets the default. old: the proposal's
    requirements, None in an older proposal."""
    kept = {(_slug(r["part"]), r["joint"]): r for r in old or []}
    out = []
    for p in parts:
        if p["joint"] != "fixed":
            r = kept.get((_slug(p["name"]), p["joint"]))
            out.append({**r, "part": p["name"]} if r else _requirement(p, "default"))
    return out


ESCAPE = re.compile(r"\\(?:u([0-9a-fA-F]{4})|.)", re.S)  # one JSON escape; an escaped backslash is one too


def unescape(text):
    """text with each JSON escape by code (u and 4 hex digits) of a printable ASCII character other than
    a quote and a backslash written as that character. The other escapes stay, so a JSON object in text
    decodes to the same."""
    def plain(m):
        c = chr(int(m[1], 16)) if m[1] else None
        return c if c and c.isascii() and c.isprintable() and c not in '"\\' else m[0]

    return ESCAPE.sub(plain, text)


def not_json(text):
    """Why the first JSON object of an answer does not decode, else None."""
    try:
        json.JSONDecoder().raw_decode(text, text.find("{"))
    except ValueError as e:
        return f"the answer is not valid JSON ({e}); a number such as range takes no units"
    return None


def parse(text):
    """The first JSON object in text. ValueError when there is none."""
    dec = json.JSONDecoder()
    i = text.find("{")
    while i >= 0:
        try:
            obj, _ = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        i = text.find("{", i + 1)
    raise ValueError(not_json(text) if "{" in text else "the answer holds no JSON object")


def from_answer(text, noun):
    """(parts, warnings) from a VLM answer {base, parts}: the base becomes the root, a part without a
    parent (or with the noun as parent) hangs from the base, a base named like the noun becomes
    "<noun> body", and a part that repeats the base is dropped. Then validate(). ValueError when the
    answer cannot be used; when the answer's object does not decode, parse() may give an object inside
    it, so the reason is the decode error."""
    d = parse(text)
    base = d.get("base")
    if not isinstance(base, str) or not base.strip():
        raise ValueError(not_json(text) or 'the answer has no "base"')
    items = d.get("parts") or []
    if not isinstance(items, list) or not all(isinstance(p, dict) for p in items):
        raise ValueError('"parts" is not a list of objects')
    warnings, said = [], _words(base)
    if _slug(said) == _slug(noun):
        base = f"{noun} body"
        warnings.append(f"the base {said!r} has the name of the object; it is called {base!r}")
    top = {_slug(said), _slug(noun), _slug(base)}
    parts = [{"name": base, "parent": None, "joint": "fixed"}]
    for p in items:
        p = dict(p)
        if isinstance(p.get("name"), str) and p["name"].strip() and _slug(p["name"]) in top:
            warnings.append(f"the part {p['name']!r} repeats the base; left out")
            continue
        parent = p.get("parent")
        if not isinstance(parent, str) or not parent.strip() or _slug(parent) in top:
            p["parent"] = base
        parts.append(p)
    if len(parts) > MAX_PARTS:
        warnings.append(f"{len(parts) - 1} parts listed; the first {MAX_PARTS - 1} are kept")
        parts = parts[:MAX_PARTS]
    if len(parts) == 1:
        warnings.append("no moving part was listed")
    return validate(parts, noun), warnings


def jpeg(path, max_side=IMAGE_SIDE):
    """The photo as JPEG bytes, at most max_side px on the long side."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def ask_opus(key, prompt, image, client=None):
    """Claude Opus 5.5's answer text. client: an anthropic.Anthropic or a stand-in (tests)."""
    import anthropic

    client = client or anthropic.Anthropic(api_key=key, max_retries=2, timeout=TIMEOUT_S)
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.standard_b64encode(image).decode("ascii")}},
               {"type": "text", "text": prompt}]
    try:
        msg = client.beta.messages.create(model=CLAUDE_MODEL, max_tokens=MAX_TOKENS, output_config={"effort": EFFORT},
                                          betas=OPUS_BETAS, fallbacks="default",
                                          messages=[{"role": "user", "content": content}])
    except anthropic.AuthenticationError:
        raise SystemExit("the Anthropic key was rejected") from None
    except (anthropic.PermissionDeniedError, anthropic.NotFoundError):
        raise SystemExit(f"this key cannot use {CLAUDE_MODEL}") from None
    except anthropic.APIStatusError as e:
        raise SystemExit(f"the Anthropic API answered HTTP {e.status_code}") from None
    except anthropic.APIConnectionError:
        raise SystemExit("could not reach the Anthropic API") from None
    if msg.stop_reason == "refusal":
        raise SystemExit(f"{CLAUDE_MODEL} declined to list the parts; type the parts instead")
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def ask_astra(key, prompt, image):
    """GPT-6-Astra's answer text, from POST /v1/responses with the photo as a data URL."""
    import urllib.error
    import urllib.request

    url = "data:image/jpeg;base64," + base64.standard_b64encode(image).decode("ascii")
    body = {"model": CODEX_MODEL, "reasoning": {"effort": EFFORT}, "max_output_tokens": MAX_TOKENS,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt},
                                                   {"type": "input_image", "image_url": url, "detail": "high"}]}]}
    req = urllib.request.Request(OPENAI_URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:  # its body can quote part of the key, so only the code is kept
        raise SystemExit(f"the OpenAI API answered HTTP {e.code}") from None
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(f"could not reach api.openai.com: {getattr(e, 'reason', e)}") from None
    except ValueError:
        raise SystemExit("the OpenAI API's answer is not JSON") from None
    if data.get("status") not in (None, "completed"):
        why = (data.get("incomplete_details") or {}).get("reason") or data.get("status")
        raise SystemExit(f"{CODEX_MODEL}'s answer is incomplete ({why})")
    content = [c for o in data.get("output") or [] if o.get("type") == "message" for c in o.get("content") or []]
    if any(c.get("type") == "refusal" for c in content):
        raise SystemExit(f"{CODEX_MODEL} declined to list the parts; type the parts instead")
    return "".join(c.get("text", "") for c in content if c.get("type") == "output_text")


def propose_vlm(noun, image, agent, key, client=None, keys=()):
    """The proposal of the job's VLM for the photo at image. agent: astra or opus. Every answer is
    scrubbed of key and keys (the other keys of the key file) before anything reads it."""
    from harpor.webapp.agent_step import scrub

    if agent not in KEY_VAR:
        raise SystemExit(f"the VLM is astra or opus, not {agent!r}")
    t0, photo = time.time(), jpeg(image)
    prompt, extra = PROMPT.format(noun=noun, most=MAX_PARTS - 1), ""
    for attempt in (1, 2):
        text = ask_opus(key, prompt + extra, photo, client) if agent == "opus" else ask_astra(key, prompt + extra, photo)
        text = scrub(unescape(text), [key, *keys])
        try:
            parts, warnings = from_answer(text, noun)
            break
        except ValueError as e:
            if attempt == 2:
                raise SystemExit(f"the VLM gave no usable part list twice; the second time: {e}") from None
            extra = f"\n\nYour previous answer could not be used: {e}. Answer with the JSON object only."
    return {"source": "vlm", "agent": agent, "model": CLAUDE_MODEL if agent == "opus" else CODEX_MODEL, "noun": noun,
            "base": parts[0]["name"], "parts": parts, "requirements": requirements_from(parse(text).get("parts") or [], parts),
            "warnings": warnings, "seconds": round(time.time() - t0, 1), "attempts": attempt, "answer": text[:4000]}


def propose_user(job):
    """The proposal of the list the user typed (job.json parts.list)."""
    from harpor.webapp import steps

    items = job["parts"]["list"]
    if not items:
        raise SystemExit("the job has no part list; type the parts")
    try:
        parts = validate(items, job["prompt"])
        agent = steps.agent(job)
    except ValueError as e:
        raise SystemExit(f"the part list: {e}") from None
    return {"source": "user", "agent": agent, "model": None, "noun": job["prompt"], "base": parts[0]["name"],
            "parts": parts, "requirements": default_requirements(parts), "warnings": [], "seconds": 0.0}


def read_key(key_file, agent):
    """The value of the key file's line for the agent's provider (KEY_VAR), without quotes, else ''."""
    return next(iter(env_values(key_file, (KEY_VAR[agent],))), "")


def run(job_dir, key_file=None, agent=None, client=None):
    """Write J/run/proposal.json: the user's list when the job's parts mode is "user" or its agent is
    "none", else the VLM's (the key from key_file), with its requirements."""
    from harpor.agent import controller
    from harpor.stage_a.mvsam3d import view_names
    from harpor.webapp import steps
    from harpor.webapp.agent_step import scrub

    J = Path(job_dir)
    job = steps.job(J)
    agent = agent or steps.agent(job)
    mode = steps.parts_mode(job)
    keys = controller.read_keys(key_file) if key_file and Path(key_file).is_file() else []
    out = J / "run" / "proposal.json"
    out.unlink(missing_ok=True)
    try:
        if mode == "user" or agent == "none":
            proposal = propose_user(job)
        else:
            if not key_file or not Path(key_file).is_file():
                raise SystemExit("the VLM's part list needs --key-file")
            key = read_key(key_file, agent)
            if not key:
                raise SystemExit(f"the key file has no {KEY_VAR[agent]} line")
            view0 = view_names(J / "run")[0]
            proposal = propose_vlm(job["prompt"], J / "run" / "input" / "images" / f"{view0}.png", agent, key, client,
                                   keys)
    except SystemExit as e:
        raise SystemExit(scrub(str(e.code), keys)) from None
    proposal = scrub({**proposal, "mode": mode}, keys)
    write_json(out, proposal)
    print(f"proposal ({proposal['source']}{', ' + proposal['model'] if proposal.get('model') else ''}): "
          f"base {proposal['base']!r}, {len(proposal['parts']) - 1} moving parts -> {out}")
    for p in proposal["parts"][1:]:
        print(f"  {p['name']} > {p['parent']} : {p['joint']}" + (f" x{p['count']}" if p["count"] > 1 else "")
              + (f" ({p['axis_hint']})" if p["axis_hint"] else ""))
    for r in proposal["requirements"]:
        print(f"  requirement of {r['part']}: {r['text']}" + (f", range {r['range']:g}" if r.get("range") else "")
              + (f", {r['material']}" if "material" in r else "") + f" ({r['source']})")
    for w in proposal["warnings"]:
        print(f"  warning: {w}")
    return proposal


def confirm(job_dir, parts):
    """Rewrite J/run/proposal.json with the list the user confirmed (the "VLM proposes, I edit" mode):
    parts goes through validate() (its ValueError passes on), its requirements through reconcile(), and
    the record gains confirmed and edited. Returns it."""
    from harpor.webapp import steps

    J = Path(job_dir)
    job = steps.job(J)
    f = J / "run" / "proposal.json"
    proposal = json.loads(f.read_text())
    parts = validate(parts, job["prompt"])
    proposal.update(base=parts[0]["name"], parts=parts, requirements=reconcile(parts, proposal.get("requirements")),
                    edited=parts != proposal.get("parts"), confirmed=True)
    write_json(f, proposal)
    return proposal
