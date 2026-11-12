"""One request with one image to the job's VLM, and the parse of its JSON answer (build_plan.md note 5).

harpor.agent.physics_review asks the job's agent model to judge MuJoCo frames with these. Opus is Claude Opus
5.5 and Fable is Claude Fable 5.2, both through the anthropic SDK (beta messages with the server-side refusal
fallback, as claude_agent; effort high; both always think and take no temperature). Astra is GPT-6-Astra
through the OpenAI Responses API, called with urllib (no openai SDK is installed). parse() takes the first JSON
object of an answer.

The API key comes from the key file (the line of the agent's provider) into memory only. It is never printed or
stored: the caller passes each answer through agent_step.scrub with every key of the file, with its JSON escapes
of plain characters written out first (unescape), so that a key spelt with escapes is found too. The API errors
report only their kind or HTTP status, because an error body can quote the key.
"""

import base64
import io
import json
import re

from harpor.agent.tools import CLAUDE_MODEL, CODEX_MODEL, FALLBACK_BETA, env_values
from harpor.webapp.steps import KEY_VAR

OPENAI_URL = "https://api.openai.com/v1/responses"
OPUS_BETAS = [FALLBACK_BETA]
EFFORT = "high"
MAX_TOKENS = 16000
IMAGE_SIDE = 1024
TIMEOUT_S = 300


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
        return f"the answer is not valid JSON ({e})"
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


def jpeg(path, max_side=IMAGE_SIDE):
    """The photo as JPEG bytes, at most max_side px on the long side."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def ask_claude(key, prompt, image, client=None, model=CLAUDE_MODEL):
    """The answer text of model, Claude Opus 5.5 by default or Claude Fable 5.2. client: an
    anthropic.Anthropic or a stand-in (tests)."""
    import anthropic

    client = client or anthropic.Anthropic(api_key=key, max_retries=2, timeout=TIMEOUT_S)
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.standard_b64encode(image).decode("ascii")}},
               {"type": "text", "text": prompt}]
    try:
        msg = client.beta.messages.create(model=model, max_tokens=MAX_TOKENS, output_config={"effort": EFFORT},
                                          betas=OPUS_BETAS, fallbacks="default",
                                          messages=[{"role": "user", "content": content}])
    except anthropic.AuthenticationError:
        raise SystemExit("the Anthropic key was rejected") from None
    except (anthropic.PermissionDeniedError, anthropic.NotFoundError):
        raise SystemExit(f"this key cannot use {model}") from None
    except anthropic.APIStatusError as e:
        raise SystemExit(f"the Anthropic API answered HTTP {e.status_code}") from None
    except anthropic.APIConnectionError:
        raise SystemExit("could not reach the Anthropic API") from None
    if msg.stop_reason == "refusal":
        raise SystemExit(f"{model} declined to answer")
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
        raise SystemExit(f"{CODEX_MODEL} declined to answer")
    return "".join(c.get("text", "") for c in content if c.get("type") == "output_text")


def read_key(key_file, agent):
    """The value of the key file's line for the agent's provider (KEY_VAR), without quotes, else ''."""
    return next(iter(env_values(key_file, (KEY_VAR[agent],))), "")
