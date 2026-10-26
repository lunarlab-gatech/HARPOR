"""What the personal worker (worker.py) needs to relay a request from the public site to server.py and to
publish a job's state back: the route table, the agent that server.py will run, the rewrite and scrub of
everything that leaves the machine, the files that the site mirrors, lean review files, the key scan,
and the live state and retry stage that the page would compute.

No I/O here but reading job files. The worker sends each request through httpx.ASGITransport to the
unchanged FastAPI app, so server.py stays the one place for request checks, and its answer goes back as
it is, after rewrite().

The rewrite, on every view, snapshot, answer and log text: the local job id becomes the job's uuid; the
job folder, the runs folder and $G become "<job>", "<runs>" and "$G"; every API key held for the job,
and anything shaped like an OpenAI or Anthropic key (sk-... of 16 or more characters), becomes "[key]".
The view loses video.path and "public", keeps in "files" and "sizes" only what the site holds (the
manifest), and takes the row's created_at as "created".

The site mirrors every path that server.ALLOW matches and server.resolve accepts (GLOBS covers ALLOW,
and a test checks that), once the stage that writes it is not running (OWNERS). Text files (stage logs,
JSON, URDF) are scrubbed before they go up, and so are the text files inside results.zip, which is
packed again; the key scan then covers what is left: the keys held for the job, and in binary data anything
shaped like a key (KEY_BYTES), so a key is found after a restart too. In lean mode (settings.lean_files) a job keeps
about LEAN_THUMBS thumbnails plus the candidates, check images and view PNGs go up as JPEG (quality 85,
long side at most 1280 px) under the same path, and a preview video over LEAN_PREVIEW is left out.
Browsers read an image by its content, so the suffix does not matter.
"""

import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from gator.webapp import jobs, server, steps

KEY_ROUTES = ("start", "overrides", "parts", "continue")  # the routes whose body may carry an API key
# route -> (method, path after /api/jobs/<local>, body): json, text (the calibration file) or none
ROUTES = {"topic": ("POST", "/topic", "json"), "calibration_put": ("PUT", "/calibration", "text"),
          "calibration_delete": ("DELETE", "/calibration", "none"), "start": ("POST", "/start", "json"),
          "overrides": ("POST", "/overrides", "json"), "parts": ("POST", "/parts", "json"),
          "continue": ("POST", "/continue", "json"), "cancel": ("POST", "/cancel", "none"),
          "delete": ("DELETE", "", "none")}
NO_BUILD = ("topic", "calibration_put", "calibration_delete", "cancel", "delete")  # near the end of a Slurm job
KEY_SHAPE = re.compile(r"sk-[A-Za-z0-9_-]{16,}")  # OpenAI sk-..., Anthropic sk-ant-...
# The same shape in binary data, standing alone (not the end of a word such as "desk-..."), for the key scan of
# files after the worker forgot the job's keys (a restart, or a job that stopped)
KEY_BYTES = re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}")
# Every ALLOW path, and the glob in the job folder that finds it. A new ALLOW path must be added here.
GLOBS = {r"frames/thumbs/\d+\.jpg": "frames/thumbs/*.jpg",
         r"frames/topic_previews/\d+\.jpg": "frames/topic_previews/*.jpg",
         r"frames/topics\.json": "frames/topics.json", r"frames/preview\.mp4": "frames/preview.mp4",
         r"pool/lidar\.json": "pool/lidar.json", r"lidar/check/f\d{5,9}\.jpg": "lidar/check/*.jpg",
         r"lidar/cloud/f\d{5,9}\.glb": "lidar/cloud/*.glb",
         r"scene/sam3_\w+/check/[\w-]+\.jpg": "scene/sam3_*/check/*.jpg",
         r"run/input/images/\d{1,3}\.png": "run/input/images/*.png", r"run/lidar/\d{1,3}\.glb": "run/lidar/*.glb",
         r"run/overlay_\d+\.png": "run/overlay_*.png",
         r"run/posed\.glb": "run/posed.glb", r"run/parts\.glb": "run/parts.glb", r"run/parts\.json": "run/parts.json",
         r"run/parts/sam3/check/\d+\.jpg": "run/parts/sam3/check/*.jpg", r"refined\.glb": "refined.glb",
         r"physics/aligned\.glb": "physics/aligned.glb", r"turntable/turntable\.mp4": "turntable/turntable.mp4",
         r"physics/summary\.json": "physics/summary.json",
         r"physics/mjcf/mjcf\.json": "physics/mjcf/mjcf.json", r"physics/urdf/model\.urdf": "physics/urdf/model.urdf",
         r"run/placement\.json": "run/placement.json", r"selection\.json": "selection.json",
         r"results\.zip": "results.zip", r"logs/\w+\.log": "logs/*.log"}
# The stage that writes each mirrored path (the first prefix that matches); logs/<stage>.log is its stage's.
OWNERS = (("frames/topics.json", "topics"), ("frames/topic_previews/", "topics"), ("frames/", "frames"),
          ("pool/lidar.json", "lidar"), ("lidar/", "lidar"), ("scene/", "masks"), ("selection.json", "select"),
          ("run/input/", "select"), ("run/lidar/", "select"), ("run/overlay_", "place"), ("run/posed.glb", "place"),
          ("run/placement.json", "place"), ("run/parts", "parts"), ("refined.glb", "agent"),
          ("physics/aligned.glb", "align"), ("physics/summary.json", "vomp"), ("physics/mjcf/", "mjcf"),
          ("physics/urdf/", "urdf"), ("turntable/", "turntable"), ("results.zip", "package"))
LEAN_THUMBS = 150
LEAN_PREVIEW = 30 << 20  # bytes
LEAN_SIDE = 1280  # px, the long side of a lean review image
LEAN_IMAGES = re.compile(r"scene/sam3_\w+/check/[\w-]+\.jpg|lidar/check/f\d+\.jpg|run/parts/sam3/check/\d+\.jpg"
                         r"|run/input/images/\d+\.png|run/overlay_\d+\.png")
LOG_FILE_CAP = 5 << 20  # bytes of a stage log file on the site: its end
TEXT = (".json", ".log", ".urdf", ".xml", ".usda", ".txt", ".csv", ".md", ".yaml", ".yml")  # scrubbed as text
# The content types of the files on the site: server.TYPES, but a URDF as plain text, so the job-files bucket takes
# no XML (a browser runs the scripts of an XHTML document, which would then come from the project's domain).
SITE_TYPES = {**server.TYPES, ".urdf": "text/plain; charset=utf-8"}
SCAN_CHUNK = 1 << 20
# The table checks of supabase/migrations: bytes of the JSON text
VIEW_LIMIT, LIVE_LIMIT, SNAPSHOT_LIMIT, ANSWER_LIMIT = 262144, 65536, 1048576, 65536
LOG_ROW = 60000  # characters of one job_logs row (the table takes 65536)
PROGRESS = re.compile(r"^PROGRESS [\d.]+\r?\n", re.M)  # app.js drops these lines too
FAILED = "the worker on your machine failed on this request"


class Refuse(Exception):
    """A request the worker answers itself: (HTTP status, detail), the way server.Bad does."""

    def __init__(self, status, detail):
        super().__init__(status, detail)
        self.status, self.detail = status, detail


class Skip(Exception):
    """A file that stays on the machine; the message is the note for the view's warnings."""


# ---- requests ----

def build(route, local, body=None, query=None, text=None):
    """(method, path, content, headers) of the request to the inner app for a relayed route. The path comes
    from the route table and the local job id, never from user text; the calibration's file name goes in
    the query string."""
    if route not in ROUTES:
        raise Refuse(400, "unknown request")
    if not isinstance(local, str) or not jobs.ID.fullmatch(local):
        raise ValueError(f"not a local job id: {local!r}")
    method, tail, kind = ROUTES[route]
    path, content, headers = f"/api/jobs/{local}{tail}", None, {}
    if kind == "json":
        content, headers = json.dumps(body or {}).encode(), {"Content-Type": "application/json"}
    elif kind == "text":
        name = (query or {}).get("name")
        if not isinstance(name, str):
            raise Refuse(400, "name the file: ?name=<file name>")
        path += "?" + urlencode({"name": name})
        content, headers = (text or "").encode("utf-8"), {"Content-Type": "application/octet-stream"}
    return method, path, content, headers


def effective_agent(route, body, job):
    """The agent that server.py will run for a relayed request, or None for a route without one: start
    takes the body's agent (astra when it names none), overrides and continue the body's, else the job's,
    and parts the job's (set_parts). None too for an agent that server.py refuses itself (it answers 400)."""
    try:
        if route == "start":
            return server.agent_of(body)
        if route in ("overrides", "continue"):
            return server.agent_of(body, job)
        if route == "parts":
            return steps.agent(job)
    except (server.Bad, ValueError):
        return None
    return None


def agent_refusal(agent, access):
    """Refuse(403) when the account may not use agent (access.agents, which never holds "none"), else None."""
    allowed = [a for a in (access or {}).get("agents") or [] if a in steps.PROVIDER]  # "none" stays in the tunnel
    if agent in allowed:
        return None
    return Refuse(403, f"this account may use the agents {', '.join(allowed)}")


# ---- the rewrite ----

def site_name(name):
    """The input_name of a job on the site: jobs.safe_name's output with no two dots in a row and at most 200
    characters (the jobs check), so that no Storage object name holds "..": "clip..final.mp4" -> "clip.final.mp4"."""
    p = Path(jobs.safe_name(name))
    stem = re.sub(r"\.{2,}", ".", p.stem.strip("."))[:max(0, 200 - len(p.suffix))].rstrip(".")
    return (stem or "video") + p.suffix


@dataclass
class Ctx:
    """What the rewrite of one job needs: its uuid, local id, $G, the runs folder, the keys held for it and
    the row's created_at."""
    uuid: str
    local: str | None = None
    G: str | None = None
    runs: str | None = None
    keys: frozenset = field(default_factory=frozenset)
    created: str | None = None

    def __post_init__(self):
        paths = {}
        for p, token in ((self.runs and self.local and Path(self.runs) / self.local, "<job>"),
                         (self.runs, "<runs>"), (self.G, "$G")):
            if not p:
                continue
            for form in {str(p), str(Path(p).resolve())}:
                if form.startswith("/") and len(form) > 1:
                    paths.setdefault(form.rstrip("/"), token)
        self._paths = paths
        self._path_re = re.compile("|".join(re.escape(p) for p in sorted(paths, key=len, reverse=True))) \
            if paths else None
        keys = sorted({k for k in self.keys if isinstance(k, str) and k}, key=len, reverse=True)
        self._key_re = re.compile("|".join(map(re.escape, keys))) if keys else None


def scrub(text, ctx):
    """text without the job's keys, key-shaped strings and the machine's paths, and with the uuid for the
    local job id."""
    if not isinstance(text, str):
        return text
    if ctx._key_re:
        text = ctx._key_re.sub("[key]", text)
    text = KEY_SHAPE.sub("[key]", text)
    if ctx._path_re:
        text = ctx._path_re.sub(lambda m: ctx._paths[m.group(0)], text)
    if ctx.local:
        text = text.replace(ctx.local, ctx.uuid)
    return text


def rewrite(obj, ctx):
    """obj (JSON) with scrub() applied to every string, keys of objects too."""
    if isinstance(obj, str):
        return scrub(obj, ctx)
    if isinstance(obj, dict):
        return {scrub(k, ctx): rewrite(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite(v, ctx) for v in obj]
    return obj


def size_of(obj):
    """The bytes of obj's JSON text, as the tables' octet_length(body::text) checks count them."""
    return len(json.dumps(obj, ensure_ascii=False).encode())


def public_view(view, ctx, manifest=(), notes=()):
    """GET /api/jobs/{id} for the site: rewritten, without video.path and "public", "files" and "sizes" only
    for paths the site holds, "created" from the row, and notes (files left out) added to the warnings."""
    v = rewrite(view, ctx)
    v.pop("public", None)
    if isinstance(v.get("video"), dict):
        v["video"] = {k: x for k, x in v["video"].items() if k != "path"}
    if isinstance(v.get("files"), list):
        v["files"] = [f for f in v["files"] if f in manifest]
    if isinstance(v.get("sizes"), dict):
        v["sizes"] = {k: n for k, n in v["sizes"].items() if k in manifest}
    if ctx.created:
        v["created"] = ctx.created
    warnings = list(v.get("warnings") or [])
    v["warnings"] = warnings + [n for n in rewrite(list(notes), ctx) if n not in warnings]
    return v


def public_snapshot(path, status, body, ctx, thumbs=None):
    """(status, body) of GET frames, selection or parts for the site, rewritten. thumbs (lean mode): the
    thumbnails the site holds, which then replace the frames' "thumbs". A body over the table's limit
    becomes a 413 with a reason."""
    body = rewrite(body, ctx)
    if path == "frames" and thumbs is not None and isinstance(body, dict) and isinstance(body.get("thumbs"), list):
        body["thumbs"] = [i for i in body["thumbs"] if i in thumbs]
    if size_of(body) > SNAPSHOT_LIMIT:
        return 413, {"detail": f"this {path} answer is too large for the site"}
    return status, body


def public_answer(status, body, ctx):
    """(status, body) of an answer for the requests row, rewritten; one over the row's limit gets a reason."""
    body = rewrite(body, ctx)
    if size_of(body) > ANSWER_LIMIT:
        return status, {"detail": "the answer is too large for the site"}
    return status, body


def state_of(view):
    """The SSE "state" event that server.state gives for the job of this view (GET /api/jobs/{id} holds the
    same fields, and its "progress" is jobs.progress)."""
    return {"status": view["status"], "current": view.get("current"), "error": view.get("error"),
            "warnings": view.get("warnings") or [], "key_ok": view.get("key_ok"), "agent": view.get("agent"),
            "articulated": view.get("articulated"), "parts_mode": view.get("parts_mode"),
            "parts_review": view.get("parts_review"),
            "stages": [{k: x for k, x in s.items() if k != "seconds"} for s in view.get("progress") or []]}


def coarse(state):
    """What makes a publish go at once: the status and each stage's status (progress alone waits 2 s)."""
    if not state:
        return None
    return state.get("status"), tuple((s.get("name"), s.get("status")) for s in state.get("stages") or [])


def stopped_at(view):
    """When the job last stopped (its newest stage end, else None), for retention; None while it is active
    or before anything ran."""
    if view.get("status") in jobs.ACTIVE + ("uploaded",):
        return None
    ends = [s["ended"] for s in view.get("progress") or [] if isinstance(s.get("ended"), (int, float))]
    return datetime.fromtimestamp(max(ends), timezone.utc).isoformat() if ends else None


def row_fields(view, queue_pos=None):
    """The jobs row's status, prompt, queue_pos and (for a stopped job) stopped_at, from a view."""
    row = {"status": view["status"], "prompt": view.get("prompt"),
           "queue_pos": queue_pos if view["status"] == "queued" else None}
    t = stopped_at(view)
    if t:
        row["stopped_at"] = t
    return row


def first_not_done(state, parts_confirmed=False, scale_stale=False):
    """app.js retryFrom: "frames" while the input is not analyzed, else the first build stage not done or
    skipped (None when all are, or before any build). A job stopped at an unconfirmed part list goes back to
    propose, and a LiDAR scale found with another calibration (scale_stale) to lidar."""
    stage = {s["name"]: s for s in state["stages"]}
    if stage["frames"]["status"] != "done":
        return "frames"
    x = next((s for s in state["stages"] if s["name"] in jobs.BUILD and s["status"] not in ("done", "skipped")), None)
    if x is None or stage["extract"]["status"] == "pending":
        return None
    unconfirmed = state.get("articulated") is True and state.get("parts_mode") == "vlm_edit" \
        and stage.get("propose", {}).get("status") == "done" and not parts_confirmed
    name = "propose" if x["name"] == "parts" and unconfirmed else x["name"]
    return "lidar" if scale_stale and jobs.BUILD.index(name) > jobs.BUILD.index("lidar") else name


# ---- logs ----

def parse_sse(text):
    """(event, data, id) of one event string of server.sse."""
    fields = {}
    for line in text.splitlines():
        k, _, v = line.partition(": ")
        fields[k] = v
    return fields.get("event"), json.loads(fields.get("data") or "null"), fields.get("id")


def log_rows(events, cursor):
    """job_logs rows ({stage, run_started, pos, text}) of the "log" events of one server.tick call, without
    PROGRESS lines. cursor: the cursor that tick returned (a call reads one stage, so its start time is the
    run's)."""
    out = []
    for ev in events:
        name, data, eid = parse_sse(ev)
        if name != "log":
            continue
        text = PROGRESS.sub("", data["text"])
        stage, _, pos = (eid or "").partition(":")
        if text and pos.isdigit():
            out.append({"stage": stage, "run_started": cursor[2] if cursor else None, "pos": int(pos), "text": text})
    return out


def split_text(text, limit=LOG_ROW):
    """text in parts of at most limit characters, cut after a newline where one is near."""
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit) + 1 or limit
        parts.append(text[:cut])
        text = text[cut:]
    return parts + [text] if text else parts


# ---- files ----

@dataclass
class Mirror:
    """A file the site should hold: its ALLOW path, the file, and its role (review, result or log)."""
    rel: str
    path: Path
    role: str


def owner(rel):
    """The stage that writes rel, else None."""
    if rel.startswith("logs/"):
        return Path(rel).stem
    return next((stage for prefix, stage in OWNERS if rel.startswith(prefix)), None)


def role_of(rel):
    return "log" if rel.startswith("logs/") else "result" if rel in server.OUTPUTS else "review"


def lean_thumbs(meta):
    """The thumbnails a lean job keeps: about LEAN_THUMBS of frames/meta.json "thumbs" at an even stride, the
    first, and every candidate."""
    thumbs = [i for i in (meta or {}).get("thumbs") or [] if type(i) is int]
    if len(thumbs) <= LEAN_THUMBS:
        return set(thumbs)
    step = len(thumbs) / LEAN_THUMBS
    keep = {thumbs[int(k * step)] for k in range(LEAN_THUMBS)} | {thumbs[0]}
    return keep | ({i for i in meta.get("candidates") or [] if type(i) is int} & set(thumbs))


def mirror_plan(J, job, lean=False):
    """The files of job folder J that the site should hold now: every ALLOW path that server.resolve accepts,
    whose stage is not running; in lean mode only the lean thumbnails."""
    J = Path(J)
    running = {name for name, st in (job.get("stages") or {}).items() if st.get("status") == "running"}
    keep = lean_thumbs(server.read_json(J / "frames" / "meta.json")) if lean else None
    out, seen = [], set()
    for glob in GLOBS.values():
        # A name without a wildcard goes to resolve as it is: refined.glb is the agent's GLB under agent/s/.
        found = sorted(p.relative_to(J).as_posix() for p in J.glob(glob)) if "*" in glob else [glob]
        for rel in found:
            if rel in seen or not server.ALLOW.fullmatch(rel) or owner(rel) in running:
                continue
            if keep is not None and rel.startswith("frames/thumbs/") and int(Path(rel).stem) not in keep:
                continue
            real = server.resolve(J, rel)
            if real is not None:
                seen.add(rel)
                out.append(Mirror(rel, real, role_of(rel)))
    return out


def file_bytes(f, ctx, lean=False):
    """(data, content type) of a mirrored file as the site gets it: text scrubbed (a stage log also cut to its
    last LOG_FILE_CAP bytes), results.zip packed again with its text scrubbed; in lean mode review images as
    JPEG, and no preview over LEAN_PREVIEW (Skip)."""
    suffix = Path(f.rel).suffix
    ctype = SITE_TYPES[suffix]
    if lean and f.rel == "frames/preview.mp4" and f.path.stat().st_size > LEAN_PREVIEW:
        raise Skip(f"the preview video ({f.path.stat().st_size / 1e6:.0f} MB) is over {LEAN_PREVIEW >> 20} MB, so "
                   "the site leaves it out")
    data = f.path.read_bytes()
    if f.rel.startswith("logs/"):
        data = scrub(data.decode("utf-8", errors="replace"), ctx).encode()
        if len(data) > LOG_FILE_CAP:
            data = data[-LOG_FILE_CAP:]
            data = b"[the start of this log is left out]\n" + data[data.find(b"\n") + 1:]
        return data, ctype
    if suffix in TEXT:
        return scrub_text(data, ctx), ctype
    if f.rel == "results.zip":
        return scrub_zip(data, ctx, f.rel), ctype
    if lean and LEAN_IMAGES.fullmatch(f.rel):
        return lean_image(data, ctype)
    return data, ctype


def scrub_text(data, ctx):
    """UTF-8 text with scrub() applied; other bytes as they are (the key scan still looks at them)."""
    try:
        return scrub(data.decode("utf-8"), ctx).encode()
    except UnicodeDecodeError:
        return data


def scrub_zip(data, ctx, rel="results.zip"):
    """A zip packed again with its text files scrubbed (names, dates and compression stay). A binary member
    that holds a key of the job, or anything shaped like a key, leaves the whole zip out (Skip). Bytes that are
    not a zip stay as they are."""
    try:
        src = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return data
    out = io.BytesIO()
    with src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            member = src.read(info)
            if Path(info.filename).suffix.lower() in TEXT:
                member = scrub_text(member, ctx)
            elif holds_key(member, ctx.keys, shape=True):
                raise Skip(f"{rel} left out: it holds the API key")
            dst.writestr(info, member)
    return out.getvalue()


def lean_image(data, ctype):
    """A review image as JPEG quality 85 with its long side at most LEAN_SIDE px. A JPEG that is small enough
    stays as it is."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        if ctype == "image/jpeg" and max(im.size) <= LEAN_SIDE:
            return data, ctype
        im = im.convert("RGB")
        im.thumbnail((LEAN_SIDE, LEAN_SIDE))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=85)
        return out.getvalue(), "image/jpeg"


def holds_key(data, keys, chunk=SCAN_CHUNK, shape=False):
    """True when data (bytes, or a path) holds any of keys, or with shape anything shaped like a key (KEY_BYTES),
    which also finds a key that the worker no longer holds. It reads chunk bytes at a time and keeps the end of each
    for the next, so a key across two chunks is found; no size limit."""
    keys = [k.encode() for k in keys if k]
    if not keys and not shape:
        return False
    keep, tail = max([len(k) - 1 for k in keys] + [32 if shape else 0]), b""
    f = open(data, "rb") if isinstance(data, (str, Path)) else io.BytesIO(data)
    with f:
        while block := f.read(chunk):
            buf = tail + block
            if any(k in buf for k in keys) or shape and KEY_BYTES.search(buf):
                return True
            tail = buf[-keep:] if keep else b""
    return False
