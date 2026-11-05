"""The demos: finished jobs that the site's owner picked, shown at demos.html as a gallery of cards, each a link to
the demo's 360 video, 3D view and files, in a standalone gallery (docs/public_site.md).

    python -m harpor.webapp.demos add SOURCE [--name N] [--title T] [--video MP4] [--glb GLB] [--zip ZIP]
                                             [--fact LABEL=VALUE ...] [--order K] [--out DIR] [--force] [--public-bag]
    python -m harpor.webapp.demos check [DIR]

The demos folder is WEBAPP_DEMOS, or --demos DIR of the server (server.py serves /demos.html and
/demos/<name>/<file>) and of the site build (site.py copies the demos into the site). There is no default folder.
A demo is a folder in it, named with a-z, 0-9 and - (NAME); names that start with "." are skipped (.DS_Store, and
add's .<name>.tmp):

  office-chair/demo.json      {"version": 1, "title": "Office chair", "order": 1, "video": "turntable.mp4",
                               "poster": "poster.jpg", "model": "aligned.glb", "facts": [[label, value], ...],
                               "downloads": ["results.zip", "aligned.glb", "turntable.mp4"]}
  office-chair/turntable.mp4  the 360 video (H.264, yuv420p, so every browser plays it)
  office-chair/poster.jpg     its first frame (optional)
  office-chair/aligned.glb    the mesh turned upright, for the 3D view
  office-chair/results.zip    the job's results.zip, cleaned
  placeholders.json           {"version": 1, "placeholders": ["Drawer cabinet", ...]} (optional): 1 to 12 titles of 1
                              to 60 characters, each once, on one line and with no space at either end; each is a
                              card that says "Coming soon" and opens nothing, after the demos' cards, unless a demo
                              has that title

load checks every rule and says which one a demo breaks (DemoError): the folder holds demo.json and the files that
it names, and nothing else; each is a regular file of a type in TYPES, at most MAX_FILE bytes (Cloudflare Pages'
limit per file), that starts with its type's magic bytes. The files are served from the site's own origin, so no type
that a browser runs as a page (HTML, SVG, XML) is allowed. load also reads every file: nothing shaped like an API key,
no path of a machine (MACHINE_PATH), a GLB with all its data inside and without Draco or KTX2 (the demos page loads no
decoder, and the CSP blocks gstatic.com), and nothing that looks like private data (PRIVATE) in demo.json, the file
names or the text inside results.zip. placeholders reads placeholders.json with the same guard. Both JSON files are
checked as written and as read, so a JSON escape cannot hide a word.

add makes a demo from a finished job folder, on the machine that holds the job: the 360 video
(turntable/turntable.mp4, else --video), physics/aligned.glb (else --glb), results.zip packed again with every key
shape and machine path taken out of its text (a binary member that holds one stops add), and the facts of the
Results tab (facts_of). It refuses a job that is not done, a rosbag job without --public-bag (the only rosbags so far
are private, and a bag's name can look harmless), and anything that looks private. From a run folder (no job.json),
--video and --glb are needed, and the facts come from the job-layout records that are there. It writes
<out>/.<name>.tmp, checks it with load, then renames it to <out>/<name>; an existing demo stays unless --force.
check runs load and placeholders on a demos folder.
"""

import argparse
import io
import html
import json
import math
import os
import re
import shutil
import struct
import sys
import zipfile
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from harpor.fileio import read_json
from harpor.webapp import steps

NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
TYPES = {".mp4": "video/mp4", ".glb": "model/gltf-binary", ".zip": "application/zip", ".jpg": "image/jpeg"}
MAGIC = {".mp4": (4, b"ftyp"), ".glb": (0, b"glTF"), ".zip": (0, b"PK\x03\x04"), ".jpg": (0, b"\xff\xd8\xff")}
MAX_FILE = 25 << 20  # bytes of one demo file
MAX_JSON = 64 << 10  # bytes of demo.json
KEYS = {"version", "title", "order", "video", "poster", "model", "facts", "downloads"}
MAX_TITLE, MAX_FACTS, MAX_LABEL, MAX_VALUE = 60, 8, 40, 160
# The private data that must never be a demo: the VisionNav scenes (Cage, Pallet), table.MOV, and the folders and host
# that hold private inputs. Checked against demo.json's text, every file name and the text inside results.zip.
PRIVATE = re.compile(r"(?<![a-z])(?:cage|pallet)|table\.mov|gator_private|private_bags|@login-ice", re.I)
# A path on a machine: PACE's storage (/storage/ice1, /storage/home, /storage/coda1, ...), a home folder, scratch, or a
# temporary folder. add puts the file name in its place (or <path> for a folder).
MACHINE_PATH = re.compile(r"(?<![\w.-])/(?:storage/[\w.-]+|home|Users|scratch|private|tmp|var)/[^\s\"'<>|;,]+")
MACHINE_BYTES = re.compile(MACHINE_PATH.pattern.encode())
NO_DECODER = ("KHR_draco_mesh_compression", "KHR_texture_basisu")
KEY_SHAPE = re.compile(r"sk-[A-Za-z0-9_-]{16,}")
KEY_BYTES = re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}")
SCAN_CHUNK = 1 << 20

def holds_key(data, keys, chunk=SCAN_CHUNK, shape=False):
    """True when data (bytes, or a path) holds any of keys, or with shape anything shaped like a key (KEY_BYTES),
    including keys from older jobs. It reads chunk bytes at a time and keeps the end of each
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

CLEAN_TEXT = (".json", ".log", ".urdf", ".xml", ".usda", ".txt", ".csv", ".md", ".yaml", ".yml", ".mtl", ".obj")  # the members of results.zip that add cleans as text
VIDEO, MODEL = "turntable/turntable.mp4", "physics/aligned.glb"  # in a job folder (server.OUTPUTS)
MARKER = "<!-- demos -->"  # where page() puts the demos in static/demos.html
PLACEHOLDERS, MAX_PLACEHOLDERS = "placeholders.json", 12  # in the demos folder, not a demo
# a placeholder's title is plain text on one line: no control character, line or paragraph separator, or bidi control
CONTROL = re.compile("[\x00-\x1f\x7f-\x9f\u2028\u2029\u202a-\u202e\u2066-\u2069]")
CARD_FACTS, SHORT = ("Size", "Mass"), 24  # the facts that a card shows first; else others of at most SHORT characters
# app.js KIND_LABEL (job.json source kind) and RAN (agent_result.agent); a test compares them.
KIND = {"video": "video", "rosbag": "rosbag", "stray": "Stray capture"}
RAN = {"codex": "GPT-6-Astra", "claude": "Claude Opus 5.5"}


class DemoError(Exception):
    """Why a demo, or the demos folder, cannot be used, in one line."""


@dataclass(frozen=True)
class Demo:
    name: str
    title: str
    order: int
    video: str
    poster: str | None
    model: str
    facts: tuple  # ((label, value), ...)
    downloads: tuple
    sizes: dict  # {file: bytes} of every listed file

    @property
    def files(self):
        """The demo's files: the video, the model, the poster and the downloads, each once."""
        return tuple(dict.fromkeys(f for f in (self.video, self.model, self.poster, *self.downloads) if f))


# ---- numbers in app.js's form ----

def fixed(x, d):
    """Number(x).toFixed(d): the exact value of the float, halves away from zero."""
    return str(Decimal(float(x) or 0.0).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def num(x, d=3):
    """app.js num: Number(x).toPrecision(d), "?" for None."""
    if x is None:
        return "?"
    v = Decimal(float(x))
    if v == 0:
        return "0" + ("." + "0" * (d - 1) if d > 1 else "")
    sign, v = "-" if v < 0 else "", abs(v)
    e = v.adjusted()
    n = int(v.scaleb(d - 1 - e).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    if n >= 10 ** d:  # 999.7 to 3 digits is 1.00e+3
        e += 1
        n = int(v.scaleb(d - 1 - e).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    digits = str(n)
    if e < -6 or e >= d:
        return f"{sign}{digits[0]}{'.' + digits[1:] if d > 1 else ''}e{'+' if e >= 0 else '-'}{abs(e)}"
    if e >= 0:
        return sign + digits[:e + 1] + ("." + digits[e + 1:] if digits[e + 1:] else "")
    return sign + "0." + "0" * (-e - 1) + digits


def size_text(b):
    """app.js fileSize: "512 B", "3.4 KB", "35 KB", "1.2 MB", "100 MB", "2.50 GB" (1 KB is 1000 bytes)."""
    if b < 1e3:
        return f"{b} B"
    if b < 999.5e3:
        return fixed(b / 1e3, 1 if b < 9.95e3 else 0) + " KB"
    if b >= 999.5e6:
        return fixed(b / 1e9, 2) + " GB"
    return fixed(b / 1e6, 0 if b >= 99.95e6 else 1) + " MB"


# ---- the checks ----

def private(text, what):
    """DemoError when text looks private (PRIVATE)."""
    m = PRIVATE.search(str(text))
    if m:
        raise DemoError(f"{what}: looks private (matches {m[0].lower()!r}); private data is never a demo")


def _guard(text, what):
    """DemoError when text looks private, or holds a key shape or a machine path."""
    private(text, what)
    if KEY_SHAPE.search(text) or MACHINE_PATH.search(text):
        raise DemoError(f"{what} holds a key shape or a machine path")


def _glb(data, what):
    """DemoError unless data is a glTF 2 binary with all its data inside and no extension that needs a decoder, whose
    JSON (the node, mesh, material and image names) does not look private."""
    try:
        version, _ = struct.unpack_from("<II", data, 4)
        n, kind = struct.unpack_from("<I4s", data, 12)
        text = data[20:20 + n].decode()
        gltf = json.loads(text)
    except (struct.error, ValueError):
        raise DemoError(f"{what} is not a GLB that can be read") from None
    if version != 2 or kind != b"JSON" or not isinstance(gltf, dict):
        raise DemoError(f"{what} is not a glTF 2 binary")
    private(text, what)
    used = set(gltf.get("extensionsUsed") or []) | set(gltf.get("extensionsRequired") or [])
    if used & set(NO_DECODER):
        raise DemoError(f"{what} uses {', '.join(sorted(used & set(NO_DECODER)))}; the demos page loads no decoder, "
                        "so save it without")
    if any(isinstance(x, dict) and "uri" in x for k in ("buffers", "images") for x in gltf.get(k) or []):
        raise DemoError(f"{what} points to files outside it (uri); a demo's GLB holds all its data")


def _zip(path, what):
    """DemoError when a member of the zip looks private, or holds a key shape or a machine path."""
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise DemoError(f"{what} is not a zip that can be read") from None
    with z:
        for info in z.infolist():
            m = f"{what}'s {info.filename}"
            private(info.filename, m)
            data = z.read(info)
            if Path(info.filename).suffix.lower() in CLEAN_TEXT:
                text = data.decode("utf-8", errors="replace")
                private(text, m)
                if KEY_SHAPE.search(text) or MACHINE_PATH.search(text):
                    raise DemoError(f"{m} holds a key shape or a machine path; make the demo again with add")
            elif holds_key(data, [], shape=True) or MACHINE_BYTES.search(data):
                raise DemoError(f"{m} holds a key shape or a machine path")


def _text(value, low, high, what):
    if not isinstance(value, str) or not low <= len(value) <= high:
        raise DemoError(f"{what} is text of {low} to {high} characters")
    return value


def demo(d, name=None):
    """The Demo in folder d (named name, else d's own name), with every rule checked; DemoError otherwise."""
    name = name or d.name
    if d.is_symlink() or not d.is_dir():
        raise DemoError(f"{name} is not a demo folder (a demo is a folder, not a link)")
    if not NAME.fullmatch(name):
        raise DemoError(f"{name}: a demo's name is a-z, 0-9 and -, at most 40 characters, and starts with a letter "
                        "or digit")
    private(name, name)
    spec = d / "demo.json"
    if spec.is_symlink() or not spec.is_file() or spec.stat().st_size > MAX_JSON:
        raise DemoError(f"{name}: no demo.json, or one over {MAX_JSON >> 10} KB")
    text = spec.read_text(errors="replace")
    _guard(text, f"{name}/demo.json")
    try:
        j = json.loads(text)
    except ValueError as e:
        raise DemoError(f"{name}/demo.json is not JSON: {e}") from None
    _guard(json.dumps(j, ensure_ascii=False), f"{name}/demo.json")  # as read: a JSON escape hides nothing
    if not isinstance(j, dict):
        raise DemoError(f"{name}/demo.json is not a JSON object")
    if set(j) - KEYS:
        raise DemoError(f"{name}/demo.json has unknown keys: {', '.join(sorted(set(j) - KEYS))}")
    if j.get("version") != 1:
        raise DemoError(f"{name}/demo.json: version is 1")
    title = _text(j.get("title"), 1, MAX_TITLE, f"{name}: title")
    order = j.get("order", 0)
    if type(order) is not int:
        raise DemoError(f"{name}: order is a whole number")
    facts = j.get("facts", [])
    if not isinstance(facts, list) or len(facts) > MAX_FACTS or not all(isinstance(f, list) and len(f) == 2 for f in facts):
        raise DemoError(f"{name}: facts is a list of at most {MAX_FACTS} [label, value] pairs; remove some with "
                        "--fact LABEL=")
    for label, value in facts:
        _text(label, 1, MAX_LABEL, f"{name}: a fact's label")
        _text(value, 1, MAX_VALUE, f"{name}: the value of {label}")
    downloads = j.get("downloads", [])
    if not isinstance(downloads, list) or len(set(map(str, downloads))) != len(downloads):
        raise DemoError(f"{name}: downloads is a list of file names, each once")
    for key, suffix, need in (("video", ".mp4", True), ("model", ".glb", True), ("poster", ".jpg", False)):
        if j.get(key) is None and not need:
            continue
        if not isinstance(j.get(key), str) or not j[key].endswith(suffix):
            raise DemoError(f"{name}: {key} is the name of a {suffix} file" + ("" if need else ", or leave it out"))
    listed = dict.fromkeys(f for f in (j["video"], j["model"], j.get("poster"), *downloads) if f)
    sizes = {}
    for f in listed:
        if not isinstance(f, str) or not FILE.fullmatch(f) or Path(f).suffix not in TYPES:
            raise DemoError(f"{name}/{f}: a demo file is named with letters, digits, _, . and -, and is one of "
                            f"{', '.join(TYPES)}")
        p, what = d / f, f"{name}/{f}"
        private(f, what)
        if p.is_symlink() or not p.is_file():
            raise DemoError(f"{what} is missing, or not a regular file")
        sizes[f] = p.stat().st_size
        if sizes[f] > MAX_FILE:
            raise DemoError(f"{what} is {size_text(sizes[f])}; a demo file is at most {MAX_FILE >> 20} MiB")
        at, magic = MAGIC[Path(f).suffix]
        with open(p, "rb") as fh:
            if fh.read(at + len(magic))[at:] != magic:
                raise DemoError(f"{what} is not a {Path(f).suffix} file inside")
        data = p.read_bytes()
        if f.endswith(".glb"):  # the GLB's and the zip's own checks first: they say more
            _glb(data, what)
        elif f.endswith(".zip"):
            _zip(p, what)
        if holds_key(data, [], shape=True) or MACHINE_BYTES.search(data):
            raise DemoError(f"{what} holds a key shape or a machine path")
    for p in sorted(d.iterdir()):
        if not p.name.startswith(".") and p.name != "demo.json" and p.name not in listed:
            raise DemoError(f"{name}/{p.name} is not listed in demo.json")
    return Demo(name, title, order, j["video"], j.get("poster"), j["model"], tuple(map(tuple, facts)),
                tuple(downloads), sizes)


def load(folder):
    """The demos in folder (each entry but placeholders.json and the names that start with "."), sorted by
    (order, name): [] for None. DemoError names the first demo that breaks a rule, or says that folder is not a folder
    (a mistyped folder would show no demos, and a site build would take the live ones away)."""
    if folder is None:
        return []
    if not Path(folder).is_dir():
        raise DemoError(f"no demos folder {folder}")
    shown = [demo(d) for d in sorted(Path(folder).iterdir())
             if not d.name.startswith(".") and d.name != PLACEHOLDERS]
    return sorted(shown, key=lambda x: (x.order, x.name))


def placeholders(folder, shown=()):
    """The titles of folder's placeholders.json, in its order, without the titles of the demos in shown (a demo that
    has come in takes its placeholder's place): [] for None or no file. DemoError says what is wrong with the file,
    or that folder is not a folder."""
    if folder is None:
        return []
    if not Path(folder).is_dir():
        raise DemoError(f"no demos folder {folder}")
    p = Path(folder) / PLACEHOLDERS
    if not p.exists() and not p.is_symlink():
        return []
    if p.is_symlink() or not p.is_file() or p.stat().st_size > MAX_JSON:
        raise DemoError(f"{PLACEHOLDERS} is not a regular file of at most {MAX_JSON >> 10} KB")
    text = p.read_text(errors="replace")
    _guard(text, PLACEHOLDERS)
    try:
        j = json.loads(text)
    except ValueError as e:
        raise DemoError(f"{PLACEHOLDERS} is not JSON: {e}") from None
    _guard(json.dumps(j, ensure_ascii=False), PLACEHOLDERS)
    if not isinstance(j, dict):
        raise DemoError(f"{PLACEHOLDERS} is not a JSON object")
    if set(j) - {"version", "placeholders"}:
        raise DemoError(f"{PLACEHOLDERS} has unknown keys: {', '.join(sorted(set(j) - {'version', 'placeholders'}))}")
    if type(j.get("version")) is not int or j["version"] != 1:  # not true or 1.0, which equal 1
        raise DemoError(f"{PLACEHOLDERS}: version is 1")
    titles = j.get("placeholders")
    if not isinstance(titles, list) or not 1 <= len(titles) <= MAX_PLACEHOLDERS:
        raise DemoError(f"{PLACEHOLDERS}: placeholders is a list of 1 to {MAX_PLACEHOLDERS} titles; delete the file "
                        "for none")
    for i, t in enumerate(titles, 1):
        if CONTROL.search(_text(t, 1, MAX_TITLE, f"{PLACEHOLDERS}: placeholder {i}")) or t != t.strip():
            raise DemoError(f"{PLACEHOLDERS}: placeholder {i} is plain text on one line, with no space at either end")
    if len({t.casefold() for t in titles}) != len(titles):
        raise DemoError(f"{PLACEHOLDERS}: each title once")
    have = {x.title.strip().casefold() for x in shown}
    return [t for t in titles if t.casefold() not in have]


def files(folder, shown):
    """{"demos/<name>/<file>": the file} of the demos' files."""
    return {f"demos/{x.name}/{f}": Path(folder) / x.name / f for x in shown for f in x.files}


# ---- the page ----

def card_facts(x):
    """The two facts of x that its card shows: Size and Mass when it has them, else its other facts of at most SHORT
    characters, in order."""
    have = dict(x.facts)
    first = [(k, have[k]) for k in CARD_FACTS if k in have]
    return (first + [(k, v) for k, v in x.facts if k not in CARD_FACTS and len(v) <= SHORT])[:2]


def card(x):
    """A demo's card in the gallery: one link to its section (#demo-<name>) with its poster, its title and two short
    facts (card_facts)."""
    e = html.escape
    img = (f'<img class="dcimg" src="{e(f"demos/{x.name}/{x.poster}")}" alt="" loading="lazy">' if x.poster
           else '<span class="dcimg"></span>')
    facts = card_facts(x)
    dl = "<dl>" + "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in facts) + "</dl>" if facts else ""
    return f'<li><a class="democard" href="#demo-{x.name}">{img}<span class="dctitle">{e(x.title)}</span>{dl}</a></li>'


def soon_card(title):
    """A placeholder's card: the same frame with an empty picture, its title and "Coming soon", and no link."""
    return (f'<li><div class="democard soon"><span class="dcimg"></span><span class="dctitle">{html.escape(title)}'
            '</span><span class="dcnote">Coming soon</span></div></li>')


def detail(x):
    """One demo's section, which style.css shows while the URL targets it: a link back to the gallery, its title, its
    360 video with its facts (no autoplay and nothing preloaded, so a video loads only when demos.js plays it or its
    controls do), the 3D view (folded, so model-viewer fetches the GLB only when it opens) and its files, as the
    Downloads tab lists them (named <demo>_<file>)."""
    e = html.escape
    at = lambda f: e(f"demos/{x.name}/{f}")  # noqa: E731  relative, so it works at / and under /<repo>/
    poster = f' poster="{at(x.poster)}"' if x.poster else ""
    out = [f'<section class="demo" id="demo-{x.name}">', '<p class="back"><a href="#">All demos</a></p>',
           f'<h2 tabindex="-1">{e(x.title)}</h2>', '<div class="demobody">',
           f'  <video src="{at(x.video)}"{poster} muted loop playsinline controls preload="none" '
           f'aria-label="360 video of {e(x.title)}"></video>']
    if x.facts:
        out.append("  <dl>" + "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in x.facts) + "</dl>")
    out += ["</div>", '<details class="sub"><summary>3D view</summary>',
            f'  <model-viewer src="{at(x.model)}" loading="lazy" camera-controls environment-image="neutral" '
            f'shadow-intensity="1" interaction-prompt="none" alt="3D model of {e(x.title)}"></model-viewer>', "</details>"]
    if x.downloads:
        out.append(f'<ul class="dlist" aria-label="Files of {e(x.title)}">')
        for f in x.downloads:
            kind = "dl primary" if f == "results.zip" else "dl"
            out.append(f'  <li class="dlrow"><a class="{kind}" href="{at(f)}" download="{e(f"{x.name}_{f}")}">{e(f)}</a>'
                       f'<span class="dlsize">{size_text(x.sizes[f])}</span></li>')
        out.append("</ul>")
    return "\n".join(out + ["</section>"])


def page(src, shown, soon=()):
    """static/demos.html (src) with the gallery in place of its one MARKER: the demos' cards, then a card per
    placeholder title (soon), then each demo's section; "No demos yet." for neither."""
    if src.count(MARKER) != 1:
        raise DemoError(f"static/demos.html has {src.count(MARKER)} copies of {MARKER}, not 1")
    if not shown and not soon:
        return src.replace(MARKER, '<p class="emptytab">No demos yet.</p>')
    cards = ['<ul class="gallery">', *map(card, shown), *map(soon_card, soon), "</ul>"]
    return src.replace(MARKER, "\n".join(cards + list(map(detail, shown))))


# ---- add and check ----

def facts_of(J, job=None):
    """[[label, value]] of the Results tab's rows (app.js renderResults) that a demo shows, from the records in folder
    J: Input (the source's kind and the views used; job.json, when there is one), Size (the upright extent), Mass,
    Links and each joint (an articulated job), and the Agent. Never the input's name, a topic, a path or a LiDAR
    value."""
    J, out = Path(J), []
    sel = read_json(J / "selection.json") or {}
    n = len(sel.get("picks") or [])
    if job:
        kind = steps.source(job)["kind"]
        out.append(["Input", KIND.get(kind, kind) + (f", {n} view{'s' if n != 1 else ''}" if n else "")])
    ext = (read_json(J / "physics" / "aligned.json") or {}).get("extent_s")
    if isinstance(ext, list) and len(ext) == 3:
        out.append(["Size", " x ".join(fixed(100 * v, 0) for v in ext) + " cm"])
    art = steps.articulated(job) if job else (J / "run" / "parts.glb").is_file()
    mj, phys = read_json(J / "physics" / "mjcf" / "mjcf.json") if art else None, read_json(J / "physics" / "summary.json")
    links = mj.get("links") if isinstance(mj, dict) else None
    if phys:
        rb = (mj.get("inertial") if links else None) or phys.get("rigid_body") or {}
        out.append(["Mass", num(rb.get("mass")) + " kg"])
    deg, cm = (lambda x: fixed(x * 180 / math.pi, 0)), (lambda x: fixed(100 * x, 1))
    if links:
        out.append(["Links", f"{len(links)}: {', '.join(links)}"])
        for j in mj.get("joints") or []:
            r = j.get("range") or [0, 0]
            out.append([f"Joint of {j['child']}", f"turns on {j['parent']}, {deg(r[0])}° to {deg(r[1])}°"
                        if j.get("type") == "revolute" else f"slides on {j['parent']}, {cm(r[0])} to {cm(r[1])} cm"
                        if j.get("type") == "prismatic" else f"soft on {j['parent']}"
                        if j.get("type") == "soft" else f"fixed to {j['parent']}"])
        out += [[f"Joint of {k}", f"none (welded to {to})"] for k, to in (mj.get("welded") or {}).items()]
    elif mj and art:
        out.append(["Links", "1 (no joints, one rigid body)"])
    r = read_json(J / "agent" / "result.json")
    if job and steps.agent(job) == "none":
        out.append(["Agent", "off"])
    elif isinstance(r, dict) and r.get("agent"):
        out.append(["Agent", RAN.get(r["agent"], r["agent"])])
    return out


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:40].strip("-")


def clean_zip(src, dst):
    """results.zip packed again: in each text member every key shape becomes [key] and every machine path its file name
    (<path> for a folder). DemoError when a binary member holds a key shape or a machine path."""
    def name(m):
        tail = m[0].rstrip("/").rsplit("/", 1)[-1]
        return tail if "." in tail.strip(".") else "<path>"

    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info)
            if Path(info.filename).suffix.lower() in CLEAN_TEXT:
                data = MACHINE_PATH.sub(name, KEY_SHAPE.sub("[key]", data.decode("utf-8", errors="replace")))
            elif holds_key(data, [], shape=True) or MACHINE_BYTES.search(data):
                raise DemoError(f"results.zip's {info.filename} holds a key shape or a machine path; give --zip without "
                                "it")
            zout.writestr(info, data)


def check_video(path):
    """DemoError unless path is H.264 in yuv420p, which every browser plays."""
    from harpor.webapp import video

    try:
        c, s = video.open_video(path)
    except SystemExit as e:
        raise DemoError(str(e)) from None
    with c:
        codec, fmt = s.codec_context.name, s.codec_context.pix_fmt
    if (codec, fmt) != ("h264", "yuv420p"):
        raise DemoError(f"{Path(path).name} is {codec} in {fmt}; a demo's video is h264 in yuv420p")


def add(source, out, name=None, title=None, video=None, glb=None, zip_=None, facts=(), order=0, force=False,
        public_bag=False):
    """Make the demo <out>/<name> from a finished job folder or a run folder (the module docstring); returns the
    Demo. facts: (label, value) pairs that add or replace a fact, or remove it when value is empty."""
    from harpor.webapp import server

    src = Path(source).expanduser().resolve()
    if not src.is_dir():
        raise DemoError(f"{source} is not a folder")
    private(src, "the source")
    for rel in ("selection.json", "lidar.json", "pool/lidar.json"):  # they can name the input's frames and topics
        if (src / rel).is_file():
            private((src / rel).read_text(errors="replace"), rel)
    pick = lambda p:(Path(p).expanduser() if Path(p).expanduser().is_absolute() else src / p).resolve()  # noqa: E731
    job = read_json(src / "job.json") if (src / "job.json").is_file() else None
    if job is not None:
        if not isinstance(job, dict) or job.get("status") != "done":
            raise DemoError(f"the job is {job.get('status') if isinstance(job, dict) else 'unreadable'}, not done")
        s = steps.source(job)
        for what, text in (("the input's name", (job.get("video") or {}).get("name")), ("the prompt", job.get("prompt")),
                           ("the source", s.get("kind")), ("the camera topic", s.get("topic"))):
            if text:
                private(text, what)
        if s["kind"] == "rosbag" and not public_bag:
            raise DemoError("a rosbag job: the rosbags so far hold private scenes; add --public-bag when this one "
                            "does not")
        mp4 = pick(video) if video else server.resolve(src, VIDEO)
        if mp4 is None:
            raise DemoError("the job has no 360 video; start it with Make a 360 video on, or give --video PATH")
        model = pick(glb) if glb else server.resolve(src, MODEL)
        if model is None:
            raise DemoError(f"the job has no {MODEL}; give --glb PATH")
        zipped = pick(zip_) if zip_ else server.resolve(src, "results.zip")
        name, title = name or slug(job.get("prompt") or ""), title or str(job.get("prompt") or "")[:MAX_TITLE]
    else:
        if not video or not glb:
            raise DemoError(f"{src} has no job.json: give --video and --glb")
        mp4, model, zipped = pick(video), pick(glb), pick(zip_) if zip_ else None
        name, title = name or slug(src.name), title or src.name.replace("_", " ")[:MAX_TITLE]
    title = title[:1].upper() + title[1:]
    if not name or not NAME.fullmatch(name):
        raise DemoError(f"the name {name!r} is a-z, 0-9 and -, at most 40 characters: give --name")
    for p in (mp4, model, zipped):
        if p is not None:
            private(p, "a file to copy")
            if not p.is_file():
                raise DemoError(f"{p} is not a file")
    have = dict(facts_of(src, job))
    for label, value in facts:
        if value:
            have[label] = value
        else:
            have.pop(label, None)
    out = Path(out).expanduser()
    final, tmp = out / name, out / f".{name}.tmp"
    if final.exists() and not force:
        raise DemoError(f"{final} exists; add --force to replace it")
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        check_video(mp4)
        model_name = model.name if FILE.fullmatch(model.name) and model.suffix == ".glb" else "model.glb"
        for p, f in ((mp4, "turntable.mp4"), (model, model_name)):
            if p.stat().st_size > MAX_FILE:
                raise DemoError(f"{p.name} is {size_text(p.stat().st_size)}; a demo file is at most {MAX_FILE >> 20} MiB")
            shutil.copyfile(p, tmp / f)
        poster(tmp / "turntable.mp4", tmp / "poster.jpg")
        downloads = [model_name, "turntable.mp4"]
        if zipped is not None:
            clean_zip(zipped, tmp / "results.zip")
            if (tmp / "results.zip").stat().st_size > MAX_FILE:
                print(f"results.zip is {size_text((tmp / 'results.zip').stat().st_size)}, over {MAX_FILE >> 20} MiB: "
                      "left out of the downloads")
                (tmp / "results.zip").unlink()
            else:
                downloads.insert(0, "results.zip")
        (tmp / "demo.json").write_text(json.dumps(
            {"version": 1, "title": title, "order": order, "video": "turntable.mp4", "poster": "poster.jpg",
             "model": model_name, "facts": [[k, v] for k, v in have.items()], "downloads": downloads}, indent=1,
            ensure_ascii=False) + "\n")
        made = demo(tmp, name=name)
        if final.exists():
            old = out / f".{name}.old"
            shutil.rmtree(old, ignore_errors=True)
            final.rename(old)
            tmp.rename(final)
            shutil.rmtree(old)
        else:
            tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return made


def poster(mp4, jpg):
    """The video's first frame as a JPEG (quality 85)."""
    from PIL import Image

    from harpor.webapp import video

    for _, _, rgb in video.frames(mp4, keep={0}):
        Image.fromarray(rgb).save(jpg, "JPEG", quality=85)
        return
    raise DemoError(f"{Path(mp4).name} has no frame")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m harpor.webapp.demos", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="make a demo from a finished job folder, or from a run folder with --video and --glb")
    a.add_argument("source", metavar="SOURCE", help="a finished job folder, or a run folder")
    a.add_argument("--name", help="the demo's folder name (default: the prompt, such as office-chair)")
    a.add_argument("--title", help="the demo's title (default: the prompt)")
    a.add_argument("--video", metavar="MP4", help=f"the 360 video (default: the job's {VIDEO})")
    a.add_argument("--glb", metavar="GLB", help=f"the GLB of the 3D view (default: the job's {MODEL})")
    a.add_argument("--zip", metavar="ZIP", help="the results.zip to offer (default: the job's)")
    a.add_argument("--fact", action="append", default=[], metavar="LABEL=VALUE",
                   help="add or replace a fact; LABEL= removes one")
    a.add_argument("--order", type=int, default=0, help="the demos show by order, then name (default 0)")
    a.add_argument("--out", help="the demos folder (default: WEBAPP_DEMOS)")
    a.add_argument("--force", action="store_true", help="replace a demo of the same name")
    a.add_argument("--public-bag", action="store_true", help="allow a rosbag job whose scene is not private")
    c = sub.add_parser("check", help="check a demos folder")
    c.add_argument("dir", nargs="?", metavar="DIR", help="the demos folder (default: WEBAPP_DEMOS)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "add":
            out = args.out or os.environ.get("WEBAPP_DEMOS")
            if not out:
                raise DemoError("give --out DIR or set WEBAPP_DEMOS")
            facts = []
            for f in args.fact:
                label, eq, value = f.partition("=")
                if not eq or not label:
                    raise DemoError(f"--fact {f!r} is LABEL=VALUE")
                facts.append((label, value))
            made = add(args.source, out, args.name, args.title, args.video, args.glb, args.zip, facts, args.order,
                       args.force, args.public_bag)
            listing = ", ".join(f"{f} {size_text(n)}" for f, n in made.sizes.items())
            print(f"wrote {Path(out).expanduser() / made.name}: {listing}")
        else:
            folder = args.dir or os.environ.get("WEBAPP_DEMOS")
            if not folder:
                raise DemoError("give DIR or set WEBAPP_DEMOS")
            shown = load(Path(folder).expanduser())
            soon = placeholders(Path(folder).expanduser(), shown)
            total = sum(n for x in shown for n in x.sizes.values())
            more = f", {len(soon)} placeholder{'s' if len(soon) != 1 else ''}" if soon else ""
            print(f"{len(shown)} demo{'s' if len(shown) != 1 else ''}{more}, {size_text(total)}, ok")
    except DemoError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
