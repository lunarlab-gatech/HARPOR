"""gator/webapp/static/help.html: the help page that the app's Help link opens. One section per part of the app, the
method notes with the numbers of the code they explain, no script, and the local-only blocks that site.py swaps for
the public site's text (HELP_PUBLIC)."""
import html
import inspect
import re
from html.parser import HTMLParser
from pathlib import Path

from gator.agent import controller, tools
from gator.parts import geometry, vote
from gator.stage_a import lidar_scale, part_masks
from gator.webapp import agent_step, jobs, site, steps, video, views

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "gator" / "webapp" / "static"
HELP = STATIC / "help.html"
# The sections in order. The ids are stable: the app's Help links point to them, and the public page keeps them.
SECTIONS = ("about", "capture", "add-job", "status", "topics", "settings", "review", "parts", "agent", "lidar",
            "inputs", "results", "joints", "physics", "downloads", "retry", "troubleshooting")
# The local-only blocks, <!-- local:NAME --> ... <!-- /local:NAME -->; toc and account are empty here, and the public
# page fills them (the account, machine and machine-status sections).
LOCAL = ("toc", "account", "add-job", "keys", "agent-none", "troubleshooting")
# Any marker, also one with other spacing, so that a marker which site.py cannot read fails here (blocks()).
MARK = re.compile(r"<!--\s*(/?)\s*local\b:?\s*([\w-]*)\s*-->")
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class Page(HTMLParser):
    """The tags of a page with their attributes; each table row's cells as text, with the section it is in; and every
    end tag that closes no open element (or the wrong one), and the elements left open."""

    def __init__(self):
        super().__init__()
        self.tags, self.rows, self.stray, self.open = [], [], [], []
        self._section, self._row, self._cell = None, None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append((tag, a))
        if tag not in VOID:
            self.open.append(tag)
        if tag == "section":
            self._section = a.get("id")
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if self.open and self.open[-1] == tag:
            self.open.pop()
        else:
            self.stray.append(tag)
        if tag == "section":
            self._section = None
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append((self._section, self._row))
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse(text):
    p = Page()
    p.feed(text)
    p.close()
    return p


def sections(page):
    """{id: the text of its <section>}: comments and tags gone, entities decoded, white space collapsed."""
    out = {}
    for sid, body in re.findall(r'<section id="([\w-]+)">(.*?)</section>', page, re.S):
        text = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<!--.*?-->", "", body, flags=re.S)))
        text = re.sub(r"\(\s+", "(", re.sub(r"\s+([,;:)]|\.(?=\s|$))", r"\1", re.sub(r"\s+", " ", text)))
        out[sid] = text.strip()
    return out


def blocks(page):
    """[(name, its text)] of the local blocks in page order; an AssertionError when a marker has no partner."""
    marks, out = list(MARK.finditer(page)), []
    assert all(m[0] == f"<!-- {m[1]}local:{m[2]} -->" for m in marks), [m[0] for m in marks]
    assert len(marks) % 2 == 0, [m[0] for m in marks]
    for start, end in zip(marks[::2], marks[1::2]):
        assert (start[1], end[1], end[2]) == ("", "/", start[2]), (start[0], end[0])
        out.append((start[2], page[start.end():end.start()]))
    return out


def test_help_page_holds_the_method_numbers():
    """Each section gives the rules and numbers of the code that it explains, read from the code, so a changed
    constant fails here: the notes that the Results and Downloads tabs once showed behind their info buttons, and the
    numbers that left the main UI (the input types, the upload limit, 2 to 12 views and the 0.25 s snap of a manual
    pick). The stage table lists every stage with its GPU mark, the list of results.zip names every file that
    jobs.package writes, and the calibration example reads."""
    page = HELP.read_text()
    text = sections(page)
    num = lambda x: f"{x:g}"  # noqa: E731
    usd = lambda x: f"${x:.2f}".removesuffix(".00")  # noqa: E731  4.0 -> $4, 0.2 -> $0.20
    iou, small = num(steps.MIN_IOU), f"{vote.MIN_AREA:.1%}"
    score = num(inspect.signature(part_masks.run).parameters["threshold"].default)
    seam_lines = inspect.signature(geometry.seams).parameters["max_lines"].default
    snap_s = num(inspect.signature(video.snap).parameters["window_s"].default)
    nviews = re.search(r'<input type="range" id="nviews" min="(\d+)" max="(\d+)"', (STATIC / "index.html").read_text())
    upload = re.search(r"^MAX_UPLOAD = (\d+) << 30$", (ROOT / "gator" / "webapp" / "server.py").read_text(), re.M)
    assert nviews and upload
    prices = [f"{usd(controller.PRICES[k])} {w}" for k, w in (("input_tokens", "input"), ("output_tokens", "output"),
                                                                ("cache_read_input_tokens", "cache read"),
                                                                ("cache_creation_input_tokens", "cache write"))]
    size_err = 0.05  # mass goes with the size cubed
    mass_err = f"{size_err:.0%} size error gives about a {(1 + size_err) ** 3 - 1:.0%} mass error"
    src = inspect.getsource(jobs.package)
    zipped = re.findall(r'\("([\w.]+\.(?:glb|json))", ', src)
    zipped += [d + "/" for d in re.findall(r'"(\w+)"', re.search(r"for d in \(([^)]*)\)", src)[1])]
    assert len(zipped) >= 13, zipped  # the package source still reads as it did; update the pattern when it changes
    L = lidar_scale
    assert jobs.KEY.pattern.startswith("[A-Za-z0-9_-]{")  # #agent: "one word of letters, digits, - and _"
    facts = {"capture": [*jobs.VIDEOS, *jobs.BAGS, *jobs.STRAY, "HDR off"],
             "add-job": [f"up to {upload[1]} GB"],
             "status": [*steps.NAMES, "one at a time"],
             "settings": [f"within {snap_s} s", f"{nviews[1]} to {nviews[2]}", "view 0"],
             "review": [f"under {views.MIN_AREA:.0%} of the image", f"below {num(views.MIN_CONSISTENCY)}",
                        f"closer than {num(views.MIN_SPACING_DEG)}°", "over a third"],
             "parts": [f"at least {score}", f"less than {small}", "not found", "still have a part",
                       "outside the object", "union", "welds them into one body"],
             "agent": [f"{num(agent_step.BUDGET_S['refinement'])} s to edit",
                       f"{num(agent_step.BUDGET_S['articulation'])} s to set", f"up to {num(tools.MAX_EXCLUDED_S)} s",
                       f"{controller.CLAUDE_EFFORT} effort", f"below {iou}", "skipped", *prices, "A ? means",
                       "refusal fallback", f"at most {tools.MAX_LIMITATIONS}", *tools.VERDICTS["refinement"],
                       *tools.VERDICTS["articulation"], "fallback_original", "checkpoint", "export", "API key",
                       "can reach the network", "no network access", "sk-ant-", "forgets the key", "key field",
                       "letters, digits, - and _"],
             "lidar": ["median", "before MV-SAM3D", "Without LiDAR", f"under {L.MIN_POINTS} points",
                       f"over {L.MAX_IQR_REL:.0%}", f"more than {L.OUTLIER_MIN_REL:.0%}",
                       f"fewer than {L.MIN_VIEWS} views", f"over {L.MAX_SPREAD_REL:.0%}",
                       f"outside {num(L.S_MIN)} to {num(L.S_MAX)}", f"at most {num(L.MAX_TRANSLATION)} m",
                       "T_cam_lidar", "camera_axes", "time_offset_s", "replaces the bag's TF"],
             "inputs": ["0.6 of that median", "16 to 32", "★"],
             "results": [f"Below {iou}", "view 0's camera frame", "aligned.glb", "larger than", "mean over the views"],
             "joints": ["A ? means", "With 0", "nearest part above it", "fixed to the world", "one rigid body"],
             "physics": ["64³", mass_err, "collision hulls", "left out", "median density", "Pa", "kg/m³",
                         "by its volume", "bottom of the object", "MuJoCo, USD and URDF", "VoMP v1.1"],
             "downloads": ["(package)", "API key", "the agent's result when the key was found in it",
                           "cameras' mean up", "fixed one to the link that physics welds it to",
                           "each joint that the agent set", "a part without one has none", "view 0's camera frame",
                           "mask IoU", "5th to 95th percentile", "could not voxelize", "ok is false",
                           "uploaded calibration", "Stray capture", "aligned with the color camera", "view 0 first",
                           "null without LiDAR",
                           "excluded", f"up to {seam_lines} fitted lines", "not split off", "only in results.zip",
                           "urdf/meshes", *zipped],
             "retry": ["first stage that is not done", "replaces the results"]}
    assert set(facts) <= set(text), set(facts) - set(text)
    for sid, want in facts.items():
        assert all(f in text[sid] for f in want), (sid, [f for f in want if f not in text[sid]])
    # Every stage, in order, and the GPU mark on the ones that run on the GPU.
    stages = [r for s, r in parse(page).rows if s == "status" and r[0] in steps.NAMES]
    assert [r[0] for r in stages] == steps.NAMES
    assert {r[0] for r in stages if r[2] == "GPU"} == set(steps.GPU) and all(r[2] in ("", "GPU") for r in stages)
    # The calibration example is a file that the lidar stage reads.
    lidar = re.search(r'<section id="lidar">.*?</section>', page, re.S)[0]
    example = re.search(r"<pre>(.*?)</pre>", lidar, re.S)[1]
    cal = lidar_scale.read_calibration(html.unescape(example), "calibration.yaml")
    assert cal["given_as"] == "T_cam_lidar" and len(cal["T_cam_lidar"]) == 4


def test_help_page_is_a_plain_page_with_every_section():
    """A page like privacy.html: the head that site.py rewrites (one charset meta, the stylesheet and the brand link
    once each), the header with the brand alone, no script, no inline style and no event attribute (the public site's
    CSP blocks them). Every section of the plan, in order, as <section id> with its <h2> first, and a contents list that
    links to each. Every link goes to a section of the page, or home, or the stylesheet."""
    page = HELP.read_text()
    p = parse(page)
    assert page.startswith("<!doctype html>\n<html lang=\"en\">")
    assert re.search(r"<title>Help · HARPOR</title>", page)
    for once in ('<meta charset="utf-8">', 'href="/static/style.css"', 'href="/"'):
        assert page.count(once) == 1, once
    assert ("meta", {"name": "viewport", "content": "width=device-width, initial-scale=1"}) in p.tags
    assert ("meta", {"name": "color-scheme", "content": "light dark"}) in p.tags
    head = re.search(r"<header>(.*?)</header>", page, re.S)[1].strip()
    assert head == '<a href="/" class="brand">HARPOR</a>'
    assert not [t for t, _ in p.tags if t in ("script", "style", "iframe", "form", "img")]
    assert not [(t, a) for t, a in p.tags if "style" in a or any(k.startswith("on") for k in a)]
    assert not p.stray and not p.open, (p.stray, p.open)
    ids = [a["id"] for _, a in p.tags if "id" in a]
    assert len(ids) == len(set(ids))
    assert [a["id"] for t, a in p.tags if t == "section"] == list(SECTIONS)
    for sid in SECTIONS:
        assert re.search(rf'<section id="{sid}">\s*<h2>[^<]+</h2>', page), sid
    nav = re.search(r'<nav class="toc" aria-label="Contents">(.*?)</nav>', page, re.S)[1]
    assert re.findall(r'href="#([\w-]+)"', nav) == list(SECTIONS)
    for _, a in p.tags:
        href = a.get("href")
        if href is not None:
            assert href in ("/", "/static/style.css") or (href.startswith("#") and href[1:] in ids), href


def test_help_page_local_blocks_are_well_formed():
    """Each local block opens and closes once, in that order, with no block inside another, and holds whole elements,
    so that site.py can swap it for its public text. The page without the blocks is whole too, and holds none of the
    local page's own text: the upload, the SSH tunnel, the ?token= link and the agent "None"."""
    page = HELP.read_text()
    found = blocks(page)
    assert sorted(name for name, _ in found) == sorted(LOCAL)
    for name, body in found:
        p = parse(body)
        assert not p.stray and not p.open, (name, p.stray, p.open)
        assert not MARK.search(body), name
    assert dict(found)["toc"].strip() == "" and dict(found)["account"].strip() == ""
    rest = MARK.sub("", re.sub(r"<!-- local:([\w-]+) -->.*?<!-- /local:\1 -->", "", page, flags=re.S))
    p = parse(rest)
    assert not p.stray and not p.open, (p.stray, p.open)
    assert [a["id"] for t, a in p.tags if t == "section"] == list(SECTIONS)
    text = sections(rest)
    for local in ("SSH tunnel", "?token=", "Choose a file", "Cancel upload", "None", "20 GB"):
        assert not any(local in t for t in text.values()), local


def test_every_help_link_has_its_section():
    """The header's Help link (app.js HELP_AT) and the error box's link go to a section of the help page, on the
    tunnel and on the public site. The links that site.py adds to the public pages (the machines panel, the privacy
    page) go to a section of the public help page."""
    app = (STATIC / "app.js").read_text()
    at = re.search(r"const HELP_AT = \{(.*?)\};", app, re.S)[1]
    anchors = set(re.findall(r':\s*"([\w-]+)"', at)) | set(re.findall(r'"help\.html#([\w-]+)"', app))
    assert {"add-job", "status", "troubleshooting"} <= anchors and len(anchors) >= 11, anchors
    public = set(re.findall(r"help\.html#([\w-]+)", Path(site.__file__).read_text()))
    assert {"machine", "agent"} <= public, public
    cfg = {"url": "https://abcdefghijklmnop.supabase.co", "publishableKey": "sb_publishable_AbCdEf_0123456789"}
    for page, want in ((HELP.read_text(), anchors), (site.public_help(HELP.read_text(), cfg), anchors | public)):
        ids = {a["id"] for _, a in parse(page).tags if "id" in a}
        assert want <= ids, want - ids
