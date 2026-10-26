"""gator/webapp/site.py: the public site's static folder."""
import base64
import functools
import hashlib
import html
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pytest

from gator.webapp import site

ROOT = Path(__file__).resolve().parents[1]
CFG = {"url": "https://abcdefghijklmnop.supabase.co", "publishableKey": "sb_publishable_AbCdEf_0123456789"}
JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIn0.c2lnbmF0dXJl"
CDN = re.compile(r"ajax\.googleapis\.com|cdn\.jsdelivr\.net|unpkg\.com|gstatic\.com|cdnjs\.")


def write_cfg(tmp_path, **kw):
    p = tmp_path / "site.json"
    p.write_text(json.dumps({**CFG, **kw}))
    return p


@pytest.fixture
def built(tmp_path):
    out = tmp_path / "site"
    site.build(out, write_cfg(tmp_path, contact="the lab's front desk"))
    return out


@pytest.fixture
def demo_dir(tmp_path):
    """A demos folder with one demo written by hand (tests/test_webapp_demos.py)."""
    pytest.importorskip("av")
    pytest.importorskip("PIL")
    from test_webapp_demos import hand_demo

    hand_demo(tmp_path / "demos")
    return tmp_path / "demos"


class Links(HTMLParser):
    """Every src and href of a page, the attributes of its tags, its inline scripts and its head's order."""

    def __init__(self):
        super().__init__()
        self.links, self.tags, self.inline, self.head, self._script, self._in_head = [], [], [], [], None, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append((tag, a))
        if tag == "head":
            self._in_head = True
        elif self._in_head:
            self.head.append((tag, a))
        self.links += [v for k, v in attrs if k in ("src", "href")]
        if tag == "script":
            self._script = a

    def handle_endtag(self, tag):
        if tag == "head":
            self._in_head = False
        if tag == "script":
            self._script = None

    def handle_data(self, data):
        if self._script is not None and "src" not in self._script and data.strip():
            self.inline.append(data)


def parse(path):
    p = Links()
    p.feed(Path(path).read_text())
    return p


def csp_of_meta(page):
    tag = next(a for t, a in parse(page).head if t == "meta" and a.get("http-equiv") == "Content-Security-Policy")
    return tag["content"]


def csp_of_headers(out):
    return re.search(r"^  Content-Security-Policy: (.+)$", (out / "_headers").read_text(), re.M).group(1)


def test_build_writes_every_part_of_the_site(built):
    files = sorted(p.relative_to(built).as_posix() for p in built.rglob("*") if p.is_file())
    assert files == sorted([".nojekyll", "_headers", "demos.html", "help.html", "index.html", "privacy.html", "static/app.js",
                            "static/config.js", "static/demos.js", "static/style.css", "static/supa.js"] +
                           [f"static/vendor/{v}" for v in site.VENDOR])
    for name in ("app.js", "style.css", "supa.js", "demos.js"):
        assert (built / "static" / name).read_bytes() == (site.STATIC / name).read_bytes()
    for rel in site.VENDOR:
        assert (built / "static/vendor" / rel).read_bytes() == (site.STATIC / "vendor" / rel).read_bytes()
    js = (built / "static/config.js").read_text()
    conf = json.loads(js[js.index("{"):js.rindex("}") + 1])
    assert conf == CFG
    assert (built / ".nojekyll").read_bytes() == b""


def test_csp_is_strict_and_the_same_in_headers_and_meta(built):
    """Scripts and styles from the site only (no CDN host, no unsafe-inline or unsafe-eval; WebAssembly may compile,
    for model-viewer's decoders, and model-viewer's own style block may apply), the project's API and socket only,
    and blob: URLs, from which model-viewer reads a GLB's own textures;
    the header policy equals the <meta> policy of every page plus frame-ancestors, which a <meta> cannot set."""
    head = csp_of_headers(built)
    rules = dict(r.split(" ", 1) for r in head.split("; "))
    assert rules["script-src"] == "'self' 'wasm-unsafe-eval'" and rules["style-src"] == "'self' " + site.MODEL_VIEWER_STYLE
    assert rules["default-src"] == "'none'"
    assert rules["connect-src"] == "'self' blob: https://abcdefghijklmnop.supabase.co wss://abcdefghijklmnop.supabase.co"
    assert rules["img-src"] == "'self' data: blob: https://abcdefghijklmnop.supabase.co"
    assert rules["media-src"] == "'self' blob: https://abcdefghijklmnop.supabase.co"  # 'self': the demos' videos
    assert rules["frame-ancestors"] == "'none'" and rules["base-uri"] == "'none'" and rules["form-action"] == "'none'"
    assert "'unsafe-inline'" not in head and "'unsafe-eval'" not in head and not CDN.search(head)
    for page in ("index.html", "help.html", "privacy.html", "demos.html"):
        assert csp_of_meta(built / page) == head.replace("frame-ancestors 'none'; ", "")
    text = (built / "_headers").read_text()
    for h in ("Referrer-Policy: no-referrer", "Strict-Transport-Security: max-age=31536000",
              "X-Content-Type-Options: nosniff", "/static/vendor/*\n  Cache-Control: public, max-age=31536000, immutable"):
        assert h in text


def test_model_viewer_style_hash_is_that_of_its_style_block():
    """The one inline style that the policy allows is model-viewer's shadow-DOM <style>, by the hash of its text."""
    text = (site.STATIC / "vendor/model-viewer-4.4.0.min.js").read_text(encoding="utf-8")
    blocks = re.findall(r"<style>(.*?)</style>", text, re.S)
    assert len(blocks) == 1 and blocks[0].lstrip().startswith(":host {")
    assert site.MODEL_VIEWER_STYLE == "'sha256-" + base64.b64encode(hashlib.sha256(blocks[0].encode()).digest()).decode() + "'"


def test_local_stack_policy_uses_its_own_origin(tmp_path):
    cfg = {**CFG, "url": "http://127.0.0.1:54321"}
    assert "connect-src 'self' blob: http://127.0.0.1:54321 ws://127.0.0.1:54321;" in site.csp(cfg)
    assert "img-src 'self' data: blob: http://127.0.0.1:54321;" in site.csp(cfg)


def test_public_index_is_the_tunnel_page_made_public(built):
    """Relative paths only, each to a file of the site; the CSP and referrer metas before anything that loads; the
    site's own model-viewer; the Supabase scripts before app.js; the markup that supa.js draws into; the header's Demos
    and Help links before the account menu, which opens from a person icon whose name holds the email; the sign-in
    panel's link to the demos; the upload area hidden, with its message line moved out of it, after the machines
    panel, which links to the help page's machine setup; no "None" agent; no inline script, style attribute or event
    attribute, which the CSP would block."""
    page = parse(built / "index.html")
    assert page.head[:3] == [("meta", {"charset": "utf-8"}),
                             ("meta", {"http-equiv": "Content-Security-Policy", "content": csp_of_meta(built / "index.html")}),
                             ("meta", {"name": "referrer", "content": "no-referrer"})]
    for link in page.links:
        assert not re.match(r"[a-z]+:|/", link), link
        target = link.split("#")[0]
        assert not target or (built / target).is_file(), link
    scripts = [a["src"] for t, a in page.tags if t == "script"]
    assert scripts == ["static/vendor/model-viewer-4.4.0.min.js", "static/vendor/supabase-2.118.0.js", "static/config.js",
                       "static/supa.js", "static/app.js"]
    ids = {a.get("id") for _, a in page.tags}
    for i in ("harpor-account", "harpor-email", "harpor-who", "harpor-storage", "harpor-signout", "harpor-banner", "harpor-signin",
              "harpor-google", "harpor-github", "harpor-signinmsg", "harpor-waiting", "harpor-request", "harpor-note",
              "harpor-send", "harpor-requestmsg", "harpor-own", "harpor-machines", "harpor-pending", "drop", "file",
              "upmsg", "jobs", "startform", "keylabel", "demolink", "helplink"):
        assert i in ids, i
    for gone in ("harpor-resume", "harpor-ownnote", "harpor-howto"):
        assert gone not in ids, gone
    for i in ("harpor-account", "harpor-banner", "harpor-signin", "harpor-waiting", "harpor-own", "harpor-pending", "drop"):
        assert "hidden" in next(a for _, a in page.tags if a.get("id") == i), i
    agents = [a["value"] for t, a in page.tags if t == "input" and a.get("name") == "agent"]
    assert agents == ["astra", "opus"]
    assert page.inline == [] and not any("style" in a or any(k.startswith("on") for k in a) for _, a in page.tags)
    html = (built / "index.html").read_text()
    assert not CDN.search(html) and 'value="none"' not in html and "tus" not in html.lower().replace("status", "")
    header = html[html.index("<header>"):html.index("</header>")]
    assert (header.index('id="demolink"') < header.index('id="helplink"') < header.index('id="conn"')
            < header.index('id="harpor-account"'))
    # The account menu's button: a person icon (an SVG that a screen reader skips) and a name in .sr-only that starts
    # with "Account: " and holds the email; the menu's first line holds the email too.
    summary = re.search(r"<summary[^>]*>(.*?)</summary>", header, re.S)[1]
    svg = re.findall(r"<svg[^>]*>", summary)
    assert len(svg) == 1 and 'aria-hidden="true"' in svg[0] and 'focusable="false"' in svg[0]
    name = re.sub(r"<svg.*?</svg>", "", summary, flags=re.S)
    assert re.fullmatch(r'<span class="sr-only">Account: <span id="harpor-email"></span></span>', name)
    menu = header[header.index('class="accountmenu"'):]
    assert re.search(r'class="accountmenu">\s*<p id="harpor-who" class="who"></p>', menu)
    gate = html[html.index('id="harpor-signin"'):html.index("</section>", html.index('id="harpor-signin"'))]
    assert '<p><a href="demos.html" target="_blank" rel="noopener">See the demos</a></p>' in gate  # as the header's link
    drop = html[html.index('<div id="drop"'):html.index("</div>", html.index('<div id="drop"'))]
    assert 'id="file"' in drop and 'id="upmsg"' not in drop and html.count('id="upmsg"') == 1
    assert html.index('id="harpor-own"') < html.index('id="upmsg"') < html.index("<h2>Jobs</h2>")
    own = html[html.index('id="harpor-own"'):html.index("</section>", html.index('id="harpor-own"'))]
    assert '<a href="help.html#machine" target="_blank" rel="noopener">Set up a machine</a>' in own
    assert "<code>" not in own and "--worker" not in html  # the steps are on the help page


def test_tunnel_index_is_unchanged_and_loads_no_supabase_script():
    """site.py derives the public page; the server's own index.html keeps its paths, its "No agent" choice and no
    Supabase script, so the tunnel works as before."""
    html = (site.STATIC / "index.html").read_text()
    assert site.MODEL_VIEWER in html and '<script src="/static/app.js"></script>' in html
    assert 'value="none"' in html and "supa.js" not in html and "config.js" not in html and "harpor-" not in html


def test_privacy_page_says_where_data_lives_and_who_can_see_keys(built):
    """The privacy page says that inputs stay on the user's machine; that state, logs, review images and results are
    copied to the owner's Supabase project, which the owner can read; how long each is kept; that keys go sealed only
    to the user's machine, and what the owner could do with the database; and how to reach the owner. The choice of
    agent (Astra's network access) is on the help page, which the header links too. Nothing of a shared GPU is
    left."""
    page = parse(built / "privacy.html")
    for link in page.links:
        assert not re.match(r"[a-z]+:|/", link) and (built / link.split("#")[0]).is_file(), link
    assert ("a", {"id": "helplink", "href": "help.html"}) in page.tags and "brandsub" not in (built / "privacy.html").read_text()
    assert ("a", {"href": "help.html#agent"}) in page.tags
    text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", (built / "privacy.html").read_text())))
    for want in ("Your input file stays on that machine and never comes to this site.",
                 "copies each job's state, the logs of its stages, its review images, the LiDAR points of its views "
                 "and its result files to this site's Supabase project",
                 "A LiDAR scan can show more of the place than the photos do.",
                 "the site's owner, who runs that project and can read it.",
                 "The log text of a job goes 14 days after the job stops.",
                 "review images, LiDAR points and log files from the site 14 days after the job stops, and its result "
                 "files after 90 days",
                 "Delete a job on the jobs page to remove its files from the site at once.",
                 "If the site's owner takes back your approval, this page no longer lists your jobs; then ask the "
                 "owner to delete them.",
                 "Your browser seals the key to the public key of the HARPOR worker on your machine.",
                 "Only that worker can open the sealed key",
                 "could list another key and read the keys that you send after that",
                 "keep the key in a file on your machine",
                 "Choosing an agent: see Help.",
                 "write to the site's owner: the lab's front desk."):
        assert want in text, want
    for gone in ("Georgia Tech", "shared GPU", "GPU server", "in clear", "upload", "72 hours", "can reach the network",
                 "spending limit"):  # the help page says these
        assert gone not in text, gone
    assert page.inline == [] and not CDN.search((built / "privacy.html").read_text())


def test_public_help_page(built):
    """help.html on the public site: the CSP and referrer metas first, as on every page; no script, inline style or
    event attribute; relative links only, each to a file of the site or a section of the page. It adds the sections
    that only the public site has (the account, setting up a machine, the machines' status) after About, in the
    contents too, with the machine steps and the agents' network access, and holds none of the tunnel page's own
    text: the upload, the SSH tunnel, the ?token= link and the agent "None"."""
    page = parse(built / "help.html")
    assert page.head[:3] == [("meta", {"charset": "utf-8"}),
                             ("meta", {"http-equiv": "Content-Security-Policy", "content": csp_of_meta(built / "help.html")}),
                             ("meta", {"name": "referrer", "content": "no-referrer"})]
    assert ("a", {"href": "index.html", "class": "brand"}) in page.tags
    assert not [t for t, _ in page.tags if t == "script"] and page.inline == []
    assert not any("style" in a or any(k.startswith("on") for k in a) for _, a in page.tags)
    ids = [a["id"] for _, a in page.tags if "id" in a]
    assert len(ids) == len(set(ids))
    for link in page.links:
        assert not re.match(r"[a-z]+:|/", link), link
        target, _, anchor = link.partition("#")
        assert (built / target).is_file() if target else anchor in ids, link
    sections = [a["id"] for t, a in page.tags if t == "section"]
    assert sections[:4] == ["about", "account", "machine", "machine-status"] and "troubleshooting" in sections
    source = (built / "help.html").read_text()
    nav = re.search(r'<nav class="toc" aria-label="Contents">(.*?)</nav>', source, re.S)[1]
    assert re.findall(r'href="#([\w-]+)"', nav) == sections
    text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", source)))
    for want in ("pip install cryptography==50.0.2", "python -m gator.webapp.worker login",
                 "ssh -L 8767:127.0.0.1:8767 <machine>", "python -m gator.webapp --worker. Add --local-keys opus",
                 "python -m gator.webapp.worker add PATH", "offline or stopping",
                 "GPT-6-Astra runs the Codex CLI in a container that can reach the network.",
                 "These commands run on the GPU machine that runs the job.",
                 "Claude Opus 5.5 runs its shell with no network access.", "Signed out. Reload to sign in.",
                 "The person icon at the top right opens your account menu: your email, the Storage line, Privacy and "
                 "Sign out."):
        assert want in text, want
    for gone in ("SSH tunnel", "?token=", "Choose a file", "Cancel upload", "20 GB", "None"):
        assert gone not in text, gone
    assert "<!-- local:" not in source and "<!-- /local:" not in source and not CDN.search(source)


def test_tunnel_help_page_and_its_public_blocks():
    """The server serves static/help.html as it is: its own paths, no CSP meta, no "harpor-" id and none of the public
    sections. Each of its local blocks has a HELP_PUBLIC text and each text has a block; the build stops when a block
    is missing, has another name or is there twice."""
    src = (site.STATIC / "help.html").read_text()
    ids = {a["id"] for _, a in parse(site.STATIC / "help.html").tags if "id" in a}
    assert not [i for i in ids if i.startswith("harpor-")] and not ids & {"account", "machine", "machine-status"}
    assert "Content-Security-Policy" not in src and 'href="/static/style.css"' in src and 'href="/"' in src
    names = [m[2] for m in site.LOCAL_BLOCK.finditer(src)]
    assert sorted(names) == sorted(site.HELP_PUBLIC) and len(names) == len(set(names))
    keys = re.search(r"<!-- local:keys -->.*?<!-- /local:keys -->", src, re.S)[0]
    for broken in (src.replace(keys, ""), src.replace("local:keys -->", "local:key -->"),
                   src.replace(keys, keys + "\n" + keys)):
        with pytest.raises(site.SiteError, match="local blocks"):
            site.public_help(broken, CFG)
    # A marker without its partner, a stray one before a block (which would swallow the text after it), and a block
    # with other spacing, which LOCAL_BLOCK does not read: its local text would ship.
    for broken in (src.replace(keys, keys + "<!-- local:keys -->"),
                   src.replace(keys, "<!-- local:keys -->\n<p>more</p>\n" + keys),
                   src.replace("</main>", "<!--local:upload--><p>Choose a file</p><!--/local:upload-->\n</main>")):
        with pytest.raises(site.SiteError, match="marker without its partner"):
            site.public_help(broken, CFG)


def test_public_help_numbers_are_the_databases(built):
    """The public help page's own numbers come from the SQL: how long a request waits for its machine
    (settings.pending_days), how long a machine that has not run still shows (the workers' row policy), the length of
    an access request and how often one can be sent. Its storage line has the form that supa.js writes."""
    sql = "\n".join(p.read_text() for p in sorted((ROOT / "supabase" / "migrations").glob("*.sql")))
    pending = re.search(r"pending_days int not null default (\d+)", sql)[1]
    seen = re.search(r"last_seen > now\(\) - interval '(\d+) days'", sql)[1]
    low, high = re.search(r"char_length\(p_note\) not between (\d+) and (\d+)", sql).groups()
    assert "request_at < now() - interval '1 day'" in sql and "'you can send one request a day'" in sql
    text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", (built / "help.html").read_text())))
    for want in (f"within {pending} days", f"in the last {seen} days", f"in {low} to {high} characters",
                 "one request a day", "Storage: 120 of 2,048 MB"):
        assert want in text, want
    assert "`Storage: ${mib(r.data)} of ${mib(access.max_result_bytes)} MB`" in (site.STATIC / "supa.js").read_text()


def served(built, tmp_path):
    """Serve the site built as GitHub Pages serves a project site (under /HARPOR/), and load every src and href of its
    four pages and the decoder files that supa.js names by their relative URL. Returns those URLs, relative to the
    site."""
    www = tmp_path / "www"
    shutil.copytree(built, www / "HARPOR")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(www))
    handler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}/HARPOR/"
        urls = {base, base + "privacy.html"}
        for page in ("index.html", "help.html", "privacy.html", "demos.html"):
            urls |= {urllib.parse.urljoin(base + page, link.split("#")[0]) for link in parse(built / page).links}
        supa = (site.STATIC / "supa.js").read_text()
        for folder in re.findall(r'new URL\("(static/vendor/[^"]+/)", location\.href\)', supa):
            for f in (site.STATIC / folder.replace("static/", "", 1)).iterdir():
                urls.add(base + folder + f.name)
        for url in sorted(urls):
            with urllib.request.urlopen(url) as r:
                assert r.status == 200, url
        return {u.removeprefix(base) for u in urls}
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_site_served_under_a_repo_path_loads_every_asset(built, tmp_path):
    """Every asset of the four pages loads under /HARPOR/ (served), without demos."""
    assert len(served(built, tmp_path)) > 12


def test_site_with_demos_served_under_a_repo_path(tmp_path, demo_dir):
    """The same with a demo: its files load too."""
    built = tmp_path / "site"
    site.build(built, write_cfg(tmp_path), demo_dir=demo_dir)
    assert "demos/office-chair/turntable.mp4" in served(built, tmp_path)


def test_build_with_demos(tmp_path, demo_dir):
    """demos.html: the CSP and referrer metas first, the site's own model-viewer and demos.js, no other script or inline style,
    relative links that each go to a file of the site or to a section of the page (a card to its demo, a demo back to
    the gallery), the placeholders' cards after the demo's, and the demo's files copied byte for byte. A demos folder
    that breaks a rule, or that looks private, stops the build, and nothing is written. Without a folder the page says
    "No demos yet."."""
    out = tmp_path / "site"
    (demo_dir / "placeholders.json").write_text(json.dumps({"version": 1, "placeholders": ["Laptop", "Door"]}))
    written = site.build(out, write_cfg(tmp_path), demo_dir=demo_dir)
    assert sorted(p for p in written if p.startswith("demos/")) == sorted(
        f"demos/office-chair/{f}" for f in ("turntable.mp4", "poster.jpg", "aligned.glb", "results.zip"))
    for rel in written:
        if rel.startswith("demos/"):
            assert (out / rel).read_bytes() == (demo_dir / rel.removeprefix("demos/")).read_bytes(), rel
    page = parse(out / "demos.html")
    assert page.head[:3] == [("meta", {"charset": "utf-8"}),
                             ("meta", {"http-equiv": "Content-Security-Policy", "content": csp_of_meta(out / "demos.html")}),
                             ("meta", {"name": "referrer", "content": "no-referrer"})]
    assert [a["src"] for t, a in page.tags if t == "script"] == ["static/vendor/model-viewer-4.4.0.min.js", "static/demos.js"]
    assert page.inline == [] and not any("style" in a or any(k.startswith("on") for k in a) for _, a in page.tags)
    assert ("a", {"href": "index.html", "class": "brand"}) in page.tags
    ids = {a["id"] for _, a in page.tags if "id" in a}
    for link in page.links:
        if link.startswith("#"):  # the gallery is the page without a target
            assert link == "#" or link[1:] in ids, link
        else:
            assert not re.match(r"[a-z]+:|/", link) and (out / link.split("#")[0]).is_file(), link
    assert {"#demo-office-chair", "#"} <= set(page.links)
    text = (out / "demos.html").read_text()
    assert not CDN.search(text) and 'id="demo-office-chair"' in text and "No demos yet." not in text
    assert "<style" not in text and "autoplay" not in text
    assert re.findall(r'<span class="dctitle">([^<]+)</span><span class="dcnote">Coming soon</span>', text) == ["Laptop", "Door"]
    assert text.index('href="#demo-office-chair"') < text.index("Laptop")
    assert "placeholders.json" not in " ".join(written)
    (demo_dir / "placeholders.json").unlink()
    bad = tmp_path / "bad"
    shutil.copytree(demo_dir, bad)
    (bad / "office-chair" / "notes.zip").write_bytes(b"PK\x03\x04")
    with pytest.raises(site.SiteError, match="the demos folder: office-chair/notes.zip is not listed in demo.json"):
        site.build(tmp_path / "other", write_cfg(tmp_path), demo_dir=bad)
    (bad / "office-chair" / "notes.zip").unlink()
    spec = json.loads((bad / "office-chair" / "demo.json").read_text())
    (bad / "office-chair" / "demo.json").write_text(json.dumps({**spec, "title": "Pallet A"}))
    with pytest.raises(site.SiteError, match="looks private"):
        site.build(tmp_path / "other", write_cfg(tmp_path), demo_dir=bad)
    assert not (tmp_path / "other").exists()
    # Over a build with demos, a build without them needs no_demos: a deploy of it would take the live demos away.
    with pytest.raises(site.SiteError, match="has demos or placeholders; give --demos DIR, or --no-demos"):
        site.build(out, write_cfg(tmp_path))
    assert (out / "demos" / "office-chair" / "turntable.mp4").is_file()
    site.build(out, write_cfg(tmp_path), no_demos=True)
    assert '<p class="emptytab">No demos yet.</p>' in (out / "demos.html").read_text() and not (out / "demos").exists()


def test_build_with_placeholders_only(tmp_path):
    """A demos folder with only placeholders.json: a gallery of placeholder cards, none a link, no demo files and no
    "No demos yet.". Over that build, a build without a folder stops, as over one with demos. A bad placeholders.json
    stops the build with its reason, and nothing is written."""
    folder = tmp_path / "demos"
    folder.mkdir()
    (folder / "placeholders.json").write_text(json.dumps({"version": 1, "placeholders": ["Drawer cabinet", "Toolbox"]}))
    out = tmp_path / "site"
    written = site.build(out, write_cfg(tmp_path), demo_dir=folder)
    assert not any(p.startswith("demos/") for p in written)
    text = (out / "demos.html").read_text()
    assert text.count('<div class="democard soon">') == 2 and "No demos yet." not in text and "<section" not in text
    assert 'class="democard"' not in text and not re.search(r'href="#', text)
    page = parse(out / "demos.html")
    assert page.inline == [] and not any("style" in a for _, a in page.tags)
    with pytest.raises(site.SiteError, match="has demos or placeholders; give --demos DIR, or --no-demos"):
        site.build(out, write_cfg(tmp_path))
    assert (out / "demos.html").read_text() == text
    (folder / "placeholders.json").write_text(json.dumps({"version": 1, "placeholders": ["Pallet B"]}))
    with pytest.raises(site.SiteError, match=re.escape("the demos folder: placeholders.json: looks private (matches "
                                                       "'pallet'); private data is never a demo; nothing was written")):
        site.build(tmp_path / "other", write_cfg(tmp_path), demo_dir=folder)
    assert not (tmp_path / "other").exists()


def test_build_refuses_secrets(tmp_path):
    """A secret key or a JWT in site.json, or in any file of the site, stops the build before anything is written."""
    for bad in ("sb_secret_0123456789abcdef", JWT):
        with pytest.raises(site.SiteError, match="publishableKey|secret key or a JWT"):
            site.build(tmp_path / "out", write_cfg(tmp_path, publishableKey=bad))
    with pytest.raises(site.SiteError, match="secret key or a JWT"):
        site.build(tmp_path / "out", write_cfg(tmp_path, contact="key " + JWT))
    static = tmp_path / "static"
    shutil.copytree(site.STATIC, static)
    (static / "app.js").write_text((static / "app.js").read_text() + f"\n// {JWT}\n")
    with pytest.raises(site.SiteError, match="static/app.js holds a secret key or a JWT"):
        site.build(tmp_path / "out", write_cfg(tmp_path), static)
    (static / "app.js").write_text((site.STATIC / "app.js").read_text() + "\nconst k = 'sb_secret_abcdefgh12345';\n")
    with pytest.raises(site.SiteError, match="static/app.js"):
        site.build(tmp_path / "out", write_cfg(tmp_path), static)
    assert not (tmp_path / "out").exists()
    # supabase-js itself holds the bare prefix (it checks keys): that is no key
    assert b"sb_secret_`" in (site.STATIC / "vendor/supabase-2.118.0.js").read_bytes()


def test_build_checks_each_vendor_file(tmp_path):
    static = tmp_path / "static"
    shutil.copytree(site.STATIC, static)
    f = static / "vendor/supabase-2.118.0.js"
    f.write_bytes(f.read_bytes() + b"\n")
    with pytest.raises(site.SiteError, match="supabase-2.118.0.js does not have its SHA-384"):
        site.build(tmp_path / "out", write_cfg(tmp_path), static)
    (static / "vendor/three-0.183.0/draco/draco_decoder.wasm").unlink()
    shutil.copy(site.STATIC / "vendor/supabase-2.118.0.js", f)
    with pytest.raises(site.SiteError, match="draco_decoder.wasm is missing"):
        site.build(tmp_path / "out", write_cfg(tmp_path), static)


def test_vendor_folder_holds_only_checked_files():
    """Every file in static/vendor is in VENDOR (and the README says where each comes from), so nothing unchecked
    ships."""
    vendor = site.STATIC / "vendor"
    have = sorted(p.relative_to(vendor).as_posix() for p in vendor.rglob("*") if p.is_file())
    assert have == sorted([*site.VENDOR, "README.md"])
    assert not any("tus" in rel for rel in site.VENDOR)  # no browser uploads, so no TUS client
    readme = (vendor / "README.md").read_text()
    assert "tus" not in readme.lower()
    for rel in site.VENDOR:
        if not rel.startswith("licenses/"):
            assert Path(rel).name in readme, rel


def test_config_checks(tmp_path):
    """site.json takes public values only, in their forms, and refuses unknown keys."""
    good = site.check_config(dict(CFG))
    assert good["url"] == CFG["url"]
    for bad, why in [({"redirectTo": "https://x/"}, "unknown keys: redirectTo"), ({"host": "cloudflare"}, "unknown keys"),
                     ({"url": "https://x.example.com"}, "url is"), ({"url": "https://abcdefghijklmnop.supabase.co/rest"}, "url is"),
                     ({"url": "https://abcdefghijklmnop.storage.supabase.co"}, "url is"),
                     ({"publishableKey": "sb_publishable_<from the dashboard>"}, "publishableKey"),
                     ({"contact": ""}, "contact"), ({"contact": "x" * 201}, "contact")]:
        with pytest.raises(site.SiteError, match=why):
            site.check_config({**CFG, **bad})
    assert site.check_config({**CFG, "contact": "room 101"})["contact"] == "room 101"
    assert site.check_config({**CFG, "_comment": "a note"})["_comment"] == "a note"
    assert "_comment" not in site.config_js({**CFG, "_comment": "a note"})
    with pytest.raises(site.SiteError, match="run python -m gator.webapp.site init"):
        site.load_config(tmp_path / "none.json")


def test_init_writes_site_json(tmp_path):
    p = tmp_path / "site.json"
    cfg = site.init("https://abcdefghijklmnop.supabase.co/", "sb_publishable_AbCdEf_0123456789", path=p)
    assert json.loads(p.read_text()) == cfg == CFG
    with pytest.raises(site.SiteError, match="exists; add --force"):
        site.init(CFG["url"], CFG["publishableKey"], path=p)
    site.init("http://127.0.0.1:54321", "sb_publishable_local_0123", path=p, force=True, contact="room 101")
    assert json.loads(p.read_text()) == {"url": "http://127.0.0.1:54321", "publishableKey": "sb_publishable_local_0123",
                                         "contact": "room 101"}
    assert site.load_config(p)["contact"] == "room 101"
    with pytest.raises(site.SiteError, match="never the secret key"):
        site.init(CFG["url"], "sb_secret_0123456789abcdef", path=tmp_path / "other.json")
    assert not (tmp_path / "other.json").exists()


def test_build_replaces_only_an_earlier_build(tmp_path, built):
    site.build(built, write_cfg(tmp_path))  # an earlier build: replaced
    (built / "stale.txt").write_text("x")
    site.build(built, write_cfg(tmp_path))
    assert not (built / "stale.txt").exists()
    other = tmp_path / "docs"
    other.mkdir()
    (other / "notes.md").write_text("mine")
    with pytest.raises(site.SiteError, match="is not an earlier build"):
        site.build(other, write_cfg(tmp_path))
    assert (other / "notes.md").read_text() == "mine"


def run_site(*args):
    """python -m gator.webapp.site with args, without WEBAPP_DEMOS."""
    env = {k: v for k, v in os.environ.items() if k != "WEBAPP_DEMOS"}
    return subprocess.run([sys.executable, "-m", "gator.webapp.site", *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=60, env=env)


def test_command_line(tmp_path):
    run = run_site
    cfg = tmp_path / "s.json"
    r = run("init", "--url", CFG["url"], "--key", CFG["publishableKey"], "--config", str(cfg))
    assert r.returncode == 0 and CFG["publishableKey"] not in r.stdout + r.stderr
    r = run("build", "--config", str(cfg), "--out", str(tmp_path / "out"))
    assert r.returncode == 0 and r.stdout.startswith("wrote ") and (tmp_path / "out/index.html").is_file()
    assert r.stdout.splitlines()[1] == "demos: none (give --demos DIR or set WEBAPP_DEMOS)"
    r = run("init", "--url", CFG["url"], "--key", "sb_secret_0123456789abcdef", "--config", str(tmp_path / "x.json"))
    assert r.returncode == 1 and r.stderr.startswith("error: ") and "sb_secret_0123456789abcdef" not in r.stderr
    assert run("build", "--config", str(tmp_path / "missing.json")).returncode == 1
    assert run("bogus").returncode == 2


def test_command_line_with_demos(tmp_path, demo_dir):
    """--demos copies the demos in, and the second line counts them and the placeholders. Over that build, a build
    without a folder stops unless --no-demos, and a folder that is not there stops the build."""
    cfg, out = tmp_path / "s.json", tmp_path / "out"
    cfg.write_text(json.dumps(CFG))
    build = lambda *a: run_site("build", "--config", str(cfg), "--out", str(out), *a)  # noqa: E731
    r = build("--demos", str(demo_dir))
    assert r.returncode == 0 and r.stdout.splitlines()[1] == f"demos: 1 from {demo_dir}"
    (demo_dir / "placeholders.json").write_text(json.dumps({"version": 1, "placeholders": ["Laptop", "Door"]}))
    r = build("--demos", str(demo_dir))
    assert r.returncode == 0 and r.stdout.splitlines()[1] == f"demos: 1 from {demo_dir}, 2 placeholders"
    assert (out / "demos/office-chair/turntable.mp4").is_file()
    r = build()
    assert r.returncode == 1 and r.stderr == (f"error: the earlier build in {out} has demos or placeholders; give "
                                              "--demos DIR, or --no-demos to build without them\n")
    r = build("--demos", str(tmp_path / "missing"))
    assert r.returncode == 1 and r.stderr == (f"error: the demos folder: no demos folder {tmp_path / 'missing'}; "
                                              "nothing was written\n")
    assert (out / "demos/office-chair/turntable.mp4").is_file()
    assert build("--demos", str(demo_dir), "--no-demos").returncode == 2
    r = build("--no-demos")
    assert r.returncode == 0 and r.stdout.splitlines()[1] == "demos: none" and not (out / "demos").exists()


def test_gitignore_keeps_secrets_and_the_built_site_out():
    lines = (ROOT / ".gitignore").read_text().splitlines()
    for want in (".env.*", "!*.env.example", "*.pem", "*.key", "supabase/.temp/", "supabase/.branches/", "/site/"):
        assert want in lines, want
