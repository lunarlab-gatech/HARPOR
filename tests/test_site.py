"""gator/webapp/site.py: the public site's static folder."""
import base64
import functools
import hashlib
import html
import http.server
import json
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
    assert files == sorted([".nojekyll", "_headers", "index.html", "privacy.html", "static/app.js", "static/config.js",
                            "static/style.css", "static/supa.js"] + [f"static/vendor/{v}" for v in site.VENDOR])
    for name in ("app.js", "style.css", "supa.js"):
        assert (built / "static" / name).read_bytes() == (site.STATIC / name).read_bytes()
    for rel in site.VENDOR:
        assert (built / "static/vendor" / rel).read_bytes() == (site.STATIC / "vendor" / rel).read_bytes()
    js = (built / "static/config.js").read_text()
    conf = json.loads(js[js.index("{"):js.rindex("}") + 1])
    assert conf == CFG
    assert (built / ".nojekyll").read_bytes() == b""


def test_csp_is_strict_and_the_same_in_headers_and_meta(built):
    """Scripts and styles from the site only (no CDN host, no unsafe-inline or unsafe-eval; WebAssembly may compile,
    for model-viewer's decoders, and model-viewer's own style block may apply), the project's API and socket only;
    the header policy equals the <meta> policy of every page plus frame-ancestors, which a <meta> cannot set."""
    head = csp_of_headers(built)
    rules = dict(r.split(" ", 1) for r in head.split("; "))
    assert rules["script-src"] == "'self' 'wasm-unsafe-eval'" and rules["style-src"] == "'self' " + site.MODEL_VIEWER_STYLE
    assert rules["default-src"] == "'none'"
    assert rules["connect-src"] == "'self' https://abcdefghijklmnop.supabase.co wss://abcdefghijklmnop.supabase.co"
    assert rules["img-src"] == "'self' data: blob: https://abcdefghijklmnop.supabase.co"
    assert rules["media-src"] == "blob: https://abcdefghijklmnop.supabase.co"
    assert rules["frame-ancestors"] == "'none'" and rules["base-uri"] == "'none'" and rules["form-action"] == "'none'"
    assert "'unsafe-inline'" not in head and "'unsafe-eval'" not in head and not CDN.search(head)
    for page in ("index.html", "privacy.html"):
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
    assert "connect-src 'self' http://127.0.0.1:54321 ws://127.0.0.1:54321;" in site.csp(cfg)
    assert "img-src 'self' data: blob: http://127.0.0.1:54321;" in site.csp(cfg)


def test_public_index_is_the_tunnel_page_made_public(built):
    """Relative paths only, each to a file of the site; the CSP and referrer metas before anything that loads; the
    site's own model-viewer; the Supabase scripts before app.js; the markup that supa.js draws into; the upload area
    hidden, with its message line moved out of it, after the machines panel; no "No agent" choice; no inline script,
    style attribute or event attribute, which the CSP would block."""
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
    for i in ("harpor-account", "harpor-email", "harpor-storage", "harpor-signout", "harpor-banner", "harpor-signin",
              "harpor-google", "harpor-github", "harpor-signinmsg", "harpor-waiting", "harpor-request", "harpor-note",
              "harpor-send", "harpor-requestmsg", "harpor-own", "harpor-ownnote", "harpor-machines", "harpor-howto",
              "harpor-pending", "drop", "file", "upmsg", "jobs", "startform", "keylabel"):
        assert i in ids, i
    assert "harpor-resume" not in ids
    for i in ("harpor-account", "harpor-banner", "harpor-signin", "harpor-waiting", "harpor-own", "harpor-pending", "drop"):
        assert "hidden" in next(a for _, a in page.tags if a.get("id") == i), i
    agents = [a["value"] for t, a in page.tags if t == "input" and a.get("name") == "agent"]
    assert agents == ["astra", "opus"]
    assert page.inline == [] and not any("style" in a or any(k.startswith("on") for k in a) for _, a in page.tags)
    html = (built / "index.html").read_text()
    assert not CDN.search(html) and "No agent" not in html and "tus" not in html.lower().replace("status", "")
    drop = html[html.index('<div id="drop"'):html.index("</div>", html.index('<div id="drop"'))]
    assert 'id="file"' in drop and 'id="upmsg"' not in drop and html.count('id="upmsg"') == 1
    assert html.index('id="harpor-own"') < html.index('id="upmsg"') < html.index("<h2>Jobs</h2>")
    assert "python -m gator.webapp --worker</code>" in html and "--worker own" not in html


def test_tunnel_index_is_unchanged_and_loads_no_supabase_script():
    """site.py derives the public page; the server's own index.html keeps its paths, its "No agent" choice and no
    Supabase script, so the tunnel works as before."""
    html = (site.STATIC / "index.html").read_text()
    assert site.MODEL_VIEWER in html and '<script src="/static/app.js"></script>' in html
    assert 'value="none"' in html and "supa.js" not in html and "config.js" not in html and "harpor-" not in html


def test_privacy_page_says_where_data_lives_and_who_can_see_keys(built):
    """The privacy page says that inputs stay on the user's machine; that state, logs, review images and results are
    copied to the owner's Supabase project, which the owner can read; how long each is kept; that keys go sealed only
    to the user's machine, and what the owner could do with the database; Astra's network access; and how to reach the
    owner. Nothing of a shared GPU is left."""
    page = parse(built / "privacy.html")
    for link in page.links:
        assert not re.match(r"[a-z]+:|/", link) and (built / link).is_file(), link
    text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", (built / "privacy.html").read_text())))
    for want in ("Your input file stays on that machine and never comes to this site.",
                 "copies each job's state, the logs of its stages, its review images and its result files to this "
                 "site's Supabase project",
                 "the site's owner, who runs that project and can read it.",
                 "The log text of a job goes 14 days after the job stops.",
                 "review images and log files from the site 14 days after the job stops, and its result files after "
                 "90 days",
                 "Delete a job on the jobs page to remove its files from the site at once.",
                 "If the site's owner takes back your approval, this page no longer lists your jobs; then ask the "
                 "owner to delete them.",
                 "Your browser seals the key to the public key of the HARPOR worker on your machine.",
                 "Only that worker can open the sealed key",
                 "could list another key and read the keys that you send after that",
                 "keep the key in a file on your machine",
                 "GPT-6-Astra runs the Codex CLI in a container that can reach the network.",
                 "These commands run on your machine.",
                 "Claude Opus 5.5 runs its shell with no network access.",
                 "write to the site's owner: the lab's front desk."):
        assert want in text, want
    for gone in ("Georgia Tech", "shared GPU", "GPU server", "in clear", "upload", "72 hours"):
        assert gone not in text, gone
    assert page.inline == [] and not CDN.search((built / "privacy.html").read_text())


def test_site_served_under_a_repo_path_loads_every_asset(built, tmp_path):
    """Served as GitHub Pages serves a project site (under /HARPOR/), every src and href of both pages, and the
    decoder files that supa.js names, load by their relative URL."""
    www = tmp_path / "www"
    shutil.copytree(built, www / "HARPOR")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(www))
    handler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}/HARPOR/"
        urls = {base, base + "privacy.html"}
        for page in ("index.html", "privacy.html"):
            urls |= {urllib.parse.urljoin(base + page, link.split("#")[0]) for link in parse(built / page).links}
        supa = (site.STATIC / "supa.js").read_text()
        for folder in re.findall(r'new URL\("(static/vendor/[^"]+/)", location\.href\)', supa):
            for f in (site.STATIC / folder.replace("static/", "", 1)).iterdir():
                urls.add(base + folder + f.name)
        assert len(urls) > 12
        for url in sorted(urls):
            with urllib.request.urlopen(url) as r:
                assert r.status == 200, url
    finally:
        httpd.shutdown()
        httpd.server_close()


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
    """site.json takes public values only, in their forms; redirectTo and host are gone (supa.js takes the sign-in
    address from the page), and so are storageUrl, planFileLimit and workerSignKey (the site takes no uploads and has
    no shared worker). The example file names every key and is refused until it is filled in."""
    good = site.check_config(dict(CFG))
    assert good["url"] == CFG["url"]
    for bad, why in [({"redirectTo": "https://x/"}, "unknown keys: redirectTo"), ({"host": "cloudflare"}, "unknown keys"),
                     ({"url": "https://x.example.com"}, "url is"), ({"url": "https://abcdefghijklmnop.supabase.co/rest"}, "url is"),
                     ({"url": "https://abcdefghijklmnop.storage.supabase.co"}, "url is"),
                     ({"storageUrl": "https://abcdefghijklmnop.storage.supabase.co"}, "unknown keys: storageUrl"),
                     ({"planFileLimit": 52428800}, "unknown keys: planFileLimit"),
                     ({"workerSignKey": None}, "unknown keys: workerSignKey"),
                     ({"publishableKey": "sb_publishable_<from the dashboard>"}, "publishableKey"),
                     ({"contact": ""}, "contact"), ({"contact": "x" * 201}, "contact")]:
        with pytest.raises(site.SiteError, match=why):
            site.check_config({**CFG, **bad})
    assert site.check_config({**CFG, "contact": "room 101"})["contact"] == "room 101"
    example = json.loads((site.HERE / "site.example.json").read_text())
    assert set(example) == site.KEYS
    with pytest.raises(site.SiteError):
        site.check_config(example)
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


def test_command_line(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, "-m", "gator.webapp.site", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=60)
    cfg = tmp_path / "s.json"
    r = run("init", "--url", CFG["url"], "--key", CFG["publishableKey"], "--config", str(cfg))
    assert r.returncode == 0 and CFG["publishableKey"] not in r.stdout + r.stderr
    r = run("build", "--config", str(cfg), "--out", str(tmp_path / "out"))
    assert r.returncode == 0 and r.stdout.startswith("wrote ") and (tmp_path / "out/index.html").is_file()
    r = run("init", "--url", CFG["url"], "--key", "sb_secret_0123456789abcdef", "--config", str(tmp_path / "x.json"))
    assert r.returncode == 1 and r.stderr.startswith("error: ") and "sb_secret_0123456789abcdef" not in r.stderr
    assert run("build", "--config", str(tmp_path / "missing.json")).returncode == 1
    assert run("bogus").returncode == 2
    for gone in ("--storage-url", "--plan-file-limit"):  # the site takes no uploads
        assert run("init", "--url", CFG["url"], "--key", CFG["publishableKey"], gone, "1", "--config",
                   str(tmp_path / "y.json")).returncode == 2


def test_gitignore_keeps_secrets_and_the_built_site_out():
    lines = (ROOT / ".gitignore").read_text().splitlines()
    for want in (".env.*", "!*.env.example", "*.pem", "*.key", "supabase/.temp/", "supabase/.branches/", "/site/"):
        assert want in lines, want
