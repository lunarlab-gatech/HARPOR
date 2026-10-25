"""The public HARPOR site: a static folder whose page talks only to Supabase.

    python -m gator.webapp.site init --url https://<ref>.supabase.co --key sb_publishable_...
    python -m gator.webapp.site build [--out DIR]

init writes gator/webapp/site.json. Its values are public by design: the project's URL and its publishable key.
build writes the folder (site/ at the repo root by default, which git ignores):

  index.html    static/index.html with relative paths, the site's own copy of model-viewer, the Supabase scripts,
                a CSP <meta>, the markup that supa.js draws into (sign-in, account menu, the banner of the user's
                machines, the waiting-for-approval panel, the panel about the machines, the pending line), the
                upload area hidden (jobs get their input on the user's own GPU machine), and no "No agent"
                choice, since the public site runs astra and opus only
  privacy.html  what the site keeps, where, for how long, who can see it, and how API keys travel
  static/       app.js, style.css, supa.js, config.js (from site.json), and the vendor files, each checked
                against its SHA-384 first
  _headers      the same CSP and the other security headers, for Cloudflare Pages
  .nojekyll     so that GitHub Pages serves the folder as it is

Every path in the pages is relative, so the same folder works at a domain root and under /<repo>/. The build
refuses an output that holds a secret key or a JWT. The tunnel mode never uses any of this: the server serves
static/index.html as it is.
"""
import argparse
import base64
import hashlib
import html
import json
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
CONFIG = HERE / "site.json"
OUT = HERE.parents[1] / "site"
# The vendor files under static/vendor, each from the exact npm release (its tarball from registry.npmjs.org,
# checked against the registry's sha512), with the SHA-384 of the file (static/vendor/README.md says where each
# comes from). A file that differs stops the build.
VENDOR = {
    "supabase-2.118.0.js": "TcHRA3zBbn+Lexh5n+Oil4guNPOS3rjNDpGRAN46u/s/mB2IZ50ceC6TYDU2yvgK",
    "model-viewer-4.4.0.min.js": "wjsPzEO+lXOf7d9csIDfAQpPt2OKuUU3m0K9qsdKIuzX4N8czSF6hdh23eAbZa3p",
    "three-0.183.0/draco/draco_wasm_wrapper.js": "mpx8Y6zmHZEVxr3r2cZhLFOKjJscyKOeHFqkK/LrOh8wC3HGDgGcy6fB37yQntEz",
    "three-0.183.0/draco/draco_decoder.wasm": "gJ9GFHPLIX0iGgdYbwba9tNN3lQhBPDtvr64s9owC5NUXLC9oXiSxHT3JqpqgqEY",
    "three-0.183.0/basis/basis_transcoder.js": "kWnnj9LTiulmvzVhfRTdAs7wcVH7SngxPFtXiPFbqGDg72LOftkS9HBZIrtDIp8C",
    "three-0.183.0/basis/basis_transcoder.wasm": "gvjbljHqyqHvdVmQwHnnsrCiXMtLTfoccEK9W5TwWJI99ZjAD+W8XubPeuqTa6xh",
    "licenses/supabase-js-2.118.0.txt": "1OoeMQHJk9w9LYfOTwEiTwvWY9LxP7/4eNz3eO4qdWKnTc7vouDoy6mMkhxBz4oo",
    "licenses/model-viewer-4.4.0.txt": "II9e1ieUDl5AxyiVq3/FflTua1Sr0kMJ25e6imG7rXg7SiAsA2VemsvEqVsLqM7/",
    "licenses/three-0.183.0.txt": "dZIEXnc8kDiJUEWlKdsN+V7yvZT/HbLEL7IFgYWNSRkA0ErUhU5jIL71XpYCHvDY",
    "licenses/three-0.183.0-draco.md": "KrJ5ZX6UJUNT15T9bLvuaAWtjQCTu1N6y3fEdsKO2i75CnsfhVZG5jDaXbTWGZUd",
    "licenses/three-0.183.0-basis.md": "I3z9ZjiXHW6PcNj118aDKY36ILvA3bhEiLV0O2/8J8+CuYUlejKeWauyO/ah61Vt",
}
# The CSP hash of the one <style> block in model-viewer-4.4.0.min.js: the styles of its shadow DOM, which a policy
# without unsafe-inline blocks otherwise (the browser names this hash). The file is pinned above, so the text is too;
# a test recomputes the hash from the file.
MODEL_VIEWER_STYLE = "'sha256-PXtZZkJMMyc18o/DN1ZEFOJ2qEGq95IQbVaJFMOXOb0='"
# A Supabase secret key, or a JWT (its header and payload are base64url JSON, so both start with eyJ). supabase-js
# itself holds the bare prefix "sb_secret_" (it checks keys), which is no key.
SECRET = re.compile(rb"sb_secret_[A-Za-z0-9_-]{8}|eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.")
# site.json's url: the Supabase API (https://<ref>.supabase.co, or the CLI's local stack on 127.0.0.1).
URL = re.compile(r"https://[a-z0-9]{8,40}\.supabase\.co|http://(127\.0\.0\.1|localhost):\d{2,5}")
KEYS = {"url", "publishableKey", "contact", "_comment"}
MODEL_VIEWER = ('<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.4.0/'
                'model-viewer.min.js"></script>')
SCRIPTS = ("static/vendor/supabase-2.118.0.js", "static/config.js", "static/supa.js", "static/app.js")

# ---- the markup that supa.js draws into, added to static/index.html ----

ACCOUNT = """
  <details id="harpor-account" class="account" hidden>
    <summary id="harpor-email">Account</summary>
    <div class="accountmenu">
      <p id="harpor-storage" class="hint"></p>
      <p><a href="privacy.html">Privacy</a></p>
      <p><button id="harpor-signout" class="small secondary" type="button">Sign out</button></p>
    </div>
  </details>"""
BANNER = """
<div id="harpor-banner" class="banner" role="status" hidden></div>"""
GATE = """

<section id="harpor-signin" class="panel gate" hidden>
  <h2>Sign in to HARPOR</h2>
  <p>HARPOR turns a phone video, a rosbag or a Stray Scanner capture of one object into simulation-ready models: its
    meshes, its parts and joints, and its mass and materials.</p>
  <p class="gatebtns"><button id="harpor-google" type="button">Sign in with Google</button>
    <button id="harpor-github" type="button">Sign in with GitHub</button></p>
  <p id="harpor-signinmsg" class="msg" role="status"></p>
  <p class="hint">The site's owner approves each new account by hand. Jobs run on your own GPU machine, and this site
    shows them. Their state, files and logs are kept in this site's Supabase project.
    <a href="privacy.html">Privacy and where your data lives</a>.</p>
</section>

<section id="harpor-waiting" class="panel gate" hidden>
  <h2>Your account is waiting for approval</h2>
  <p>The site's owner approves each account by hand. Until then your machines cannot show jobs here.</p>
  <form id="harpor-request" class="request">
    <label for="harpor-note">Request access: say who you are and what you want to build.</label>
    <textarea id="harpor-note" rows="3" maxlength="500" required></textarea>
    <p><button id="harpor-send" type="submit">Send the request</button>
      <span id="harpor-requestmsg" class="msg" role="status"></span></p>
  </form>
</section>"""
OWN = """<section id="harpor-own" class="panel own" hidden>
    <h3>Your GPU machines</h3>
    <p id="harpor-ownnote"></p>
    <ul id="harpor-machines" class="machines"></ul>
    <details id="harpor-howto" class="sub">
      <summary>How to run jobs on your own GPU machine</summary>
      <ol class="steps">
        <li>Set up HARPOR on the machine, with its environments, containers and model checkouts. Then add the package
          that opens sealed keys to its web environment: <code>pip install cryptography==50.0.2</code>.</li>
        <li>Sign in there: <code>python -m gator.webapp.worker login</code>. On a machine without a browser, forward
          the port first with <code>ssh -L 8767:127.0.0.1:8767 &lt;machine&gt;</code>, add
          <code>--no-browser</code>, and open the address that it prints on your own computer.</li>
        <li>Start HARPOR there: <code>python -m gator.webapp --worker</code>. Add
          <code>--local-keys opus</code> to use the Anthropic key in <code>~/.config/gator/anthropic.env</code> on
          the machine, or <code>--local-keys astra</code> for the OpenAI key in <code>openai.env</code>. This page
          then asks for no key for the machine's jobs.</li>
        <li>Add a file in the machine's own HARPOR page, or with
          <code>python -m gator.webapp.worker add PATH</code>. Its job shows here within a few seconds, and the file
          stays on the machine.</li>
      </ol>
    </details>
  </section>
  """
# The upload area's message line, out of the hidden upload area: app.js says there why a delete or the list failed.
UPMSG = """
    <p id="upmsg" class="msg" role="status"></p>"""
PENDING = """
    <div id="harpor-pending" class="pending" role="status" hidden></div>"""

# ---- privacy.html ----

PRIVACY = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
{meta}
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>HARPOR privacy</title>
<meta name="description" content="What the HARPOR site keeps, where it keeps it, and who can see it">
<link rel="stylesheet" href="static/style.css">
</head>
<body>
<header>
  <a href="index.html" class="brand">HARPOR
    <span class="brandsub">High-fidelity Articulated sim-Ready Object Reconstruction</span></a>
</header>
<main class="privacy">
<div class="panel">
<h2>Privacy and where your data lives</h2>
<p>This page says what the HARPOR site keeps, where it keeps it, for how long, and who can see it.</p>

<h3>Your account</h3>
<p>You sign in with Google or GitHub. The site gets your email address and the name on that account. It keeps no
  password. The site's owner approves each account by hand, and sees your email address and the note of a request for
  access.</p>

<h3>Where your data lives</h3>
<p>Your jobs run on your own GPU machine. Your input file stays on that machine and never comes to this site.</p>
<p>So that you can see your jobs here, HARPOR on your machine copies each job's state, the logs of its stages, its
  review images and its result files to this site's Supabase project. The project also keeps your account and the
  requests that you send to your machine. Supabase stores all of this for the site's owner, who runs that project and
  can read it.</p>

<h3>How long it is kept</h3>
<p>A request that your machine does not take within 7 days goes. The log text of a job goes 14 days after the job
  stops. HARPOR on your machine deletes a job's review images and log files from the site 14 days after the job stops,
  and its result files after 90 days. It does this while it runs, so files can stay longer while your machine is off.
  Delete a job on the jobs page to remove its files from the site at once. The job's folder on your machine stays yours
  to delete. If the site's owner takes back your approval, this page no longer lists your jobs; then ask the owner to
  delete them.</p>

<h3>Your API keys</h3>
<p>Each job uses your own key: an OpenAI key for GPT-6-Astra, or an Anthropic key for Claude Opus 5.5. Your provider
  bills you for it.</p>
<p>Your browser seals the key to the public key of the HARPOR worker on your machine. Only that worker can open the
  sealed key, so Supabase never holds it in a form that can be read. The worker checks the key with your provider and
  keeps it in memory while the job needs it, at most 24 hours while the job waits for your review. You can also keep
  the key in a file on your machine; then this page asks for none.</p>
<p>Your browser trusts the public key that this site's database lists for your machine. Someone who can change that
  database, such as the site's owner, could list another key and read the keys that you send after that. If this
  matters to you, keep the key in a file on your machine. Use a key with a spending limit, and delete it at your
  provider when you no longer need it.</p>

<h3>GPT-6-Astra and the network</h3>
<p>GPT-6-Astra runs the Codex CLI in a container that can reach the network. Text in your video or images, such as a
  sign or a label, could steer it to run commands that reach the network. These commands run on your machine. Claude
  Opus 5.5 runs its shell with no network access. If this matters to you, choose Opus. Each job uses your own API key,
  and your provider bills you for it.</p>

<h3>What your browser keeps</h3>
<p>Your browser keeps your sign-in session and a short list of your requests whose answers you have not read yet. The
  site loads no analytics and no script from another site. Its scripts come from this site, and the page talks only to
  its Supabase project.</p>

<h3>Questions, and deleting your account</h3>
<p>To delete your account and all its jobs, or to ask about your data, write to the site's owner{contact}.</p>
<p><a href="index.html">Back to HARPOR</a></p>
</div>
</main>
</body>
</html>
"""


class SiteError(Exception):
    """Why the site cannot be built or configured, in one line."""


def b64sha384(data):
    return base64.b64encode(hashlib.sha384(data).digest()).decode()


def check_config(cfg):
    """site.json as build uses it; SiteError when a value is missing, unknown, secret or malformed."""
    if not isinstance(cfg, dict):
        raise SiteError("site.json is not a JSON object")
    extra = sorted(set(cfg) - KEYS)
    if extra:  # redirectTo and host are gone: supa.js takes the sign-in address from the page itself
        allowed = ", ".join(sorted(KEYS - {"_comment"}))
        raise SiteError(f"site.json has unknown keys: {', '.join(extra)}; it takes {allowed}")
    if not isinstance(cfg.get("url"), str) or not URL.fullmatch(cfg["url"]):
        raise SiteError("url is https://<ref>.supabase.co, or http://127.0.0.1:<port> for the local stack")
    key = cfg.get("publishableKey")
    if not isinstance(key, str) or not re.fullmatch(r"sb_publishable_[A-Za-z0-9_-]{8,200}", key):
        raise SiteError("publishableKey is the project's publishable key (sb_publishable_...); never the secret key")
    if cfg.get("contact") is not None and (not isinstance(cfg["contact"], str) or not 0 < len(cfg["contact"]) <= 200):
        raise SiteError("contact is how users reach the site's owner, in at most 200 characters")
    if SECRET.search(json.dumps(cfg).encode()):
        raise SiteError("site.json holds a secret key or a JWT; it may hold public values only")
    return cfg


def load_config(path=CONFIG):
    path = Path(path)
    if not path.is_file():
        raise SiteError(f"no {path}: run python -m gator.webapp.site init --url ... --key ... first")
    try:
        cfg = json.loads(path.read_text())
    except ValueError as e:
        raise SiteError(f"{path} is not JSON: {e}") from None
    return check_config(cfg)


def init(url, key, contact=None, path=CONFIG, force=False):
    """Write site.json from the project's URL and publishable key (both public). Returns the config."""
    cfg = {"url": url.rstrip("/"), "publishableKey": key}
    if contact:
        cfg["contact"] = contact
    check_config(cfg)
    path = Path(path)
    if path.exists() and not force:
        raise SiteError(f"{path} exists; add --force to replace it")
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg


def csp(cfg, meta=False):
    """The Content-Security-Policy of every page. Scripts and styles come only from the site itself (app.js sets
    styles through the CSSOM, which the policy allows), plus model-viewer's own shadow-DOM styles by their hash.
    'wasm-unsafe-eval' lets WebAssembly compile, and nothing else: model-viewer compiles its meshopt decoder when it
    loads, and its Draco and KTX2 decoders are WebAssembly too; without it the browser refuses them. The page talks to
    the project's API and its Realtime socket; images and the preview video come from signed Storage URLs. A <meta>
    policy cannot set frame-ancestors (browsers ignore it there and say so), so meta=True leaves it out; supa.js stops
    in a frame."""
    api = urlsplit(cfg["url"])
    origin = f"{api.scheme}://{api.netloc}"
    ws = ("wss" if api.scheme == "https" else "ws") + "://" + api.netloc
    rules = [("default-src", "'none'"), ("script-src", "'self' 'wasm-unsafe-eval'"),
             ("style-src", f"'self' {MODEL_VIEWER_STYLE}"), ("img-src", f"'self' data: blob: {origin}"),
             ("media-src", f"blob: {origin}"), ("connect-src", f"'self' {origin} {ws}"),
             ("worker-src", "'self' blob:"), ("font-src", "'self'"), ("manifest-src", "'self'"),
             ("frame-ancestors", "'none'"), ("base-uri", "'none'"), ("form-action", "'none'")]
    return "; ".join(f"{k} {v}" for k, v in rules if not (meta and k == "frame-ancestors"))


def head_meta(cfg):
    return (f'<meta http-equiv="Content-Security-Policy" content="{html.escape(csp(cfg, meta=True), quote=False)}">\n'
            '<meta name="referrer" content="no-referrer">')


def once(text, old, new):
    """text with old (which must occur exactly once) replaced by new; SiteError names the anchor otherwise, so a
    change in static/index.html stops the build instead of leaving a part out."""
    n = text.count(old)
    if n != 1:
        raise SiteError(f"static/index.html has {n} copies of {old.strip()[:80]!r}, not 1; update site.py")
    return text.replace(old, new)


def public_index(src, cfg):
    """static/index.html made into the public site's index.html."""
    out = once(src, '<meta charset="utf-8">', '<meta charset="utf-8">\n' + head_meta(cfg))
    out = once(out, 'href="/static/style.css"', 'href="static/style.css"')
    out = once(out, MODEL_VIEWER, '<script type="module" src="static/vendor/model-viewer-4.4.0.min.js"></script>')
    out = once(out, '<script src="/static/app.js"></script>',
               "\n".join(f'<script src="{s}"></script>' for s in SCRIPTS))
    none = re.findall(r'\n *<label><input type="radio" name="agent" value="none">[^<]*</label>', out)
    if len(none) != 1:
        raise SiteError("static/index.html has no single \"No agent\" choice to leave out; update site.py")
    out = out.replace(none[0], "")  # the public site runs astra and opus only
    out = once(out, '<span id="conn" class="conn" role="status" hidden></span>',
               '<span id="conn" class="conn" role="status" hidden></span>' + ACCOUNT)
    out = once(out, "</header>", "</header>" + BANNER)
    out = once(out, "<main>", "<main>" + GATE)
    out = once(out, '<div id="drop" class="drop">', '<div id="drop" class="drop" hidden>')
    out = once(out, UPMSG, "")
    out = once(out, "<h2>Jobs</h2>", OWN + UPMSG.strip() + "\n  <h2>Jobs</h2>")
    out = once(out, '<p id="jnext" class="next"></p>', '<p id="jnext" class="next"></p>' + PENDING)
    if re.search(r'(?:src|href)="(?:/|https?:)', out):
        raise SiteError("the public index.html still has an absolute path or another host; update site.py")
    return out


def config_js(cfg):
    public = {k: cfg[k] for k in ("url", "publishableKey")}
    return ("// Written by gator/webapp/site.py from gator/webapp/site.json. Public values only.\n"
            f"window.HARPOR_CONFIG = {json.dumps(public, indent=2)};\n")


def headers_file(cfg):
    return ("# Response headers for Cloudflare Pages, written by gator/webapp/site.py. GitHub Pages ignores this\n"
            "# file; the pages carry the same policy in a <meta> tag, without frame-ancestors, which a <meta> cannot\n"
            "# set.\n"
            "/*\n"
            f"  Content-Security-Policy: {csp(cfg)}\n"
            "  Referrer-Policy: no-referrer\n"
            "  Strict-Transport-Security: max-age=31536000\n"
            "  X-Content-Type-Options: nosniff\n"
            "  Permissions-Policy: camera=(), microphone=(), geolocation=()\n"
            "\n"
            "# The vendor files have their version in their name, so they never change.\n"
            "/static/vendor/*\n"
            "  Cache-Control: public, max-age=31536000, immutable\n")


def vendor_files(static=STATIC):
    """{path under static/vendor: bytes}, each checked against VENDOR."""
    out = {}
    for rel, want in VENDOR.items():
        p = Path(static) / "vendor" / rel
        if not p.is_file():
            raise SiteError(f"static/vendor/{rel} is missing (static/vendor/README.md says where it comes from)")
        data = p.read_bytes()
        if b64sha384(data) != want:
            raise SiteError(f"static/vendor/{rel} does not have its SHA-384; take it again from its npm release")
        out[rel] = data
    return out


def files(cfg, static=STATIC):
    """{path in the site folder: bytes} of the whole site."""
    out = {"index.html": public_index((Path(static) / "index.html").read_text(), cfg).encode(),
           "privacy.html": PRIVACY.format(meta=head_meta(cfg), contact=": " + html.escape(cfg["contact"], quote=False)
                                          if cfg.get("contact") else "").encode(),
           "static/config.js": config_js(cfg).encode(), "_headers": headers_file(cfg).encode(), ".nojekyll": b""}
    for name in ("app.js", "style.css", "supa.js"):
        out[f"static/{name}"] = (Path(static) / name).read_bytes()
    out.update({f"static/vendor/{rel}": data for rel, data in vendor_files(static).items()})
    for path, data in out.items():
        if SECRET.search(data):
            raise SiteError(f"{path} holds a secret key or a JWT; nothing was written")
    return out


def build(out=OUT, config=CONFIG, static=STATIC):
    """Write the site into out. An existing out is replaced only when it is an earlier build (or empty)."""
    out = Path(out)
    site = files(load_config(config), static)
    if out.exists():
        if any(out.iterdir()) and not ((out / ".nojekyll").is_file() and (out / "static" / "config.js").is_file()):
            raise SiteError(f"{out} is not an earlier build of the site; choose another --out")
        shutil.rmtree(out)
    for path, data in site.items():
        p = out / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return sorted(site)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gator.webapp.site", description="Build the public HARPOR site.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init", help="write gator/webapp/site.json")
    a.add_argument("--url", required=True, help="the project URL, https://<ref>.supabase.co")
    a.add_argument("--key", required=True, help="the publishable key, sb_publishable_...")
    a.add_argument("--contact", help="how users reach you, for the privacy page")
    a.add_argument("--force", action="store_true", help="replace an existing site.json")
    a.add_argument("--config", default=str(CONFIG), help=argparse.SUPPRESS)
    b = sub.add_parser("build", help="write the site folder")
    b.add_argument("--out", default=str(OUT), help=f"the folder to write (default {OUT})")
    b.add_argument("--config", default=str(CONFIG), help="site.json")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "init":
            cfg = init(args.url, args.key, args.contact, args.config, args.force)
            print(f"wrote {args.config} for {cfg['url']}; build the site with python -m gator.webapp.site build")
        else:
            written = build(args.out, args.config)
            print(f"wrote {len(written)} files to {args.out}")
    except SiteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
