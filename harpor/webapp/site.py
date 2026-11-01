"""The public HARPOR site: a static folder whose page talks only to Supabase.

    python -m harpor.webapp.site init --url https://<ref>.supabase.co --key sb_publishable_...
    python -m harpor.webapp.site build [--out DIR] [--demos DIR | --no-demos]

init writes harpor/webapp/site.json. Its values are public by design: the project's URL and its publishable key.
build writes the folder (site/ at the repo root by default, which git ignores):

  index.html    static/index.html with relative paths, the site's own copy of model-viewer, the Supabase scripts,
                a CSP <meta>, the markup that supa.js and account.js draw into (sign-in, account menu, the banner of
                the user's machines that are offline or stopping, the waiting-for-approval panel, the panel about the
                machines, the pending line), the upload area hidden (jobs get their input on the user's own GPU
                machine), and no "None" agent, since the public site runs astra and opus only
  help.html     static/help.html with relative paths, a CSP <meta>, the account menu with the scripts that fill it
                (ACCOUNT_SCRIPTS), and the public text (HELP_PUBLIC) in place of its local-only blocks: no upload,
                SSH tunnel or "None" agent, and the sections on the account, setting up a machine and the machines'
                status
  privacy.html  what the site keeps, where, for how long, who can see it, and how API keys travel, with the header
                of the other pages and the account menu's scripts
  demos.html    static/demos.html with relative paths, a CSP <meta>, the site's own model-viewer, demos.js, the
                account menu with the scripts that fill it (ACCOUNT_SCRIPTS), and the demos of --demos DIR (default:
                WEBAPP_DEMOS), each checked by demos.load, with the cards of its placeholders.json
                (demos.placeholders); "No demos yet." without a folder.
                A build without a folder refuses an --out whose earlier build has demos or placeholders, unless
                --no-demos: a deploy of it would take the live gallery away
  demos/        each demo's files (demos/<name>/<file>), copied as they are
  static/       app.js, style.css, supa.js, account.js, demos.js, config.js (from site.json), and the vendor files,
                each checked against its SHA-384 first
  _headers      the same CSP and the other security headers, for Cloudflare Pages
  .nojekyll     so that GitHub Pages serves the folder as it is

Every path in the pages is relative, so the same folder works at a domain root and under /<repo>/. The build
refuses an output that holds a secret key or a JWT. The tunnel mode never uses any of this: the server serves
static/index.html and static/help.html as they are, and static/demos.html with the server's own demos folder.
"""
import argparse
import base64
import hashlib
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

from harpor.webapp import demos

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
KEYS = {"url", "publishableKey", "contact", "_comment"}  # _comment: a note for people, which build ignores
MODEL_VIEWER = ('<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.4.0/'
                'model-viewer.min.js"></script>')
# The scripts of the account menu, at the end of help.html, demos.html and privacy.html; the app adds supa.js and
# app.js.
ACCOUNT_SCRIPTS = ("static/vendor/supabase-2.118.0.js", "static/config.js", "static/account.js")
SCRIPTS = ACCOUNT_SCRIPTS + ("static/supa.js", "static/app.js")

# ---- the markup that supa.js and account.js draw into, added to static/index.html (ACCOUNT also to the others) ----

# The account menu opens from a person icon, at the end of the header of every page. A screen reader names it
# "Account: <email>"; the menu's first line shows the email. The SVG takes its stroke and fill from style.css (the CSP
# blocks style attributes).
ACCOUNT = """
  <details id="harpor-account" class="account" hidden>
    <summary title="Account"><svg class="person" viewBox="0 0 20 20" width="20" height="20" aria-hidden="true" focusable="false"><circle cx="10" cy="7" r="3.25"/><path d="M3.75 17c.6-3.3 3.1-5.25 6.25-5.25s5.65 1.95 6.25 5.25"/></svg><span class="sr-only">Account: <span id="harpor-email"></span></span></summary>
    <div class="accountmenu">
      <p id="harpor-who" class="who"></p>
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
  <p>Turn a capture of one object into a simulation-ready model, on your own GPU machine.</p>
  <p><a href="demos.html" target="_blank" rel="noopener">See the demos</a></p>
  <p class="gatebtns"><button id="harpor-google" type="button">Sign in with Google</button>
    <button id="harpor-github" type="button">Sign in with GitHub</button></p>
  <p id="harpor-signinmsg" class="msg" role="status"></p>
  <p class="hint"><a href="privacy.html">Privacy</a></p>
</section>

<section id="harpor-waiting" class="panel gate" hidden>
  <h2>Waiting for approval</h2>
  <form id="harpor-request" class="request">
    <label for="harpor-note">Request access: who you are and what you will build</label>
    <textarea id="harpor-note" rows="3" maxlength="500" required></textarea>
    <p><button id="harpor-send" type="submit">Send request</button>
      <span id="harpor-requestmsg" class="msg" role="status"></span></p>
  </form>
</section>"""
# How to set up a machine is in help.html (HELP_PUBLIC's account block), in a new tab like the header's Help link.
OWN = """<section id="harpor-own" class="panel own" hidden>
    <h3>Your GPU machines</h3>
    <ul id="harpor-machines" class="machines"></ul>
    <p><a href="help.html#machine" target="_blank" rel="noopener">Set up a machine</a></p>
  </section>
  """
# The upload area's message line, out of the hidden upload area: app.js says there why a delete or the list failed.
UPMSG = """
    <p id="upmsg" class="msg" role="status"></p>"""
PENDING = """
    <div id="harpor-pending" class="pending" role="status" hidden></div>"""

# ---- help.html: the public text of each local-only block of static/help.html ----

# static/help.html marks what only the tunnel page has as <!-- local:NAME -->...<!-- /local:NAME -->: the upload, the
# SSH tunnel, the ?token= link and the agent "None". The build puts HELP_PUBLIC[NAME] in each block's place, and stops
# when a name is on one side only. The section ids stay, so the app's Help links work on both sites. toc and account
# are empty on the tunnel page: here they add the sections that only the public site has.
HELP_PUBLIC = {
    "toc": """<li><a href="#account">Account and access</a></li>
    <li><a href="#machine">Set up a machine</a></li>
    <li><a href="#machine-status">Machine status</a></li>""",
    "account": """<section id="account">
<h2>Account and access</h2>
<p>This site shows the HARPOR jobs that run on your own GPU machine. Sign in with Google or GitHub. The person icon at
  the top right opens your account menu: your email, the Storage line, Privacy and Sign out.</p>
<p>The site's owner approves each account by hand. Until then, the page says Waiting for approval, and your machines
  cannot show jobs here. To ask for access, write who you are and what you will build, in 1 to 500 characters, and
  press Send request. You can send one request a day.</p>
<p>The owner also sets which agents your account may use. The Settings tab offers only those.</p>
<p><a href="privacy.html">Privacy</a> says what the site keeps, where it keeps it, and who can see it.</p>
</section>

<section id="machine">
<h2>Set up a machine</h2>
<p>Jobs run on your own GPU machine. To set one up:</p>
<ol>
  <li>Set up HARPOR on the machine, with its environments, containers and model checkouts. Then add the package that
    opens sealed keys to its web environment: <code>pip install cryptography==50.0.2</code>.</li>
  <li>Sign in there: <code>python -m harpor.webapp.worker login</code>. On a machine without a browser, forward the
    port first with <code>ssh -L 8767:127.0.0.1:8767 &lt;machine&gt;</code>, add <code>--no-browser</code>, and open
    the address that it prints on your own computer.</li>
  <li>Start HARPOR there: <code>python -m harpor.webapp --worker</code>. Add <code>--local-keys opus</code> to use the
    Anthropic key in <code>~/.config/harpor/anthropic.env</code> on the machine, or <code>--local-keys astra</code> for
    the OpenAI key in <code>openai.env</code>. This site then asks for no key for the machine's jobs.</li>
  <li>Add jobs on the machine: see <a href="#add-job">Add a job</a>.</li>
</ol>
</section>

<section id="machine-status">
<h2>Machine status</h2>
<p>The machines panel on the jobs list gives the state of each machine. The banner under the header shows only the
  machines that are offline or stopping.</p>
<dl>
  <dt>online</dt>
  <dd>The machine runs HARPOR and takes jobs.</dd>
  <dt>online until 05:35</dt>
  <dd>The machine runs HARPOR until that time.</dd>
  <dt>stopping at 05:35, stopping soon</dt>
  <dd>The machine stops soon and takes no new builds. New builds wait until it runs again.</dd>
  <dt>offline since Oct 24 05:35</dt>
  <dd>The machine stopped then. Its jobs wait until it runs again.</dd>
  <dt>offline</dt>
  <dd>The machine has not run in the last 30 days. Its jobs wait until it runs again.</dd>
</dl>
<p>Has its own Anthropic key, or OpenAI key, means that the machine was started with <code>--local-keys</code>. Its
  jobs need no key from this site.</p>
<p>A request that the machine does not take within 7 days expires.</p>
<p>The Storage line in the account menu, such as Storage: 120 of 2,048 MB, counts the files that your machines copied
  to this site, against the limit of your account. When it is full, new files stay on the machine: delete old jobs to
  make room. <a href="privacy.html">Privacy</a> says how long the files stay.</p>
</section>""",
    "add-job": """<p>Jobs run on your own GPU machine, so you add the file there: in the machine's own HARPOR page, or with
  <code>python -m harpor.webapp.worker add PATH</code>. The job shows on this site within a few seconds, and the file
  stays on the machine. To set up a machine, see <a href="#machine">Set up a machine</a>.</p>""",
    "keys": """<p>The page seals the key so that only the HARPOR worker on the job's machine can open it.
  <a href="privacy.html">Privacy</a> says how keys travel and how long the worker keeps them.</p>
<p>A machine started with <code>--local-keys</code> uses its own key file (see <a href="#machine">Set up a
  machine</a>), and the page asks for no key for its jobs.</p>
<p>When the job's machine is offline or stopping, the page does not send the key, and the line under the status says
  Key not sent.</p>""",
    "agent-none": "",  # the public site runs astra and opus only
    "troubleshooting": """<h3>The site</h3>
<dl>
  <dt>Cannot reach Supabase. Check your connection.</dt>
  <dd>The page cannot reach the site's database. Check your network, then reload the page.</dd>
  <dt>Signed out. Reload to sign in.</dt>
  <dd>Your sign-in ended. Reload the page and sign in again.</dd>
  <dt>The machine went offline. Try again.</dt>
  <dd>The job's machine stopped while the page sent the key. Send the request again: the page then leaves the key out
    (see <a href="#agent">The agent and API keys</a>).</dd>
  <dt>The machine is stopping. Try again.</dt>
  <dd>The machine began to stop while the page sent the key. Send the request again: the page then leaves the key
    out.</dd>
  <dt>The machine restarted. Paste the key again.</dt>
  <dd>The worker on the machine restarted while the page sent the key, so it cannot open a key sealed for the old one.
    Paste the key again and send.</dd>
  <dt>you can send one request a day</dt>
  <dd>You can ask for access once a day. Send the request again tomorrow.</dd>
  <dt>too many requests: at most N an hour</dt>
  <dd>Your account sent too many requests to its machines in the last hour. Wait, then try again.</dd>
  <dt>an earlier request for this job still waits for its machine</dt>
  <dd>The machine has not taken your last request for this job. Wait for it, or withdraw it.</dd>
  <dt>Withdraw</dt>
  <dd>Each request that waits for its machine shows in a line under the status of the job, such as Start, sent 10:42:
    waits for lab-box. Withdraw takes it back while the machine has not taken it.</dd>
  <dt>File not on the site yet.</dt>
  <dd>The machine has not copied the file to the site yet, or the file expired. <a href="privacy.html">Privacy</a>
    says how long files stay.</dd>
</dl>""",
}
# A local block, with the indent before it and the line break after it, which an empty public text leaves out too.
LOCAL_BLOCK = re.compile(r"([ \t]*)<!-- local:([\w-]+) -->.*?<!-- /local:\2 -->(\n?)", re.S)
# Any local-block marker, also one with other spacing that LOCAL_BLOCK does not take.
LOCAL_MARK = re.compile(r"<!--\s*/?\s*local\b")

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
  <a href="index.html" class="brand">HARPOR</a>
  <a id="demolink" href="demos.html">Demos</a>
  <a id="helplink" href="help.html">Help</a>
</header>
<main class="privacy">
<div class="panel">
<h2>Privacy and where your data lives</h2>

<h3>Your account</h3>
<p>You sign in with Google or GitHub. The site gets your email address and the name on that account. It keeps no
  password. The site's owner approves each account by hand, and sees your email address and the note of a request for
  access.</p>

<h3>Where your data lives</h3>
<p>Your jobs run on your own GPU machine. Your input file stays on that machine and never comes to this site.</p>
<p>So that you can see your jobs here, HARPOR on your machine copies each job's state, the logs of its stages, its
  review images, the LiDAR points of its views and its result files to this site's Supabase project. A LiDAR scan can
  show more of the place than the photos do. The project also keeps your account and the requests that you send to
  your machine. Supabase stores all of this for the site's owner, who runs that project and can read it.</p>

<h3>How long it is kept</h3>
<p>A request that your machine does not take within 7 days goes. The log text of a job goes 14 days after the job
  stops. HARPOR on your machine deletes a job's review images, LiDAR points and log files from the site 14 days after
  the job stops, and its result files after 90 days. It does this while it runs, so files can stay longer while your machine is off.
  Delete a job on the jobs page to remove its files from the site at once. The job's folder on your machine stays yours
  to delete. If the site's owner takes back your approval, this page no longer lists your jobs; then ask the owner to
  delete them.</p>

<h3>Your API keys</h3>
<p>Your browser seals the key to the public key of the HARPOR worker on your machine. Only that worker can open the
  sealed key, so Supabase never holds it in a form that can be read. The worker checks the key with your provider and
  keeps it in memory while the job needs it, at most 24 hours while the job waits for your review.</p>
<p>Your browser trusts the public key that this site's database lists for your machine. Someone who can change that
  database, such as the site's owner, could list another key and read the keys that you send after that. If this
  matters to you, keep the key in a file on your machine.</p>
<p>Choosing an agent: see <a href="help.html#agent">Help</a>.</p>

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
    if extra:
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
        raise SiteError(f"no {path}: run python -m harpor.webapp.site init --url ... --key ... first")
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
    the project's API and its Realtime socket; images and the videos come from signed Storage URLs, and the demos'
    videos from the site itself. blob: in connect-src lets model-viewer read a GLB's own textures, which it fetches
    from blob: URLs that the page made; without it a textured model shows white. A <meta>
    policy cannot set frame-ancestors (browsers ignore it there and say so), so meta=True leaves it out; supa.js stops
    in a frame."""
    api = urlsplit(cfg["url"])
    origin = f"{api.scheme}://{api.netloc}"
    ws = ("wss" if api.scheme == "https" else "ws") + "://" + api.netloc
    rules = [("default-src", "'none'"), ("script-src", "'self' 'wasm-unsafe-eval'"),
             ("style-src", f"'self' {MODEL_VIEWER_STYLE}"), ("img-src", f"'self' data: blob: {origin}"),
             ("media-src", f"'self' blob: {origin}"), ("connect-src", f"'self' blob: {origin} {ws}"),
             ("worker-src", "'self' blob:"), ("font-src", "'self'"), ("manifest-src", "'self'"),
             ("frame-ancestors", "'none'"), ("base-uri", "'none'"), ("form-action", "'none'")]
    return "; ".join(f"{k} {v}" for k, v in rules if not (meta and k == "frame-ancestors"))


def head_meta(cfg):
    return (f'<meta http-equiv="Content-Security-Policy" content="{html.escape(csp(cfg, meta=True), quote=False)}">\n'
            '<meta name="referrer" content="no-referrer">')


def once(text, old, new, page="static/index.html"):
    """text with old (which must occur exactly once) replaced by new; SiteError names the anchor otherwise, so a
    change in the page stops the build instead of leaving a part out."""
    n = text.count(old)
    if n != 1:
        raise SiteError(f"{page} has {n} copies of {old.strip()[:80]!r}, not 1; update site.py")
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
        raise SiteError("static/index.html has no single \"None\" agent to leave out; update site.py")
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


def account_menu(out, page):
    """help.html, demos.html or privacy.html with the account menu at the end of its header and its scripts at the end
    of the body."""
    out = once(out, "\n</header>", ACCOUNT + "\n</header>", page)
    scripts = "".join(f'\n<script src="{s}"></script>' for s in ACCOUNT_SCRIPTS)
    return once(out, "\n</body>", scripts + "\n</body>", page)


def public_help(src, cfg):
    """static/help.html made into the public site's help.html: the account menu, and the public text of each
    local-only block."""
    page = "static/help.html"
    out = once(src, '<meta charset="utf-8">', '<meta charset="utf-8">\n' + head_meta(cfg), page)
    out = once(out, 'href="/static/style.css"', 'href="static/style.css"', page)
    out = once(out, 'href="/"', 'href="index.html"', page)  # the brand link
    out = account_menu(out, page)
    names = [m[2] for m in LOCAL_BLOCK.finditer(out)]
    if sorted(names) != sorted(HELP_PUBLIC):
        raise SiteError(f"{page} has the local blocks {', '.join(names) or 'none'}, not one of each name in "
                        "HELP_PUBLIC; update site.py")
    # Two markers per block and no other: a stray marker (or one that LOCAL_BLOCK cannot read) would ship local text.
    if len(LOCAL_MARK.findall(out)) != 2 * len(names):
        raise SiteError(f"{page} has a local-block marker without its partner; update site.py")
    out = LOCAL_BLOCK.sub(lambda m: HELP_PUBLIC[m[2]] and m[1] + HELP_PUBLIC[m[2]] + m[3], out)
    if re.search(r'(?:src|href)="(?:/|https?:)', out):
        raise SiteError("the public help.html still has an absolute path or another host; update site.py")
    return out


def public_demos(src, cfg, shown, soon=()):
    """static/demos.html made into the public site's demos.html, with the demos shown and the placeholders' cards."""
    page = "static/demos.html"
    out = once(src, '<meta charset="utf-8">', '<meta charset="utf-8">\n' + head_meta(cfg), page)
    out = once(out, 'href="/static/style.css"', 'href="static/style.css"', page)
    out = once(out, 'href="/"', 'href="index.html"', page)  # the brand link
    out = once(out, MODEL_VIEWER, '<script type="module" src="static/vendor/model-viewer-4.4.0.min.js"></script>', page)
    out = once(out, 'src="/static/demos.js"', 'src="static/demos.js"', page)
    out = account_menu(out, page)
    try:
        out = demos.page(out, shown, soon)
    except demos.DemoError as e:
        raise SiteError(str(e)) from None
    if re.search(r'(?:src|href)="(?:/|https?:)', out):
        raise SiteError("the public demos.html still has an absolute path or another host; update site.py")
    return out


def public_privacy(cfg):
    """privacy.html, with the account menu."""
    contact = ": " + html.escape(cfg["contact"], quote=False) if cfg.get("contact") else ""
    return account_menu(PRIVACY.format(meta=head_meta(cfg), contact=contact), "privacy.html")


def config_js(cfg):
    public = {k: cfg[k] for k in ("url", "publishableKey")}
    return ("// Written by harpor/webapp/site.py from harpor/webapp/site.json. Public values only.\n"
            f"window.HARPOR_CONFIG = {json.dumps(public, indent=2)};\n")


def headers_file(cfg):
    return ("# Response headers for Cloudflare Pages, written by harpor/webapp/site.py. GitHub Pages ignores this\n"
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


def files(cfg, static=STATIC, demo_dir=None):
    """{path in the site folder: bytes} of the whole site, with the demos of demo_dir."""
    try:
        shown = demos.load(demo_dir)
        soon = demos.placeholders(demo_dir, shown)
    except demos.DemoError as e:
        raise SiteError(f"the demos folder: {e}; nothing was written") from None
    out = {"index.html": public_index((Path(static) / "index.html").read_text(), cfg).encode(),
           "help.html": public_help((Path(static) / "help.html").read_text(), cfg).encode(),
           "privacy.html": public_privacy(cfg).encode(),
           "demos.html": public_demos((Path(static) / "demos.html").read_text(), cfg, shown, soon).encode(),
           "static/config.js": config_js(cfg).encode(), "_headers": headers_file(cfg).encode(), ".nojekyll": b""}
    out.update({rel: p.read_bytes() for rel, p in demos.files(demo_dir, shown).items()})
    for name in ("app.js", "style.css", "supa.js", "account.js", "demos.js"):
        out[f"static/{name}"] = (Path(static) / name).read_bytes()
    out.update({f"static/vendor/{rel}": data for rel, data in vendor_files(static).items()})
    for path, data in out.items():
        if SECRET.search(data):
            raise SiteError(f"{path} holds a secret key or a JWT; nothing was written")
    return out


def build(out=OUT, config=CONFIG, static=STATIC, demo_dir=None, no_demos=False):
    """Write the site into out, with the demos of demo_dir. An existing out is replaced only when it is an earlier
    build (or empty), and one with demos or placeholders' cards only when demo_dir is given or no_demos is true."""
    out = Path(out)
    site = files(load_config(config), static, demo_dir)
    if out.exists():
        if any(out.iterdir()) and not ((out / ".nojekyll").is_file() and (out / "static" / "config.js").is_file()):
            raise SiteError(f"{out} is not an earlier build of the site; choose another --out")
        page = out / "demos.html"
        cards = page.is_file() and 'class="democard' in page.read_text(errors="replace")
        if demo_dir is None and not no_demos and ((out / "demos").is_dir() or cards):
            raise SiteError(f"the earlier build in {out} has demos or placeholders; give --demos DIR, or --no-demos to "
                            "build without them")
        shutil.rmtree(out)
    for path, data in site.items():
        p = out / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return sorted(site)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m harpor.webapp.site", description="Build the public HARPOR site.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init", help="write harpor/webapp/site.json")
    a.add_argument("--url", required=True, help="the project URL, https://<ref>.supabase.co")
    a.add_argument("--key", required=True, help="the publishable key, sb_publishable_...")
    a.add_argument("--contact", help="how users reach you, for the privacy page")
    a.add_argument("--force", action="store_true", help="replace an existing site.json")
    a.add_argument("--config", default=str(CONFIG), help=argparse.SUPPRESS)
    b = sub.add_parser("build", help="write the site folder")
    b.add_argument("--out", default=str(OUT), help=f"the folder to write (default {OUT})")
    b.add_argument("--config", default=str(CONFIG), help="site.json")
    d = b.add_mutually_exclusive_group()
    d.add_argument("--demos", metavar="DIR", default=os.environ.get("WEBAPP_DEMOS"),
                   help="the demos folder (default: WEBAPP_DEMOS)")
    d.add_argument("--no-demos", action="store_true", help="build without demos, also over a build that had them")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "init":
            cfg = init(args.url, args.key, args.contact, args.config, args.force)
            print(f"wrote {args.config} for {cfg['url']}; build the site with python -m harpor.webapp.site build")
        else:
            folder = Path(args.demos).expanduser() if args.demos and not args.no_demos else None
            written = build(args.out, args.config, demo_dir=folder, no_demos=args.no_demos)
            print(f"wrote {len(written)} files to {args.out}")
            # A build without the folder publishes an empty gallery, so the line says where the demos came from.
            n = len({p.split("/")[1] for p in written if p.startswith("demos/")})
            soon = (Path(args.out) / "demos.html").read_text().count('class="democard soon"')
            more = f", {soon} placeholder{'s' if soon != 1 else ''}" if soon else ""
            print(f"demos: {n} from {folder}{more}" if folder else "demos: none" if args.no_demos
                  else "demos: none (give --demos DIR or set WEBAPP_DEMOS)")
    except SiteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
