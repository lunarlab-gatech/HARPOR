"""static/supa.js (the public site's transport), static/account.js (the account menu) and app.js's seam, in node with
a stand-in DOM and a stand-in supabase client. No network, no Supabase project."""
import base64
import json
import re
import shutil
import subprocess
import textwrap
import time

import pytest

from harpor.webapp import jobs, server, steps

SUPA = server.STATIC / "supa.js"
ACCOUNT = server.STATIC / "account.js"
APP = server.STATIC / "app.js"
CONFIG = {"url": "https://abcdefghijklmnop.supabase.co", "publishableKey": "sb_publishable_test"}
USER = "11111111-1111-4111-8111-111111111111"
JOB, JOB2, JOB3 = ("aaaaaaaa-0000-4000-8000-000000000001", "aaaaaaaa-0000-4000-8000-000000000002",
                   "aaaaaaaa-0000-4000-8000-000000000003")
W1, W2, W3 = ("bbbbbbbb-0000-4000-8000-000000000001", "bbbbbbbb-0000-4000-8000-000000000002",
              "bbbbbbbb-0000-4000-8000-000000000003")
MACHINE, MACHINE2 = "cccccccc-0000-4000-8000-000000000001", "cccccccc-0000-4000-8000-000000000002"
MACHINES = ((MACHINE, "lab-box"), (MACHINE2, "desk-pc"))

# The page in node: a small DOM in which every selector gives one element that keeps what the page sets on it (its
# attributes, children, text, hidden, style, value, checked, disabled and handlers); querySelectorAll answers from
# `lists`. A fake clock drives setTimeout, setInterval and Date.now (advance(ms) runs what is due). fetch answers from
# `net` ({url: [status, body]}) and records `fetched`. localStorage is a Map (`store`; `storeBroken` makes it throw).
# location is a URL that history.replaceState and hash assignments change; location.assign goes to `assigned`.
# The stand-in supabase client works on `db`: tables of rows, rpc handlers by name (job_objects lists a job's Storage
# objects unless a test sets its own), Storage objects that can be signed and removed (db.fail.remove makes a remove
# fail), Realtime channels that a test feeds with emit(table, event, row), and an auth session. Every call goes to
# `calls`. The test's setup code runs before account.js, supa.js and app.js load, its code after; the code prints JSON
# last. settle(promise) advances the clock until the promise settles (a relayed request waits on timers); loaded()
# fires DOMContentLoaded, which a browser fires once it has run the page's scripts.
HARNESS = r"""
const fs = require("fs"), vm = require("vm"), { webcrypto } = require("crypto");
const [accountPath, supaPath, appPath, optsJson, setupCode, testCode] = process.argv.slice(2);
const opts = JSON.parse(optsJson);
const ctx = vm.createContext({ console, URL, URLSearchParams, TextEncoder, TextDecoder, Response, Headers, Blob, atob, btoa,
  crypto: webcrypto, realImmediate: setImmediate, opts });
vm.runInContext(String.raw`
var window = globalThis, self = globalThis;
globalThis.top = opts.framed ? {} : globalThis;
const made = [], byId = {}, lists = {}, docListeners = [], winListeners = [], fetched = [], assigned = [], net = {};
let clock = Date.now(), timerSeq = 0, reloads = 0;
const timers = [];
Date.now = () => clock;
globalThis.setTimeout = (fn, ms) => { timers.push({ id: ++timerSeq, fn, due: clock + (ms || 0), live: true }); return timerSeq; };
globalThis.setInterval = (fn, ms) => { timers.push({ id: ++timerSeq, fn, due: clock + ms, every: ms, live: true }); return timerSeq; };
globalThis.clearTimeout = globalThis.clearInterval = (id) => { const t = timers.find((x) => x.id === id); if (t) t.live = false; };
const flush = async () => { for (let i = 0; i < 20; i++) await new Promise((r) => realImmediate(r)); };
async function advance(ms) {
  const end = clock + ms;
  for (;;) {
    await flush();
    const due = timers.filter((t) => t.live && t.due <= end).sort((a, b) => a.due - b.due || a.id - b.id)[0];
    if (!due) break;
    clock = Math.max(clock, due.due);
    if (due.every) due.due += due.every; else due.live = false;
    due.fn();
  }
  clock = end;
  await flush();
}
class Elem {
  constructor(tag, sel) {
    Object.assign(this, { tag, sel, attrs: {}, kids: [], parent: null, style: {}, cls: new Set(), dataset: {}, listeners: {},
      value: "", checked: false, disabled: false, open: false });
    made.push(this);
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  get hidden() { return "hidden" in this.attrs; }
  set hidden(v) { if (v) this.attrs.hidden = ""; else delete this.attrs.hidden; }
  get className() { return [...this.cls].join(" "); }
  set className(v) { this.cls = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get classList() {
    const c = this.cls;
    return { add: (x) => c.add(x), remove: (x) => c.delete(x), contains: (x) => c.has(x),
      toggle: (x, on) => { on = on === undefined ? !c.has(x) : !!on; if (on) c.add(x); else c.delete(x); return on; } };
  }
  append(...k) { for (const x of k) { if (x && typeof x === "object") x.parent = this; this.kids.push(x); } }
  prepend(...k) { for (const x of k) if (x && typeof x === "object") x.parent = this; this.kids.unshift(...k); }
  replaceChildren(...k) { this.kids = []; this.append(...k); }
  replaceWith(x) { if (this.parent) { const i = this.parent.kids.indexOf(this); this.parent.kids[i] = x; x.parent = this.parent; } }
  remove() { if (this.parent) this.parent.kids = this.parent.kids.filter((x) => x !== this); this.parent = null; }
  contains(x) { for (let e = x; e; e = e.parent) if (e === this) return true; return false; }
  closest(sel) {
    for (let e = this; e; e = e.parent) {
      const m = /^([a-z]+)?(\[([a-z-]+)\])?$/.exec(sel);
      if (m && (!m[1] || e.tag === m[1]) && (!m[3] || m[3] in e.attrs)) return e;
    }
    return null;
  }
  get parentElement() { return this.parent; }
  get textContent() { return this.kids.map((k) => typeof k === "string" ? k : k.textContent).join(""); }
  set textContent(v) { this.kids = [String(v)]; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  focus() {}
  scrollIntoView() {}
  pause() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: 0, height: 0, right: 0, bottom: 0 }; }
  querySelectorAll() { return []; }
}
const text = (e) => typeof e === "string" ? e : (e && e.kids || []).map(text).join("");
const walk = (e, out = []) => { if (e && typeof e === "object") { out.push(e); e.kids.forEach((k) => walk(k, out)); } return out; };
const radio = (name, value, checked) => {
  const label = new Elem("label"), r = new Elem("input");
  Object.assign(r, { name, value, checked: !!checked });
  label.append(r);
  return r;
};
lists["input[name=agent]"] = (opts.agents || ["astra", "opus"]).map((a, i) => radio("agent", a, i === 0));
lists["input[name=mode]"] = [radio("mode", "auto", true), radio("mode", "manual")];
lists["input[name=partsmode]"] = ["vlm", "vlm_edit", "user"].map((v, i) => radio("partsmode", v, i === 0));
const document = globalThis.document = {
  title: "", activeElement: null, body: new Elem("body"), documentElement: new Elem("html"),
  querySelector: (sel) => sel === "input[name=mode]:checked" ? lists["input[name=mode]"].find((r) => r.checked)
    : byId[sel] || (byId[sel] = new Elem("div", sel)),
  querySelectorAll: (sel) => lists[sel] || [],
  createElement: (tag) => new Elem(tag), createElementNS: (ns, tag) => new Elem(tag),
  addEventListener: (type, fn, capture) => docListeners.push({ type, fn, capture }),
};
// What index.html and site.py's markup hide at first.
for (const id of ["#list", "#job", "#harpor-account", "#harpor-banner", "#harpor-signin", "#harpor-waiting", "#harpor-own",
  "#harpor-pending"]) document.querySelector(id).hidden = true;
// A promise that needs the clock (a relayed request's wait): advance until it settles.
async function settle(p, ms = 200000) {
  let done = false;
  p.then(() => { done = true; }, () => { done = true; });
  for (let t = 0; t < ms && !done; t += 250) await advance(250);
  return p;
}
const loaded = () => docListeners.filter((l) => l.type === "DOMContentLoaded").forEach((l) => l.fn());
globalThis.addEventListener = (type, fn) => winListeners.push({ type, fn });
globalThis.matchMedia = () => ({ matches: false });
globalThis.confirm = () => true;
globalThis.innerWidth = 1280;
globalThis.innerHeight = 800;
class Loc {
  constructor(href) { this.u = new URL(href); }
  get href() { return this.u.href; }
  get origin() { return this.u.origin; }
  get pathname() { return this.u.pathname; }
  get search() { return this.u.search; }
  get hash() { return this.u.hash; }
  set hash(v) { this.u.hash = v; }
  assign(u) { assigned.push(u); }
  reload() { reloads++; }
}
globalThis.location = new Loc(opts.href || "https://x.pages.dev/");
globalThis.history = { state: null, replaceState(state, title, url) { this.state = state; location.u = new URL(url, location.href); } };
const store = new Map();
let storeBroken = false;
globalThis.localStorage = {
  getItem: (k) => { if (storeBroken) throw new Error("blocked"); return store.has(k) ? store.get(k) : null; },
  setItem: (k, v) => { if (storeBroken) throw new Error("blocked"); store.set(k, String(v)); },
  removeItem: (k) => { if (storeBroken) throw new Error("blocked"); store.delete(k); },
};
globalThis.fetch = async (url, o) => {
  fetched.push({ url, method: (o && o.method) || "GET" });
  if (!(url in net)) throw new TypeError("Failed to fetch");
  const [status, body] = net[url];
  return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
};
const MV = {};
globalThis.customElements = { whenDefined: () => Promise.resolve(), get: () => MV };

// ---- the stand-in supabase client ----
const db = { tables: {}, rpc: {}, objects: new Set(), session: null, channels: [], realtime: "SUBSCRIBED", offline: false,
  fail: {} };
const calls = [];
let signSeq = 0;
class Query {
  constructor(table) { Object.assign(this, { table, cols: "*", filters: [], ord: null, lim: null, one: false }); }
  select(cols) { this.cols = cols; return this; }
  eq(c, v) { this.filters.push(["eq", c, v]); return this; }
  in(c, v) { this.filters.push(["in", c, v]); return this; }
  gt(c, v) { this.filters.push(["gt", c, v]); return this; }
  is(c, v) { this.filters.push(["is", c, v]); return this; }
  order(c, o) { this.ord = [c, o && o.ascending === false ? -1 : 1]; return this; }
  limit(n) { this.lim = n; return this; }
  maybeSingle() { this.one = true; return this; }
  single() { this.one = true; return this; }
  then(res, rej) { return Promise.resolve().then(() => this.run()).then(res, rej); }
  run() {
    calls.push({ table: this.table, cols: this.cols, filters: this.filters, ord: this.ord, lim: this.lim });
    if (db.offline) return { data: null, error: { message: "TypeError: Failed to fetch" }, status: 0 };
    if (db.fail[this.table]) return { data: null, error: { message: db.fail[this.table][1] }, status: db.fail[this.table][0] };
    const cols = this.cols.split(",");
    let rows = (db.tables[this.table] || []).filter((r) => this.filters.every(([op, c, v]) =>
      op === "eq" ? r[c] === v : op === "in" ? v.includes(r[c]) : op === "gt" ? r[c] > v : op === "is" ? r[c] === v : true));
    if (this.ord) rows = [...rows].sort((a, b) => (a[this.ord[0]] > b[this.ord[0]] ? 1 : a[this.ord[0]] < b[this.ord[0]] ? -1 : 0) * this.ord[1]);
    if (this.lim != null) rows = rows.slice(0, this.lim);
    if (!cols.includes("*")) rows = rows.map((r) => Object.fromEntries(cols.map((c) => [c, r[c] === undefined ? null : r[c]])));
    return { data: this.one ? rows[0] || null : rows, error: null, status: 200 };
  }
}
const client = {
  from: (t) => new Query(t),
  rpc: async (name, args) => {
    calls.push({ rpc: name, args: JSON.parse(JSON.stringify(args || {})) });
    if (db.offline) return { data: null, error: { message: "TypeError: Failed to fetch" }, status: 0 };
    const fn = db.rpc[name] || stock[name];
    if (!fn) return { data: null, error: { message: "no handler for " + name }, status: 404 };
    const out = await fn(args || {});
    return out && out.error ? { data: null, error: { message: out.error }, status: 400 } : { data: out ? out.data : null, error: null, status: 200 };
  },
  storage: { from: (bucket) => ({
    createSignedUrls: async (paths, seconds) => {
      calls.push({ sign: bucket, paths: [...paths], seconds });
      return { data: paths.map((p) => ({ path: p, signedUrl: db.objects.has(p) ? "https://abcdefghijklmnop.supabase.co/storage/v1/object/sign/" + bucket + "/" + p + "?token=t" + (++signSeq) : null })), error: null };
    },
    remove: async (names) => {
      calls.push({ remove: bucket, names: [...names] });
      if (db.offline) throw new TypeError("Failed to fetch");
      if (db.fail.remove) return { data: null, error: { message: db.fail.remove } };
      const gone = names.filter((n) => db.objects.delete(n));
      return { data: gone.map((name) => ({ name })), error: null };
    } }) },
  channel: (name) => {
    const ch = { name, binds: [], removed: false, status: null };
    ch.on = (type, filter, cb) => { ch.binds.push({ filter, cb }); return ch; };
    ch.subscribe = (cb) => { ch.status = cb || null; db.channels.push(ch); if (cb && db.realtime) setTimeout(() => cb(db.realtime), 0); return ch; };
    return ch;
  },
  removeChannel: (ch) => { ch.removed = true; calls.push({ removeChannel: ch.name }); },
  auth: {
    getSession: async () => ({ data: { session: db.session }, error: null }),
    signInWithOAuth: async (a) => { calls.push({ oauth: a }); return { error: null }; },
    signOut: async (a) => { calls.push({ signOut: a }); return { error: null }; },
    onAuthStateChange: (cb) => { db.authCb = cb; return { data: { subscription: { unsubscribe() {} } } }; },
  },
};
// A Realtime change for every channel that listens to the table (and whose job_id or user_id filter matches).
function emit(table, event, row) {
  for (const ch of db.channels) for (const b of ch.binds) {
    if (ch.removed || b.filter.table !== table || (b.filter.event !== "*" && b.filter.event !== event)) continue;
    const m = /^(\w+)=eq\.(.+)$/.exec(b.filter.filter || "");
    if (m && String(row[m[1]]) !== m[2]) continue;
    b.cb({ eventType: event, new: row });
  }
}
// The rpc handlers that a test does not set: job_objects gives the objects under <user>/<job>/ in the order they came.
const stock = {
  job_objects: (a) => ({ data: [...db.objects].filter((o) => o.startsWith(db.session.user.id + "/" + a.p_job + "/"))
    .slice(0, a.p_limit || 1000).map((name) => ({ name, created_at: "2026-10-24T14:42:05+00:00" })) }),
};
globalThis.supabase = { createClient: (url, key, o) => { calls.push({ createClient: [url, key, o] }); return client; } };
if (opts.config) globalThis.HARPOR_CONFIG = opts.config;
const out = (x) => console.log(JSON.stringify(x));
`, ctx);
const run = (code, name) => vm.runInContext(code, ctx, { filename: name });
try {
  run(setupCode, "setup");
  run(fs.readFileSync(accountPath, "utf8"), "account.js");
  if (supaPath) run(fs.readFileSync(supaPath, "utf8"), "supa.js");
  if (appPath) run(fs.readFileSync(appPath, "utf8"), "app.js");
  Promise.resolve(run(testCode, "test")).catch((e) => { console.error(e.stack); process.exitCode = 1; });
} catch (e) { console.error(e.stack); process.exitCode = 1; }
"""


def run_js(code, setup="", config=CONFIG, app=True, href="https://x.pages.dev/", framed=False, agents=None, supa=True):
    """Load account.js, supa.js (with HARPOR_CONFIG unless config is None) and app.js in HARNESS after setup; run code,
    an async function body; return the JSON it printed last. supa=False loads account.js alone, as help.html and
    demos.html do."""
    if shutil.which("node") is None:
        pytest.skip("no node")
    opts = {"config": config, "href": href, "framed": framed, **({"agents": agents} if agents else {})}
    p = subprocess.run(["node", "-e", HARNESS, "-", str(ACCOUNT), str(SUPA) if supa else "",
                        str(APP) if app and supa else "", json.dumps(opts),
                        textwrap.dedent(setup), "(async () => {" + textwrap.dedent(code) + "})()"],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.splitlines()[-1])


def iso(seconds_ago):
    """A timestamptz as PostgREST gives it, seconds_ago before now."""
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - seconds_ago))


def local(t):
    import datetime
    return datetime.datetime.fromisoformat(t).astimezone()


def hm(t):
    """supa.js's hm(): the local "HH:MM" of a timestamptz."""
    return local(t).strftime("%H:%M")


def when(t):
    """supa.js's when(): "Oct 24 05:35" in local time."""
    d = local(t)
    return f"{d.strftime('%b')} {d.day} {d.strftime('%H:%M')}"


# A signed-in account, with the tables that the gate reads; js() of it is setup code. Its machines are lab-box and
# desk-pc unless said. Workers' last_seen is "now" unless said, since the harness's clock starts at the real time.
def account(approved=True, agents=("astra", "opus"), machines=MACHINES, workers=(), jobs_=(), extra=""):
    tables = {
        "access": [{"user_id": USER, "approved": approved, "agents": list(agents), "max_result_bytes": 2147483648,
                    "request_at": None}],
        "settings": [{"worker_online_s": 30, "max_machines": 5}],
        "machines": [{"id": m, "name": n, "created_at": iso(100 - i)} for i, (m, n) in enumerate(machines)],
        "workers": list(workers), "jobs": list(jobs_), "job_views": [], "job_files": [], "job_snapshots": [],
        "job_live": [], "job_logs": [], "requests": [],
    }
    return (f"db.session = {{ access_token: 'tok-1', user: {{ id: {json.dumps(USER)}, email: 'ana@example.org' }} }};\n"
            f"db.tables = {json.dumps(tables)};\n"
            "db.rpc.storage_used = () => ({ data: 125829120 });\n" + extra)


def worker(wid=W1, machine=MACHINE, seen_ago=2, accepting=True, ends_in=3600, started_ago=600, local_keys=(),
           public_key="A" * 88, stopped=False):
    return {"id": wid, "machine_id": machine, "public_key": public_key, "local_keys": list(local_keys),
            "started_at": iso(started_ago), "last_seen": iso(seen_ago), "accepting": accepting,
            "ends_at": iso(-ends_in) if ends_in is not None else None, "stopped_at": iso(seen_ago) if stopped else None}


def job_row(jid=JOB, status="review", machine=MACHINE, **kw):
    return {"id": jid, "user_id": USER, "created_at": "2026-10-24T14:42:05+00:00", "status": status, "note": None,
            "prompt": "chair", "queue_pos": None, "input_name": "chair.mp4", "input_kind": "video", "input_bytes": 1000,
            "machine_id": machine, **kw}


def view(jid=JOB, agent="opus", status="review", **kw):
    return {"job_id": jid, "user_id": USER, "body": {"id": jid, "status": status, "agent": agent, "video": {"name": "chair.mp4"},
                                                      "warnings": [], "files": [], "stages": {}, "progress": [], **kw}}


# ---- app.js: the seam in tunnel mode, and its public-site parts ----


def test_tunnel_page_has_no_transport_and_routes_at_once():
    """Without HARPOR_CONFIG (the tunnel page loads neither file anyway) supa.js does nothing, and app.js's default
    transport is today's: plain fetch of the API paths, the paths themselves as URLs, XMLHttpRequest for uploads, ACTIVE
    as the live statuses, keyless() false, the tunnel texts (an empty list asks for an upload), and the list loads at
    once."""
    out = run_js("""
        await advance(0);
        out({ transport: typeof HARPOR_TRANSPORT, client: calls.length, pub: PUBLIC, keyless: [T.keyless("j", "opus"), T.keyless()],
          url: T.url("/api/jobs/j/file/run/posed.glb", "j_posed.glb"), live: T.live === ACTIVE, liveLog: T.liveLog,
          offline: T.offline, unauthorized: T.unauthorized, xhr: typeof T.xhr, fetched, routing, empty: text($("#jobs tbody")) });
        """, setup='net["/api/jobs"] = [200, []];', config=None)
    assert out["transport"] == "undefined" and out["client"] == 0 and out["pub"] is False
    assert out["keyless"] == [False, False] and out["url"] == "/api/jobs/j/file/run/posed.glb" and out["live"] is True
    assert out["liveLog"] == "live log" and out["offline"] == "Cannot reach the server. Is the SSH tunnel up?"
    assert out["unauthorized"] == "Not logged in. Open the ?token= link the server printed."
    assert out["fetched"] == [{"url": "/api/jobs", "method": "GET"}] and out["routing"] is True and out["xhr"] == "function"
    assert out["empty"] == "No jobs yet."


def test_app_js_list_shows_where_and_the_queue_place():
    """The jobs list: a queued job's place, where the job runs under its input's kind, and the Delete button by the
    row's deletable (the public site: a job whose machine is offline). Tunnel rows (no queue_pos, where or deletable)
    keep today's badges, marks and buttons."""
    rows = [{"id": "a", "created": "2026-10-24T10:42:05", "status": s, "prompt": "", "video": f"{s}.mp4", "kind": "video", **kw}
            for s, kw in [("queued", {"queue_pos": 2, "where": "on lab-box"}), ("queued", {"queue_pos": 11}),
                          ("queued", {"queue_pos": 23}), ("queued", {}), ("review", {"where": "on desk-pc", "deletable": False}),
                          ("failed", {}), ("uploaded", {}), ("uploading", {}), ("uploading", {"deletable": True}),
                          ("running", {"where": "on lab-box", "deletable": True}), ("running", {})]]
    out = run_js("""
        renderJobs(""" + json.dumps(rows) + """);
        const btn = (td) => td.kids.find((k) => typeof k === "object");  // the trash can button, when the row has one
        out($("#jobs tbody").kids.map((tr) => ({ act: tr.cls.has("act"), badge: text(tr.kids[3]), hint: text(tr.kids[1].kids[1] || ""),
          del: btn(tr.kids[5]) ? btn(tr.kids[5]).attrs.title : "", label: btn(tr.kids[5]) ? btn(tr.kids[5]).attrs["aria-label"] : "",
          icon: btn(tr.kids[5]) ? btn(tr.kids[5]).kids[0].tag : "", thumb: tr.kids[0].kids[0].attrs.src })));
        """, config=None)
    assert [r["badge"] for r in out] == ["queued (2nd)", "queued (11th)", "queued (23rd)", "queued", "needs review", "failed",
                                         "uploaded", "uploading", "uploading", "running", "running"]
    assert [r["act"] for r in out] == [False] * 4 + [True, True] + [False] * 5
    assert [r["del"] for r in out] == ["", "", "", "", "Delete", "Delete", "Delete", "", "Delete", "Delete", ""]
    assert out[4]["label"] == "Delete review.mp4" and out[4]["icon"] == "svg"  # a trash can, named for screen readers
    assert out[0]["hint"] == "video, on lab-box" and out[4]["hint"] == "video, on desk-pc" and out[1]["hint"] == "video"
    assert out[0]["thumb"] == "/api/jobs/a/thumb/0"


def test_app_js_job_page_where_and_queue_place():
    """The job page: #jid says where the job runs, a queued job shows its place; the status words are the tunnel's, and
    a status outside T.live opens no live view."""
    job = {"id": "j", "status": "uploaded", "video": {"name": "chair.mp4"}, "where": "on lab-box", "queue_pos": None,
           "stages": {"frames": {"status": "pending"}, "select": {"status": "pending"}},
           "progress": [{"name": "frames", "status": "pending"}], "files": [], "warnings": [], "agent": "opus", "key_ok": False, "current": None, "error": None, "source": {"kind": "video"},
           "kind": "video"}
    out = run_js("""
        V = { id: "j", picks: [] };
        await refresh();
        const page = { jid: $("#jid").textContent, status: text($("#jstatus")), next: $("#jnext").textContent,
          where: text($("#where")), es: !!V.es, title: document.title };
        V.job = { queue_pos: 2, video: { name: "chair.mp4" } };
        renderState({ status: "queued", agent: "opus", stages: [{ name: "frames", status: "done" }], warnings: [] }, false);
        out({ page, queued: text($("#jstatus").kids[0]), where: text($("#where")), title: document.title });
        """, setup='net["/api/jobs/j"] = [200, ' + json.dumps(job) + "];", config=None)
    assert out["page"]["jid"] == "job j, on lab-box" and out["page"]["es"] is False
    assert out["page"]["status"] == "Uploaded" and out["page"]["next"] == ""
    assert out["page"]["where"] == "/ chair.mp4" and out["page"]["title"] == "uploaded · chair.mp4 · HARPOR"
    assert out["queued"] == "queued (2nd)" and out["where"] == "/ chair.mp4"  # the status shows under the title only
    assert out["title"] == "Queued · chair.mp4 · HARPOR"


def test_app_js_style_goes_through_the_cssom():
    """h() sets style through the CSSOM (allowed by a CSP without unsafe-inline), not as an attribute; an element
    without a CSSOM gets the attribute, as before."""
    out = run_js("""
        const a = h("img", { style: "aspect-ratio: 4 / 3", alt: "" });
        const plain = document.createElement;
        document.createElement = (tag) => ({ attrs: {}, setAttribute(k, v) { this.attrs[k] = v; }, append() {} });
        const b = h("img", { style: "--ar: 16 / 9" });
        document.createElement = plain;
        out({ css: a.style.cssText, attr: a.attrs.style || null, fallback: b.attrs.style });
        """, config=None)
    assert out == {"css": "aspect-ratio: 4 / 3", "attr": None, "fallback": "--ar: 16 / 9"}


# A transport that only says which agent's key the job's machine holds (opus), for app.js alone.
KEYLESS = """
globalThis.HARPOR_TRANSPORT = { fetch: (u, o) => fetch(u, o), url: (p) => p, events: () => null,
  live: ["analyzing", "queued", "running"], keyless: (id, a) => a === "opus" ? "lab-box uses the Anthropic key stored on it." : false,
  noJobs: "none", ready: Promise.resolve(), liveLog: "live log", offline: "off", unauthorized: "out" };
"""


def test_app_js_keyless_job_asks_for_no_key():
    """T.keyless: the start form shows the machine's sentence in place of the key field (not required), the Start
    body sends no key, and the key row of review and Retry stays hidden; another agent asks for its key as before."""
    out = run_js("""
        await advance(0);
        V = { id: "j", picks: [], view0: null };
        const radios = lists["input[name=agent]"], form = [];
        for (const [i, r] of radios.entries()) {
          radios.forEach((x, k) => { x.checked = k === i; });
          agentToggle();
          form.push([r.value, $("#keyname").textContent, $("#key").hidden, $("#key").required]);
        }
        Object.assign($("#key"), { value: "sk-ant-typed" });
        Object.assign($("#prompt"), { value: "chair" });
        Object.assign($("#nviews"), { value: "4" });
        globalThis.fetch = async (url, o) => { fetched.push({ url, body: o && o.body }); return new Response("{}", { status: 200 }); };
        await $("#startform").onsubmit({ preventDefault() {} });
        const start = JSON.parse(fetched.find((f) => f.url === "/api/jobs/j/start").body);
        V = { id: "j" };
        const rows = ["opus", "astra"].map((agent) => {
          renderState({ status: "review", agent, key_ok: false, stages: [{ name: "frames", status: "done" }], warnings: [] }, false);
          return $("#keyrow").hidden;
        });
        out({ form, start: [start.agent, start.api_key], rows });
        """, setup=KEYLESS, config=None)
    assert out["form"] == [["astra", "OpenAI API key", False, True],
                           ["opus", "lab-box uses the Anthropic key stored on it.", True, False]]
    assert out["start"] == ["opus", None] and out["rows"] == [True, False]


def test_app_js_offers_only_the_agents_the_site_allows():
    """loadFrames never checks a disabled agent radio: a job whose agent is not offered (none, or astra for an account
    that may use opus only) starts from the first agent offered."""
    meta = {"video": {"n_frames": 10, "duration_s": 1.0, "fps": 10, "width": 64, "height": 48, "codec": "h264", "kind": "video"},
            "times": [0.1 * i for i in range(10)], "rel": [1.0] * 10, "candidates": [0, 5], "thumbs": [0, 5],
            "thumb_size": [64, 48]}
    out = run_js("""
        await advance(0);
        const radios = lists["input[name=agent]"], seen = [];
        for (const [agent, off] of [["none", null], ["astra", "astra"], ["opus", "astra"]]) {
          radios.forEach((r) => { r.disabled = r.value === off; r.checked = false; });
          V = { id: "j", picks: [], view0: null, job: { id: "j", agent, kind: "video", parts: {}, mode: "auto", picks: [] } };
          await loadFrames();
          seen.push(radios.filter((r) => r.checked).map((r) => r.value));
        }
        out(seen);
        """, setup=KEYLESS + 'net["/api/jobs/j/frames"] = [200, ' + json.dumps(meta) + "];", config=None)
    assert out == [["astra"], ["opus"], ["opus"]]


def test_app_js_public_page_routes_only_when_ready():
    """On the public site app.js routes once T.ready resolves (supa.js signed the user in), and a hash change before
    that does not route."""
    out = run_js("""
        await advance(0);
        const before = fetched.length;
        location.hash = "#/job/j";
        winListeners.filter((l) => l.type === "hashchange").forEach((l) => l.fn());
        await advance(0);
        const early = fetched.length;
        location.hash = "";
        letGo();
        await advance(0);
        out({ before, early, after: fetched.map((f) => f.url), routing });
        """, setup="""
        let letGo;
        globalThis.HARPOR_TRANSPORT = { fetch: (u, o) => fetch(u, o), url: (p) => p, events: () => null, live: [],
          keyless: () => false, noJobs: "", ready: new Promise((r) => { letGo = r; }), liveLog: "", offline: "off", unauthorized: "out" };
        net["/api/jobs"] = [200, []];
        """, config=None)
    assert out == {"before": 0, "early": 0, "after": ["/api/jobs"], "routing": True}


# ---- supa.js: sign-in, approval, the banner and the account ----


def test_signed_out_page_signs_in_with_pkce_back_to_the_site_folder():
    """Signed out: the sign-in panel, PKCE, and redirectTo is the page's folder (index.html, the query and the hash
    dropped), at a domain root and under /<repo>/. app.js does not route. A refused sign-in says why and leaves no
    trace in the address bar."""
    code = """
        await advance(0);
        const msg = $("#harpor-signinmsg").textContent;
        $("#harpor-google").onclick();
        $("#harpor-github").onclick();
        await advance(0);
        out({ signin: $("#harpor-signin").hidden, waiting: $("#harpor-waiting").hidden, account: $("#harpor-account").hidden,
          auth: calls.find((c) => c.createClient).createClient[2].auth, oauth: calls.filter((c) => c.oauth).map((c) => c.oauth),
          tables: calls.filter((c) => c.table).length, msg, href: location.href });
        """
    root = run_js(code, setup="db.session = null;")
    assert root["signin"] is False and root["waiting"] is True and root["account"] is True and root["tables"] == 0
    assert root["auth"]["flowType"] == "pkce" and root["auth"]["detectSessionInUrl"] is True
    assert root["oauth"] == [{"provider": p, "options": {"redirectTo": "https://x.pages.dev/"}} for p in ("google", "github")]
    sub = run_js(code, href="https://x.github.io/HARPOR/index.html?code=1#/job/2")
    assert {o["options"]["redirectTo"] for o in sub["oauth"]} == {"https://x.github.io/HARPOR/"}
    bad = run_js(code, href="https://x.github.io/HARPOR/?error=access_denied&error_description=The+user+said+no#/")
    assert bad["msg"] == "Sign-in failed: The user said no" and bad["href"] == "https://x.github.io/HARPOR/#/"


def test_account_waiting_for_approval_can_request_access():
    """Signed in without approval: the waiting panel and its request box (rpc request_access); the account icon's
    name and the menu's first line hold the email; app.js never routes, and no machine or job is read."""
    out = run_js("""
        await advance(0);
        $("#harpor-note").value = "  I build robot props for the lab.  ";
        await $("#harpor-request").onsubmit({ preventDefault() {} });
        out({ waiting: $("#harpor-waiting").hidden, signin: $("#harpor-signin").hidden, email: $("#harpor-email").textContent,
          who: $("#harpor-who").textContent,
          account: $("#harpor-account").hidden, rpc: calls.filter((c) => c.rpc).map((c) => [c.rpc, c.args]),
          msg: $("#harpor-requestmsg").textContent, own: $("#harpor-own").hidden,
          read: calls.filter((c) => c.table).map((c) => [c.table, c.cols]) });
        """, setup=account(approved=False) + "db.rpc.request_access = () => ({ data: 'sent' });")
    assert out["waiting"] is False and out["signin"] is True and out["account"] is False
    assert out["email"] == out["who"] == "ana@example.org"
    assert out["rpc"] == [["request_access", {"p_note": "I build robot props for the lab."}]]
    assert out["msg"] == "Request sent." and out["own"] is True
    assert out["read"] == [["access", "approved,agents,max_result_bytes,request_at"], ["settings", "worker_online_s,max_machines"]]


def test_approved_account_gets_the_machine_panel():
    """Approved: the machines panel, where the tunnel page has its upload area: each machine with its state and the
    keys it holds (site.py adds the link to help.html#machine). Agent radios outside access.agents are disabled and
    hidden. The account menu shows the storage."""
    code = """
        await advance(0);
        const radios = lists["input[name=agent]"];
        out({ own: $("#harpor-own").hidden, machines: $("#harpor-machines").kids.map(text), storage: $("#harpor-storage").textContent,
          radios: radios.map((r) => [r.value, r.disabled, r.parentElement.hidden, r.checked]),
          routed: calls.some((c) => c.table === "jobs") });
        """
    none = run_js(code, setup=account(agents=("opus",), machines=()))
    assert none["own"] is False and none["routed"] is True
    assert none["machines"] == ["No machines yet."] and none["storage"] == "Storage: 120 of 2,048 MB"
    assert none["radios"] == [["astra", True, True, False], ["opus", False, False, True]]
    two = run_js(code, setup=account(workers=[worker(local_keys=["opus"], ends_in=None)]))
    assert two["machines"] == ["lab-box: online. Has its own Anthropic key.", "desk-pc: offline."]
    assert two["radios"] == [["astra", False, False, True], ["opus", False, False, False]]
    both = run_js(code, setup=account(machines=MACHINES[:1], workers=[worker(local_keys=["astra", "opus"], accepting=False)]))
    assert both["machines"] == [f"lab-box: stopping at {hm(worker()['ends_at'])}. Has its own OpenAI and Anthropic keys."]


def test_banner_shows_the_machines_that_are_offline_or_stopping():
    """The banner: one line for each of the user's machines that is offline or stopping, in the order they were added:
    stopping at the end of its Slurm job, or soon (builds wait); offline since when, or offline (not run in the last 30
    days). An online machine has no line (the machines panel lists it); with none to show, no banner. A stopping
    machine's line has the warning mark, an offline one the hollow mark."""
    def lines(workers, machines=MACHINES[:1]):
        return run_js("await advance(0); out({ lines: $('#harpor-banner').kids.map((p) => [text(p), p.className]), "
                      "hidden: $('#harpor-banner').hidden });", setup=account(machines=machines, workers=workers))
    for on in (worker(ends_in=7200), worker(ends_in=None)):
        assert lines([on]) == {"lines": [], "hidden": True}
    ending = worker(accepting=False, ends_in=1800)
    assert lines([ending]) == {"lines": [[f"lab-box: stopping at {hm(ending['ends_at'])}", "stopping"]], "hidden": False}
    assert lines([worker(accepting=False, ends_in=None)])["lines"] == [["lab-box: stopping soon", "stopping"]]
    gone = worker(seen_ago=4000, started_ago=9000)
    out = lines([gone, worker(W2, MACHINE2, seen_ago=5)], machines=MACHINES)
    assert out["lines"] == [[f"lab-box: offline since {when(gone['last_seen'])}", "off"]]
    away = worker(seen_ago=600, stopped=True)
    assert lines([away])["lines"] == [[f"lab-box: offline since {when(away['stopped_at'])}", "off"]]
    assert lines([])["lines"] == [["lab-box: offline", "off"]]
    assert lines([], machines=()) == {"lines": [], "hidden": True}
    css = (server.STATIC / "style.css").read_text()
    assert ".banner p.stopping::before { background: var(--warn);" in css and ".banner p.on" not in css


def test_banner_and_machine_panel_keep_still_between_reads():
    """The read every 20 s draws the banner (a status region) only when a line changes, and finds a machine that was
    added after the page opened."""
    out = run_js("""
        await advance(0);
        const first = $("#harpor-banner").kids[0];
        await advance(41000);
        const same = $("#harpor-banner").kids[0] === first;
        db.tables.workers[0].stopped_at = new Date(Date.now()).toISOString();
        db.tables.machines.push({ id: "cccccccc-0000-4000-8000-000000000009", name: "new-box", created_at: new Date(Date.now()).toISOString() });
        await advance(21000);
        out({ same, reads: calls.filter((c) => c.table === "workers").length, now: $("#harpor-banner").kids.map(text),
          panel: $("#harpor-machines").kids.map(text) });
        """, setup=account(machines=MACHINES[:1], workers=[worker(seen_ago=-100, accepting=False)]))
    assert out["same"] is True and out["reads"] >= 4
    assert out["now"][0].startswith("lab-box: offline since ") and out["now"][1:] == ["new-box: offline"]
    assert out["panel"][0].startswith("lab-box: offline since ") and out["panel"][1:] == ["new-box: offline."]


def test_a_live_machine_stays_found_behind_many_restarts_of_another():
    """readWorkers reads the user's workers in one query, the last beat first: 60 newer rows of another machine
    (restarts) do not hide a machine whose worker started long ago and still beats, so its jobs still get their key
    sealed to it, not key_later."""
    priv, pub = p256()
    restarts = [worker(wid=f"5e5e5e5e-0000-4000-8000-{i:012d}", machine=MACHINE2, started_ago=10 + i, seen_ago=10 + i,
                       stopped=True) for i in range(60)]
    machine = worker(started_ago=40 * 86400, seen_ago=2, public_key=pub)
    out = run_js(f"""
        await advance(0);
        const lines = $("#harpor-machines").kids.map(text);
        setTimeout(() => answer(7, 200, {{ status: "queued" }}), 1000);
        await settle(T.fetch("/api/jobs/{JOB}/continue", {{ method: "POST", body: JSON.stringify({{ api_key: "sk-ant-key1" }}) }}));
        out({{ lines, queries: calls.filter((c) => c.table === "workers").map((c) => [c.cols, c.filters, c.ord, c.lim]),
          sent: relayed.map((a) => [a.p_body, (a.p_sealed || {{}}).worker_id || null]) }});
        """, setup=relay_setup(restarts + [machine], [job_row()], [view()]))
    assert out["lines"][0] == f"lab-box: online until {hm(machine['ends_at'])}."
    assert out["lines"][1].startswith("desk-pc: offline since ")
    assert out["queries"][0] == ["id,machine_id,public_key,local_keys,started_at,last_seen,ends_at,accepting,stopped_at", [],
                                 ["last_seen", -1], 50]
    assert all(q == out["queries"][0] for q in out["queries"]) and len(out["queries"]) >= 2
    assert out["sent"] == [[{"api_key": None}, W1]]


def test_page_in_a_frame_shows_nothing():
    """GitHub Pages cannot send frame-ancestors: in a frame supa.js hides the page, makes no client and app.js never
    routes. Its stand-in transport has the keys of the real one."""
    out = run_js("await advance(0); out({ hidden: document.documentElement.hidden, calls, fetched, pub: PUBLIC, routing, "
                 "keys: Object.keys(HARPOR_TRANSPORT).sort() });", setup=account(), framed=True)
    assert out == {"hidden": True, "calls": [], "fetched": [], "pub": True, "routing": False,
                   "keys": sorted(["fetch", "url", "events", "live", "keyless", "noJobs", "ready", "liveLog", "offline",
                                   "unauthorized"])}


# ---- account.js: the account menu on the app, the help page and the demos page ----

# The menu as account.js left it: hidden, its lines, the client it made, what it read and whether the page reloaded.
# q, not app.js's $: help.html and demos.html have no app.js.
MENU = """((q) => ({ hidden: q("#harpor-account").hidden, open: q("#harpor-account").open,
  email: q("#harpor-email").textContent, who: q("#harpor-who").textContent, storage: q("#harpor-storage").textContent,
  clients: calls.filter((c) => c.createClient).map((c) => c.createClient), signOut: calls.filter((c) => c.signOut),
  reads: calls.filter((c) => c.table).map((c) => [c.table, c.cols]), rpc: calls.filter((c) => c.rpc).map((c) => c.rpc),
  reloads }))((sel) => document.querySelector(sel))"""


def test_account_menu_on_the_app_comes_from_account_js():
    """On the app supa.js takes account.js's client (one client, PKCE) and has it show the menu after its own sign-in:
    the email twice and the storage line. The end of the page's parse starts nothing more (the help pages' path does
    not run there). Sign out signs this browser out and reloads the page."""
    out = run_js(f"""
        await advance(0);
        loaded();
        await advance(0);
        const before = {MENU};
        await $("#harpor-signout").onclick();
        out({{ before, after: {MENU} }});
        """, setup=account())
    before, after = out["before"], out["after"]
    assert before["hidden"] is False and before["email"] == before["who"] == "ana@example.org"
    assert before["storage"] == "Storage: 120 of 2,048 MB" and before["reloads"] == 0
    assert len(before["clients"]) == 1 and before["clients"][0][2]["auth"]["flowType"] == "pkce"
    assert [r for r in before["reads"] if r[0] == "access"] == [["access", "approved,agents,max_result_bytes,request_at"]]
    assert before["rpc"] == ["storage_used"]
    assert after["signOut"] == [{"signOut": {"scope": "local"}}] and after["reloads"] == 1


def test_account_menu_on_the_help_and_demos_pages():
    """account.js alone (help.html and demos.html have no supa.js): once the page is parsed it reads the stored session
    with the app's client options, and so the app's sign-in. Signed in, it fills the menu as the app does (the email
    twice, the storage line of an approved account) and shows it; a click elsewhere and Escape close it, a click in it
    does not. Sign out signs this browser out, hides the menu and the page stays; a sign-out in another tab hides it
    too. Nothing runs before the end of the parse."""
    out = run_js(f"""
        await advance(0);
        const early = calls.length;
        loaded();
        await advance(0);
        const $ = (sel) => document.querySelector(sel);
        const shown = {MENU}, menu = $("#harpor-account"), inner = new Elem("p"), click = (target) =>
          docListeners.filter((l) => l.type === "click").forEach((l) => l.fn({{ target }}));
        menu.append(inner);
        menu.open = true;
        click(inner);
        const inside = menu.open;
        click(document.body);
        const outside = menu.open;
        menu.open = true;
        docListeners.filter((l) => l.type === "keydown").forEach((l) => l.fn({{ key: "Escape" }}));
        const escape = menu.open;
        menu.open = true;
        await $("#harpor-signout").onclick();
        const gone = {MENU};
        menu.hidden = false;
        db.authCb("SIGNED_OUT");
        out({{ early, shown, inside, outside, escape, gone, elsewhere: menu.hidden }});
        """, setup=account(), supa=False)
    shown, gone = out["shown"], out["gone"]
    assert out["early"] == 0
    assert shown["hidden"] is False and shown["email"] == shown["who"] == "ana@example.org"
    assert shown["storage"] == "Storage: 120 of 2,048 MB"
    assert shown["clients"] == [[CONFIG["url"], CONFIG["publishableKey"], {"auth": {
        "flowType": "pkce", "persistSession": True, "autoRefreshToken": True, "detectSessionInUrl": True}}]]
    assert shown["reads"] == [["access", "approved,max_result_bytes"]] and shown["rpc"] == ["storage_used"]
    assert (out["inside"], out["outside"], out["escape"]) == (True, False, False)
    assert gone["signOut"] == [{"signOut": {"scope": "local"}}] and gone["hidden"] is True and gone["open"] is False
    assert gone["reloads"] == 0 and out["elsewhere"] is True


def test_account_menu_stays_hidden_signed_out_and_in_a_frame():
    """account.js alone: signed out the menu stays hidden and nothing is read; an account that waits for approval gets
    the menu without the storage line; in a frame (a click trap) it makes no client and shows nothing; without
    HARPOR_CONFIG (no site) it does nothing."""
    code = f"loaded(); await advance(0); out({MENU});"
    out = run_js(code, setup="db.session = null;", supa=False)
    assert out["hidden"] is True and len(out["clients"]) == 1 and out["reads"] == [] and out["rpc"] == []
    out = run_js(code, setup=account(approved=False), supa=False)
    assert out["hidden"] is False and out["who"] == "ana@example.org" and out["storage"] == "" and out["rpc"] == []
    for kw in ({"framed": True}, {"config": None}):
        out = run_js(code, setup=account(), supa=False, **kw)
        assert out["hidden"] is True and out["clients"] == [] and out["reads"] == [] and out["email"] == ""


def test_account_menu_follows_a_sign_in_in_another_tab():
    """account.js alone, signed out at first: a sign-in in another tab (supabase-js's SIGNED_IN) shows the menu, and
    the storage line comes after the callback ends. A later SIGNED_IN (the tab gets the focus back) reads nothing
    again. A sign-out hides the menu and takes the storage line away, and the next sign-in shows it again. The click
    and Escape handlers are added once."""
    out = run_js(f"""
        loaded();
        await advance(0);
        const before = {MENU};
        const session = {{ user: {{ id: "u2", email: "bo@example.org" }} }};
        db.authCb("SIGNED_IN", session);
        const at_once = {MENU};
        await advance(0);
        const shown = {MENU};
        db.authCb("SIGNED_IN", session);
        await advance(0);
        const again = {MENU};
        db.authCb("SIGNED_OUT", null);
        const gone = {MENU};
        db.authCb("SIGNED_IN", session);
        db.authCb("TOKEN_REFRESHED", session);
        await advance(0);
        const back = {MENU};
        const handlers = ["click", "keydown"].map((t) => docListeners.filter((l) => l.type === t).length);
        out({{ before, at_once, shown, again, gone, back, handlers }});
        """, setup=account() + "db.session = null;", supa=False)
    assert out["before"]["hidden"] is True and out["before"]["reads"] == []
    assert out["at_once"]["hidden"] is False and out["at_once"]["who"] == "bo@example.org"
    assert out["at_once"]["reads"] == [] and out["at_once"]["storage"] == ""
    assert out["shown"]["reads"] == [["access", "approved,max_result_bytes"]]
    assert out["shown"]["storage"] == "Storage: 120 of 2,048 MB"
    assert out["again"]["reads"] == out["shown"]["reads"] and out["again"]["rpc"] == ["storage_used"]
    assert out["gone"]["hidden"] is True and out["gone"]["storage"] == ""
    assert out["back"]["hidden"] is False and len(out["back"]["reads"]) == 2 and out["back"]["storage"] != ""
    assert out["handlers"] == [1, 1] and out["back"]["reloads"] == 0


def test_account_menu_keeps_the_keyboard_focus():
    """Escape from a line of the open menu closes it and puts the focus on the person icon; Escape with the menu shut
    does nothing, and Escape with the focus elsewhere closes the menu and leaves the focus there. Sign out on the help
    or demos page moves the focus from the hidden menu to the Help link."""
    out = run_js("""
        loaded();
        await advance(0);
        const $ = (sel) => document.querySelector(sel), menu = $("#harpor-account"), btn = $("#harpor-signout");
        const focused = [], esc = () =>
          docListeners.filter((l) => l.type === "keydown").forEach((l) => l.fn({ key: "Escape" }));
        $("#harpor-account > summary").focus = () => focused.push("icon");
        $("#helplink").focus = () => focused.push("help");
        menu.append(btn);
        document.activeElement = btn;
        esc();
        const shut = [menu.open, [...focused]];
        menu.open = true;
        esc();
        const closed = [menu.open, [...focused]];
        menu.open = true;
        document.activeElement = document.body;
        esc();
        const still = [menu.open, [...focused]];
        menu.open = true;
        document.activeElement = btn;
        await btn.onclick();
        out({ shut, closed, still, after: [...focused], hidden: menu.hidden });
        """, setup=account(), supa=False)
    assert out["shut"] == [False, []]
    assert out["closed"] == [False, ["icon"]] and out["still"] == [False, ["icon"]]
    assert out["after"] == ["icon", "help"] and out["hidden"] is True


# ---- supa.js: reads and files ----


def test_job_list_rows_thumbnails_and_missing_files():
    """GET /api/jobs: the user's rows (named columns), in the tunnel's shape plus where (the machine's name), the queue
    place and deletable (its machine is offline); thumbnail 0 is signed for 1 h, so T.url answers at once; a file that
    is not on the site gives a URL that fails at once, and T.fetch gives 404 for it."""
    rows = [job_row(JOB, "queued", queue_pos=2), job_row(JOB2, "running", MACHINE2, input_name="desk.mov"),
            job_row(JOB3, "done", input_name="bag.mcap", input_kind="rosbag")]
    setup = account(workers=[worker()], jobs_=rows) + f"""
        db.tables.job_files = [{{ job_id: "{JOB}", path: "frames/thumbs/0.jpg", object: "{USER}/{JOB}/0123456789abcdef/frames/thumbs/0.jpg" }}];
        db.objects.add("{USER}/{JOB}/0123456789abcdef/frames/thumbs/0.jpg");
        """
    out = run_js(f"""
        await advance(0);
        const r = await T.fetch("/api/jobs");
        const list = await r.json(), miss = await T.fetch(T.url("/api/jobs/{JOB2}/thumb/0"));
        out({{ list, thumb: T.url("/api/jobs/{JOB}/thumb/0"), other: T.url("/api/jobs/{JOB2}/thumb/0"),
          missing: [miss.status, await miss.json()], cols: calls.filter((c) => c.table === "jobs").map((c) => c.cols),
          signed: calls.filter((c) => c.sign).map((c) => c.seconds),
          drawn: $("#jobs tbody").kids.map((tr) => [text(tr.kids[1].kids[1] || ""),
            (tr.kids[5].kids.find((k) => typeof k === "object") || {{ attrs: {{}} }}).attrs.title || ""]) }});
        """, setup=setup)
    assert out["list"][0]["created"] == local("2026-10-24T14:42:05+00:00").strftime("%Y-%m-%dT%H:%M:%S")
    assert [(x["id"], x["status"], x["video"], x["kind"], x["where"], x["queue_pos"], x["deletable"]) for x in out["list"]] == [
        (JOB, "queued", "chair.mp4", "video", "on lab-box", 2, False),
        (JOB2, "running", "desk.mov", "video", "on desk-pc", None, True),  # its machine never ran: offline
        (JOB3, "done", "bag.mcap", "rosbag", "on lab-box", None, False)]
    assert out["thumb"].startswith("https://abcdefghijklmnop.supabase.co/storage/v1/object/sign/job-files/")
    assert out["other"] == "data:,missing:frames%2Fthumbs%2F0.jpg"
    assert out["missing"] == [404, {"detail": "File not on the site yet."}]
    assert out["cols"] and set(out["cols"]) == {"id,created_at,status,note,prompt,queue_pos,input_name,input_kind,input_bytes,machine_id"}
    assert 3600 in out["signed"]
    assert out["drawn"] == [["video, on lab-box", ""], ["video, on desk-pc", "Delete"], ["rosbag, on lab-box", "Delete"]]


def test_an_empty_list_says_where_jobs_come_from():
    """The public site has no upload area (the transport has no xhr): an empty jobs list says that jobs come from the
    user's GPU machine."""
    out = run_js("""
        await advance(0);
        out({ xhr: "xhr" in T, text: text($("#jobs tbody")), noJobs: T.noJobs });
        """, setup=account())
    assert out == {"xhr": False, "text": "No jobs yet. Add a file on your GPU machine.",
                   "noJobs": "No jobs yet. Add a file on your GPU machine."}


def test_job_view_signs_its_files_and_download_links():
    """GET /api/jobs/{id}: the worker's view with where, the queue place and the row's note; its files signed for 1 h
    and its results and logs also as 10-minute download links named by T.url; a preview that is not on the site shows
    as missing; a path that went from job_files is missing again."""
    files = [("frames/thumbs/0.jpg", "review"), ("run/posed.glb", "result"), ("logs/sam3d.log", "log")]
    obj = {p: f"{USER}/{JOB}/{i:016x}/{p}" for i, (p, _) in enumerate(files)}
    note = "lab-box stopped. This job goes on when HARPOR runs there again."
    setup = account(workers=[worker()], jobs_=[job_row(note=note, queue_pos=3, status="queued")]) + f"""
        db.tables.job_views = [{json.dumps(view(status="queued", files=["run/posed.glb"], preview={"file": "frames/preview.mp4"}))}];
        db.tables.job_files = {json.dumps([{"job_id": JOB, "path": p, "object": obj[p], "role": r} for p, r in files])};
        for (const o of {json.dumps(list(obj.values()))}) db.objects.add(o);
        """
    out = run_js(f"""
        await advance(0);
        const v = await (await T.fetch("/api/jobs/{JOB}")).json();
        const urls = {{ glb: T.url("/api/jobs/{JOB}/file/run/posed.glb"), dl: T.url("/api/jobs/{JOB}/file/run/posed.glb", "{JOB}_posed.glb"),
          thumbdl: T.url("/api/jobs/{JOB}/file/frames/thumbs/0.jpg", "0.jpg") }};
        const signs = calls.filter((c) => c.sign).map((c) => [c.seconds, c.paths]);
        db.tables.job_files = db.tables.job_files.filter((f) => f.path !== "logs/sam3d.log");
        await T.fetch("/api/jobs/{JOB}");
        out({{ v, urls, signs, gone: T.url("/api/jobs/{JOB}/file/logs/sam3d.log") }});
        """, setup=setup)
    v = out["v"]
    assert v["where"] == "on lab-box" and v["queue_pos"] == 3 and v["status"] == "queued"
    assert v["warnings"] == [note] and v["preview"] == {"missing": True}
    assert obj["run/posed.glb"] + "?token=t" in out["urls"]["glb"]
    assert re.search(r"\?token=t\d+&download=" + JOB + "_posed.glb$", out["urls"]["dl"])
    assert out["urls"]["dl"].split("&")[0] != out["urls"]["glb"]  # a link of its own, not the 1 h URL
    assert out["urls"]["thumbdl"].endswith("&download=0.jpg")  # a review file: its 1 h URL with the name
    assert sorted(sum((x for t, x in out["signs"] if t == 3600), [])) == sorted(obj.values())
    assert sorted(sum((x for t, x in out["signs"] if t == 600), [])) == sorted([obj["run/posed.glb"], obj["logs/sam3d.log"]])
    assert out["gone"] == "data:,missing:logs%2Fsam3d.log"


def test_pending_view_has_the_keys_of_the_servers_view_of_a_fresh_job(tmp_path):
    """A job whose machine has not sent its view yet gets a view built from its jobs row with the same keys as GET
    /api/jobs/{id} of a fresh job (the worker's rewrite drops video.path), the same stages in order, and no warning."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    cfg = steps.Config(G=tmp_path / "G", runs=tmp_path / "runs", fake=True)
    job_id, _ = jobs.create_job(cfg.runs, "clip.mp4")
    with TestClient(server.make_app(cfg, "tok", key_dir=tmp_path / "keys")) as c:
        c.get("/?token=tok", follow_redirects=False)
        fresh = c.get(f"/api/jobs/{job_id}").json()
    out = run_js(f"""
        await advance(0);
        out(await (await T.fetch("/api/jobs/{JOB}")).json());
        """, setup=account(workers=[worker(seen_ago=999)], jobs_=[job_row(status="uploaded")]))
    assert sorted(out) == sorted(set(fresh) | {"where", "queue_pos"})
    assert sorted(out["video"]) == sorted(set(fresh["video"]) - {"path"})
    assert [(s["name"], s["gpu"]) for s in out["progress"]] == [(s["name"], s["gpu"]) for s in fresh["progress"]]
    assert sorted(out["progress"][0]) == sorted(fresh["progress"][0])
    assert list(out["stages"]) == list(fresh["stages"]) and out["stages"]["frames"] == fresh["stages"]["frames"]
    assert out["status"] == "uploaded" and out["files"] == [] and out["source"] == {"kind": "video", "topic": None}
    assert out["warnings"] == [] and out["where"] == "on lab-box"


def test_snapshots_keep_the_routes_status_and_body():
    out = run_js(f"""
        await advance(0);
        const got = [];
        for (const p of ["frames", "selection", "parts"]) {{ const r = await T.fetch("/api/jobs/{JOB}/" + p); got.push([r.status, await r.json()]); }}
        out(got);
        """, setup=account(jobs_=[job_row()]) + f"""
        db.tables.job_snapshots = [{{ job_id: "{JOB}", path: "frames", status: 200, body: {{ candidates: [1, 2] }} }},
          {{ job_id: "{JOB}", path: "parts", status: 404, body: {{ detail: "no such job" }} }}];
        """)
    assert out == [[200, {"candidates": [1, 2]}], [404, {"detail": "the views are not selected yet"}],
                   [404, {"detail": "no such job"}]]


# ---- supa.js: relayed requests and sealed keys ----


def p256():
    """A worker's key pair: (private key, base64 of the 65-byte public point)."""
    ec = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ec")
    from cryptography.hazmat.primitives import serialization
    priv = ec.generate_private_key(ec.SECP256R1())
    pub = priv.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return priv, base64.b64encode(pub).decode()


def open_sealed(priv, sealed, job, user, agent, wid):
    """The worker's side of the seal, with the request's ref in the AAD."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    epk, iv, ct = (base64.b64decode(sealed[k]) for k in ("epk", "iv", "ct"))
    aad = f"harpor-key-v1|{job}|{user}|{agent}|{wid}|{sealed['ref']}".encode()
    peer = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), epk)
    k = HKDF(hashes.SHA256(), 32, salt=epk, info=aad).derive(priv.exchange(ec.ECDH(), peer))
    return AESGCM(k).decrypt(iv, ct, aad).decode()


# rpc relay: record the call; answer {id, worker_online, accepting} from the job's worker as the test sets it, and add
# the request row. answer(id, status, response) is the worker's answer, as a Realtime update.
RELAY = """
let nextId = 7;
const relayed = [];
db.rpc.relay = (a) => {
  relayed.push(a);
  if (db.refuse.length) return { error: db.refuse.shift() };
  const id = nextId++;
  db.tables.requests.push({ id, job_id: a.p_job, user_id: USER_ID, route: a.p_route, body: a.p_body, state: "pending", status: null,
    response: null, created_at: new Date(Date.now()).toISOString() });
  return { data: { id, worker_online: db.online, accepting: db.accepting } };
};
db.refuse = []; db.online = true; db.accepting = true;
function answer(id, status, response, state) {
  const r = db.tables.requests.find((x) => x.id === id);
  Object.assign(r, { state: state || "done", status, response });
  emit("requests", "UPDATE", { ...r });
}
db.rpc.withdraw_request = (a) => { const r = db.tables.requests.find((x) => x.id === a.p_id); Object.assign(r, { state: "withdrawn", status: 409, response: { detail: "you withdrew this request" } }); return { data: "withdrawn" }; };
"""


def relay_setup(workers, rows, views=(), extra=""):
    return (account(workers=workers, jobs_=rows) + f"const USER_ID = {json.dumps(USER)};\n" + RELAY +
            f"db.tables.job_views = {json.dumps(list(views))};\n" + extra)


def test_a_key_is_sealed_to_the_machines_worker_and_opens_there():
    """With the job's machine online, a Build with a key seals it to the machine's worker (the agent from the job's
    view), sends api_key null, waits for the answer through Realtime and gives the server's status and JSON. The worker
    opens it with its own id and the request's ref in the AAD; another worker id does not open it."""
    from cryptography.exceptions import InvalidTag
    priv, pub = p256()
    key = "sk-ant-" + "k" * 40
    out = run_js(f"""
        await advance(0);
        setTimeout(() => answer(7, 200, {{ status: "queued", from: "sam3d" }}), 1500);
        const r = await settle(T.fetch("/api/jobs/{JOB}/continue", {{ method: "POST", body: JSON.stringify({{ api_key: "{key}" }}) }}));
        out({{ status: r.status, body: await r.json(), relayed, pending: $("#harpor-pending").hidden }});
        """, setup=relay_setup([worker(public_key=pub)], [job_row()], [view()]))
    assert out["status"] == 200 and out["body"] == {"status": "queued", "from": "sam3d"} and out["pending"] is True
    (a,) = out["relayed"]
    assert a["p_route"] == "continue" and a["p_body"] == {"api_key": None} and a["p_text"] is None
    s = a["p_sealed"]
    assert s["worker_id"] == W1 and s["agent"] == "opus" and re.fullmatch(r"[0-9a-f-]{36}", s["ref"])
    assert len(s["epk"]) == 88 and len(s["iv"]) == 16 and key not in json.dumps(a)
    assert open_sealed(priv, s, JOB, USER, "opus", W1) == key
    with pytest.raises(InvalidTag):
        open_sealed(priv, s, JOB, USER, "opus", W2)


def test_a_jobs_key_goes_to_its_machine_not_to_the_newest_worker():
    """workerFor: a job's key is sealed to the live worker of its machine, even when another machine's worker started
    later, and never to a stopped worker of the same machine."""
    priv, pub = p256()
    key = "sk-" + "o" * 40
    out = run_js(f"""
        await advance(0);
        setTimeout(() => answer(7, 200, {{ status: "queued" }}), 1000);
        await settle(T.fetch("/api/jobs/{JOB2}/start", {{ method: "POST", body: JSON.stringify({{ agent: "astra", api_key: "{key}", mode: "auto" }}) }}));
        out(relayed[0]);
        """, setup=relay_setup([worker(started_ago=10), worker(W2, MACHINE2, public_key=pub, started_ago=900),
                                worker(W3, MACHINE2, started_ago=2000, seen_ago=950, stopped=True)],
                               [job_row(JOB2, "ready", MACHINE2)]))
    assert out["p_sealed"]["worker_id"] == W2 and out["p_body"]["api_key"] is None
    assert open_sealed(priv, out["p_sealed"], JOB2, USER, "astra", W2) == key


def test_a_machine_that_holds_the_key_gets_none_from_the_page():
    """A live worker whose local_keys hold the agent: no key, no key_later and nothing sealed; T.keyless names the
    machine for that agent only, and never for a job of another machine."""
    out = run_js(f"""
        await advance(0);
        await T.fetch("/api/jobs");
        setTimeout(() => answer(7, 200, {{ status: "queued" }}), 1000);
        await settle(T.fetch("/api/jobs/{JOB}/start", {{ method: "POST", body: JSON.stringify({{ agent: "opus", api_key: "sk-ant-typed", key_later: true }}) }}));
        out({{ a: relayed[0], keyless: [T.keyless("{JOB}", "opus"), T.keyless("{JOB}", "astra"), T.keyless("{JOB2}", "opus")] }});
        """, setup=relay_setup([worker(local_keys=["opus"]), worker(W2, MACHINE2)],
                               [job_row(JOB, "ready"), job_row(JOB2, "ready", MACHINE2)]))
    assert out["a"]["p_body"] == {"agent": "opus", "api_key": None} and out["a"]["p_sealed"] is None
    assert out["keyless"] == ["lab-box uses the Anthropic key stored on it.", False, False]


def test_with_the_machine_offline_a_request_waits_as_pending_with_a_withdraw_button():
    """With the job's machine offline the key is not sent: the body says key_later (not for overrides), relay() keeps
    the request pending, the page answers 202, and the job page's line says what waits and for which machine, with
    Withdraw (rpc withdraw_request). Each line names the request as its button does."""
    out = run_js(f"""
        await advance(0);
        db.online = false;
        const r = await T.fetch("/api/jobs/{JOB}/continue", {{ method: "POST", body: JSON.stringify({{ api_key: "sk-ant-zzz" }}) }});
        const line = $("#harpor-pending").kids.map(text), hidden = $("#harpor-pending").hidden;
        const btn = walk($("#harpor-pending")).find((e) => e.tag === "button");
        await btn.onclick();
        await advance(0);
        const after = $("#harpor-pending").hidden;
        await T.fetch("/api/jobs/{JOB}/overrides", {{ method: "POST", body: JSON.stringify({{ locked: [], api_key: null }}) }});
        location.hash = "#/job/{JOB2}";
        await T.fetch("/api/jobs/{JOB2}");
        await T.fetch("/api/jobs/{JOB2}/start", {{ method: "POST", body: JSON.stringify({{ agent: "astra", api_key: "sk-x1" }}) }});
        const other = $("#harpor-pending").kids.map(text);
        location.hash = "#/job/{JOB}";
        await T.fetch("/api/jobs/{JOB}/parts", {{ method: "POST", body: JSON.stringify({{ list: [] }}) }});
        await T.fetch("/api/jobs/{JOB}/calibration", {{ method: "DELETE" }});
        const named = $("#harpor-pending").kids.map((p) => text(p).split(",")[0]);
        out({{ status: r.status, body: await r.json(), line, hidden, after, other, named,
          relayed: relayed.map((a) => [a.p_route, a.p_body, a.p_sealed]), withdrawn: calls.filter((c) => c.rpc === "withdraw_request").map((c) => c.args) }});
        """, setup=relay_setup([worker(seen_ago=999, started_ago=2000)],
                               [job_row(), job_row(JOB2, "ready", MACHINE2)], [view()]),
        href=f"https://x.pages.dev/#/job/{JOB}")
    assert out["status"] == 202 and out["body"] == {"status": "pending", "request": 7}
    assert re.fullmatch(r"Build, sent \d\d:\d\d: waits for lab-box Withdraw", out["line"][0])
    assert out["line"][1] == "Key not sent: lab-box is offline. The job will ask for it later."
    assert out["hidden"] is False and out["withdrawn"] == [{"p_id": 7}] and out["after"] is True
    assert out["relayed"][0] == ["continue", {"api_key": None, "key_later": True}, None]
    assert out["relayed"][1] == ["overrides", {"locked": [], "api_key": None}, None]
    assert out["relayed"][2] == ["start", {"agent": "astra", "api_key": None, "key_later": True}, None]
    assert re.fullmatch(r"Start, sent \d\d:\d\d: waits for desk-pc Withdraw", out["other"][0])
    assert out["other"][1] == "Key not sent: desk-pc is offline. The job will ask for it later."
    assert out["named"] == ["Select again", "Build", "Remove calibration"]


def test_a_machine_that_stops_soon_gets_no_key_for_a_build():
    """A machine near the end of its Slurm job (online, takes no builds): a Build sends no key and waits as pending, the
    line says that the machine stops soon; a cancel still waits for its answer."""
    out = run_js(f"""
        await advance(0);
        db.accepting = false;
        const r = await T.fetch("/api/jobs/{JOB}/continue", {{ method: "POST", body: JSON.stringify({{ api_key: "sk-ant-zzz" }}) }});
        const line = $("#harpor-pending").kids.map(text);
        setTimeout(() => answer(8, 200, {{ status: "cancelled" }}), 1000);
        const c = await settle(T.fetch("/api/jobs/{JOB}/cancel", {{ method: "POST", body: "" }}));
        out({{ status: r.status, line, cancel: [c.status, await c.json()], relayed: relayed.map((a) => [a.p_route, a.p_body, a.p_sealed]) }});
        """, setup=relay_setup([worker(accepting=False, public_key=p256()[1])], [job_row()], [view()]),
        href=f"https://x.pages.dev/#/job/{JOB}")
    assert out["status"] == 202 and re.fullmatch(r"Build, sent \d\d:\d\d: waits for lab-box Withdraw", out["line"][0])
    assert out["line"][1] == "Key not sent: lab-box stops soon. The job will ask for it later."
    assert out["relayed"] == [["continue", {"api_key": None, "key_later": True}, None], ["cancel", {}, None]]
    assert out["cancel"] == [200, {"status": "cancelled"}]


def test_an_answer_that_comes_later_shows_on_the_job_page_once():
    """No answer within 60 s while the machine is online: 202 and the line (the request is never sent again). The
    answer that comes later changes the line and refreshes the page; OK dismisses it for good."""
    out = run_js(f"""
        await advance(0);
        let refreshed = 0;
        refresh = async () => {{ refreshed++; }};
        const p = T.fetch("/api/jobs/{JOB}/topic", {{ method: "POST", body: JSON.stringify({{ topic: "/cam/image" }}) }});
        await advance(61000);
        const r = await p;
        const waiting = $("#harpor-pending").kids.map(text);
        answer(7, 409, {{ detail: "the topic is not one of the bag's camera topics (frames/topics.json)" }});
        await advance(0);
        const later = $("#harpor-pending").kids.map((e) => [text(e), e.className]);
        const ok = walk($("#harpor-pending")).find((e) => e.tag === "button");
        ok.onclick();
        await advance(0);
        out({{ status: r.status, waiting, later, refreshed, gone: $("#harpor-pending").hidden, relays: relayed.length,
          kept: JSON.parse(store.get("harpor-waiting")) }});
        """, setup=relay_setup([worker()], [job_row(input_name="run.bag", input_kind="rosbag")]),
        href=f"https://x.pages.dev/#/job/{JOB}")
    assert out["status"] == 202 and out["relays"] == 1
    assert re.fullmatch(r"Camera topic, sent \d\d:\d\d: waits for lab-box Withdraw", out["waiting"][0])
    assert re.fullmatch(r"Camera topic, sent \d\d:\d\d: refused: the topic is not one of the bag's camera topics "
                        r"\(frames/topics\.json\) OK", out["later"][0][0])
    assert out["later"][0][1] == "refused" and out["refreshed"] >= 1 and out["gone"] is True and out["kept"] == []


def test_relay_refusals_and_a_machine_that_restarted():
    """A key of the wrong provider is refused in the page (jobs.key_kind_error) and never sent; relay()'s refusals keep
    their message with a matching status; worker_restarted, worker_offline and worker_ending read the workers again and
    send once more, and a second refusal says in plain words that the machine went or stops, or that it restarted."""
    out = run_js(f"""
        await advance(0);
        const send = async (route, body) => {{ const r = await settle(T.fetch("/api/jobs/{JOB}/" + route, {{ method: "POST", body: JSON.stringify(body) }})); return [r.status, (await r.json()).detail || null]; }};
        const got = [];
        for (const [agent, key] of [["astra", "sk-ant-abc"], ["opus", "sk-abc"], ["astra", "pk-abc"], ["astra", "sk-a b"]])
          got.push(await send("start", {{ agent, api_key: key }}));
        const before = relayed.length;
        for (const refusal of ["too many requests: at most 60 an hour", "the job is running; wait until it stops, or cancel it",
          "this account may use the agents opus", "an earlier request for this job still waits for its machine",
          "a cancel for this job already waits", "your account is waiting for approval", "no such job"]) {{
          db.refuse = [refusal];
          got.push(await send(refusal.startsWith("a cancel") ? "cancel" : "continue", {{}}));
        }}
        const reads = calls.filter((c) => c.table === "workers").length, sent = relayed.length;
        db.refuse = ["worker_restarted: the machine restarted; paste the key again"];
        setTimeout(() => answer(7, 200, {{ status: "queued" }}), 1000);
        got.push(await send("continue", {{ api_key: "sk-ant-ok" }}));
        const restarted = {{ sent: relayed.length - sent, reads: calls.filter((c) => c.table === "workers").length - reads,
          sealed: relayed.slice(-2).map((a) => !!a.p_sealed) }};
        db.refuse = ["worker_offline: the machine that runs this job is offline; send the request without the key",
          "worker_offline: the machine that runs this job is offline; send the request without the key"];
        got.push(await send("continue", {{ api_key: "sk-ant-ok" }}));
        for (const why of ["worker_ending: the machine stops soon; send the request without the key",
          "worker_restarted: the machine restarted; paste the key again"]) {{
          db.refuse = [why, why];
          got.push(await send("continue", {{ api_key: "sk-ant-ok" }}));
        }}
        out({{ got, before, restarted }});
        """, setup=relay_setup([worker(public_key=p256()[1])], [job_row()], [view()]))
    assert out["got"][:4] == [[400, jobs.key_kind_error("sk-ant-abc", "astra")], [400, jobs.key_kind_error("sk-abc", "opus")],
                              [400, jobs.key_kind_error("pk-abc", "astra")],
                              [400, "Not a valid API key."]]
    assert out["before"] == 0
    assert out["got"][4:11] == [[429, "too many requests: at most 60 an hour"],
                                [409, "the job is running; wait until it stops, or cancel it"],
                                [403, "this account may use the agents opus"],
                                [409, "an earlier request for this job still waits for its machine"],
                                [409, "a cancel for this job already waits"],
                                [403, "your account is waiting for approval"], [404, "no such job"]]
    # two reads of the workers (before the request, and once after the refusal), one query each
    assert out["got"][11] == [200, None] and out["restarted"] == {"sent": 2, "reads": 2, "sealed": [True, True]}
    assert out["got"][12:] == [[503, "The machine went offline. Try again."], [503, "The machine is stopping. Try again."],
                               [503, "The machine restarted. Paste the key again."]]


def test_the_sealed_agent_is_the_one_the_server_will_run():
    """effectiveAgent follows server.agent_of: start names its agent (astra when it names none);
    overrides and continue name it or take the job's; parts always takes the job's."""
    cases = [("start", {"mode": "auto"}, "astra"), ("start", {"agent": "opus"}, "opus"),
             ("continue", {}, "opus"), ("continue", {"agent": "astra"}, "astra"), ("continue", {"agent": None}, "opus"),
             ("overrides", {"locked": []}, "opus"), ("parts", {"agent": "astra", "parts": []}, "opus")]
    out = run_js(f"""
        await advance(0);
        const got = [];
        for (const [route, body] of {json.dumps([[r, b] for r, b, _ in cases])}) {{
          setTimeout(() => {{ for (const r of db.tables.requests) if (r.state === "pending") answer(r.id, 200, {{}}); }}, 500);
          await settle(T.fetch("/api/jobs/{JOB}/" + route, {{ method: "POST", body: JSON.stringify({{ ...body, api_key: "sk-ant-x" }}) }}));
          got.push(relayed.length ? (relayed[relayed.length - 1].p_sealed || {{}}).agent || null : null);
          relayed.length = 0;
        }}
        out(got);
        """, setup=relay_setup([worker(public_key=p256()[1])], [job_row()], [view(agent="opus")]))
    # A Start that resolves to astra with an Anthropic key is refused in the page, before any seal.
    assert out == [None if a == "astra" else a for _, _, a in cases]
    agent_of = getattr(server, "agent_of", None)
    if callable(agent_of):  # server.py's own function, once it is at module level
        job = {"agent": "opus"}
        for route, body, want in cases:
            assert (steps.agent(job) if route == "parts" else agent_of(body, None if route == "start" else job)) == want


def test_delete_calibration_and_cancel_requests():
    """DELETE /api/jobs/{id} is rpc delete_job, then the job's files; a refusal is 409 and removes nothing. A calibration
    upload is relayed with its text, name and LiDAR topic (none when app.js sends an empty one; not UTF-8: refused in
    the page), its removal with no text; cancel sends an empty body."""
    out = run_js(f"""
        await advance(0);
        const got = [];
        const del = await T.fetch("/api/jobs/{JOB}", {{ method: "DELETE" }});
        got.push([del.status, await del.json()]);
        db.rpc.delete_job = () => ({{ error: "the job is running; cancel it first" }});
        const no = await T.fetch("/api/jobs/{JOB}", {{ method: "DELETE" }});
        got.push([no.status, await no.json()]);
        const answerAll = () => setTimeout(() => {{ for (const r of db.tables.requests) if (r.state === "pending") answer(r.id, 200, {{ ok: true }}); }}, 500);
        answerAll();
        await settle(T.fetch("/api/jobs/{JOB}/calibration?name=cal%20one.yaml&lidar_topic=", {{ method: "PUT", body: new Blob(["T_cam_lidar: [1]\\n"]) }}));
        answerAll();
        await settle(T.fetch("/api/jobs/{JOB}/calibration?name=c.yaml&lidar_topic=%2Flivox%2Flidar", {{ method: "PUT", body: new Blob(["T_cam_lidar: [2]\\n"]) }}));
        const bad = await T.fetch("/api/jobs/{JOB}/calibration?name=x.yaml", {{ method: "PUT", body: new Blob([new Uint8Array([0xff, 0xfe])]) }});
        got.push([bad.status, await bad.json()]);
        answerAll();
        await settle(T.fetch("/api/jobs/{JOB}/calibration", {{ method: "DELETE" }}));
        answerAll();
        await settle(T.fetch("/api/jobs/{JOB}/cancel", {{ method: "POST", body: "" }}));
        out({{ got, deletes: calls.filter((c) => c.rpc === "delete_job").map((c) => c.args),
          listed: calls.filter((c) => c.rpc === "job_objects").map((c) => c.args),
          relayed: relayed.map((a) => [a.p_route, a.p_query, a.p_body, a.p_text, a.p_sealed]) }});
        """, setup=relay_setup([worker()], [job_row(input_name="run.bag", input_kind="rosbag")]) +
        "db.rpc.delete_job = () => ({ data: 'deleting' });")
    assert out["got"] == [[200, {"deleted": JOB}], [409, {"detail": "the job is running; cancel it first"}],
                          [400, {"detail": "the calibration file is not UTF-8 text"}]]
    assert out["deletes"] == [{"p_job": JOB}] * 2 and out["listed"] == [{"p_job": JOB}]
    assert out["relayed"] == [["calibration_put", {"name": "cal one.yaml"}, {}, "T_cam_lidar: [1]\n", None],
                              ["calibration_put", {"name": "c.yaml", "lidar_topic": "/livox/lidar"}, {},
                               "T_cam_lidar: [2]\n", None],
                              ["calibration_delete", {}, {}, None, None], ["cancel", {}, {}, None, None]]


def test_a_delete_removes_the_jobs_files_from_storage():
    """After delete_job the page removes the job's objects itself, so a machine that never comes back leaves no files:
    job_objects, then the Storage remove of those names, again while 1,000 came back. Other jobs' files stay. A remove
    that fails, or a lost connection, stops the rounds; the delete still answers 200, and the machine removes the
    rest."""
    many = [f"{USER}/{JOB}/{i:016x}/frames/thumbs/{i}.jpg" for i in range(1500)]
    keep = [f"{USER}/{JOB2}/00000000000000aa/results.zip", f"{USER}/{JOB3}0/00000000000000ab/results.zip"]
    stuck = [f"{USER}/{JOB3}/{i:016x}/logs/x.log" for i in range(1200)]
    out = run_js(f"""
        await advance(0);
        for (const o of [...{json.dumps(many)}, ...{json.dumps(keep)}, ...{json.dumps(stuck)}]) db.objects.add(o);
        const r = await T.fetch("/api/jobs/{JOB}", {{ method: "DELETE" }});
        const first = {{ status: r.status, body: await r.json(), removes: calls.filter((c) => c.remove).map((c) => [c.remove, c.names.length]),
          listed: calls.filter((c) => c.rpc === "job_objects").length, left: [...db.objects].filter((o) => o.includes("{JOB}/")).length }};
        calls.length = 0;
        db.fail.remove = "the object is not yours";
        const r2 = await T.fetch("/api/jobs/{JOB3}", {{ method: "DELETE" }});
        const second = {{ status: r2.status, removes: calls.filter((c) => c.remove).length, listed: calls.filter((c) => c.rpc === "job_objects").length }};
        calls.length = 0;
        db.fail.remove = null;
        db.rpc.job_objects = () => {{ db.offline = true; return {{ data: [{{ name: "{stuck[0]}" }}] }}; }};
        const r3 = await T.fetch("/api/jobs/{JOB3}", {{ method: "DELETE" }});
        db.offline = false;
        out({{ first, second, third: [r3.status, calls.filter((c) => c.remove).length], kept: {json.dumps(keep)}.every((o) => db.objects.has(o)),
          stuck: {json.dumps(stuck)}.filter((o) => db.objects.has(o)).length }});
        """, setup=account(jobs_=[job_row(), job_row(JOB2), job_row(JOB3)]) + "db.rpc.delete_job = () => ({ data: 'deleting' });")
    assert out["first"] == {"status": 200, "body": {"deleted": JOB}, "removes": [["job-files", 1000], ["job-files", 500]],
                            "listed": 2, "left": 0}
    assert out["second"] == {"status": 200, "removes": 1, "listed": 1}
    assert out["third"] == [200, 1] and out["kept"] is True and out["stuck"] == 1200


def test_offline_and_signed_out_reads_say_so():
    """A lost connection makes T.fetch throw, so app.js's api() says that Supabase cannot be reached; PostgREST's 401
    gives app.js's signed-out text."""
    out = run_js("""
        await advance(0);
        db.offline = true;
        const off = await api("/api/jobs").catch((e) => e.message);
        db.offline = false;
        db.fail.jobs = [401, "JWT expired"];
        const out1 = await api("/api/jobs").catch((e) => e.message);
        out({ off, out1, conn: $("#conn").textContent });
        """, setup=account())
    assert out == {"off": "Cannot reach Supabase. Check your connection.",
                   "out1": "Signed out. Reload to sign in.",
                   "conn": "Cannot reach Supabase. Check your connection."}


# ---- supa.js: the live view and signed URLs again ----


def test_live_view_gives_state_then_the_newest_runs_log_then_follows_and_ends():
    """RelayEventSource: on open the job's state, then the log rows of the newest stage run (lastEventId <stage>:<pos>),
    then Realtime inserts and state changes; when the status leaves the live statuses it gives the last log rows,
    then end, and closes its channel."""
    logs = [{"id": 1, "job_id": JOB, "stage": "masks", "run_started": 100.0, "pos": 10, "text": "old run\n"},
            {"id": 2, "job_id": JOB, "stage": "sam3d", "run_started": 200.0, "pos": 5, "text": "== sam3d\n"},
            {"id": 3, "job_id": JOB, "stage": "sam3d", "run_started": 200.0, "pos": 9, "text": "step 1\n"}]
    out = run_js(f"""
        await advance(0);
        const got = [];
        const es = T.events("/api/jobs/{JOB}/events");
        for (const t of ["state", "log", "end"]) es.addEventListener(t, (e) => got.push([t, JSON.parse(e.data), e.lastEventId]));
        es.onopen = () => got.push(["open"]);
        await advance(100);
        emit("job_logs", "INSERT", {{ id: 4, job_id: "{JOB}", stage: "sam3d", run_started: 200.0, pos: 15, text: "step 2\\n" }});
        db.tables.job_logs.push({{ id: 5, job_id: "{JOB}", stage: "sam3d", run_started: 200.0, pos: 22, text: "done\\n" }});
        emit("job_live", "UPDATE", {{ job_id: "{JOB}", state: {{ status: "failed", stages: [], warnings: [] }}, updated_at: "2026-10-24T15:00:00+00:00" }});
        await advance(100);
        out({{ got, state: es.readyState, removed: calls.filter((c) => c.removeChannel).length }});
        """, setup=account(jobs_=[job_row(status="running")]) + f"""
        db.tables.job_live = [{{ job_id: "{JOB}", state: {{ status: "running", current: "sam3d", stages: [], warnings: [] }}, updated_at: "2026-10-24T14:59:00+00:00" }}];
        db.tables.job_logs = {json.dumps(logs)};
        """)
    got = out["got"]
    assert got[0] == ["state", {"status": "running", "current": "sam3d", "stages": [], "warnings": []}, ""]
    assert got[1:3] == [["log", {"stage": "sam3d", "text": "== sam3d\n"}, "sam3d:5"], ["log", {"stage": "sam3d", "text": "step 1\n"}, "sam3d:9"]]
    assert got[3] == ["open"]
    assert got[4] == ["log", {"stage": "sam3d", "text": "step 2\n"}, "sam3d:15"]
    assert got[5][0] == "state" and got[5][1]["status"] == "failed"
    assert got[6] == ["log", {"stage": "sam3d", "text": "done\n"}, "sam3d:22"] and got[7] == ["end", {"status": "failed"}, ""]
    assert len(got) == 8 and out["state"] == 2 and out["removed"] == 1


def test_live_view_polls_when_realtime_does_not_connect():
    """No SUBSCRIBED within 5 s: the live view polls job_logs (id > last) and job_live (updated_at > last) every 2 s."""
    out = run_js(f"""
        await advance(0);
        const got = [];
        const es = T.events("/api/jobs/{JOB}/events");
        es.addEventListener("log", (e) => got.push(JSON.parse(e.data).text));
        es.onopen = () => got.push("open");
        await advance(4000);
        const early = [...got];
        db.tables.job_logs.push({{ id: 9, job_id: "{JOB}", stage: "da3", run_started: 1.0, pos: 4, text: "polled\\n" }});
        await advance(3500);
        out({{ early, got, polls: calls.filter((c) => c.table === "job_logs" && c.filters.some((f) => f[0] === "gt")).length }});
        """, setup=account(jobs_=[job_row(status="running")]) + f"""
        db.realtime = null;
        db.tables.job_live = [{{ job_id: "{JOB}", state: {{ status: "running", stages: [], warnings: [] }}, updated_at: "2026-10-24T14:59:00+00:00" }}];
        """)
    assert out["early"] == [] and out["got"] == ["polled\n", "open"] and out["polls"] >= 1


def test_urls_are_signed_again_before_they_end_and_old_download_links_on_click():
    """The open job's URLs are signed again before they end (every 5 min, those with less than 15 min left); a click on
    a download link older than 8 min is held, the link signed again, and the browser sent to the new one."""
    obj = f"{USER}/{JOB}/00000000000000aa/results.zip"
    out = run_js(f"""
        await advance(0);
        await T.fetch("/api/jobs/{JOB}");
        const first = T.url("/api/jobs/{JOB}/file/results.zip", "j_results.zip");
        await advance(46 * 60000);
        const views = calls.filter((c) => c.sign && c.seconds === 3600).length;
        const a = new Elem("a");
        a.setAttribute("href", first);
        let held = false;
        for (const l of docListeners.filter((x) => x.type === "click")) l.fn({{ target: a, preventDefault() {{ held = true; }} }});
        await advance(0);
        out({{ first, views, held, href: a.getAttribute("href"), assigned }});
        """, setup=account(jobs_=[job_row(status="done")]) + f"""
        db.tables.job_views = [{json.dumps(view(status="done", files=["results.zip"]))}];
        db.tables.job_files = [{{ job_id: "{JOB}", path: "results.zip", object: "{obj}", role: "result" }}];
        db.objects.add("{obj}");
        """, href=f"https://x.pages.dev/#/job/{JOB}")
    assert out["views"] >= 2 and out["held"] is True
    assert out["href"] != out["first"] and out["href"].endswith("&download=j_results.zip") and out["assigned"] == [out["href"]]


def test_model_viewer_decoders_come_from_the_sites_folder():
    """supa.js points model-viewer's Draco and KTX2 decoders at the vendor folder next to the page, under any path."""
    out = run_js("await advance(0); out(MV);", setup=account(), href="https://x.github.io/HARPOR/index.html#/job/1")
    assert out == {"dracoDecoderLocation": "https://x.github.io/HARPOR/static/vendor/three-0.183.0/draco/",
                   "ktx2TranscoderLocation": "https://x.github.io/HARPOR/static/vendor/three-0.183.0/basis/"}
    for path in ("draco/draco_wasm_wrapper.js", "draco/draco_decoder.wasm", "basis/basis_transcoder.js",
                 "basis/basis_transcoder.wasm"):
        assert (server.STATIC / "vendor" / "three-0.183.0" / path).is_file()


def test_page_works_without_local_storage():
    """With localStorage blocked (a private window) the page still signs in, lists jobs and answers a pending request."""
    out = run_js(f"""
        storeBroken = true;
        await advance(0);
        const list = await (await T.fetch("/api/jobs")).json();
        db.online = false;
        const r = await T.fetch("/api/jobs/{JOB}/continue", {{ method: "POST", body: "{{}}" }});
        out({{ n: list.length, status: r.status }});
        """, setup=relay_setup([worker(seen_ago=999)], [job_row()], [view()]), href=f"https://x.pages.dev/#/job/{JOB}")
    assert out == {"n": 1, "status": 202}
