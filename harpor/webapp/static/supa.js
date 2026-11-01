// HARPOR web app on the public site: the transport that app.js uses (its T) when the page comes from the static site
// that site.py builds and talks to Supabase, not to the server behind the SSH tunnel. Every job runs on one of the
// user's own GPU machines. Reads come from the tables that the personal workers on those machines keep up to date (job
// views, live state, logs, snapshots and file lists). Writes go through rpc("relay") to the job's machine, and its
// answer is the server's own status and JSON. Files are Storage objects behind short signed URLs, and an API key is
// sealed in the page to the public key of the machine's worker (WebCrypto) before it leaves. This page uploads
// nothing: inputs go in on the machine. This file also draws the sign-in, the banner of the user's machines that are
// offline or stopping, the waiting-for-approval panel, the line of requests that wait for a machine, and the panel
// about the machines, and it shows the account menu through account.js; site.py writes their markup into the public
// index.html.
// site.py loads static/vendor/supabase-2.118.0.js, then config.js (HARPOR_CONFIG), then account.js, then this file,
// then app.js. In tunnel mode none of them load, HARPOR_TRANSPORT stays undefined and app.js uses its default.
"use strict";
(function () {
  const C = typeof HARPOR_CONFIG === "undefined" ? null : HARPOR_CONFIG;
  if (!C) return;
  const $ = (sel) => document.querySelector(sel);
  const never = new Promise(() => {});
  // GitHub Pages cannot send frame-ancestors. In a frame (a click trap) the page shows nothing and never routes.
  if (window.top !== window.self) {
    document.documentElement.hidden = true;
    window.HARPOR_TRANSPORT = { fetch: () => never, url: () => "", events: () => null, live: [], keyless: () => false,
      noJobs: "", ready: never, liveLog: "", offline: "", unauthorized: "" };
    return;
  }

  const VIEW_S = 3600, DL_S = 600;  // signed URLs: 1 h for what the page shows, 10 min for download links
  const RESIGN_MS = 15 * 60000, DL_OLD_MS = 8 * 60000;  // sign again this long before the end; a download link this old
  const WAIT_MS = 60000, WAIT_KEY_MS = 120000;  // how long a relayed request waits for its answer; with a key check
  const LIVE = ["analyzing", "queued", "running"];  // the statuses that keep the live view open
  const OPEN = ["pending", "taken"];  // a request that has no answer yet
  const BUILD = ["start", "overrides", "parts", "continue"];  // the routes that may carry an API key
  const KEY = /^[A-Za-z0-9_-]{1,512}$/;  // jobs.KEY
  const PROVIDER = { astra: "OpenAI", opus: "Anthropic" };  // steps.PROVIDER
  // steps.STAGES: each stage's name and whether it runs on the GPU, for the view of a job that no worker has seen yet.
  const STAGES = [["topics", false], ["frames", false], ["extract", false], ["masks", true], ["prep", false], ["da3", true],
    ["lidar", false], ["select", false], ["sam3d", true], ["place", false], ["propose", false], ["parts", true], ["agent", true],
    ["align", false], ["vomp", true], ["mjcf", false], ["usd", false], ["urdf", false], ["turntable", true], ["package", false]];
  const SNAP_NONE = { frames: "the frames are not analyzed yet", selection: "the views are not selected yet",
    parts: "the part list is not on the site yet" };
  const ROUTE_NAME = { topic: "Camera topic", calibration_put: "Calibration", calibration_delete: "Remove calibration",
    start: "Start", overrides: "Select again", parts: "Build", continue: "Build", cancel: "Cancel", delete: "Delete" };
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const OFFLINE = "Cannot reach Supabase. Check your connection.";
  const MISSING = "data:,missing:";  // the URL of a file that is not on the site: an image fails at once, T.fetch gives 404
  // What a relay refusal says when the machine went, stops or restarted (relay() has already tried once more); others as
  // they are.
  const WORKER_GONE = { offline: "The machine went offline. Try again.", ending: "The machine is stopping. Try again.",
    restarted: "The machine restarted. Paste the key again." };
  const JOB_COLS = "id,created_at,status,note,prompt,queue_pos,input_name,input_kind,input_bytes,machine_id";
  const WORKER_COLS = "id,machine_id,public_key,local_keys,started_at,last_seen,ends_at,accepting,stopped_at";

  const A = HARPOR_ACCOUNT, sb = A.claim();  // account.js's one client (PKCE); this file draws the menu
  let user = null, access = null, settings = { worker_online_s: 30 }, readyFn = null, seq = 0;
  const own = new Map();  // machine id -> the worker row of that machine that beat last
  const machines = new Map();  // machine id -> name
  const rows = new Map();  // job id -> its jobs row, for workerFor and keyless
  const views = new Map();  // job id -> the view that GET /api/jobs/{id} last gave
  const signed = new Map();  // "<job>/<path>" -> {url, exp, object}: what the page shows
  const dls = new Map();  // "<job>/<path>" -> {url, at, object}: download links
  const issued = new Map();  // every download link given out -> {k, at, object}, so an old one on the page is known
  const notes = new Map();  // job id -> what the pending line adds, such as a key that was not sent
  const waiters = new Map();  // request id -> the function that ends its wait

  // ---- small helpers ----

  const reply = (status, body) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  class Offline extends Error {}
  // A PostgREST or RPC call: its {data, error, status}. A lost connection throws, so app.js's api() says so.
  async function q(call) {
    const r = await call;
    if (r.status === 0) throw new Offline(OFFLINE);
    return r;
  }
  // The answer to a call that the database refused: its status, and its message as the server's "detail".
  const fail = (r) => reply(r.status === 401 ? 401 : httpOf(r.error.message), { detail: plain(r.error.message) });
  function httpOf(msg) {
    return /^too many/.test(msg) ? 429 : /^no such/.test(msg) ? 404
      : /waiting for approval|may use the agents|runs the agents/.test(msg) ? 403 : /^worker_/.test(msg) ? 503
        : /^the job is|still waits|already waits|does not wait/.test(msg) ? 409 : 400;
  }
  function plain(msg) {
    const s = String(msg || ""), m = /^worker_(offline|ending|restarted): /.exec(s);
    return m ? WORKER_GONE[m[1]] : s;
  }
  // localStorage, which a private window or blocked site data can take away: the page works without it.
  const keep = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* no remembered answers */ } },
  };
  function el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k.startsWith("on")) e[k] = v;
      else if (k === "class") e.className = v;
      else if (v === true) e.setAttribute(k, "");
      else if (v !== false && v != null) e.setAttribute(k, v);
    }
    e.append(...kids.flat().filter((x) => x != null && x !== false));
    return e;
  }
  function say(e, text, info) {  // a message line: red for a problem, muted for progress
    e.textContent = text;
    e.className = info ? "msg info" : "msg";
  }
  const pad = (n) => String(n).padStart(2, "0");
  const hm = (t) => { const d = new Date(t); return `${pad(d.getHours())}:${pad(d.getMinutes())}`; };  // "05:35", local time
  const when = (t) => { const d = new Date(t); return `${MONTHS[d.getMonth()]} ${d.getDate()} ${hm(t)}`; };  // "Oct 24 05:35"
  // A timestamp as job.json's "created" has it (local time), which app.js shows without Date.
  const localIso = (t) => {
    const d = new Date(t);
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  };
  const viewing = (id) => new RegExp(`^#/job/${id}(/|$)`).test(location.hash);
  const appRefresh = () => { if (typeof refresh === "function") refresh().catch(() => {}); };  // app.js's refresh()

  // ---- machines: the banner, and which worker serves a job ----

  const isOnline = (w) => !!w && !w.stopped_at && Date.now() - Date.parse(w.last_seen) < 1000 * (settings.worker_online_s || 30);
  const machineName = (id) => machines.get(id) || "your machine";
  const whereOf = (row) => "on " + machineName(row.machine_id);
  const workerFor = (row) => !row ? null : own.get(row.machine_id) || null;
  // Every 20 s, and before each relayed request: the user's machines (one may have been added or renamed since the page
  // opened), and their workers (RLS gives the user's own), the last beat first. A live worker beats every 10 s, so many
  // restarts of another machine never push it out of the list, and it is the one that relay() seals keys to (the only
  // worker of its machine that is not stopped).
  async function readWorkers() {
    const [m, w] = await Promise.all([q(sb.from("machines").select("id,name").order("created_at", { ascending: true })),
      q(sb.from("workers").select(WORKER_COLS).order("last_seen", { ascending: false })
        .limit(10 * Math.max(machines.size, settings.max_machines || 5)))]);
    if (m.error || w.error) return;
    machines.clear();
    for (const x of m.data) machines.set(x.id, x.name);
    own.clear();
    for (const x of w.data) if (!own.has(x.machine_id)) own.set(x.machine_id, x);
    banner();
    ownPanel();
  }
  // [text, online, stopping]; help.html#machine-status says what each state means.
  function machineLine(name, w) {
    if (isOnline(w)) {
      if (w.accepting === false) return [`${name}: stopping ${w.ends_at ? "at " + hm(w.ends_at) : "soon"}`, true, true];
      return [w.ends_at ? `${name}: online until ${hm(w.ends_at)}` : `${name}: online`, true, false];
    }
    // no worker row: RLS shows only the workers seen in the last 30 days
    return [w ? `${name}: offline since ${when(w.stopped_at || w.last_seen)}` : `${name}: offline`, false, false];
  }
  function banner() {  // a line for each of the user's machines that is offline or stopping; the machines panel lists all
    const box = $("#harpor-banner");
    if (!box || !access) return;
    const lines = [...machines].map(([id, name]) => machineLine(name, own.get(id))).filter(([, on, stopping]) => !on || stopping);
    const key = JSON.stringify(lines);
    if (box.dataset.lines === key) return;  // a status region: only a change is read out
    box.dataset.lines = key;
    box.replaceChildren(...lines.map(([text, on]) => el("p", { class: on ? "stopping" : "off" }, text)));
    box.hidden = !lines.length;
  }
  // The panel about the user's GPU machines, where the tunnel page has its upload area: each machine with its state
  // and the keys it holds. site.py adds the link to help.html#machine, which says how to set one up.
  function ownPanel() {
    const box = $("#harpor-own");
    if (!box || !access || !access.approved) return;
    box.hidden = false;
    const list = [...machines].map(([id, name]) => {
      const w = own.get(id), keys = isOnline(w) ? (w.local_keys || []).map((a) => PROVIDER[a]).filter(Boolean) : [];
      const state = machineLine(name, w)[0].slice(name.length + 2);  // "online", "offline since Oct 24 05:35", ...
      return el("li", {}, el("b", {}, name), `: ${state}.`,
        keys.length ? ` Has its own ${keys.join(" and ")} key${keys.length > 1 ? "s" : ""}.` : null);
    });
    $("#harpor-machines").replaceChildren(...(list.length ? list : [el("li", { class: "hint" }, "No machines yet.")]));
  }
  // T.keyless: a job on a machine whose live worker holds that agent's key needs no key from the page.
  function keyless(id, agent) {
    const row = rows.get(id), w = workerFor(row);
    if (!row || !isOnline(w) || !(w.local_keys || []).includes(agent)) return false;
    return `${machineName(row.machine_id)} uses the ${PROVIDER[agent]} key stored on it.`;
  }

  // ---- sign-in and approval (site.py writes the markup; account.js draws the account menu) ----

  const redirectTo = () => new URL("./", location.href).href;  // the site's folder: at a domain root and under /<repo>/
  function panel(name) {
    for (const p of ["signin", "waiting"]) $("#harpor-" + p).hidden = p !== name;
  }
  async function signIn(provider) {
    say($("#harpor-signinmsg"), "", true);
    const { error } = await sb.auth.signInWithOAuth({ provider, options: { redirectTo: redirectTo() } });
    if (error) say($("#harpor-signinmsg"), error.message);
  }
  // A refused sign-in comes back as ?error=...&error_description=...; the address bar keeps no trace of it.
  function urlError() {
    const u = new URL(location.href), hash = new URLSearchParams(u.hash.replace(/^#/, ""));
    const text = u.searchParams.get("error_description") || hash.get("error_description");
    if (text === null) return null;
    for (const k of ["error", "error_code", "error_description"]) u.searchParams.delete(k);
    try { history.replaceState(history.state, "", u.pathname + u.search + (hash.get("error_description") ? "" : u.hash)); }
    catch (e) { /* the message still shows */ }
    return text;
  }
  function waiting() {  // signed in, not approved: the request-access box, once a day
    panel("waiting");
    if (access && access.request_at) say($("#harpor-requestmsg"), `Request sent ${when(access.request_at)}.`, true);
    $("#harpor-request").onsubmit = async (e) => {
      e.preventDefault();
      $("#harpor-send").disabled = true;
      const r = await q(sb.rpc("request_access", { p_note: $("#harpor-note").value.trim() })).catch(() => ({ error: { message: OFFLINE } }));
      $("#harpor-send").disabled = false;
      say($("#harpor-requestmsg"), r.error ? plain(r.error.message) : "Request sent.", !r.error);
    };
  }
  // The agent radios: only the agents the account may use (site.py has already removed the "None" agent).
  function agentRadios() {
    const radios = [...document.querySelectorAll("input[name=agent]")];
    for (const r of radios) {
      const off = !(access.agents || []).includes(r.value);
      r.disabled = off;
      if (off) r.checked = false;
      const label = r.closest && r.closest("label");
      if (label) label.hidden = off;
    }
    if (!radios.some((r) => r.checked)) {
      const first = radios.find((r) => !r.disabled);
      if (first) first.checked = true;
    }
  }
  function decoders() {  // model-viewer's Draco and KTX2 decoders from the site's own vendor folder, not from gstatic.com
    if (typeof customElements === "undefined") return;
    customElements.whenDefined("model-viewer").then(() => {
      const MV = customElements.get("model-viewer");
      MV.dracoDecoderLocation = new URL("static/vendor/three-0.183.0/draco/", location.href).href;
      MV.ktx2TranscoderLocation = new URL("static/vendor/three-0.183.0/basis/", location.href).href;
    });
  }
  async function gate() {
    const err = urlError();
    $("#harpor-google").onclick = () => signIn("google");
    $("#harpor-github").onclick = () => signIn("github");
    let session = null;
    try { session = (await sb.auth.getSession()).data.session; } catch (e) { /* signed out */ }
    if (!session) {
      panel("signin");
      if (err) say($("#harpor-signinmsg"), "Sign-in failed: " + err);
      return;
    }
    user = session.user;
    sb.auth.onAuthStateChange((event) => { if (event === "SIGNED_OUT") location.reload(); });
    A.show(user, () => location.reload());  // the account menu; after Sign out the page reloads to the sign-in
    const [a, s] = await Promise.all([q(sb.from("access").select("approved,agents,max_result_bytes,request_at").maybeSingle()),
      q(sb.from("settings").select("worker_online_s,max_machines").maybeSingle())]);
    access = a.data;
    if (s.data) settings = s.data;
    if (!access || !access.approved) return waiting();
    panel(null);
    A.storage(access).catch(() => {});
    agentRadios();
    decoders();
    sb.channel("harpor-requests-" + user.id).on("postgres_changes",
      { event: "UPDATE", schema: "public", table: "requests", filter: `user_id=eq.${user.id}` }, (msg) => answered(msg.new)).subscribe();
    await readWorkers().catch(() => {});
    ownPanel();
    setInterval(() => { readWorkers().catch(() => {}); }, 20000);
    setInterval(() => { resign().catch(() => {}); }, 5 * 60000);
    readyFn();  // app.js routes now
  }

  // ---- reads ----

  const listRow = (r) => ({ id: r.id, created: localIso(r.created_at), status: r.status, prompt: r.prompt || "", video: r.input_name,
    kind: r.input_kind, queue_pos: r.queue_pos, where: whereOf(r), deletable: canDelete(r) });
  // delete_job takes a job in any state while its machine is offline, since that machine may never come back.
  const canDelete = (r) => !isOnline(own.get(r.machine_id));
  async function listJobs() {  // RLS gives the user's own rows
    const r = await q(sb.from("jobs").select(JOB_COLS).order("created_at", { ascending: false }));
    if (r.error) return fail(r);
    for (const row of r.data) rows.set(row.id, row);
    await signThumbs(r.data.map((row) => row.id));
    return reply(200, r.data.map(listRow));
  }
  async function readRow(id) {
    const r = await q(sb.from("jobs").select(JOB_COLS).eq("id", id).maybeSingle());
    if (r.data) rows.set(id, r.data);
    return r.data || null;
  }
  async function readView(id) {  // the job's agent, for a request sent before the job page was drawn
    const r = await q(sb.from("job_views").select("body").eq("job_id", id).maybeSingle());
    return r.data ? r.data.body : null;
  }
  async function getJob(id) {
    const box = $("#harpor-pending");
    if (box && box.dataset.job !== id) { box.replaceChildren(); box.hidden = true; }  // another job's lines
    const [r, v] = await Promise.all([q(sb.from("jobs").select(JOB_COLS).eq("id", id).maybeSingle()),
      q(sb.from("job_views").select("body").eq("job_id", id).maybeSingle())]);
    if (r.error) return fail(r);
    const row = r.data;
    if (!row) return reply(404, { detail: "no such job" });
    rows.set(id, row);
    const body = v.data ? v.data.body : pendingView(row);
    body.queue_pos = row.queue_pos;
    body.where = whereOf(row);
    if (row.note && !(body.warnings || []).includes(row.note)) body.warnings = [...(body.warnings || []), row.note];
    views.set(id, body);
    await signJob(id);
    // A preview that the worker left out (too large for the site) shows as missing, not as a video that fails.
    if (body.preview && body.preview.file && !signed.has(id + "/frames/preview.mp4")) body.preview = { missing: true };
    await showPending(id).catch(() => {});
    return reply(200, body);
  }
  // GET /api/jobs/{id} of a job whose machine has not sent its view yet: the keys of the server's answer for a fresh job
  // (a test compares them), every stage pending, and no files.
  function pendingView(row) {
    const stage = { status: "pending", started: null, ended: null, returncode: null, note: null };
    return { id: row.id, created: localIso(row.created_at), video: { name: row.input_name, bytes: row.input_bytes },
      source: { kind: row.input_kind, topic: null }, agent: "astra", status: row.status, mode: "auto", prompt: row.prompt,
      slug: null, n_views: 6, picks: [], view0: null, auto_continue: false, overrides: { locked: [], excluded: [], instance: {} },
      articulated: false, parts: { mode: "vlm", list: [], confirmed: false },
      lidar: row.input_kind === "stray" ? { kind: "stray" } : null, calibration: null, lidar_current: null,
      lidar_stale: null, turntable: false,
      stages: Object.fromEntries(STAGES.map(([name]) => [name, { ...stage }])), current: null, error: null,
      warnings: [], key_ok: false, kind: row.input_kind,
      parts_mode: "vlm", parts_review: false, preview: null,
      progress: STAGES.map(([name, gpu]) => ({ ...stage, name, gpu, seconds: null, progress: null })),
      files: [], sizes: {}, agent_result: null };
  }
  async function snapshot(id, path) {  // frames, selection and parts: the route's own status and body
    const r = await q(sb.from("job_snapshots").select("status,body").eq("job_id", id).eq("path", path).maybeSingle());
    if (r.error) return fail(r);
    return r.data ? reply(r.data.status, r.data.body) : reply(404, { detail: SNAP_NONE[path] });
  }

  // ---- files: signed URLs made ahead, so that T.url answers at once ----

  // [{job_id, path, object}] -> signed URLs, 100 a call; a file whose object is gone gets none.
  async function sign(files, seconds, put) {
    for (let i = 0; i < files.length; i += 100) {
      const part = files.slice(i, i + 100), by = new Map(part.map((f) => [f.object, f]));
      const { data, error } = await sb.storage.from("job-files").createSignedUrls(part.map((f) => f.object), seconds);
      if (error || !Array.isArray(data)) continue;
      for (const d of data) if (d.signedUrl && by.has(d.path)) put(by.get(d.path), d.signedUrl);
    }
  }
  const signView = (files) => sign(files, VIEW_S, (f, url) => signed.set(f.job_id + "/" + f.path,
    { url, exp: Date.now() + VIEW_S * 1000, object: f.object }));
  const signDownload = (files) => sign(files, DL_S, (f, url) => {
    const k = f.job_id + "/" + f.path, d = { url, at: Date.now(), object: f.object };
    dls.set(k, d);
    issued.set(url, { k, ...d });
    if (issued.size > 2000) issued.delete(issued.keys().next().value);
  });
  // A URL counts while it has more than RESIGN_MS left and names the same object (a changed file gets a new name).
  const fresh = (k, object) => { const s = signed.get(k); return !!s && s.exp - Date.now() > RESIGN_MS && (!object || s.object === object); };
  async function signThumbs(ids) {  // the list shows frame 0 of each job
    if (!ids.length) return;
    const r = await q(sb.from("job_files").select("job_id,path,object").in("job_id", ids).eq("path", "frames/thumbs/0.jpg"));
    if (!r.error) await signView(r.data.filter((f) => !fresh(f.job_id + "/" + f.path, f.object)));
  }
  async function signJob(id) {
    const r = await q(sb.from("job_files").select("job_id,path,object,role").eq("job_id", id));
    if (r.error) return;
    await signView(r.data.filter((f) => !fresh(id + "/" + f.path, f.object)));
    // Results and logs also get short download links; an old one is signed again when it is clicked.
    await signDownload(r.data.filter((f) => f.role !== "review").filter((f) => {
      const d = dls.get(id + "/" + f.path);
      return !d || d.object !== f.object || Date.now() - d.at > DL_OLD_MS;
    }));
    const have = new Set(r.data.map((f) => id + "/" + f.path));  // a file that went is missing again
    for (const m of [signed, dls]) for (const k of [...m.keys()]) if (k.startsWith(id + "/") && !have.has(k)) m.delete(k);
  }
  async function resign() {  // every 5 min: the open job's URLs that end within 15 min
    for (const id of views.keys()) if (viewing(id)) await signJob(id);
  }
  // T.url: /api/jobs/<id>/file/<path> or /api/jobs/<id>/thumb/<i> as a signed URL; with a name, a download link
  // (Storage sends the file with that name, which the download attribute cannot do across origins).
  function tUrl(path, name) {
    const m = /^\/api\/jobs\/([^/]+)\/(file|thumb)\/(.+)$/.exec(path);
    if (!m) return path;
    const p = m[2] === "thumb" ? `frames/thumbs/${m[3]}.jpg` : m[3], k = m[1] + "/" + p;
    const d = name ? dls.get(k) || signed.get(k) : signed.get(k);
    if (!d) return MISSING + encodeURIComponent(p);
    return name ? d.url + "&download=" + encodeURIComponent(name) : d.url;
  }
  // A download link older than 8 min is signed again when it is clicked (download links last 10 min): the link on
  // the page may be older than the newest one, which the background signing keeps for the next draw.
  document.addEventListener("click", (e) => {
    const a = e.target && e.target.closest ? e.target.closest("a[href]") : null, href = a ? a.getAttribute("href") || "" : "";
    const cut = href.indexOf("&download="), old = cut > 0 ? issued.get(href.slice(0, cut)) : null;
    if (!old || Date.now() - old.at < DL_OLD_MS) return;
    e.preventDefault();
    const name = href.slice(cut + "&download=".length), slash = old.k.indexOf("/"), now = dls.get(old.k);
    const go = () => {
      const d = dls.get(old.k);
      if (!d || Date.now() - d.at >= DL_OLD_MS) return;
      a.setAttribute("href", d.url + "&download=" + name);
      location.assign(d.url + "&download=" + name);
    };
    if (now && Date.now() - now.at < DL_OLD_MS) return go();
    signDownload([{ job_id: old.k.slice(0, slash), path: old.k.slice(slash + 1), object: old.object }]).then(go).catch(() => {});
  }, true);

  // ---- writes: the relay ----

  // The agent that server.py will run for a request (server.agent_of): start names it (astra when it names none);
  // overrides and continue name it or take the job's; parts always takes the job's.
  function effectiveAgent(route, body, view) {
    const job = view ? view.agent : "astra";
    if (route === "start") return "agent" in body ? body.agent : "astra";
    if (route === "parts" || body.agent == null) return job;
    return body.agent;
  }
  function keyKindError(key, agent) {  // jobs.key_kind_error: the key's provider by its prefix alone
    if (agent === "opus") return key.startsWith("sk-ant-") ? null : "Anthropic keys start with sk-ant-";
    if (key.startsWith("sk-ant-")) return "this is an Anthropic key (sk-ant-); this agent needs an OpenAI key";
    return key.startsWith("sk-") ? null : "OpenAI keys start with sk-";
  }
  // Seal an API key to a worker's ECDH P-256 key: a one-time key pair, HKDF-SHA256 with the one-time public key as
  // salt, AES-256-GCM with a random 12-byte IV. The AAD ties it to the job, the user, the agent, the worker and this
  // request's ref, so it opens nowhere else.
  async function seal(apiKey, job, uid, agent, w) {
    const ref = crypto.randomUUID(), enc = new TextEncoder(), subtle = crypto.subtle;
    const b64 = (u8) => btoa(String.fromCharCode(...u8)), unb64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
    const wk = await subtle.importKey("raw", unb64(w.public_key), { name: "ECDH", namedCurve: "P-256" }, false, []);
    const eph = await subtle.generateKey({ name: "ECDH", namedCurve: "P-256" }, false, ["deriveBits"]);
    const epk = new Uint8Array(await subtle.exportKey("raw", eph.publicKey));
    const secret = await subtle.deriveBits({ name: "ECDH", public: wk }, eph.privateKey, 256);
    const aad = enc.encode(`harpor-key-v1|${job}|${uid}|${agent}|${w.id}|${ref}`);
    const hk = await subtle.importKey("raw", secret, "HKDF", false, ["deriveKey"]);
    const aes = await subtle.deriveKey({ name: "HKDF", hash: "SHA-256", salt: epk, info: aad }, hk,
      { name: "AES-GCM", length: 256 }, false, ["encrypt"]);
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const ct = new Uint8Array(await subtle.encrypt({ name: "AES-GCM", iv, additionalData: aad }, aes, enc.encode(apiKey)));
    return { ref, worker_id: w.id, agent, epk: b64(epk), iv: b64(iv), ct: b64(ct) };
  }
  function body(opts) {  // the JSON object that app.js sent, or null
    try {
      const b = JSON.parse(opts.body || "{}");
      return b && typeof b === "object" && !Array.isArray(b) ? b : null;
    } catch (e) { return null; }
  }
  async function utf8(data) {  // a calibration file as text, or null when it is not UTF-8
    if (typeof data === "string") return data;
    try { return new TextDecoder("utf-8", { fatal: true }).decode(await data.arrayBuffer()); } catch (e) { return null; }
  }
  // One relayed request. With the job's machine online and taking builds, an API key is sealed to its worker, or none
  // is sent when the machine holds the agent's key. With the machine offline (or about to stop, for a build) the key
  // cannot be sealed: the request says key_later and waits as pending, and the job asks for the key when it needs it.
  // The answer is the server's status and JSON; with no answer in time, 202 {"status": "pending"} and the pending line
  // on the job page.
  async function relay(id, route, opts, query, again) {
    const raw = ["calibration_put", "calibration_delete", "cancel"].includes(route) ? {} : body(opts);
    if (raw === null) return reply(400, { detail: "the request body is not a JSON object" });
    const text = route === "calibration_put" ? await utf8(opts.body) : null;
    if (route === "calibration_put" && text === null) return reply(400, { detail: "the calibration file is not UTF-8 text" });
    const row = rows.get(id) || await readRow(id);
    if (!row) return reply(404, { detail: "no such job" });
    await readWorkers();
    const w = workerFor(row), live = isOnline(w) && w.accepting === true;
    const b = { ...raw };
    let sealed = null, note = null;
    if (BUILD.includes(route)) {
      const agent = effectiveAgent(route, raw, views.get(id) || await readView(id));
      const key = typeof raw.api_key === "string" && raw.api_key ? raw.api_key : null;
      if ("api_key" in b || key) b.api_key = null;
      if (live && (w.local_keys || []).includes(agent)) delete b.key_later;  // the machine reads its own key file
      else if (live && key) {
        if (!KEY.test(key)) return reply(400, { detail: "Not a valid API key." });
        const bad = keyKindError(key, agent);
        if (bad) return reply(400, { detail: bad });
        sealed = await seal(key, id, user.id, agent, w);
      } else if (!live) {
        const who = machineName(row.machine_id) + (isOnline(w) ? " stops soon" : " is offline");
        if (key) note = `Key not sent: ${who}. The job will ask for it later.`;
        if (route !== "overrides") b.key_later = true;
      }
    }
    const { data, error } = await q(sb.rpc("relay", { p_job: id, p_route: route, p_query: query || {}, p_body: b,
      p_text: text, p_sealed: sealed }));
    // The machine went, came back or restarted since the banner read it: read it again and send once more.
    if (error && !again && /^worker_(offline|ending|restarted)/.test(error.message)) return relay(id, route, opts, query, true);
    if (error) return reply(httpOf(error.message), { detail: plain(error.message) });
    if (note) notes.set(id, note);
    else notes.delete(id);
    const waits = data.worker_online && (data.accepting || !BUILD.includes(route));
    const done = waits ? await waitFor(data.id, sealed ? WAIT_KEY_MS : WAIT_MS) : null;
    if (done) return reply(done.status || 409, done.response || { detail: "the request " + done.state });
    keep.set("harpor-waiting", [...keep.get("harpor-waiting", []), data.id].slice(-50));  // its answer shows when it comes
    await showPending(id).catch(() => {});
    return reply(202, { status: "pending", request: data.id });  // ok for api(); act() then refreshes
  }
  // The answer to a request: a Realtime update on requests (RLS: the user's own rows), with a 1 s poll as backup.
  function waitFor(rid, ms) {
    return new Promise((resolve) => {
      const t0 = Date.now();
      let timer = null;
      const finish = (r) => { clearInterval(timer); waiters.delete(rid); resolve(r); };
      waiters.set(rid, finish);
      timer = setInterval(async () => {
        if (Date.now() - t0 > ms) return finish(null);
        const r = await sb.from("requests").select("id,job_id,state,status,response").eq("id", rid).maybeSingle();
        if (r && r.data && !OPEN.includes(r.data.state)) finish(r.data);
      }, 1000);
    });
  }
  function answered(x) {  // a request row changed
    if (!x || OPEN.includes(x.state)) return;
    const wait = waiters.get(x.id);
    if (wait) return wait(x);
    if (keep.get("harpor-waiting", []).includes(x.id) && viewing(x.job_id)) {
      showPending(x.job_id).catch(() => {});
      appRefresh();
    }
  }
  const seen = (rid) => keep.set("harpor-waiting", keep.get("harpor-waiting", []).filter((x) => x !== rid));
  const routeName = (x) => x.route === "continue" && x.body && x.body.from ? "Retry" : ROUTE_NAME[x.route] || x.route;
  // The job page's line for each request that waits ("Start, sent 10:42: waits for lab-box [Withdraw]"), and for each
  // answer that came while the page was away or after its wait ran out, until the user dismisses it.
  async function showPending(id) {
    const box = $("#harpor-pending");
    if (!box) return;
    const r = await q(sb.from("requests").select("id,job_id,route,body,state,status,response,created_at")
      .eq("job_id", id).order("id", { ascending: false }).limit(10));
    if (r.error || !viewing(id)) return;
    const mine = keep.get("harpor-waiting", []), row = rows.get(id), note = notes.get(id);
    const who = row ? machineName(row.machine_id) : "your machine";
    const lines = r.data.filter((x) => OPEN.includes(x.state) || mine.includes(x.id)).reverse().map((x) => {
      const what = `${routeName(x)}, sent ${hm(x.created_at)}: `;
      if (x.state === "pending") {
        return el("p", {}, what + "waits for " + who, x.route === "delete" ? null
          : [" ", el("button", { type: "button", class: "small secondary", onclick: () => withdraw(id, x.id) }, "Withdraw")]);
      }
      if (x.state === "taken") return el("p", {}, what + who + " works on it");
      const res = x.response || {}, ok = x.status >= 200 && x.status < 300;
      // "queued" (the server's status), "refused: <why>", or why it never ran ("you withdrew this request")
      const said = ok ? (typeof res.status === "string" ? res.status : "done")
        : x.state === "done" ? "refused: " + (res.detail || x.status) : res.detail || x.state;
      return el("p", { class: ok ? null : "refused" }, what + said, " ",
        el("button", { type: "button", class: "small secondary", onclick: () => { seen(x.id); showPending(id).catch(() => {}); } }, "OK"));
    });
    if (note && lines.length) lines.push(el("p", { class: "hint" }, note));
    box.dataset.job = id;
    box.replaceChildren(...lines);
    box.hidden = !lines.length;
  }
  async function withdraw(id, rid) {
    const r = await q(sb.rpc("withdraw_request", { p_id: rid })).catch(() => ({ error: { message: OFFLINE } }));
    if (r.error) notes.set(id, plain(r.error.message));
    else { seen(rid); notes.delete(id); }
    await showPending(id).catch(() => {});
    appRefresh();
  }

  // ---- the transport that app.js calls ----

  // The files of a job that delete_job took, so that a machine that never comes back leaves none on the site:
  // job_objects names them, 1,000 a call, and the Storage policy lets users remove what is in their own folder. It
  // stops after 10 rounds or at the first call that fails; the machine removes what is left when it runs again.
  async function removeObjects(id) {
    for (let round = 0; round < 10; round++) {
      const r = await q(sb.rpc("job_objects", { p_job: id }));
      if (r.error || !Array.isArray(r.data) || !r.data.length) return;
      const { error } = await sb.storage.from("job-files").remove(r.data.map((o) => o.name));
      if (error || r.data.length < 1000) return;
    }
  }
  async function tFetch(url, opts = {}) {
    try {
      if (/^https?:\/\//.test(url)) return await fetch(url, opts);  // a signed Storage URL
      if (url.startsWith(MISSING)) return reply(404, { detail: "File not on the site yet." });
      const u = new URL(url, location.href), m = (opts.method || "GET").toUpperCase();
      const p = u.pathname.replace(/^.*?\/api\//, "api/").split("/");  // any prefix before /api/: the site's folder
      const [api, jobs, id, sub] = p;
      if (api !== "api" || jobs !== "jobs") return reply(404, { detail: "not on the public site" });
      if (m === "GET" && p.length === 2) return await listJobs();
      if (m === "GET" && p.length === 3) return await getJob(id);
      if (m === "GET" && p.length === 4 && ["frames", "selection", "parts"].includes(sub)) return await snapshot(id, sub);
      if (m === "GET" && (sub === "file" || sub === "thumb")) {  // a file by its API path: its signed URL
        return await tFetch(tUrl(u.pathname.slice(u.pathname.indexOf("/api/"))), opts);
      }
      if (m === "DELETE" && p.length === 3) {
        const r = await q(sb.rpc("delete_job", { p_job: id }));
        if (r.error) return reply(409, { detail: plain(r.error.message) });
        for (const map of [rows, views]) map.delete(id);
        await removeObjects(id).catch(() => {});
        return reply(200, { deleted: id });
      }
      // A calibration upload names its file and, when one is chosen, the LiDAR topic it is for (app.js calibrate()).
      if (sub === "calibration" && m === "PUT") {
        const topic = u.searchParams.get("lidar_topic");
        return await relay(id, "calibration_put", opts,
          { name: u.searchParams.get("name") || "", ...(topic ? { lidar_topic: topic } : {}) });
      }
      if (sub === "calibration" && m === "DELETE") return await relay(id, "calibration_delete", opts, {});
      if (m === "POST" && p.length === 4 && ["topic", "start", "overrides", "parts", "continue", "cancel"].includes(sub)) {
        return await relay(id, sub, opts);
      }
      return reply(404, { detail: "not on the public site" });
    } catch (e) {
      if (e instanceof Offline || e instanceof TypeError) throw e;  // api() says that Supabase cannot be reached
      return reply(500, { detail: "the page failed on this request: " + e.message });
    }
  }

  // ---- the live view: job_live and job_logs as app.js's EventSource ----

  // It has the surface that openStream() uses. On open it gives the job's state and the log of its newest stage run
  // (as server.tick gives a new client), then follows Realtime on both tables. If Realtime does not connect within 5 s
  // it polls every 2 s. When the status leaves LIVE and no newer log waits, it gives "end" and closes.
  class RelayEventSource {
    constructor(id) {
      Object.assign(this, { id, readyState: 0, onopen: null, onerror: null, on: {}, last: null, at: null, logId: 0,
        ch: null, poller: null, timer: null, ending: false });
      this.start().catch(() => this.lost());
    }
    addEventListener(type, fn) { (this.on[type] = this.on[type] || []).push(fn); }
    emit(type, data, lastEventId) {
      if (this.readyState === 2) return;
      for (const fn of this.on[type] || []) fn({ type, data: JSON.stringify(data), lastEventId: lastEventId || "" });
    }
    async start() {
      const live = await q(sb.from("job_live").select("state,updated_at").eq("job_id", this.id).maybeSingle());
      const top = await q(sb.from("job_logs").select("id,stage,run_started").eq("job_id", this.id)
        .order("id", { ascending: false }).limit(1));
      if (live.data) this.state(live.data.state, live.data.updated_at, true);
      const t = top.data && top.data[0];
      if (t) {
        const run = sb.from("job_logs").select("id,stage,run_started,pos,text").eq("job_id", this.id).eq("stage", t.stage);
        const r = await q((t.run_started == null ? run.is("run_started", null) : run.eq("run_started", t.run_started))
          .order("id", { ascending: true }));
        for (const x of r.data || []) this.log(x);
        this.logId = Math.max(this.logId, t.id);
      }
      if (this.readyState === 2) return;
      if (this.status != null && !LIVE.includes(this.status)) return this.checkEnd();  // the job stopped before
      const id = this.id, filter = `job_id=eq.${id}`;
      this.ch = sb.channel(`harpor-live-${id}-${++seq}`)
        .on("postgres_changes", { event: "*", schema: "public", table: "job_live", filter }, (m) => { if (m.new) this.state(m.new.state, m.new.updated_at); })
        .on("postgres_changes", { event: "INSERT", schema: "public", table: "job_logs", filter }, (m) => { if (m.new) this.log(m.new); })
        .subscribe((status) => {
          if (status === "SUBSCRIBED") this.open();
          else if (["CHANNEL_ERROR", "TIMED_OUT", "CLOSED"].includes(status) && this.readyState !== 2) this.poll();
        });
      this.timer = setTimeout(() => { if (!this.poller && this.readyState === 0) this.poll(); }, 5000);
    }
    open() {
      clearTimeout(this.timer);
      if (this.readyState === 2) return;
      this.readyState = 1;
      if (this.onopen) this.onopen();
      this.catchUp().catch(() => this.lost());  // what changed between the first read and the subscription
    }
    poll() {
      if (this.poller || this.readyState === 2) return;
      this.poller = setInterval(() => {
        this.catchUp().then(() => {
          if (this.readyState === 0) { this.readyState = 1; if (this.onopen) this.onopen(); }
        }).catch(() => this.lost());
      }, 2000);
    }
    lost() {  // app.js says "reconnecting"; the poll goes on
      if (this.readyState === 2) return;
      this.readyState = 0;
      if (this.onerror) this.onerror();
      this.poll();
    }
    async catchUp() {
      const [logs, live] = await Promise.all([
        q(sb.from("job_logs").select("id,stage,run_started,pos,text").eq("job_id", this.id).gt("id", this.logId)
          .order("id", { ascending: true }).limit(200)),
        q(sb.from("job_live").select("state,updated_at").eq("job_id", this.id).gt("updated_at", this.at || "1970-01-01T00:00:00Z")
          .maybeSingle())]);
      for (const x of logs.data || []) this.log(x);
      if (live.data) this.state(live.data.state, live.data.updated_at);
    }
    state(st, at, first) {
      if (!st || typeof st !== "object") return;
      if (at) this.at = at;
      const note = (rows.get(this.id) || {}).note;  // the jobs row's note, as GET /api/jobs/{id} gives it
      const s = note && Array.isArray(st.warnings) && !st.warnings.includes(note) ? { ...st, warnings: [...st.warnings, note] } : st;
      const key = JSON.stringify(s);
      if (key !== this.last) {
        this.last = key;
        this.status = s.status;
        this.emit("state", s);
      }
      if (!first) this.checkEnd();
    }
    log(x) {
      if (!x || x.id <= this.logId) return;
      this.logId = x.id;
      this.emit("log", { stage: x.stage, text: x.text }, `${x.stage}:${x.pos}`);
    }
    checkEnd() {
      if (this.ending || this.status == null || LIVE.includes(this.status) || this.readyState === 2) return;
      this.ending = true;
      const status = this.status;
      q(sb.from("job_logs").select("id,stage,run_started,pos,text").eq("job_id", this.id).gt("id", this.logId)
        .order("id", { ascending: true }).limit(200)).then((r) => { for (const x of r.data || []) this.log(x); }).catch(() => {})
        .then(() => { this.emit("end", { status }); this.close(); });
    }
    close() {
      this.readyState = 2;
      clearTimeout(this.timer);
      clearInterval(this.poller);
      if (this.ch) sb.removeChannel(this.ch);
    }
  }

  window.HARPOR_TRANSPORT = {
    fetch: tFetch,
    url: tUrl,
    events: (url) => new RelayEventSource(String(url).split("/")[3]),  // /api/jobs/<id>/events
    live: LIVE,
    keyless,
    noJobs: "No jobs yet. Add a file on your GPU machine.",
    ready: new Promise((resolve) => { readyFn = resolve; }),
    liveLog: "live log",
    offline: OFFLINE,
    unauthorized: "Signed out. Reload to sign in.",
  };
  gate().catch(() => {
    $("#conn").hidden = false;
    $("#conn").textContent = OFFLINE;
  });
})();
