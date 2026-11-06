// HARPOR web app UI: jobs list and upload; a job page with the panels that wait for the user (camera topic, review of
// the views and of the part list, a LiDAR failure) above five tabs: Results (the videos, the viewers and the facts),
// Downloads (the files, by group), Inputs (the input's facts, its preview video, the views the build used and their LiDAR points),
// Settings (the build settings, with LiDAR for the metric scale) and Log. The header's Help link opens help.html at the screen's section.
// Plain DOM code, no build step. The API key is only read from the password fields and sent once.
"use strict";

const $ = (sel) => document.querySelector(sel);
const ACTIVE = ["analyzing", "queued", "running"];
const BUILD = ["extract", "masks", "prep", "da3", "lidar", "select", "sam3d", "vomp", "place", "propose", "parts", "vomp_parts", "agent", "align",
  "mjcf", "usd", "urdf", "turntable", "softvideo", "package"];
const OPTIONAL = ["turntable", "softvideo"];  // steps.OPTIONAL: when it fails, the job still ends done
// job.json "agent" (steps.AGENTS): the key it takes and the placeholder of the key field.
const AGENTS = { astra: { key: "OpenAI API key", placeholder: "sk-..." },
  opus: { key: "Anthropic API key", placeholder: "sk-ant-..." }, none: { key: null, placeholder: "" } };
const RAN = { codex: "GPT-6-Astra", claude: "Claude Opus 5.5" };  // agent_result.agent, the controller's name
const VLM = { astra: RAN.codex, opus: RAN.claude };  // proposal.json "agent": the model that listed the parts
// What each stage does, shown muted next to its name.
const STAGE_LABEL = { topics: "camera topics", frames: "frame analysis", extract: "extract frames", masks: "SAM 3 masks",
  prep: "prepare views", da3: "depth (Depth Anything 3)", lidar: "metric scale (LiDAR)", select: "view selection",
  sam3d: "mesh (MV-SAM3D)", place: "pose and scale", propose: "part list", parts: "split into parts", vomp_parts: "part materials (VoMP)", agent: "agent refinement", align: "align upright", vomp: "materials (VoMP)",
  mjcf: "MuJoCo model", usd: "USD", urdf: "URDF", turntable: "360 video", softvideo: "soft-part video",
  package: "results.zip" };
// A stage's label; the agent's names its task.
const stageLabel = (name, articulated) => name !== "agent" ? STAGE_LABEL[name] || ""
  : articulated ? "agent sets the joints" : STAGE_LABEL.agent;
const JOINTS = ["revolute", "prismatic", "fixed", "soft", "unknown"];  // harpor.parts.propose.JOINTS
// agent_result.verdict of an articulation (the agent sets joints); a refinement's verdicts show as they are.
const VERDICT = { unchanged: "unchanged (no joints)", needs_review: "needs review" };
const LIST_STATUS = { review: "needs review" };  // badge text in the jobs list; the others are the status
const NEEDS_YOU = ["review", "ready", "choose_topic", "failed"];  // the list marks these rows
// The next step under the status; a status without one has none.
const NEXT = { ready: "Name the object, then Start.", queued: "You can close this page.", running: "You can close this page.",
  review: "Check the picked views, then Build.", failed: "Check the log, then Retry.",
  parts: "Check the part list, then Build.",  // a review of the part list
  parts_stale: "The scale is out of date: cancel the job, then Retry from lidar.",  // ... that the server cannot confirm
  // The lidar stage failed (a rosbag); ... that has a calibration.
  lidar: "In Settings, upload a calibration and Retry, or turn LiDAR off and build again.",
  lidar_cal: "In Settings, replace the calibration and Retry, or turn LiDAR off and build again.",
  // A Stray capture takes no calibration, and a retry reads the same depth and masks again.
  lidar_stray: "In Settings, turn LiDAR off or change the views, then build again.",
  // A done job whose scale is stale (V.scaleStale: another calibration or an older estimator): Retry runs again from lidar.
  stale: "Run lidar and later stages again to update them. It runs the lidar stage and every stage after it, GPU stages " +
    "included." };
// The part list's review with a stale scale: the server refuses to confirm the list (jobs.Runner.resume), and a cancelled
// job's Retry starts from lidar (retryFrom).
const STALE_PARTS = " so Build cannot confirm this list. Cancel the job, then press Retry from lidar. The job finds the " +
  "scale again, and the VLM lists the parts again.";
// Why a scale is stale, by its cause (staleBy): the warning of a done job (V.scaleStale), the line when the server
// starts from lidar, the line in the review, and the line in the part list's review.
const SCALE_STALE = {
  calibration: { done: "The calibration changed after this build, so the scale and the masses are out of date.",
    start: "Starting from lidar (calibration changed).",
    review: "Calibration changed. Build finds the scale again, then stops here.",
    parts: "The calibration changed after the scale was found," + STALE_PARTS },
  estimator: { done: "The app now finds the LiDAR scale in a new way, so this build's scale and masses are out of date.",
    start: "Starting from lidar (the app now finds the scale in a new way).",
    review: "The app now finds the scale in a new way. Build finds the scale again, then stops here.",
    parts: "The app now finds the scale in a new way," + STALE_PARTS },
  unknown: { done: "The scale and the masses of this build are out of date.",
    start: "Starting from lidar (the scale is out of date).",
    review: "The scale is out of date. Build finds the scale again, then stops here.",
    parts: "The scale is out of date," + STALE_PARTS } };
// The cause the server gives (job.lidar_stale): "calibration", or "estimator" when an older version of the app found
// the scale. A Stray capture has no calibration. A page that does not know the cause (the server was updated while a
// stopped job's page was open) names none.
const staleBy = () => !V.job ? "unknown" : V.job.kind === "stray" ? "estimator"
  : ["calibration", "estimator"].includes(V.job.lidar_stale) ? V.job.lidar_stale : "unknown";
// The Retry button's label of a done job whose scale is stale.
const RERUN_LIDAR = "Run lidar and later stages again";
const KIND_LABEL = { video: "video", rosbag: "rosbag", stray: "Stray capture" };  // job.json source kind, in the jobs list
// Local server transport; factories keep the UI usable in the DOM test harness.
const T = {
  fetch: (url, opts) => fetch(url, opts),
  url: (path) => path,
  xhr: () => new XMLHttpRequest(),
  events: (url) => new EventSource(url),
  live: ACTIVE,
  noJobs: "No jobs yet.",
  liveLog: "live log",
  offline: "Cannot reach the server. Is the SSH tunnel up?",
  unauthorized: "Not logged in. Open the ?token= link the server printed.",
};
const OFFLINE = T.offline;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
// The job page's tabs by their name in the hash (#/job/<id>/<name>), with the id of each one's panel. The tab buttons
// are #tab-<name>.
const TABS = { results: "results", downloads: "downloads", inputs: "input", settings: "frames", log: "run" };
const TAB_NAMES = Object.keys(TABS);
// The tab that shows first for each job status. A tab that the user picks stays until the status changes.
const TAB_OF = { ready: "settings", analyzing: "log", queued: "log", running: "log", review: "inputs", choose_topic: "inputs",
  done: "results", failed: "log", cancelled: "log" };
const tabOf = (status) => TAB_OF[status] || "log";
// The statuses of a job that runs (analyzing, queued, running) count as one for the tabs: a tab picked while the job
// waits for the GPU stays when it starts.
const phase = (status) => ACTIVE.includes(status) ? "active" : status;
// What the Results, Settings and Downloads tabs and the views strip of the Inputs tab say while they have nothing to
// show. After a failed or cancelled job, Results and Downloads say that it stopped: nothing more comes until Retry or
// a new build.
const RESULTS_LATER = "No results yet.";  // index.html's first text
const RESULTS_STOPPED = "Job stopped before any results.";
const SETTINGS_LATER = "Available after the input is read.";
const SETTINGS_BUSY = "Locked while the job runs.";
const FILES_LATER = "No files yet.";
const FILES_BUSY = "Files show when the build stops.";  // a build replaces the files as it runs
const FILES_NONE_STOPPED = "Job stopped before any files.";
// What the Downloads tab says above its files when results.zip is not one of them: before the build (in review), and
// after a build that stopped.
const FILES_PARTIAL = "More files after the build.";
const FILES_STOPPED = "Build stopped before results.zip.";
const RESULTS_UNREAD = "Could not read the results: ";  // a result file that does not parse (drawJob)
const NO_VIEWS = "No views selected yet.";
// The open job page: id, job, meta, picks, view0, sel, ov and ov0 (the review's overrides as edited and as saved),
// cards (the review cards: renderCards), es, state, scaleStale (the LiDAR scale is stale: scaleStale()); tab (the
// tab shown), pick (the tab the user picked), tabPhase (the phase of the status that the tab follows) and hashTab (the
// hash's tab when the page opened); logAtEnd and logTop (where the log was left); and input, inputKey, videoKey, video,
// selected, inputPlaced, inputAsked, placeAsked and overlay (the Inputs tab: renderInput); spin, spinKey, spinFig and
// spinOn (the 360 video), and soft, softKey, softFig and softOn (the soft-part video: renderSpin).
let V = {};
let listKey = null, upXhr = null;  // the jobs list as last drawn; the running upload

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k.startsWith("on")) e[k] = v;
    else if (k === "class") e.className = v;
    // Through the CSSOM, which a strict CSP allows; it blocks style attributes.
    else if (k === "style" && v != null && v !== false && e.style) e.style.cssText = v;
    else if (v === true) e.setAttribute(k, "");
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  e.append(...kids.flat().filter((x) => x != null && x !== false));
  return e;
}
function s(tag, attrs) {  // an SVG element
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  return e;
}
const fill = (el, ...kids) => el.replaceChildren(...kids.flat().filter((x) => x != null && x !== false));  // skips empty slots
const badge = (t, text) => h("span", { class: "badge " + t }, text || t.replace("_", " "));
const fmt = (sec) => sec < 60 ? sec.toFixed(1) + " s" : sec < 3600 ? Math.floor(sec / 60) + " min " + Math.floor(sec % 60) + " s"
  : Math.floor(sec / 3600) + " h " + Math.floor((sec % 3600) / 60) + " min";
const file = (p, dl) => T.url(`/api/jobs/${V.id}/file/${p}`, dl);
const shortTime = (t) => {  // "2026-10-17T21:02:50" -> "Oct 17 21:02", without Date (the server's clock and zone)
  const m = /^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d)/.exec(t || "");
  return m ? `${MONTHS[+m[2] - 1]} ${+m[3]} ${m[4]}:${m[5]}` : t || "";
};
// "12.3 MB", "100 MB", "1.00 GB": the bounds go by the rounded number, so 99.96 MB shows as "100 MB" and 999.6 MB as
// "1.00 GB".
const size = (b) => b >= 999.5e6 ? (b / 1e9).toFixed(2) + " GB" : (b / 1e6).toFixed(b >= 99.95e6 ? 0 : 1) + " MB";
const count = (x) => Number(x || 0).toLocaleString("en-US");
const pct = (x) => x == null ? "?" : (100 * x).toFixed(1) + "%";
function say(el, text, info) {  // a message line: red for a problem, muted for progress
  el.textContent = text;
  el.className = info ? "msg info" : "msg";
}

// A part list as the user types it, one part per line: "name > parent : joint", or "name x3 > parent : joint"
// for three identical parts. The base is the line without a parent; blank lines are skipped. Returns
// [{name, parent, joint, count}] (parent and joint null when not given). harpor.parts.propose.validate checks
// the rest on the server (one base, parents in the list, names). Throws an Error that names the line.
function parseParts(text) {
  const out = [];
  String(text || "").split(/\r?\n/).forEach((line, i) => {
    if (!line.trim()) return;
    const m = /^([^>:]*)(?:>([^>:]*))?(?::(.*))?$/.exec(line);
    if (!m) throw new Error(`line ${i + 1}: write the part as name > parent : joint`);
    const c = /^(.*?)\s+[x×]\s*(\d+)$/i.exec(m[1].trim());
    const name = (c ? c[1] : m[1]).trim(), parent = m[2] == null ? null : m[2].trim();
    const joint = m[3] == null ? null : m[3].trim().toLowerCase();
    if (!name) throw new Error(`line ${i + 1}: the part needs a name`);
    if (parent === "") throw new Error(`line ${i + 1}: write the parent's name after >`);
    if (joint !== null && !JOINTS.includes(joint)) throw new Error(`line ${i + 1}: the joint is one of ${JOINTS.join(", ")}`);
    out.push({ name, parent, joint, count: c ? Number(c[2]) : 1 });
  });
  return out;
}
// The text of a part list (proposal.json "parts") in parseParts' form; the base comes without a joint.
const formatParts = (parts) => (parts || []).map((p) => p.name + (p.count > 1 ? " x" + p.count : "") +
  (p.parent ? ` > ${p.parent} : ${p.joint || "unknown"}` : "")).join("\n");

async function api(path, opts) {
  let r;
  try { r = await T.fetch(path, opts); } catch (e) { online(false); throw new Error(OFFLINE); }
  online(true);
  const j = await r.json().catch(() => ({}));
  if (r.status === 401) throw new Error(T.unauthorized);
  if (!r.ok) throw new Error(j.detail || `${r.status} ${r.statusText}`);
  return j;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
function online(ok) {
  $("#conn").hidden = ok;
  if (!ok) $("#conn").textContent = OFFLINE;
}
// Run one action: disable its buttons, show what it does, then restore them; refresh on success (or always).
async function act(btns, msg, text, fn, after, always) {
  const was = btns.map((b) => b.disabled);
  btns.forEach((b) => { b.disabled = true; });
  say(msg, text, true);
  let ok = true;
  try { await fn(); say(msg, "", true); } catch (e) { ok = false; say(msg, e.message); }
  btns.forEach((b, i) => { b.disabled = was[i]; });
  if (ok && after) after();
  if (ok || always) await refresh().catch((e) => say(msg, e.message));
  refocus();
}
// A button that had the focus was disabled or hidden by its action, so the focus fell to the page: the tab shown takes
// it (for Start, Build, Retry and Cancel, the tab that the new status shows).
function refocus() {
  const a = document.activeElement;
  if ((!a || a === document.body) && V.tab && $("#tabs").hidden === false) $("#tab-" + V.tab).focus({ preventScroll: true });
}

function show(id) {
  for (const x of ["list", "job"]) $("#" + x).hidden = x !== id;
  helpAt();
}
// The section of help.html that the header's Help link opens: the jobs list's, a waiting panel's (the first one shown,
// in page order), else the shown tab's. Before app.js routes, the link stays help.html.
const HELP_AT = { list: "add-job", lidarfail: "lidar", topics: "topics", review: "review", partsreview: "parts",
  results: "results", downloads: "downloads", inputs: "inputs", settings: "settings", log: "status" };
function helpAt() {
  const panel = ["lidarfail", "topics", "review", "partsreview"].find((x) => $("#" + x).hidden === false);
  const at = $("#list").hidden === false ? "list" : $("#job").hidden === false ? panel || V.tab : null;
  $("#helplink").setAttribute("href", "help.html" + (HELP_AT[at] ? "#" + HELP_AT[at] : ""));
}
// A job page's hash, "#/job/<id>" or "#/job/<id>/<tab>": [id, the tab's name or null]; null for another hash.
function jobHash(hash) {
  const m = /^#\/job\/([\w-]+)(?:\/([a-z]+))?$/.exec(hash || "");
  return m ? [m[1], TAB_NAMES.includes(m[2]) ? m[2] : null] : null;
}
function route() {
  const at = jobHash(location.hash);
  if (at && at[0] === V.id && V.state && !$("#job").hidden) {  // the same job: only the tab changed (a typed URL)
    selectTab(at[1] || tabOf(V.state.status), !!at[1]);
    return;
  }
  closeStream();
  if (at) openJob(at[0], at[1]).then(() => { if (V.id === at[0]) $("#jtitle").focus({ preventScroll: true }); })
    .catch((e) => { $("#jstatus").textContent = e.message; });
  else showList();
}
window.addEventListener("hashchange", route);

// ---- 1. jobs list and upload ----

async function showList() {
  show("list");
  V = {};
  $("#where").replaceChildren();
  document.title = "HARPOR";
  let rows = null;
  try { rows = await api("/api/jobs"); } catch (e) { say($("#upmsg"), e.message); }
  renderJobs(rows);
}
function renderJobs(rows) {
  listKey = rows ? JSON.stringify(rows) : null;
  if (rows && !rows.length) {
    $("#jobs tbody").replaceChildren(h("tr", { class: "empty" },
      h("td", { colspan: 6 }, T.noJobs)));
    return;
  }
  const hint = (j) => KIND_LABEL[j.kind] || j.kind || "";
  $("#jobs tbody").replaceChildren(...(rows || []).map((j) => h("tr", { class: NEEDS_YOU.includes(j.status) ? "act" : null },
    h("td", { class: "thumb" }, h("img", { src: T.url(`/api/jobs/${j.id}/thumb/0`), alt: "", loading: "lazy",
      onerror: (e) => e.target.replaceWith(h("span", { class: "noimg" })) })),
    h("td", { class: "input" }, h("a", { href: "#/job/" + j.id }, j.video), hint(j) ? h("div", { class: "hint" }, hint(j)) : null),
    h("td", { class: "obj" }, j.prompt || ""),
    h("td", { class: "stat" }, badge(j.status, LIST_STATUS[j.status])),
    h("td", { class: "when", title: j.created.replace("T", " ") }, shortTime(j.created)),
    h("td", { class: "acts" }, !deletable(j) ? ""
      : h("button", { class: "icon danger", type: "button", title: "Delete", "aria-label": "Delete " + j.video,
        onclick: () => remove(j) }, trashIcon())))));
}
function trashIcon() {  // the row's Delete button: a trash can drawn in the text color
  const svg = s("svg", { viewBox: "0 0 24 24", width: 16, height: 16, "aria-hidden": "true", focusable: "false" });
  svg.append(s("path", { d: "M4 7h16M10 7V4h4v3M6 7l1 13h10l1-13M10 11v6M14 11v6", fill: "none", stroke: "currentColor",
    "stroke-width": 1.6, "stroke-linecap": "round", "stroke-linejoin": "round" }));
  return svg;
}
// A row's Delete button is disabled while the job runs or uploads.
const deletable = (j) => !(ACTIVE.includes(j.status) || j.status === "uploading");
async function pollList() {  // the 5 s refresh: draws the list again only when it changed
  if ($("#list").hidden || document.hidden === true) return;
  let rows;
  try { rows = await api("/api/jobs"); } catch (e) {  // the header shows a lost connection; a new login shows here
    if (e.message !== OFFLINE) { say($("#upmsg"), e.message); V.listError = true; }
    return;
  }
  if (V.listError) { V.listError = false; say($("#upmsg"), "", true); }
  if (JSON.stringify(rows) !== listKey) renderJobs(rows);
}
setInterval(pollList, 5000);
async function remove(j) {
  if (!confirm(`Delete ${j.video}${j.prompt ? ` (${j.prompt})` : ""} from ${shortTime(j.created)} and all its files?`)) return;
  try { await api("/api/jobs/" + j.id, { method: "DELETE" }); } catch (e) { say($("#upmsg"), e.message); }
  showList();
}
const KINDS = /\.(mp4|mov|m4v|avi|mkv|webm|bag|mcap|db3|zip)$/i;  // jobs.VIDEOS, jobs.BAGS and jobs.STRAY
function upload(f) {
  if (!f || upXhr) return;
  if (!KINDS.test(f.name)) {
    say($("#upmsg"), `${f.name}: file type not supported.`);
    return;
  }
  const xhr = upXhr = T.xhr(), prog = $("#upprog"), t0 = Date.now();
  const done = () => {
    upXhr = null;
    prog.hidden = $("#upcancel").hidden = true;
    $("#uprate").textContent = "";
    $("#file").disabled = false;
    window.onbeforeunload = null;
  };
  xhr.open("PUT", "/api/upload?name=" + encodeURIComponent(f.name));
  prog.value = 0;
  prog.hidden = $("#upcancel").hidden = false;
  $("#file").disabled = true;
  window.onbeforeunload = (e) => { e.preventDefault(); return ""; };  // leaving would cut the upload off
  say($("#upmsg"), `uploading ${f.name} (${size(f.size)})`, true);
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    prog.value = e.loaded / e.total;
    const rate = e.loaded / Math.max((Date.now() - t0) / 1000, 0.001), left = (e.total - e.loaded) / Math.max(rate, 1);
    // not a live region: it changes many times a second
    $("#uprate").textContent = `${size(e.loaded)} of ${size(e.total)}, about ${fmt(Math.round(left))} left`;
  };
  xhr.onload = () => {
    done();
    let j = {};
    try { j = JSON.parse(xhr.responseText); } catch (e) { /* not JSON */ }
    if (xhr.status === 200) {
      say($("#upmsg"), "", true);
      if (!$("#list").hidden) location.hash = "#/job/" + j.id;
    } else say($("#upmsg"), "upload failed: " + (j.detail || xhr.status));
  };
  xhr.onerror = () => { done(); say($("#upmsg"), "upload failed: the connection dropped"); };
  xhr.onabort = () => { done(); say($("#upmsg"), "upload cancelled", true); };
  xhr.send(f);
}
$("#upcancel").onclick = () => { if (upXhr) upXhr.abort(); };
$("#file").onchange = (e) => { upload(e.target.files[0]); e.target.value = ""; };
const drop = $("#drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = (e) => { if (!drop.contains(e.relatedTarget)) drop.classList.remove("over"); };
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); upload(e.dataTransfer.files[0]); };
// A file dropped next to the drop zone would make the browser leave the page.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

// ---- the job page ----

async function openJob(id, tab) {
  V = { id, picks: [], view0: null, hashTab: tab || null };
  // The last job's, until refresh draws this one's: the waiting panels, the tabs and what the tabs show. They hide
  // before show(), so the Help link does not keep the last job's section when this one does not load.
  for (const x of ["review", "partsreview", "lidarfail", "topics", "tabs", ...Object.values(TABS), "resultsnone", "resultsbody",
    "downloadsnone", "downloadsbody", "inputnone", "inputbody", "framesnone", "framesbody"]) $("#" + x).hidden = true;
  show("job");
  clearInput();
  clearSpin();
  $("#log").textContent = "";
  $("#logname").textContent = "";
  $("#rawlog").hidden = true;
  await refresh();
}
async function refresh() {
  const job = V.job = await api(`/api/jobs/${V.id}`);
  fill($("#jtitle"), job.video.name, job.prompt ? [" ", h("span", { class: "jobobj" }, job.prompt)] : null);
  $("#jid").textContent = "job " + job.id;
  if (T.live.includes(job.status)) openStream();
  // A stale scale (another calibration, or an older estimator): Retry then starts from lidar (retryFrom), also on a
  // done job.
  V.scaleStale = ["failed", "cancelled", "done"].includes(job.status) && scaleStale(job);
  renderState({ status: job.status, current: job.current, error: job.error, warnings: job.warnings, key_ok: job.key_ok,
    agent: job.agent, articulated: job.articulated, parts_mode: job.parts_mode, parts_review: job.parts_review,
    stages: job.progress }, false);
  // The tab follows the status at the end, once its panel is drawn; also when a panel could not be drawn. The Help
  // link follows the waiting panel that shows.
  try { await drawJob(job); } finally { if (V.job === job) { followStatus(job.status); helpAt(); } }
}
async function drawJob(job) {  // refresh's panels: the waiting panels and what the tabs show
  const analyzed = job.stages.frames.status === "done", rosbag = job.source.kind === "rosbag";
  // The lidar stage failed: its record (pool/lidar.json, ok false) with the check images; the status line gives the
  // reason. The panel is drawn first and shows only once drawn, so it does not come late or grow above the tabs.
  const lidarFail = lidarFailed(job.status, job.progress);
  if (lidarFail) await renderLidarFail(job).catch((e) => fill($("#lidarfailbody"), h("p", { class: "msg" }, e.message)));
  $("#lidarfail").hidden = !lidarFail;
  // A rosbag job chooses its camera topic when it has several, after a failed frames run, or on "Change topic".
  const idle = !ACTIVE.includes(job.status) && !["review", "done"].includes(job.status);
  $("#topics").hidden = !rosbag || job.stages.topics.status !== "done" || !(job.status === "choose_topic" ||
    (idle && (V.pickTopic || (["failed", "cancelled"].includes(job.status) && !analyzed))));
  if (!$("#topics").hidden) await loadTopics(job).catch((e) => { say($("#topicmsg"), e.message); });
  $("#tkeep").hidden = !(V.pickTopic && analyzed);
  // "Change topic", in the Inputs tab (whose label names the topic), while the table does not show.
  $("#topicrow").hidden = !rosbag || !idle || !$("#topics").hidden;
  // Once the frames are read, the Inputs tab has the input's facts and the Settings tab the build settings. The
  // settings change only while the job does not run; in review they start over.
  if (analyzed) await settings(job);
  else V.meta = null;  // the frames are read again
  const form = analyzed && !ACTIVE.includes(job.status);
  $("#framesbody").hidden = !form;
  $("#framesnone").hidden = form;
  $("#framesnone").textContent = analyzed ? SETTINGS_BUSY : SETTINGS_LATER;
  // A job in review waits for the views, or (parts_review) for its part list after the VLM listed the parts.
  const views = job.status === "review" && job.parts_review !== true, partList = job.status === "review" && job.parts_review === true;
  $("#review").hidden = !views;
  $("#partsreview").hidden = !partList;
  $("#manualstart").hidden = !!V.manual;
  if (views) await loadReview().catch((e) => { $("#selsum").textContent = e.message; });
  if (partList) await loadParts().catch((e) => { say($("#partsmsg"), e.message); });
  // The files, by group, once the job stops. They come first, so a result file that does not parse (renderResults)
  // does not hide them.
  renderDownloads(job);
  // The results, once a build made them; a new build hides the old ones. A file that does not parse (a cut write
  // after a kill or a full disk) gives a message in the facts, and the Inputs tab still draws.
  const results = !ACTIVE.includes(job.status) && job.files.includes("run/posed.glb");
  if (results) await renderResults(job).catch((e) => {
    fill($("#facts2"), h("p", { class: "msg" }, RESULTS_UNREAD + e.message));
    fill($("#partchecks"));  // the check images of an earlier draw
  });
  if (!results) clearSpin();  // a new build replaces the video
  $("#resultsbody").hidden = !results;
  $("#resultsnone").hidden = results;
  $("#resultsnone").textContent = ["failed", "cancelled"].includes(job.status) ? RESULTS_STOPPED : RESULTS_LATER;
  // The input's preview video, and the views used once they are selected (in review, during the build and after).
  V.inputAsked = false;
  V.selected = job.stages.select.status === "done";
  if (analyzed) await renderInput(job).catch((e) => fill($("#inputviews"), h("p", { class: "msg" }, e.message)));
  else clearInput();
  $("#inputbody").hidden = !analyzed;
  $("#inputnone").hidden = analyzed;
  if (job.status === "failed" && !V.es && !$("#log").textContent) {
    const failed = job.progress.find((x) => x.status === "failed");
    if (failed) await showLog(failed.name);
  }
}
async function settings(job) {  // the build settings (Settings tab) and the input's facts (Inputs tab)
  $("#start").textContent = job.status === "ready" ? "Start" : job.status === "review" ? "Start over" : "Build again";
  await loadFrames();
}

// ---- the tabs (the WAI-ARIA tabs pattern) ----

// Show the tab `name` and hide the other panels. pick: the user chose it, so the hash keeps it (a reload shows it
// again) until the job's status changes. When the focus was in the panel that hides, or on a tab that is no longer
// selected (the status changed), the new tab takes it. The preview video stops when its tab hides, and the videos
// of the Results tab play only while it shows (spinTab).
function selectTab(name, pick) {
  const old = V.tab && V.tab !== name ? $("#" + TABS[V.tab]) : null, had = document.activeElement, tab = $("#tab-" + name);
  const lost = !!had && had !== tab && ((!!old && old.contains(had) === true) || $("#tabs").contains(had) === true);
  if (V.tab === "inputs" && name !== "inputs" && V.video) V.video.pause();
  spinTab(name);
  V.tab = name;
  V.pick = pick ? name : null;
  for (const n of TAB_NAMES) {
    const on = n === name, tab = $("#tab-" + n);
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
    $("#" + TABS[n]).hidden = !on;
  }
  tabHash(V.pick);
  if (lost) tab.focus({ preventScroll: true });
  revealTab(tab);
  helpAt();
  if (name === "log") showRun();
}
// The hash of the open job with the picked tab (or none), by replaceState: a tab adds no step to the back button.
// history.state keeps the status that the tab was picked in, so a reload after a change of status shows its own tab.
function tabHash(pick) {
  const at = jobHash(location.hash);
  if (!at || at[0] !== V.id) return;  // another page came meanwhile
  try {
    history.replaceState(pick ? { tab: pick, status: V.state ? V.state.status : null } : null, "",
      `#/job/${V.id}` + (pick ? "/" + pick : ""));
  } catch (e) { /* no history: the tab still shows */ }
}
// The tab follows the job's status: at first the hash's tab (unless it was picked while the job had a status of another
// phase), else the status's own tab; after a change of phase, the new status's own tab.
function followStatus(status) {
  if (V.tabPhase === phase(status)) return;
  const first = V.tabPhase == null;
  V.tabPhase = phase(status);
  $("#tabs").hidden = false;
  let was = null;
  try { was = history.state; } catch (e) { /* no history */ }
  const keep = first && V.hashTab && !(was && was.status && phase(was.status) !== phase(status)) ? V.hashTab : null;
  selectTab(keep || tabOf(status), !!keep);
}
const EDGE = 24;  // px: the faded edge of a tab bar that scrolls (style.css .tabs.more-l and .more-r)
function revealTab(tab) {  // on a phone the tab bar scrolls sideways: bring the tab into it, without moving the page
  const bar = $("#tabs"), x = tab.offsetLeft, w = tab.offsetWidth, left = bar.scrollLeft, bw = bar.clientWidth;
  if ([x, w, left, bw, bar.scrollWidth].every((v) => typeof v === "number") && bar.scrollWidth > bw) {
    if (x < left + EDGE) bar.scrollLeft = Math.max(0, x - EDGE);
    else if (x + w > left + bw - EDGE) bar.scrollLeft = x + w - bw + EDGE;
  }
  tabEdges();
}
function tabEdges() {  // a faded edge where more tabs are off the bar
  const bar = $("#tabs"), left = bar.scrollLeft, sw = bar.scrollWidth, bw = bar.clientWidth;
  if (![left, sw, bw].every((v) => typeof v === "number")) return;
  bar.classList.toggle("more-l", left > 1);
  bar.classList.toggle("more-r", left < sw - bw - 1);
}
$("#tabs").addEventListener("scroll", tabEdges, { passive: true });
// A narrower window (a phone turned) can push the shown tab off the bar.
window.addEventListener("resize", () => { if (V.tab && $("#job").hidden === false) revealTab($("#tab-" + V.tab)); });
$("#tabs").onclick = (e) => {
  const tab = e.target && e.target.closest ? e.target.closest("[role=tab]") : null;
  if (tab) selectTab(tab.dataset.tab, true);
};
$("#tabs").onkeydown = (e) => {  // Left and Right go to the tab before or after (round), Home and End to the first or last
  const i = TAB_NAMES.indexOf(e.target && e.target.dataset ? e.target.dataset.tab : null);
  const to = i < 0 || e.altKey || e.ctrlKey || e.metaKey ? undefined
    : { ArrowLeft: i - 1, ArrowRight: i + 1, Home: 0, End: TAB_NAMES.length - 1 }[e.key];
  if (to === undefined) return;
  e.preventDefault();
  const name = TAB_NAMES[(to + TAB_NAMES.length) % TAB_NAMES.length];
  selectTab(name, true);
  $("#tab-" + name).focus();
};

// ---- the camera topic of a rosbag (frames/topics.json) ----

async function loadTopics(job) {
  const r = await api(file("frames/topics.json")), cur = job.source.topic;
  const inUse = job.stages.frames.status === "done";
  const lens = (t) => badge(t.rectified === true ? "rectified" : t.rectified === false ? "raw" : "unknown");
  $("#ttable tbody").replaceChildren(...r.topics.map((t) => h("tr", { class: t.topic === cur ? "current" : null, "aria-current": t.topic === cur ? "true" : null },
    h("td", {}, t.preview ? h("img", { src: file(t.preview), alt: "", loading: "lazy" }) : ""),
    h("td", { class: "topic" }, h("code", {}, t.topic.split("/").flatMap((x, i) => i > 1 ? [h("wbr"), "/" + x] : i ? ["/" + x] : [x]))),
    h("td", { "data-label": "Type" }, `${t.type} ${t.format || ""}`), h("td", { "data-label": "Messages" }, String(t.count)),
    h("td", { "data-label": "Rate" }, t.rate_hz != null ? t.rate_hz.toFixed(1) + " Hz" : "?"),
    h("td", { "data-label": "Size" }, t.width ? `${t.width}x${t.height}` : "?"),
    h("td", { class: "lens" }, lens(t)),
    h("td", {}, t.topic === cur && inUse ? h("span", { class: "hint" }, "in use")
      : h("button", { class: "small", type: "button", "aria-label": "Use " + t.topic, onclick: (e) => useTopic(t.topic, e.target) },
        "Use")))));
  $("#tskipped").textContent = r.skipped.length ? "Not usable: " + r.skipped.map((x) => `${x.topic} (${x.why})`).join(", ") : "";
}
async function useTopic(topic, btn) {
  await act([btn], $("#topicmsg"), "reading the frames of " + topic, () => post(`/api/jobs/${V.id}/topic`, { topic }), () => {
    V.pickTopic = false;
    V.meta = null;
    $("#log").textContent = "";
  });
}
// True when the system asks for reduced motion: scrolls jump, and the videos of the Results tab do not start by themselves.
const reducedMotion = () => typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
const smooth = () => reducedMotion() ? "auto" : "smooth";
$("#chtopic").onclick = async () => {  // the topic table, above the tabs, takes the focus
  V.pickTopic = true;
  await refresh().catch((e) => say($("#topicmsg"), e.message));
  if ($("#topics").hidden) return;
  $("#topicshead").focus({ preventScroll: true });
  $("#topics").scrollIntoView({ block: "start", behavior: smooth() });
};
$("#tkeep").onclick = async () => {  // back to "Change topic", in the Inputs tab
  V.pickTopic = false;
  await refresh().catch((e) => say($("#topicmsg"), e.message));
  if ($("#input").hidden === false && $("#topicrow").hidden === false) $("#chtopic").focus({ preventScroll: true });
  else refocus();  // the button hid with the table: the tab shown takes the focus
};

// ---- 3. stages, log and the event stream ----

// The job's status, its stages, the actions and the next step. follow (true unless false): the tab follows the status.
function renderState(st, follow) {
  V.state = st;
  const build = st.stages.filter((x) => BUILD.includes(x.name));
  const nDone = build.filter((x) => ["done", "skipped"].includes(x.status)).length;
  const art = st.articulated === true, partList = st.status === "review" && st.parts_review === true;
  // "Running: mesh (MV-SAM3D)": the stage's label, or its name when it has none.
  const cur = st.current;
  const sum = { analyzing: cur === "topics" ? "Listing topics" : "Reading input", choose_topic: "Choose topic", ready: "Ready",
    queued: "Waiting for GPU", running: cur ? "Running: " + (stageLabel(cur, art) || cur) : "Running",
    review: partList ? "Part list waiting" : "Needs review",
    done: "Done", failed: "Failed", cancelled: "Cancelled", uploaded: "Uploaded" }[st.status] || st.status;
  const overall = ["queued", "running"].includes(st.status) ? h("div", { class: "overall" },
    h("progress", { max: build.length, value: nDone, "aria-label": "build progress" }),
    h("span", { class: "hint" }, `${nDone}/${build.length} stages`)) : null;
  // The badge says the state once: the summary ("Done"), or a queued job's place in the queue. A failed job's error ends
  // with the help page's troubleshooting section.
  fill($("#jstatus"), badge(st.status, sum), overall, st.error ? h("div", { class: "msg errbox" }, st.error, " ",
    h("a", { href: "help.html#troubleshooting", target: "_blank", rel: "noopener" }, "Help")) : null);
  const from = retryFrom(st);
  // A done job whose scale is stale (V.scaleStale) runs again from lidar.
  const rerun = st.status === "done" && from === "lidar";
  let next = ["failed", "cancelled"].includes(st.status) && from === null ? "Build again from the Settings tab."
    : rerun ? NEXT.stale + (V.job && V.job.auto_continue === false ? " It pauses for your review." : "")
    : partList ? NEXT[V.job && scaleStale(V.job) ? "parts_stale" : "parts"] : lidarFailed(st.status, st.stages)
      ? NEXT[!V.job ? "lidar" : V.job.kind === "stray" ? "lidar_stray" : V.job.calibration ? "lidar_cal" : "lidar"]
      : NEXT[st.status] || "";
  const ran = (name) => build.some((x) => x.name === name && x.status === "done");
  if (["queued", "running"].includes(st.status) && V.job && V.job.auto_continue === false && !ran("select"))
    next += " It pauses for your review.";
  if (["queued", "running"].includes(st.status) && art && st.parts_mode === "vlm_edit" && !ran("propose"))
    next += " It pauses for the part list.";
  $("#jnext").textContent = next;
  const said = sum + (st.error ? ". " + st.error : "");
  if (said !== V.said) { V.said = said; $("#announce").textContent = said; }
  $("#jwarn").replaceChildren(...[rerun ? SCALE_STALE[staleBy()].done : null, ...(st.warnings || [])].filter(Boolean)
    .map((w) => h("li", {}, w)));
  const name = V.job ? V.job.video.name : V.id || "";
  $("#where").replaceChildren(h("span", { class: "crumb" }, "/ " + name));  // the status shows once, under the title
  document.title = ({ running: `${cur || "running"} ${Math.min(nDone + 1, build.length)}/${build.length}`, queued: "Queued",
    review: partList ? "Part list waiting" : "Review waiting", done: "Done", failed: "Failed", cancelled: "Cancelled", ready: "Ready", analyzing: "Reading",
    choose_topic: "Choose topic" }[st.status] || st.status) + (name ? " · " + name : "") + " · HARPOR";

  const started = st.stages.some((x) => x.started || x.status !== "pending");
  $("#runbody").hidden = !started;
  $("#runnone").hidden = started;
  // The stage that Cancel stopped is recorded as failed with the note "cancelled" (jobs.Runner._run_stage): it shows as
  // cancelled. A failed stage's note that the status line's error already gives ("<stage>: <note>") is left out.
  const stages = st.stages.map((x) => x.status === "failed" && x.note === "cancelled" ? { ...x, status: "cancelled", note: null }
    : x.status === "failed" && st.error === `${x.name}: ${x.note}` ? { ...x, note: null } : x);
  const nOf = (status) => stages.filter((x) => x.status === status).length, nFailed = nOf("failed"), nCancelled = nOf("cancelled");
  $("#runsum").textContent = `${nDone}/${build.length} done` + (nFailed ? ` · ${nFailed} failed` : "") +
    (nCancelled ? ` · ${nCancelled} cancelled` : "") + (st.status === "running" && cur ? ` · running ${cur}` : "");
  const now = Date.now() / 1000, live = !!V.es;
  const focused = document.activeElement && document.activeElement.dataset && document.activeElement.dataset.stage;
  $("#stages").replaceChildren(...stages.map((x) => h("li", { class: x.status },
    h("div", { class: "nm" }, live ? h("span", { class: "name" }, x.name)
      : h("button", { type: "button", class: "name link", "data-stage": x.name, "aria-controls": "log", "aria-pressed": String(x.name === V.logStage),
        onclick: () => showLog(x.name) }, x.name),
    h("span", { class: "label" }, stageLabel(x.name, art))),
    h("span", { class: "gpu" }, x.gpu ? "GPU" : ""),
    badge(x.status),  // a stage that is turned off says so in its note ("turned off: rigid object")
    h("span", { class: "secs", "data-started": x.started || null, "data-ended": x.ended || null },
      x.started ? fmt(Math.max(0, (x.ended || now) - x.started)) : ""),
    x.status === "running" && x.progress != null ? h("progress", { max: 1, value: x.progress, "aria-label": x.name + " progress" }) : null,
    x.note ? h("span", { class: "note" }, x.note) : null)));
  if (typeof focused === "string") {  // the list was drawn again under a focused stage button
    const b = [...document.querySelectorAll("#stages button.name")].find((y) => y.dataset.stage === focused);
    if (b) b.focus();
  }
  // A finished job's total, under the clocks: their sum, so a wait in the queue or for a review does not count.
  $("#runtotal").hidden = st.status !== "done";
  $("#runtotalsecs").textContent = fmt(stages.reduce((sum, x) => sum + (x.started && x.ended ? Math.max(0, x.ended - x.started) : 0), 0));

  $("#cancel").hidden = !(ACTIVE.includes(st.status) || st.status === "review");
  // A done job retries only a failed optional stage (Runner.resume runs package again after it), or runs again from lidar
  // (rerun).
  $("#retry").hidden = from === null || !(["failed", "cancelled"].includes(st.status) || rerun ||
    st.status === "done" && OPTIONAL.includes(from));
  if (!$("#retry").hidden) $("#retry").textContent = rerun ? RERUN_LIDAR : from === "frames" ? "Read input again"
    : from === "propose" && (st.stages.find((y) => y.name === "propose") || {}).status === "done" ? "Back to part list"
      : "Retry from " + from;
  // A rerun from lidar runs the agent stage again, and a done job's key is gone (jobs.Runner._status).
  const agent = st.agent;
  $("#keyrow").hidden = st.key_ok || agent === "none" || !(["review", "failed", "cancelled"].includes(st.status) || rerun) ||
    from === "frames";
  $("#key2name").textContent = AGENTS[agent].key || "API key";
  // The key goes next to the button that sends it.
  const home = st.status === "review" ? partList ? $("#partsbar") : $("#revbar") : $("#actions");
  if ($("#keyrow").parentElement !== home) home.prepend($("#keyrow"));
  if (follow !== false) followStatus(st.status);
}
function retryFrom(st) {  // "frames" analyzes the input again (no key needed); else the first build stage not done
  const stage = (name) => st.stages.find((y) => y.name === name);
  if (stage("frames").status !== "done") return "frames";
  const x = st.stages.find((x) => BUILD.includes(x.name) && !["done", "skipped"].includes(x.status));
  if (!x) return V.scaleStale === true ? "lidar" : null;  // a done job: again from lidar when its scale is stale
  if (stage("extract").status === "pending") return null;
  // A job stopped before its part list was confirmed goes back to that list (the server resumes from propose).
  const unconfirmed = st.articulated === true && st.parts_mode === "vlm_edit" && (stage("propose") || {}).status === "done"
    && !(V.job && V.job.parts.confirmed === true);
  const name = x.name === "parts" && unconfirmed ? "propose" : x.name;
  // A stale scale (V.scaleStale): the server starts from lidar (Runner.resume).
  return V.scaleStale === true && BUILD.indexOf(name) > BUILD.indexOf("lidar") ? "lidar" : name;
}
// The clocks of running stages, without drawing the list again. They only set text, so a hidden tab costs no layout.
function tick() {
  if (!V.state || !ACTIVE.includes(V.state.status)) return;
  const now = Date.now() / 1000;
  for (const e of document.querySelectorAll("#stages .secs[data-started]:not([data-ended])"))
    e.textContent = fmt(Math.max(0, now - Number(e.dataset.started)));
}
setInterval(tick, 1000);

const quiet = (text) => text.replace(/^PROGRESS [\d.]+\r?\n/gm, "");  // the stage list shows progress already
// The live log follows its end unless the user scrolled up. While its tab is hidden it only takes the text, with no
// layout; showRun puts it at its end (or where the user left it) when the tab opens.
function appendLog(text) {
  const log = $("#log"), shown = $("#run").hidden === false;
  const atEnd = shown ? log.scrollTop + log.clientHeight >= log.scrollHeight - 30 : V.logAtEnd !== false;
  log.textContent = (log.textContent + quiet(text)).slice(-200000);
  if (shown && atEnd) log.scrollTop = log.scrollHeight;
}
$("#log").onscroll = () => {  // where the user left the log, for when its tab opens again
  const log = $("#log");
  if ($("#run").hidden !== false) return;
  V.logTop = log.scrollTop;
  V.logAtEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
};
function showRun() {  // the Log tab opens: the log where it was left, the clocks current
  const log = $("#log");
  log.scrollTop = V.logAtEnd === false ? V.logTop || 0 : log.scrollHeight;
  tick();
}
function openStream() {
  if (V.es) return;
  const id = V.id, es = V.es = T.events(`/api/jobs/${id}/events`);
  $("#logname").textContent = "live log (connecting)";
  V.logStage = null;
  V.logAtEnd = true;
  $("#rawlog").hidden = true;
  es.onopen = () => { if (V.id === id) $("#logname").textContent = T.liveLog; };
  es.onerror = () => {
    if (V.id === id) $("#logname").textContent = es.readyState === 2 ? "live log lost: reload the page" : "live log (reconnecting)";
  };
  es.addEventListener("state", (e) => {
    if (V.id !== id) return;
    const st = JSON.parse(e.data);
    // The tab stays: the refresh after "end" (the job stopped) draws the new tab's panel, then shows it.
    renderState(st, false);
    // The views were selected while the page is open: refresh draws them in the Inputs tab. place ended: refresh adds
    // the mesh fit toggle to them (once per place run).
    const sel = st.stages.find((x) => x.name === "select"), place = st.stages.find((x) => x.name === "place");
    if (sel && sel.status === "done" && !V.selected && !V.inputAsked) {
      V.inputAsked = true;
      refresh().catch(() => { V.inputAsked = false; });
    } else if (place && place.status === "done" && V.input && !V.input.placed && V.placeAsked !== place.ended) {
      V.placeAsked = place.ended;
      refresh().catch(() => { V.placeAsked = null; });
    }
  });
  es.addEventListener("log", (e) => {
    if (V.id !== id) return;
    const d = JSON.parse(e.data);
    appendLog(d.text);
    if (d.stage) $("#logname").textContent = "live log: " + d.stage;
  });
  // The job stopped: the log keeps the lines that came, and the stage names (drawn again) load each stage's own log.
  es.addEventListener("end", () => {
    closeStream();
    if (V.id !== id) return;
    $("#logname").textContent = $("#log").textContent ? "log until the job stopped" : "";
    refresh();
  });
}
function closeStream() {
  if (V.es) V.es.close();
  V.es = null;
}
async function showLog(stage) {
  if (V.es) return;  // the live log is showing
  const r = await T.fetch(file(`logs/${stage}.log`));
  $("#log").textContent = r.ok ? quiet(await r.text()).slice(-200000) : `no log of ${stage} yet`;
  $("#logname").textContent = `log of ${stage}`;
  V.logStage = stage;
  $("#rawlog").href = file(`logs/${stage}.log`);
  $("#rawlog").hidden = !r.ok;
  V.logAtEnd = true;  // the error lines are at the end; a hidden log goes there when its tab opens
  $("#log").scrollTop = 1e9;
  for (const b of document.querySelectorAll("#stages button.name")) b.setAttribute("aria-pressed", String(b.dataset.stage === stage));
}
$("#cancel").onclick = () => {
  if (!confirm("Cancel this job?")) return;
  act([$("#cancel"), $("#retry")], $("#runmsg"), "cancelling", () => post(`/api/jobs/${V.id}/cancel`), null, true);
};
$("#retry").onclick = () => {
  const from = retryFrom(V.state), rerun = V.state.status === "done" && from === "lidar";
  // A done job's run from lidar replaces its results, as Build again does.
  if (rerun && !confirm("Run lidar and every stage after it again? This replaces the current results.")) return;
  let r = null;
  return act([$("#retry"), $("#cancel")], $("#runmsg"), (rerun ? "running again from " : "retrying from ") + from,
    async () => { r = await post(`/api/jobs/${V.id}/continue`, { from, api_key: $("#key2").value || null }); },
    () => {
      $("#key2").value = "";
      $("#log").textContent = "";
      // The server may start earlier than the page asked: say where the run starts.
      if (r && r.from && r.from !== from) say($("#runmsg"), r.from === "lidar"
        ? SCALE_STALE[staleBy()].start : `Starting from ${r.from}.`, true);
    });
};

// ---- 2. frames and build settings ----

function snap(i) {  // video.snap: the sharpest frame within 0.25 s, ties to the nearest
  const t = V.meta.times, rel = V.meta.rel;
  let best = i;
  for (let j = 0; j < t.length; j++) {
    if (Math.abs(t[j] - t[i]) > 0.25 + 1e-9) continue;
    if (rel[j] > rel[best] || (rel[j] === rel[best] && Math.abs(j - i) < Math.abs(best - i))) best = j;
  }
  return best;
}
async function loadFrames() {
  if (!V.meta) {
    V.meta = await api(`/api/jobs/${V.id}/frames`);
    const job = V.job;
    // A bag's LiDAR topics and TF links, for the LiDAR box; an error shows in the box.
    V.topics = job.kind === "rosbag" ? await api(file("frames/topics.json")).catch((e) => ({ error: e.message })) : null;
    $("#prompt").value = job.prompt || "";  // a job without a name shows the placeholder, not the last job's name
    $("#nviews").value = $("#nval").textContent = job.n_views || 6;
    $("#pause").checked = !job.auto_continue;
    $("#turntable").checked = job.turntable === true;
    // Fall back to the first available agent for an unrecognized saved agent.
    const radios = [...document.querySelectorAll("input[name=agent]")];
    for (const r of radios) r.checked = r.value === job.agent && r.disabled !== true;
    if (!radios.some((r) => r.checked)) {
      const first = radios.find((r) => r.disabled !== true);
      if (first) first.checked = true;
    }
    $("#articulated").checked = job.articulated === true;
    partsPick = job.parts.mode;
    $("#partlist").value = formatParts(job.parts.list);
    agentToggle();
    lidarSetup(job);
    document.querySelector(`input[name=mode][value=${V.manual ? "manual" : job.mode || "auto"}]`).checked = true;
    V.picks = job.mode === "manual" ? [...job.picks] : [];
    // job.view0 is a clicked frame, or a pool name ("f00012", a snapped pick) once view 0 was chosen in the review.
    const v0 = typeof job.view0 === "string" ? Number(job.view0.slice(1)) : job.view0;
    V.view0 = V.picks.includes(v0) ? v0 : V.picks.find((p) => snap(p) === v0) ?? null;
    $("#gridbox").open = mode() === "manual";
  }
  const m = V.meta, v = m.video, stray = v.kind === "stray";
  $("#facts").textContent = `${v.fps} fps, ${v.width}x${v.height}, ${v.codec}${v.bit_depth ? " " + v.bit_depth + "-bit" : ""}`;
  $("#vsum").textContent = `${stray ? "Stray capture" : "Video"}: ${v.n_frames} frames, ${v.duration_s.toFixed(1)} s, ` +
    `${m.candidates.length} candidate views`;
  $("#hdr").hidden = !v.hdr;
  // A Stray Scanner capture: its turn from gravity, its depth images and the export's warnings.
  $("#strayinfo").hidden = !stray;
  $("#strayfacts").textContent = stray ? strayFacts(v) : "";
  $("#straywarn").replaceChildren(...(stray ? v.warnings || [] : []).map((w) => h("li", {}, w)));
  // The Settings tab points to the input's warnings in the Inputs tab: some bear on the settings, such as LiDAR.
  const nWarn = (v.hdr ? 1 : 0) + (stray ? (v.warnings || []).length : 0);
  $("#inputwarn").hidden = !nWarn;
  $("#inputwarntext").textContent = nWarn === 1 ? "1 input warning" : `${nWarn} input warnings`;
  lidarToggle();
  sparkline(m);
  renderFrames();
}
$("#towarn").onclick = () => {  // the Inputs tab at the input's first warning
  const to = $("#hdr").hidden === false ? $("#hdr") : $("#straywarn");
  selectTab("inputs", true);
  to.scrollIntoView({ block: "center", behavior: smooth() });
  to.focus({ preventScroll: true });
};
function strayFacts(v) {  // frames/meta.json "video" of a Stray capture (stray.probe)
  const a = v.upright_agreement, d = v.depth_size || [];
  // stray.upright_turns: counterclockwise quarter turns; the agreement counts only the frames that show gravity.
  return [a == null ? "No gravity data: frames as stored."
    : (v.rotation ? `Turned ${v.rotation}° to upright` : "Upright as stored") + ` (gravity agrees: ${Math.round(100 * a)}%).`,
  `Depth: ${v.n_depth}/${v.n_frames} frames${d.length === 2 ? ` (${d[0]}x${d[1]})` : ""}.`,
  `Confidence: ${v.n_confidence}.`].join(" ");
}
function sparkline(m) {
  const t0 = m.times[0], span = (m.times[m.times.length - 1] - t0) || 1;
  const x = (t) => (600 * (t - t0) / span).toFixed(1), y = (r) => (58 - 54 * Math.min(r, 1.5) / 1.5).toFixed(1);
  const svg = $("#spark");
  svg.setAttribute("aria-label", `sharpness over ${span.toFixed(1)} s with ${m.candidates.length} candidate views marked`);
  svg.replaceChildren(s("line", { x1: 0, x2: 600, y1: y(0.6), y2: y(0.6), stroke: "#c62828", "stroke-dasharray": "4 4", "vector-effect": "non-scaling-stroke" }),
    s("polyline", { points: m.times.map((t, i) => x(t) + "," + y(m.rel[i])).join(" "), fill: "none", stroke: "#1a66d6", "vector-effect": "non-scaling-stroke" }),
    ...m.candidates.map((i) => s("circle", { cx: x(m.times[i]), cy: y(m.rel[i]), r: 2.5, fill: "#ffb300" })));
}
function mode() { return document.querySelector("input[name=mode]:checked").value; }
function chosenAgent() {
  for (const r of document.querySelectorAll("input[name=agent]")) if (r.checked) return r.value;
  return "astra";
}
// The key field names the chosen agent's provider.
function agentToggle() {
  const a = AGENTS[chosenAgent()];
  $("#keylabel").hidden = !a.key;
  $("#key").hidden = false;
  $("#key").required = !!a.key;
  $("#keyname").textContent = a.key || "";
  $("#key").placeholder = a.placeholder;
  partsToggle();
}
for (const r of document.querySelectorAll("input[name=agent]")) r.onchange = agentToggle;
let partsPick = "vlm";  // the parts mode the user chose; agent "None" shows "I type them" and keeps this choice
const chosenPartsMode = () => chosenAgent() === "none" ? "user" : partsPick;  // steps.parts_mode: no agent, no VLM
function partsToggle() {  // the parts settings of an articulated object; the typed list only for "I type them"
  const art = $("#articulated").checked === true, none = chosenAgent() === "none", pm = chosenPartsMode();
  for (const r of document.querySelectorAll("input[name=partsmode]")) {
    r.disabled = none && r.value !== "user";
    r.checked = r.value === pm;
  }
  $("#partsbox").hidden = !art;
  $("#partsnoagent").hidden = !none;
  $("#partlistbox").hidden = pm !== "user";
  $("#partlist").required = art && pm === "user";
}
for (const r of document.querySelectorAll("input[name=partsmode]")) r.onchange = () => { partsPick = r.value; partsToggle(); };
$("#articulated").onchange = partsToggle;

// ---- 2b. LiDAR for the metric scale: a bag's LiDAR topic with its TF link or the calibration; a Stray capture's depth ----

// A topics.json "lidar" entry: "/livox/lidar (Livox, 10 Hz, livox_frame)".
const lidarLabel = (t) => `${t.topic} (${t.type === "CustomMsg" ? "Livox" : t.type || "?"}, ` +
  `${t.rate_hz != null ? Number(Number(t.rate_hz).toFixed(1)) + " Hz" : "rate unknown"}, ${t.frame_id || "no frame_id"})`;
const tfName = (f) => String(f || "").trim().replace(/^\/+/, "");  // rosbag.frame_name
// The frames from target to source along topics.json "tf" ([{parent, child, static}]), both ways along each link, as
// rosbag.tf_chain finds them: the static links alone first, then with the dynamic ones. null without a path.
function tfChain(edges, target, source) {
  target = tfName(target);
  source = tfName(source);
  if (!target || !source) return null;
  for (const use of [edges.filter((e) => e.static === true), edges]) {
    const adj = new Map(), add = (a, b) => { if (!adj.has(a)) adj.set(a, []); adj.get(a).push(b); };
    for (const e of use) { add(tfName(e.parent), tfName(e.child)); add(tfName(e.child), tfName(e.parent)); }
    const prev = new Map([[target, null]]), queue = [target];
    while (queue.length && !prev.has(source)) {
      const f = queue.shift();
      for (const g of adj.get(f) || []) if (!prev.has(g)) { prev.set(g, f); queue.push(g); }
    }
    if (prev.has(source)) {
      const out = [];
      for (let f = source; f != null; f = prev.get(f)) out.push(f);
      return out.reverse();
    }
  }
  return null;
}
// The status line of the camera topic's link to a LiDAR topic: [text, true when the user must act]. A calibration
// replaces the bag's TF (lidar.extrinsic). One uploaded for another LiDAR topic (its lidar_topic) is refused at the start
// (jobs.lidar_settings); one without lidar_topic is not checked.
function lidarLink(found, camTopic, lidarTopic, cal) {
  if (!lidarTopic) return ["", false];
  const cam = (found.topics || []).find((t) => t.topic === camTopic), lid = (found.lidar || []).find((t) => t.topic === lidarTopic);
  const how = cam && cam.tf_links[lidarTopic];
  const chain = how && ((lid && tfChain(found.tf || [], cam.frame_id, lid.frame_id)) || [cam.frame_id, lid ? lid.frame_id : lidarTopic]);
  const tf = how ? `${how === "static" ? "/tf_static" : "/tf"}: ${chain.join(" <- ")}` : null;
  if (cal && cal.lidar_topic && cal.lidar_topic !== lidarTopic)
    return [`The calibration ${cal.name} is for ${cal.lidar_topic}: choose that topic, or upload a calibration for ${lidarTopic}`, true];
  if (cal) return [`Linked by the calibration ${cal.name}` + (tf ? ` (it replaces ${tf})` : ""), false];
  if (tf) return ["Linked by " + tf, false];
  return ["No TF link: upload a calibration", true];
}
// job.json "calibration" in short (server.calibration_summary): the LiDAR's distance from the camera and its turn. A
// time_offset_s of 0 is a given offset (the two header clocks are one); null means that the file gives none. info is
// the camera topic's camera_info in topics.json: null when the bag has none, so K must come from the file
// (jobs.lidar_settings refuses the start).
const calText = (c, info) => [`${c.name}: ${(100 * c.translation_m).toFixed(1)} cm, ${Number(c.rotation_deg).toFixed(1)}°.`,
  c.has_intrinsics ? "K from file." : info === null ? "No K: the bag has no camera_info, so give K in the file."
    : "K from camera_info.", c.camera_axes === "body" ? "Body axes." : null,
  c.time_offset_s != null ? `Offset ${c.time_offset_s} s.` : null].filter(Boolean).join(" ");
function lidarSetup(job) {  // the job's own setting, each time the frames load: a Stray capture starts with LiDAR on
  const list = V.topics && Array.isArray(V.topics.lidar) ? V.topics.lidar : [];
  const had = job.lidar && job.lidar.kind === "rosbag" && list.some((t) => t.topic === job.lidar.topic) ? job.lidar.topic : null;
  const pick = had || (list.length === 1 ? list[0].topic : "");  // the only topic is preselected
  fill($("#lidartopic"), pick ? null : h("option", { value: "" }, "Choose a LiDAR topic"),
    list.map((t) => h("option", { value: t.topic, selected: t.topic === pick }, lidarLabel(t))));
  $("#lidartopic").value = pick;
  $("#uselidar").checked = job.kind === "stray" ? !!(job.lidar && job.lidar.kind === "stray") : !!had;
}
function lidarToggle() {  // the LiDAR box follows the input kind, the checkbox, the topic and the calibration
  const job = V.job || {}, kind = job.kind, stray = kind === "stray", found = V.topics || {};
  $("#lidarbox").hidden = !stray && kind !== "rosbag";  // a video has no LiDAR
  if ($("#lidarbox").hidden) return;
  const list = found.lidar || [], usable = stray || list.length > 0;
  $("#uselidar").disabled = !usable;
  if (!usable) $("#uselidar").checked = false;
  const on = $("#uselidar").checked === true;
  const skipped = (found.lidar_skipped || []).map((x) => `${x.topic} (${x.why})`).join(", ");
  // Only why the box is off: what LiDAR on and off do is in help.html.
  $("#lidarhint").textContent = usable ? ""
    : found.error ? "Cannot read the bag's topics: " + found.error
      : "No LiDAR topic in this bag." + (skipped ? " Not usable: " + skipped + "." : "");
  // The topic, its link and the calibration: a bag's only. No "required" on the select: lidarChoice says what to do.
  $("#lidaropts").hidden = stray || !usable || !on;
  if ($("#lidaropts").hidden) return;
  const cal = job.calibration || null;
  const [link, act] = lidarLink(found, job.source.topic, $("#lidartopic").value, cal);
  if ($("#lidarlink").textContent !== link) $("#lidarlink").textContent = link;  // a live region: only a change is read out
  $("#lidarlink").className = act ? "warn" : "hint";
  $("#lidarskipped").textContent = skipped ? "Not usable: " + skipped : "";
  $("#calsum").hidden = !cal;
  const camera = (found.topics || []).find((t) => t.topic === job.source.topic);
  $("#calsum").textContent = cal ? calText(cal, camera ? camera.camera_info : undefined) : "";
  $("#calbtn").textContent = cal ? "Replace calibration" : "Upload calibration";
  $("#calremove").hidden = !cal;
}
$("#uselidar").onchange = lidarToggle;
$("#lidartopic").onchange = lidarToggle;
// The start body's "lidar": null, {topic} (a LiDAR topic of the bag) or {kind: "stray"} (the capture's depth).
function lidarChoice() {
  const kind = V.job && V.job.kind;
  if (!["rosbag", "stray"].includes(kind) || $("#uselidar").checked !== true || $("#uselidar").disabled === true) return null;
  if (kind === "stray") return { kind: "stray" };
  const topic = $("#lidartopic").value;
  if (typeof topic !== "string" || !topic) throw new Error("choose a LiDAR topic, or turn LiDAR off");
  return { topic };
}
// Whether a job's done lidar stage is stale: the job's lidar_current (steps.done("lidar"), which compares the
// calibrations' sha256 and wants the record of the current estimator) is false. A rosbag's scale goes stale when its
// calibration changes, and any scale, a Stray capture's too, when an older version of the app found it
// (job.lidar_stale). The server then retries from lidar.
function scaleStale(job) { return !!job.lidar && job.lidar_current === false; }
// Upload a calibration file at once (PUT, with the LiDAR topic it is for: the server checks and summarises it), or
// remove it (f null). The summary or the server's reason shows in the box. A review shows again whether the scale is
// current; after a stop (failed, cancelled or done), Retry names lidar when the scale is stale.
async function calibrate(f) {
  const focus = document.activeElement;
  // Retry and Build wait for the upload: they would run with the old calibration.
  const btns = [$("#calremove"), $("#start"), $("#calfile"), $("#retry"), $("#build")], was = btns.map((b) => b.disabled);
  btns.forEach((b) => { b.disabled = true; });
  say($("#calmsg"), f ? "checking " + f.name : "removing the calibration", true);
  const url = `/api/jobs/${V.id}/calibration`, topic = $("#lidartopic").value;  // "" when no topic is chosen
  let done = null;
  try {
    const r = await (f ? api(url + "?name=" + encodeURIComponent(f.name) + "&lidar_topic=" +
      encodeURIComponent(typeof topic === "string" ? topic : ""), { method: "PUT", body: f }) : api(url, { method: "DELETE" }));
    if (V.job) V.job.calibration = f ? r : null;
    done = f ? `Saved ${r.name}.` : "Removed.";
  } catch (e) { say($("#calmsg"), e.message); }
  if (done !== null && V.sel && $("#review").hidden === false) {
    try { V.sel.lidar = (await api(`/api/jobs/${V.id}/selection`)).lidar; renderSelLidar(); } catch (e) { /* the old summary stays */ }
  } else if (done !== null && V.job && ["failed", "cancelled", "done"].includes(V.job.status)) {
    try {  // the cause (lidar_stale) can change with the upload too
      const j = await api(`/api/jobs/${V.id}`);
      Object.assign(V.job, { lidar_current: j.lidar_current, lidar_stale: j.lidar_stale });
    } catch (e) { /* the old values stay */ }
    V.scaleStale = scaleStale(V.job);
    if (V.state) renderState(V.state);  // the Retry button names the stage it starts from; a done job's warning shows
  }
  btns.forEach((b, i) => { b.disabled = was[i]; });
  lidarToggle();
  if (done !== null) say($("#calmsg"), done, true);
  // Focus goes back to the control that had it; Remove hides after a delete, so the upload button takes it.
  const now = document.activeElement, back = focus === $("#calremove") && $("#calremove").hidden ? $("#calfile") : focus;
  if (btns.includes(focus) && (!now || now === document.body || now === focus)) back.focus({ preventScroll: true });
}
$("#calfile").onchange = (e) => {
  const f = e.target.files && e.target.files[0];
  e.target.value = "";  // the same file again uploads again
  return f ? calibrate(f) : undefined;
};
$("#calremove").onclick = () => calibrate(null);

function renderFrames() {
  const m = V.meta, manual = mode() === "manual", cand = new Set(m.candidates), have = new Set(m.thumbs);
  $("#autobox").hidden = manual;
  $("#manbox").hidden = !manual;
  $("#picks").hidden = !manual;
  $("#grid").classList.toggle("manual", manual);
  const shown = new Map(V.picks.map((p, k) => [have.has(snap(p)) ? snap(p) : p, k]));
  // Auto mode shows only the candidates it chooses from; manual mode shows every thumbnail, as buttons.
  const ratio = `aspect-ratio: ${m.thumb_size[0]} / ${m.thumb_size[1]}`;  // keeps cells sized before lazy images load
  const fi = document.activeElement && document.activeElement.dataset && document.activeElement.dataset.i;
  $("#grid").replaceChildren(...(manual ? m.thumbs : m.candidates).map((i) => {
    const picked = manual && shown.has(i), k = shown.get(i), at = m.times[i].toFixed(2) + " s";
    const kids = [h("img", { src: T.url(`/api/jobs/${V.id}/thumb/${i}`), loading: "lazy", alt: manual ? "" : "frame " + i, style: ratio }),
      picked ? h("b", {}, String(k + 1) + (V.picks[k] === V.view0 ? " ★" : "")) : null, h("span", {}, at)];
    const cls = "cell" + (cand.has(i) ? " cand" : "") + (picked ? " pick" : "");
    return manual ? h("button", { type: "button", class: cls, "data-i": i, "aria-pressed": String(picked), "aria-label": `frame ${i} at ${at}` }, ...kids)
      : h("div", { class: cls, "data-i": i }, ...kids);
  }));
  if (typeof fi === "string") {  // the grid was drawn again under a focused frame button
    const b = document.querySelector('#grid [data-i="' + fi + '"]');
    if (b) b.focus();
  }
  $("#pickcount").textContent = `${V.picks.length} picked (2 to 12)`;
  $("#picks").replaceChildren(...V.picks.map((p, k) => h("li", {}, `${k + 1}. frame ${snap(p)}, ${m.times[snap(p)].toFixed(2)} s `,
    h("button", { class: "small secondary", type: "button", "aria-pressed": String(V.view0 === p), onclick: () => { V.view0 = p; renderFrames(); } },
      V.view0 === p ? "view 0 ★" : "make view 0"), " ",
    h("button", { class: "small secondary", type: "button", onclick: () => togglePick(p) }, "remove"))));
}
function togglePick(i) {
  const k = V.picks.findIndex((p) => p === i || snap(p) === i || snap(p) === snap(i));  // i may be a shown (snapped) cell
  if (k >= 0) {
    if (V.view0 === V.picks[k]) V.view0 = null;
    V.picks.splice(k, 1);
  } else if (V.picks.length >= 12) {
    $("#pickcount").textContent = "12 is the maximum. Remove one first.";
    return;
  } else V.picks.push(i);
  say($("#startmsg"), "");
  renderFrames();
}
$("#grid").onclick = (e) => {
  const cell = e.target.closest(".cell");
  if (cell && mode() === "manual") togglePick(+cell.dataset.i);
};
for (const r of document.querySelectorAll("input[name=mode]")) r.onchange = () => {
  if (mode() === "manual") $("#gridbox").open = true;
  renderFrames();
};
$("#nviews").oninput = (e) => { $("#nval").textContent = e.target.value; };
$("#startform").onsubmit = async (e) => {
  e.preventDefault();
  const manual = mode() === "manual";
  if (manual && V.picks.length < 2) { say($("#startmsg"), "pick at least 2 frames"); return; }
  // An articulated object sends who lists its parts, and the typed list for "I type them".
  const articulated = $("#articulated").checked === true, pm = chosenPartsMode();
  let parts = [];
  if (articulated && pm === "user") {
    try { parts = parseParts($("#partlist").value); } catch (err) { say($("#startmsg"), err.message); return; }
  }
  let lidar = null;
  try { lidar = lidarChoice(); } catch (err) {  // no topic chosen: the select takes the focus
    say($("#startmsg"), err.message);
    $("#lidartopic").focus();
    return;
  }
  if (V.job && V.job.files && (V.job.files.includes("run/posed.glb") || V.job.files.includes("results.zip")) &&
      !confirm("Build again? This replaces the current results.")) return;
  const agent = chosenAgent(), keyed = agent !== "none", typed = keyed;
  $("#start").disabled = true;
  say($("#startmsg"), keyed ? "checking the key" : "", true);
  try {
    await post(`/api/jobs/${V.id}/start`, { mode: mode(), prompt: $("#prompt").value, n_views: +$("#nviews").value,
      picks: manual ? V.picks : [], view0: manual ? V.view0 : null, auto_continue: !$("#pause").checked,
      agent, api_key: typed ? $("#key").value : null, articulated, ...(articulated ? { parts_mode: pm, parts } : {}), lidar,
      turntable: $("#turntable").checked === true });
    $("#key").value = "";
    say($("#startmsg"), "", true);
    $("#log").textContent = "";
    V.meta = null;
    V.manual = false;
    await refresh();
  } catch (err) { say($("#startmsg"), err.message); }
  $("#start").disabled = false;
  // Disabled while it ran, Start lost the focus: it takes it back when it still shows (a refused start), else the tab.
  if (document.activeElement === document.body && $("#frames").hidden === false && $("#framesbody").hidden === false)
    $("#start").focus({ preventScroll: true });
  else refocus();
};

// ---- 4. review ----

async function loadReview() {
  const sel = V.sel = await api(`/api/jobs/${V.id}/selection`);
  // job.view0 is a pool name once set here; from the start form it is a clicked frame, which became the anchor.
  const view0 = typeof sel.view0 === "string" ? sel.view0 : sel.view0 != null ? sel.anchor : null;
  const saved = () => ({ locked: [...sel.overrides.locked], excluded: [...sel.overrides.excluded],
    instance: { ...sel.overrides.instance }, view0 });
  V.ov = saved();
  V.ov0 = saved();  // what this selection ran with: change() compares the edits with it
  const pick = (x) => x == null ? "?" : x;
  // Coverage: every candidate is within that angle of a pick. Spacing: the closest picks are that far apart.
  $("#selsum").textContent = sel.ok
    ? `${sel.picks.length} of ${sel.n_requested} views · coverage ${pick(sel.coverage_deg)}° · spacing ${pick(sel.min_spacing_deg)}°`
    : "Selection failed. Change some views and select again, or pick frames manually.";
  renderSelLidar();
  $("#selwarn").replaceChildren(...sel.warnings.filter((w) => !V.state.warnings.includes(w)).map((w) => h("li", {}, w)));
  $("#ring").setAttribute("aria-label", `camera positions seen from above: ${sel.picks.length} picked of ${sel.candidates.length} candidates`);
  ring(sel);
  renderCards();
  reviewBar();
}

// ---- 4a. the metric scale from LiDAR: in the review header, and in a panel when the lidar stage failed ----

const NO_LIDAR = "Scale: Depth Anything 3's estimate (no LiDAR)";
// lidar_scale's view statuses; "used" views say nothing more.
const LIDAR_STATUS = { used: "", outlier: "outlier", scattered: "scattered", few_points: "few points", no_mask: "no mask",
  no_samples: "no points" };
const LIDAR_COUNTS = ["views", "views_with_points", "views_used", "outliers", "scattered", "few_points", "no_mask", "no_samples"];
// pool/lidar.json in short, as server.lidar_compact gives it in /selection.
function lidarCompact(rec) {
  if (!rec || typeof rec !== "object") return null;
  const c = rec.combine || {};
  return { ok: rec.ok === true, s: rec.s, reason: rec.reason, counts: Object.fromEntries(LIDAR_COUNTS.map((k) => [k, c[k]])),
    spread_rel: c.spread_rel, source: rec.source, extrinsic: (rec.extrinsic || {}).source, calibration_changed: false,
    warnings: rec.warnings || [], check_dir: "lidar/check",
    views: (rec.views || []).map((v) => ({ name: v.name, s: v.s, points_used: v.points_used, status: v.status })) };
}
// One view's scale and point count: "×1.05 (280 pts)", "×1.21 (90 pts, outlier)", "few points (12 pts)", "no points".
function lidarViewText(v) {
  const st = v.status in LIDAR_STATUS ? LIDAR_STATUS[v.status] : String(v.status || "").replace(/_/g, " ");
  const pts = count(v.points_used) + " pts";
  return v.s != null ? `×${Number(v.s).toFixed(2)} (${pts}${st ? ", " + st : ""})`
    : (st || "no scale") + (v.points_used ? ` (${pts})` : "");
}
// The scale (or why there is none), why it is stale (a changed calibration, or an older estimator), the warnings, and the
// check images folded. head false leaves out the first line: the panel of a failed lidar stage, whose reason the status
// line gives.
function lidarView(lid, head) {
  const c = lid.counts || {}, views = lid.views || [], dir = lid.check_dir || "lidar/check";
  return [head === false ? null : lid.ok ? h("p", { class: "lidarhead" }, `LiDAR scale ×${Number(lid.s).toFixed(3)} ` +
      `(${c.views_used}/${c.views} views, spread ${pct(lid.spread_rel)})`)
    : h("p", { class: "msg" }, "No metric scale from LiDAR: " + (lid.reason || "no reason given")),
  lid.calibration_changed ? h("p", { class: "warn" }, SCALE_STALE.calibration.review)
    : lid.estimator_changed ? h("p", { class: "warn" }, SCALE_STALE.estimator.review) : null,
  (lid.warnings || []).length ? h("ul", { class: "warnings" }, lid.warnings.map((w) => h("li", {}, w))) : null,
  views.length ? h("details", { class: "sub", open: V.lidarOpen === true, ontoggle: (e) => { V.lidarOpen = e.target.open; } },
    h("summary", {}, `LiDAR check images (${views.length})`),
    // lidar_scale.check_image: r / s against the combined s; without one, each view's own s, and white without that.
    h("p", { class: "hint" }, "Green: mask outline. Blue: nearer. Red: farther. White: agrees."),
    h("div", { class: "checks" }, views.map((v) => h("figure", {},
      h("a", { href: file(`${dir}/${v.name}.jpg`), target: "_blank", rel: "noopener" },
        h("img", { src: file(`${dir}/${v.name}.jpg`), loading: "lazy", alt: "LiDAR points in " + v.name })),
      h("figcaption", {}, `${v.name}: ${lidarViewText(v)}`))))) : null];
}
function renderSelLidar() {  // the review header: LiDAR's scale, or DA3's own
  const lid = V.sel.lidar, job = V.job || {};
  fill($("#sellidar"), lid ? lidarView(lid) : h("p", {}, job.lidar
    ? "LiDAR scale: not found yet. Build finds it, then stops here." : NO_LIDAR),
  // A bag's calibration can change in review, in the Settings tab, without a start over.
  job.kind === "rosbag" && job.lidar ? h("p", {}, h("button", { class: "small secondary", type: "button",
    onclick: toLidarSettings }, "Change the calibration")) : null);
}
// The job failed at the lidar stage (stages: job.progress or the state's stages).
const lidarFailed = (status, stages) => status === "failed"
  && ((stages || []).find((x) => x.name === "lidar") || {}).status === "failed";
async function renderLidarFail(job) {
  const rec = job.files.includes("pool/lidar.json") ? lidarCompact(await api(file("pool/lidar.json"))) : null;
  fill($("#lidarfailbody"), rec ? lidarView(rec, false) : h("p", { class: "hint" }, "No LiDAR record was written. See the log."),
    h("p", {}, h("button", { class: "small secondary", type: "button", onclick: toLidarSettings }, "LiDAR settings")));
}
function toLidarSettings() {  // the LiDAR box in the Settings tab: the calibration upload, or LiDAR off
  selectTab("settings", true);
  $("#lidarbox").scrollIntoView({ block: "center", behavior: smooth() });
  ($("#lidaropts").hidden ? $("#uselidar") : $("#calfile")).focus({ preventScroll: true });
}

// A view's use in overrides ov: "in" (always use: locked), "out" (never use: excluded) or "auto".
const useOf = (ov, name) => ov.locked.includes(name) ? "in" : ov.excluded.includes(name) ? "out" : "auto";
// A candidate's status word in the review: "locked" for a pick that is set to "always use", else its status.
const candStatus = (c, sel) => c.status === "picked" && sel.overrides.locked.includes(c.name) ? "locked"
  : c.status.replace("_", " ");
// The cameras seen from above, as this selection ran: view 0 is a star, a pick a filled dot (a locked pick a square),
// each with its view number outward from the object, an available view a hollow dot, and a view that the app leaves
// out (small, inconsistent, face-on or truncated) a cross. style.css (#ring) gives the colors; the attributes here use only currentColor,
// so the marks still read without the stylesheet.
function ring(sel) {
  const svg = $("#ring"), c = sel.ring.center, pts = sel.ring.points;
  svg.replaceChildren();
  if (!c) return;
  // 1.3: room for the view numbers outside the outermost camera
  const r = 1.3 * Math.max(1e-3, ...pts.filter(Boolean).map((p) => Math.hypot(p[0] - c[0], p[1] - c[1])));
  svg.setAttribute("viewBox", `${c[0] - r} ${-c[1] - r} ${2 * r} ${2 * r}`);
  const d = r / 30, order = new Map(sel.picks.map((p, k) => [p.name, k]));
  const line = { fill: "none", stroke: "currentColor", "stroke-width": d / 3 };
  svg.append(s("path", { d: `M${c[0] - d} ${-c[1]}h${2 * d}M${c[0]} ${-c[1] - d}v${2 * d}`, ...line }));
  sel.candidates.forEach((cand, i) => {
    const p = pts[i];
    if (!p) return;  // no camera direction: an excluded view, or one without a mask
    const [x, y] = [p[0], -p[1]], picked = order.has(cand.name), word = candStatus(cand, sel);
    let mark, size = 1.2 * d;  // how far the mark reaches from the camera
    if (cand.name === sel.anchor) {
      size = 2.4 * d;
      const star = [...Array(10)].map((_, k) => { const a = Math.PI * k / 5 - Math.PI / 2, rr = k % 2 ? d : size; return `${x + rr * Math.cos(a)},${y + rr * Math.sin(a)}`; });
      mark = s("polygon", { points: star.join(" "), fill: "currentColor", stroke: "currentColor", "stroke-width": d / 4 });
    } else if (word === "locked") {
      mark = s("rect", { class: "lock", x: x - d, y: y - d, width: 2 * d, height: 2 * d, ...line, fill: "currentColor" });
    } else if (picked || cand.status === "available") {
      mark = s("circle", { class: picked ? "pick" : "cand", cx: x, cy: y, r: size, ...line, fill: picked ? "currentColor" : "none" });
    } else mark = s("path", { class: "out", d: `M${x - d} ${y - d}l${2 * d} ${2 * d}m0 ${-2 * d}l${-2 * d} ${2 * d}`, ...line });
    const title = s("title");
    title.textContent = `${cand.name} (${word})`;
    mark.append(title);
    svg.append(mark);
    if (picked) {  // outward from the object: clear of the mark and of the cameras beside it
      const ux = x - c[0], uy = y + c[1], n = Math.hypot(ux, uy), at = size + 2.2 * d;
      const label = s("text", { x: n ? x + at * ux / n : x, y: n ? y + at * uy / n : y - at, "font-size": 2.4 * d,
        "text-anchor": "middle", "dominant-baseline": "central", fill: "currentColor" });
      label.textContent = String(order.get(cand.name));
      svg.append(label);
    }
  });
}
function renderCards() {
  const sel = V.sel, ov = V.ov, order = new Map(sel.picks.map((p, k) => [p.name, k]));
  const lviews = new Map(((sel.lidar && sel.lidar.views) || []).map((v) => [v.name, v]));  // each view's LiDAR scale
  V.cards = new Map();  // candidate -> its card, its change line and its view 0 choice with its radio (markCards)
  const card = (c) => {
    const sam = sel.sam3[c.name] || { instances: [] }, k = order.get(c.name), lv = lviews.get(c.name);
    const use = useOf(ov, c.name), inst = ov.instance[c.name] || "", word = candStatus(c, sel);
    const note = h("div", { class: "change", id: "change-" + c.name });
    const radio = h("input", { type: "radio", name: "view0", checked: (ov.view0 || sel.anchor) === c.name,
      onchange: () => edit(c.name, "view0") });
    const zero = h("label", { class: "check" }, radio, " view 0");
    const el = h("div", { class: "card " + c.status, role: "group", "aria-labelledby": "card-" + c.name },
      h("div", { class: "cimg" }, h("img", { src: file(`${sel.check_dir}/${c.name}.jpg`), loading: "lazy", alt: "" }),
        k != null ? h("b", { class: "chip" }, `view ${k}${c.name === sel.anchor ? " ★" : ""}`) : null),
      h("div", { class: "meta", id: "card-" + c.name }, `${c.name} at ${c.time_s.toFixed(2)} s`),
      // A locked pick's badge has its own mark (style.css .badge.locked), so it differs from a pick without the word.
      h("div", {}, badge(word === "locked" ? "locked" : c.status, word), c.q != null ? ` q ${c.q.toFixed(2)}` : "",
        c.consistency != null ? ` agree ${c.consistency.toFixed(2)}` : "",
        c.angle_to_nearest_pick != null && k == null ? ` ${c.angle_to_nearest_pick}° from a pick` : ""),
      lv ? h("div", { class: "lidar" + (lv.status === "used" ? "" : " off") }, (lv.s != null ? "LiDAR " : "LiDAR: ") + lidarViewText(lv)) : null,
      note,
      h("select", { "aria-label": c.name + ": use", "aria-describedby": "change-" + c.name, onchange: (e) => edit(c.name, "use", e.target.value) },
        h("option", { value: "auto", selected: use === "auto" }, "auto"),
        h("option", { value: "in", selected: use === "in" }, "always use"),
        h("option", { value: "out", selected: use === "out" }, "never use")),
      h("select", { "aria-label": c.name + ": mask", onchange: (e) => edit(c.name, "instance", e.target.value) },
        h("option", { value: "", selected: !inst }, "top mask"),
        ...sam.instances.map((x) => h("option", { value: String(x.index), selected: inst === String(x.index) }, `mask ${x.index} (${x.score.toFixed(2)})`)),
        h("option", { value: "none", selected: inst === "none" }, "not in frame")),
      zero);
    V.cards.set(c, { el, note, zero, radio });
    return el;
  };
  const picked = sel.candidates.filter((c) => order.has(c.name)).sort((a, b) => order.get(a.name) - order.get(b.name));
  const others = sel.candidates.filter((c) => !order.has(c.name)).sort((a, b) => a.frame - b.frame);
  fill($("#cards"), h("h4", {}, `Picked views (${picked.length})`), h("div", { class: "cards" }, ...picked.map(card)),
    others.length ? h("h4", {}, `Other candidates (${others.length})`) : null,
    others.length ? h("div", { class: "cards" }, ...others.map(card)) : null);
  markCards();
}
// What a card says about its edit until Select again (change()).
const CHANGED = { adds: "Added when you select again.", drops: "Left out when you select again.",
  edited: "Changed. Select again to apply it." };
// A candidate's edit that waits for Select again (V.ov against V.ov0): "adds" (always use, on a view that is not
// picked), "drops" (never use, on a pick), "edited" (another change of its use, its mask or view 0) or "".
function change(c) {
  const n = c.name, use = useOf(V.ov, n), picked = c.status === "picked", v0 = (o) => (o.view0 || V.sel.anchor) === n;
  if (use !== useOf(V.ov0, n)) return use === "out" && picked ? "drops" : use === "in" && !picked ? "adds" : "edited";
  return (V.ov.instance[n] || "") !== (V.ov0.instance[n] || "") || v0(V.ov) !== v0(V.ov0) ? "edited" : "";
}
// Where view 0 can go, with the overrides ov: a pick that is not set to "never use", or a view set to "always use".
const offers0 = (c, ov) => { const use = useOf(ov, c.name); return use === "in" || (use === "auto" && c.status === "picked"); };
// Each card shows its edit, and offers view 0 only where it can go (offers0); its radio follows V.ov.
function markCards() {
  for (const [c, e] of V.cards) {
    const what = change(c);
    e.el.classList.toggle("adds", what === "adds");
    e.el.classList.toggle("drops", what === "drops");
    e.note.textContent = CHANGED[what] || "";
    e.zero.hidden = !offers0(c, V.ov);
    e.radio.checked = (V.ov.view0 || V.sel.anchor) === c.name;
  }
}
// The bar: while edits wait, "Select again" is the primary button and Build is off; else Build is the primary button
// (off when the selection failed).
function reviewBar() {
  const edited = V.sel.candidates.some((c) => change(c));
  $("#build").disabled = edited || !V.sel.ok;
  $("#reselect").disabled = !edited;
  $("#build").classList.toggle("secondary", edited);
  $("#reselect").classList.toggle("secondary", !edited);
  $("#discard").hidden = !edited;
  say($("#revmsg"), edited ? "Select again to apply changes." : "", true);
}
function edit(name, what, value) {
  const ov = V.ov;
  if (what === "use") {
    ov.locked = ov.locked.filter((n) => n !== name);
    ov.excluded = ov.excluded.filter((n) => n !== name);
    if (value === "in") ov.locked.push(name);
    if (value === "out") ov.excluded.push(name);
  } else if (what === "instance") {
    if (value) ov.instance[name] = value; else delete ov.instance[name];
  } else ov.view0 = name;
  // A card that holds view 0 and no longer offers it (an added view set back to "auto") gives it back to the saved choice.
  const held = V.sel.candidates.find((c) => c.name === ov.view0);
  if (held && !offers0(held, ov)) ov.view0 = V.ov0.view0;
  markCards();
  reviewBar();
}
$("#discard").onclick = () => loadReview().catch((e) => say($("#revmsg"), e.message));
$("#manualstart").onclick = async () => {  // the Settings tab in manual mode, with the frames open
  V.manual = true;
  V.meta = null;
  await refresh().catch((e) => say($("#revmsg"), e.message));
  selectTab("settings", true);
  $("#gridbox").open = true;
  $("#gridsum").focus({ preventScroll: true });
  $("#tabs").scrollIntoView({ block: "start", behavior: smooth() });
};
// Build or Select again closes the waiting panel, which would leave the page scrolled past the job's head: the
// title, the status and the progress come back into view, with the tabs under them.
const toHead = () => $("#jtitle").scrollIntoView({ block: "nearest" });
// A view 0 that no card offers now (a saved choice set to "never use") goes: the app takes the best view again.
$("#reselect").onclick = () => act([$("#reselect"), $("#build")], $("#revmsg"), "selecting",
  () => post(`/api/jobs/${V.id}/overrides`, { ...V.ov,
    view0: V.sel.candidates.some((c) => c.name === V.ov.view0 && offers0(c, V.ov)) ? V.ov.view0 : null,
    api_key: $("#key2").value || null }),
  () => { $("#key2").value = ""; $("#log").textContent = ""; toHead(); });
$("#build").onclick = () => act([$("#build"), $("#reselect")], $("#revmsg"), "checking the key",
  () => post(`/api/jobs/${V.id}/continue`, { api_key: $("#key2").value || null }),
  () => { $("#key2").value = ""; $("#log").textContent = ""; toHead(); });

// ---- 4b. the part list ("VLM, then I check": the job waits after propose) ----

async function loadParts() {
  const r = V.parts = await api(`/api/jobs/${V.id}/parts`), p = r.proposal || { parts: [], warnings: [] };
  const moving = p.parts.filter((x) => x.parent), n = moving.length;
  const who = p.source === "vlm" ? VLM[p.agent] || "The VLM" : "You";
  fill($("#partssum"), !p.parts.length ? "No part list yet." : n ? `${who}: ${n} moving ${n === 1 ? "part" : "parts"}. Base: ${p.base}.`
    : `${who} found no moving parts. Add them under the base (${p.base}).`);
  // A stale scale comes first: the server refuses Build then, so the line says what to do. While the list waits, the
  // calibration cannot change, so the cause is mostly an older estimator.
  const stale = !!V.job && scaleStale(V.job) ? h("li", { class: "warn" }, SCALE_STALE[staleBy()].parts) : null;
  fill($("#partswarn"), stale, (p.warnings || []).map((w) => h("li", {}, w)));
  fill($("#partshints"), moving.filter((x) => x.axis_hint).map((x) => h("li", {}, h("b", {}, x.name), ": " + x.axis_hint)));
  $("#partsfig").hidden = !r.photo;
  if (r.photo) $("#partsphoto").src = file(r.photo);
  const key = JSON.stringify(p.parts);  // a new proposal replaces the text; the same one keeps the user's edits
  if (key !== V.partsKey) {
    V.partsKey = key;
    V.partsText = formatParts(p.parts);
    $("#partsedit").value = V.partsText;
  }
  $("#partsreset").hidden = $("#partsedit").value === V.partsText;
}
// The parsed list for POST /parts: a part the user kept (same name, same joint) keeps the VLM's axis hint.
function keepHints(parts, proposed) {
  const norm = (s) => String(s || "").trim().toLowerCase().replace(/\s+/g, " ");
  const was = new Map(proposed.map((p) => [norm(p.name), p]));
  return parts.map((p) => {
    const q = was.get(norm(p.name));
    return q && q.axis_hint && (p.joint || "unknown") === q.joint ? { ...p, axis_hint: q.axis_hint } : p;
  });
}
$("#partsedit").oninput = () => { $("#partsreset").hidden = $("#partsedit").value === V.partsText; say($("#partsmsg"), "", true); };
$("#partsreset").onclick = () => {
  $("#partsedit").value = V.partsText;
  $("#partsreset").hidden = true;
  say($("#partsmsg"), "", true);
};
$("#partsbuild").onclick = () => {  // the VLM's list as it is (continue), or the edited list (POST /parts)
  // The text as shown confirms the VLM's list without parsing it here: a VLM's name may hold ":" or ">".
  const same = $("#partsedit").value.trim() === (V.partsText || "").trim();
  let parts;
  if (!same) try { parts = parseParts($("#partsedit").value); } catch (e) { say($("#partsmsg"), e.message); return; }
  const proposed = (V.parts && V.parts.proposal && V.parts.proposal.parts) || [];
  const api_key = $("#key2").value || null;
  return act([$("#partsbuild"), $("#partsreset")], $("#partsmsg"), "checking the parts",
    () => same ? post(`/api/jobs/${V.id}/continue`, { api_key }) : post(`/api/jobs/${V.id}/parts`, { parts: keepHints(parts, proposed), api_key }),
    () => { $("#key2").value = ""; $("#log").textContent = ""; V.partsKey = null; toHead(); });
};

// ---- 5. results ----

// model-viewer comes from a module script: a CDN on the tunnel page, the site's own copy on the public site. When that
// script does not load, a viewer would say "loading" for ever, so it waits for the element at most MV_WAIT_MS.
const MV_WAIT_MS = 10000;
function mvDefined() {  // true once <model-viewer> is defined; false when it is not after MV_WAIT_MS
  const ce = window.customElements;
  if (!ce) return Promise.resolve(false);
  if (ce.get("model-viewer")) return Promise.resolve(true);
  return Promise.race([ce.whenDefined("model-viewer").then(() => true),
    new Promise((r) => setTimeout(() => r(false), MV_WAIT_MS))]);
}
async function renderResults(job) {
  renderSpin(job);
  const has = (p) => job.files.includes(p), json = async (p) => has(p) ? (await T.fetch(file(p))).json() : null;
  const cm = (x) => (100 * x).toFixed(0);
  // A viewer: until its mesh shows, the box says "Loading the mesh" (model-viewer's poster slot, which it hides then)
  // and the caption "loading" where the size goes. The caption says it to a screen reader, so the box's text is hidden.
  // When the mesh does not load, or model-viewer itself does not, the caption is the title and a line offers the file.
  const viewer = (p, title) => {
    const cap = h("figcaption", {}, title + ": loading"), fig = h("figure", {}, cap), name = p.split("/").pop();
    const failed = () => {
      cap.textContent = title;
      fig.replaceChildren(cap, h("p", { class: "msg" }, `Could not show ${name}: `,
        h("a", { href: file(p, name), download: name }, "download it"), "."));
    };
    fig.append(h("model-viewer", { src: file(p), "camera-controls": true, "environment-image": "neutral", "shadow-intensity": "1",
      "interaction-prompt": "none", alt: title,
      onload: (e) => {  // the size of the model, in the GLB's meters
        const d = e.target.getDimensions && e.target.getDimensions();
        cap.textContent = d ? `${title}: ${cm(d.x)} x ${cm(d.y)} x ${cm(d.z)} cm` : title;
      },
      onerror: failed }, h("p", { slot: "poster", class: "mvload", "aria-hidden": "true" }, "Loading the mesh")));
    mvDefined().then((ok) => { if (!ok) failed(); });
    return fig;
  };
  const off = job.agent === "none", a = job.agent_result, art = job.articulated === true;
  const ran = a && RAN[a.agent];
  const rigged = !!a && a.mode === "articulation";  // the agent set the joints of the split parts
  const by = (title) => `${title} (${ran})`;
  const notRefined = art ? "off (parts welded into one body)" : "off";
  const refined = has("refined.glb") ? viewer("refined.glb", !rigged ? by("Refined")
    : by(a.joints_delivered > 0 ? "Parts with joints" : "Agent result, no joints")) : null;
  const split = has("run/parts.glb") ? viewer("run/parts.glb", "Split parts") : null;
  $("#viewers").classList.toggle("single", !refined && !split);
  $("#resultsbody").classList.toggle("onefig", !refined && !split);  // a wide screen puts the facts beside it
  // With the agent off there is no agent viewer; the Agent row of the facts says so.
  fill($("#viewers"), viewer("run/posed.glb", "Posed mesh"), split, refined);
  // The evidence alone may fail to read: its rows are left out then, not every row.
  const [place, phys, mj, parts, pr, lid, ev] = await Promise.all([json("run/placement.json"), json("physics/summary.json"),
    art ? json("physics/mjcf/mjcf.json") : null, art ? json("run/parts.json") : null,
    art && has("run/parts.json") ? api(`/api/jobs/${V.id}/parts`).catch(() => null) : null, json("pool/lidar.json"),
    art ? json("agent/evidence.json").catch(() => null) : null]);
  // [label, value]: what each one means is in help.html.
  const rows = [], num = (x, d = 3) => x == null ? "?" : Number(x).toPrecision(d);
  // A position in m, to the mm. Rounding first turns float noise such as -1.51e-18 into 0.000, not -0.000.
  const mm = (x) => x == null ? "?" : (Math.round(1000 * x) / 1000).toFixed(3);
  if (place) rows.push(["Metric scale", metricScale(place, lid, job)]);
  if (place) rows.push(["Mask IoU", num(place.check && place.check.mean_mask_iou)]);
  const lost = parts ? (parts.parts || []).filter((x) => x.status !== "found") : [];  // not_found, dropped_small, unconfirmed
  if (lost.length) rows.push(["Parts not split off", lost.map((x) => `${x.prompt} (${x.why || x.status.replace("_", " ")})`).join("; ")]);
  if (job.stages.agent.status === "skipped") rows.push(["Agent", off ? notRefined : "skipped: " + job.stages.agent.note]);
  else if (a) {
    rows.push(["Agent", `${ran} (${a.model})`]);
    rows.push(["Agent verdict", `${rigged && VERDICT[a.verdict] || a.verdict} (${a.source})`], ["Agent reason", a.reason || ""]);
    for (const l of a.limitations || []) rows.push(["Limitation", l]);
    if (rigged) rows.push(["Joints", a.joints_delivered != null ? String(a.joints_delivered) : "?"]);
    if (rigged) rows.push(["Joint changes", String(a.counts.joints)]);
    rows.push(...roundRows(a, ev));
    if (a.tokens) {
      const t = a.tokens;
      rows.push(["Agent tokens", `${count(t.input_tokens)} input, ${count(t.output_tokens)} output, ` +
        `${count(t.cache_read_input_tokens)} cache read, ${count(t.cache_creation_input_tokens)} cache write` +
        (a.turns != null ? ` in ${a.turns} turns` : "")],
        ["Agent cost (estimate)", a.cost_usd != null ? "about $" + Number(a.cost_usd).toFixed(2) : "?"]);
    }
  }
  if (mj && mj.links) {  // the MJCF of an object with joints: a body per link
    const names = Object.keys(mj.links), deg = (x) => (x * 180 / Math.PI).toFixed(0), cmr = (x) => (100 * x).toFixed(1);
    rows.push(["Links", `${names.length}: ${names.join(", ")}`],
      ["Base", `${mj.root}, ${mj.fixed_base ? "fixed to the world" : "free to move"}`]);
    for (const j of mj.joints || []) rows.push([`Joint of ${j.child}`, j.type === "revolute"
      ? `turns on ${j.parent}, ${deg(j.range[0])}° to ${deg(j.range[1])}°` : j.type === "prismatic"
        ? `slides on ${j.parent}, ${cmr(j.range[0])} to ${cmr(j.range[1])} cm` : j.type === "soft"
          ? `soft on ${j.parent}` : `fixed to ${j.parent}`]);
    for (const [k, to] of Object.entries(mj.welded || {})) rows.push([`Joint of ${k}`, `none (welded to ${to})`]);
    rows.push(...softRows(mj));
  } else if (mj && art) rows.push(["Links", "1 (no joints, one rigid body)"]);
  if (phys) {
    const whole = mj && mj.links ? mj.inertial : null;  // every link, also one that VoMP could not voxelize
    const rb = whole || phys.rigid_body || {}, md = phys.median || {}, skipped = Object.keys(phys.links_skipped || {});
    rows.push(["Mass", num(rb.mass) + " kg"], ["Center of mass", (rb.com || []).map(mm).join(", ") + " m"]);
    if (whole) rows.push(["Link masses", Object.entries(mj.links).map(([k, l]) => `${k} ${num(l.inertial.mass)} kg` +
      (l.source === "hull" ? " (from hulls)" : "")).join(", ")]);
    else if (skipped.length) rows.push(["Mass note", `Leaves out ${skipped.join(", ")} (no VoMP result)`]);
    rows.push(["Young's modulus (median)", num(md.youngs_modulus_pa) + " Pa"], ["Poisson ratio (median)", num(md.poisson_ratio)],
      ["Density (median)", num(md.density_kg_m3) + " kg/m³"]);
  }
  $("#facts2").replaceChildren(h("dl", {}, ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])));
  const checks = (pr && pr.checks) || [];  // run/parts/sam3/check/<view>.jpg
  fill($("#partchecks"), checks.length ? h("details", { class: "sub" }, h("summary", {}, `Part masks (${checks.length})`),
    h("p", { class: "hint" }, "One color per part."),
    h("div", { class: "checks" }, ...checks.map((c) => h("a", { href: file(c), target: "_blank", rel: "noopener" },
      h("img", { src: file(c), loading: "lazy", alt: "part masks in view " + /(\d+)\.jpg$/.exec(c)[1] }))))) : null);
}
// The "Metric scale" row: placement.json metric_scale (select applied it), with lidar.json's source, views and spread.
function metricScale(place, rec, job) {
  const m = place.metric_scale;
  if (typeof m !== "number") return "Depth Anything 3's estimate (no LiDAR)";
  const src = (rec && rec.source) || {}, c = (rec && rec.combine) || {};
  const same = !!rec && typeof rec.s === "number" && Math.abs(rec.s - m) <= 1e-4 * m;  // lidar.json of this build
  const facts = [src.kind === "stray" ? "iPhone LiDAR depth" : src.topic || (job.lidar && job.lidar.topic) || null,
    same && c.views_used != null ? `${c.views_used} views` : null, same && c.spread_rel != null ? "spread " + pct(c.spread_rel) : null];
  const shown = facts.filter(Boolean);
  return `×${m.toFixed(3)} from LiDAR` + (shown.length ? ` (${shown.join(", ")})` : "");
}
// The outcomes of a requirement (harpor.agent.controller.OUTCOMES), whose counts agent_result.requirements gives.
const OUTCOMES = ["pass", "fail", "inconclusive", "unassessed"];
const degrees = (x) => (x * 180 / Math.PI).toFixed(0);
// The rows of a session in rounds (controller.run_session) and of the surface check: the rounds, how many requirements
// the delivered candidate meets, each requirement of agent/evidence.json (ev) with its outcome and what the host's
// check measured, and how much of the original's surface the result keeps. A key that the result lacks gives no row:
// old jobs may have none of them.
function roundRows(a, ev) {
  const rows = [], rounds = Array.isArray(a.rounds) ? a.rounds.filter((r) => r && typeof r === "object") : [];
  if (rounds.length) rows.push(["Rounds", roundsText(rounds, a.candidate)]);
  const n = a.requirements && typeof a.requirements === "object" ? a.requirements : null;
  if (n) {
    const total = OUTCOMES.reduce((sum, k) => sum + (Number(n[k]) || 0), 0);
    rows.push(["Requirements", `${Number(n.pass) || 0} of ${total} met`]);
    for (const r of chosenRecords(ev)) rows.push([`Requirement ${r.requirement}`, requirementText(r, ev)]);
  }
  if (a.surface && typeof a.surface === "object") rows.push(["Surface retained", surfaceText(a.surface)]);
  return rows;
}
// "2 (build fail, repair pass); delivered round 2's candidate". cand: agent_result.candidate, null when the session
// delivered no stored candidate.
function roundsText(rounds, cand) {
  const each = rounds.map((r) => r.outcome ? `${r.kind} ${r.outcome}` : r.kind).join(", ");
  const sent = cand && typeof cand === "object" ? `; delivered round ${cand.round}'s candidate`
    : cand === null ? "; no candidate delivered" : "";
  return `${rounds.length} (${each})${sent}`;
}
// The records of the candidate that the session delivered (evidence.json selected), else none.
function chosenRecords(ev) {
  const sel = ev && ev.selected, cands = ev && Array.isArray(ev.candidates) ? ev.candidates : [];
  const c = sel && sel.version != null ? cands.find((x) => x && x.version === sel.version) : null;
  return c && Array.isArray(c.records) ? c.records.filter((r) => r && typeof r === "object") : [];
}
// "lid, revolute: pass, clear from 0° to 86°", "pad, soft: pass, dents 4.0 mm", or for a part that was not split off
// "drawer (not split off), prismatic: unassessed".
function requirementText(r, ev) {
  const m = r.measured && typeof r.measured === "object" ? r.measured : {};
  const target = Array.isArray(r.target) ? r.target[0] : null;
  const req = (Array.isArray(ev.requirements) ? ev.requirements : []).find((q) => q && q.id === r.requirement) || {};
  let text = `${target || `${req.prompt || "?"} (not split off)`}, ${r.joint}: ${r.outcome}`;
  const [lo, hi] = Array.isArray(m.achieved) ? m.achieved : [];
  if (typeof lo === "number" && typeof hi === "number") {
    text += m.type === "prismatic" ? `, clear from ${(100 * lo).toFixed(1)} to ${(100 * hi).toFixed(1)} cm`
      : `, clear from ${degrees(lo)}° to ${degrees(hi)}°`;
  }
  if (typeof m.dmax_m === "number") text += `, dents ${(1000 * m.dmax_m).toFixed(1)} mm`;
  return text + (r.required === false ? ", not required" : "");
}
// "99.5% kept: 98.0% as it was, 1.5% changed", with the share whose UVs cannot be checked and the surface findings.
function surfaceText(sf) {
  if (sf.error != null) return "not compared (the check failed)";
  const sh = sf.shares || {}, found = Array.isArray(sf.findings) ? sf.findings : [];
  const codes = [...new Set(found.map((f) => f && f.code).filter(Boolean))];
  return `${pct(sh.present)} kept: ${pct(sh.retained)} as it was, ${pct(sh.modified)} changed`
    + (sh.unresolved > 0 ? `, ${pct(sh.unresolved)} without UVs to check` : "")
    + (codes.length ? `; ${codes.join(", ")} (${sf.blocks ? "blocks articulated" : "report only"})` : "");
}
// The Soft part rows of mjcf.json: each soft link with its material and Young's modulus, and each one that stayed rigid
// in MuJoCo too (soft_rigid), with the reason.
function softRows(mj) {
  const kpa = (y) => typeof y === "number" ? Math.round(y / 1e3).toLocaleString("en-US") : "?";
  const soft = Object.entries(mj.soft || {}).map(([k, sp]) => ["Soft part",
    `${k}: ${(sp && sp.material) || "?"}, about ${kpa(sp && sp.young)} kPa; rigid in URDF and USD`]);
  const rigid = Object.entries(mj.soft_rigid || {}).map(([k, r]) => ["Soft part",
    `${k}: rigid in MuJoCo too (${(r && r.reason) || "?"})`]);
  return [...soft, ...rigid];
}

// ---- 5a. the Downloads tab: the job's files by group ----

// The groups in order, each with its files as [path in the job folder, the name it downloads as]. They are
// server.OUTPUTS, which server.ALLOW serves; the tab lists those that exist (job.files). The MJCF and USD models, the
// URDF's meshes and the agent's result are not served alone: they come only in results.zip (help.html says what it holds).
const DOWNLOADS = [
  ["Everything", [["results.zip", "results.zip"]]],
  ["Meshes", [["run/posed.glb", "posed.glb"], ["run/parts.glb", "parts.glb"], ["refined.glb", "refined.glb"],
    ["physics/aligned.glb", "aligned.glb"]]],
  ["Video", [["turntable/turntable.mp4", "turntable.mp4"], ["softvideo/softvideo.mp4", "softvideo.mp4"]]],
  ["Simulation", [["physics/urdf/model.urdf", "model.urdf"]]],
  ["Records", [["run/placement.json", "placement.json"], ["physics/summary.json", "summary.json"], ["pool/lidar.json", "lidar.json"],
    ["selection.json", "selection.json"], ["run/parts.json", "parts.json"], ["physics/mjcf/mjcf.json", "mjcf.json"],
    ["agent/evidence.json", "evidence.json"]]]];
// What the Simulation group says above its one file: model.urdf does not load alone.
const SIM_TEXT = "Needs the meshes in results.zip.";
// A file's size: "512 B", "3.4 KB", "35 KB", "1.2 MB", "100 MB", "2.50 GB" (1 KB is 1000 bytes, as in size()).
const fileSize = (b) => b < 1e3 ? b + " B" : b < 999.5e3 ? (b / 1e3).toFixed(b < 9.95e3 ? 1 : 0) + " KB" : size(b);
// What the tab says above its files when results.zip is not one of them: before a build ends (FILES_PARTIAL), or
// after a build that stopped (FILES_STOPPED). A done job has results.zip (package is the last stage).
const filesNote = (status) => ["failed", "cancelled"].includes(status) ? FILES_STOPPED : status === "done" ? null : FILES_PARTIAL;
// The files of the job that exist, by group: each a download link (named <job id>_<file>, so files of two jobs do not
// clash) and its size (job.sizes). While a build runs the tab lists none, since the build replaces them as it goes
// (jobs.DERIVED); it says one sentence then, and when there is no file yet.
function renderDownloads(job) {
  const files = ACTIVE.includes(job.status) ? [] : job.files || [], sizes = job.sizes || {};
  const has = (p) => files.includes(p);
  const groups = DOWNLOADS.map(([title, rows]) => [title, rows.filter(([p]) => has(p))]).filter(([, rows]) => rows.length);
  const row = ([p, name]) => h("li", { class: "dlrow" },
    h("a", { href: file(p, `${job.id}_${name}`), download: `${job.id}_${name}`, class: p === "results.zip" ? "dl primary" : "dl" }, name),
    typeof sizes[p] === "number" ? h("span", { class: "dlsize" }, fileSize(sizes[p])) : null);
  const note = groups.length && !has("results.zip") ? filesNote(job.status) : null;
  fill($("#downloadsbody"), note ? h("p", { class: "dlnote" }, note) : null, groups.map(([title, rows]) => {
    const id = "dl-" + title.toLowerCase(), text = title === "Simulation" ? SIM_TEXT : null;
    return h("div", { class: "dlgroup" }, h("h3", { id }, title), text ? h("p", { class: "hint" }, text) : null,
      h("ul", { class: "dlist", "aria-labelledby": id }, rows.map(row)));
  }));
  $("#downloadsbody").hidden = !groups.length;
  $("#downloadsnone").hidden = groups.length > 0;
  $("#downloadsnone").textContent = ["queued", "running"].includes(job.status) ? FILES_BUSY
    : ["failed", "cancelled"].includes(job.status) ? FILES_NONE_STOPPED : FILES_LATER;
}

// ---- 5b. the input: its preview video and the views the build used ----

// job.preview (server.preview_of): the record of frames/preview.mp4 (video.Preview) with "seek", where each view is in
// it; {error} when the frames stage could not write it; {missing} when the file is gone since. It is null until the
// frames are read, and renderInput runs only after that.
const MISSING_PREVIEW = "Preview video missing.";
// The input's name and kind: "table.MOV (video)", "run.bag (rosbag, camera topic /cam/image_rect)".
function inputLabel(job) {
  const src = job.source, kind = job.kind;
  return `${job.video.name} (${KIND_LABEL[kind] || kind}${kind === "rosbag" && src.topic ? ", camera topic " + src.topic : ""})`;
}
// Where the preview p shows view pick pk: {frame, at}, the frame it shows (the preview keeps every stride-th frame, so a
// skipped frame shows the kept one before it) and its time in the video (p.seek). Without p.seek, frame is null and at
// is the pick's time less t0, since the preview starts at frame 0.
function previewSpot(pk, p) {
  const s = p.seek && p.seek[pk.frame];
  return s ? { frame: s.frame, at: Number(s.at) } : { frame: null, at: Math.max(0, Number(pk.time_s) - (Number(p.t0) || 0)) };
}
const viewText = (pk, i) => `view ${i}, frame ${pk.frame}, ${Number(pk.time_s).toFixed(1)} s`;  // a selection.json pick
const viewSrc = (i) => file(V.overlay === true && V.input.placed ? `run/overlay_${i}.png` : `run/input/images/${i}.png`);
function clearInput() {  // the frames are read again, or another job opens: the old video, views and points go
  V.input = V.inputKey = V.videoKey = V.video = null;
  fill($("#inputvideo"));
  fill($("#inputviews"));
  freeCloud();
  fill($("#inputlidar"));
  say($("#inputmsg"), "", true);
}
// The Inputs tab once the frames are read: the preview video, and the views used once they are selected, with their
// LiDAR points for a job with LiDAR.
async function renderInput(job) {
  const p = job.preview.file ? job.preview : null;
  // The video is drawn once per preview file, so a refresh or a new selection does not stop it. Its "seek" follows the
  // selection, so it does not count.
  const vkey = JSON.stringify([job.id, { ...job.preview, seek: null }]);
  if (vkey !== V.videoKey) {
    V.videoKey = vkey;
    V.inputKey = null;  // the views are buttons only next to a video
    const video = V.video = p ? h("video", { src: file("frames/preview.mp4"), controls: true, preload: "metadata",
      playsinline: true, muted: true, width: p.width, height: p.height, style: `--ar: ${p.width} / ${p.height}`,
      "aria-label": "Preview of " + job.video.name }) : null;
    if (video) video.muted = true;  // the attribute only sets the default
    const why = job.preview.missing ? MISSING_PREVIEW : `No preview video: ${job.preview.error}`;
    fill($("#inputvideo"), video ? h("figure", { class: "inputfig" }, video, h("figcaption", {}, inputLabel(job)))
      : [h("p", {}, inputLabel(job)), h("p", { class: "hint" }, why)]);
  }
  V.inputPlaced = job.files.includes("run/placement.json");  // place drew run/overlay_<i>.png for every view first
  if (job.stages.select.status !== "done") {  // no views yet, or a new selection runs: the old views go
    const pool = job.lidar ? poolClouds(job) : [];  // a LiDAR scale that failed: the points of the pool's views
    const key = pool.length ? "pool " + pool.map((x) => x.path).join() : "none";
    if (V.inputKey !== key) {
      const page = V;
      V.input = null;
      V.inputKey = key;
      fill($("#inputviews"), h("p", { class: "hint" }, NO_VIEWS));
      say($("#inputmsg"), "", true);
      if (pool.length) await renderCloud(job, pool, () => V !== page || V.inputKey !== key);
      else {
        freeCloud();
        fill($("#inputlidar"));
      }
    }
    return;
  }
  // Redraw views when the selection or LiDAR files change; placement changes only the heading.
  const key = JSON.stringify([job.id, job.stages.select.ended, vkey, usedClouds(job).map((x) => x.path).join()]);
  if (key === V.inputKey) {
    if (V.input && V.input.placed !== V.inputPlaced) {
      V.input.placed = V.inputPlaced;
      inputHead();
      if (V.overlay === true) V.input.imgs.forEach((img, i) => img.setAttribute("src", viewSrc(i)));
    }
    return;
  }
  const page = V;  // openJob makes a new V for another job
  V.inputKey = key;
  const gone = () => V !== page || V.inputKey !== key;  // another job's page, or another selection, came meanwhile
  let sel;
  try { sel = await api(file("selection.json")); } catch (e) {
    if (gone()) return;
    V.inputKey = null;
    throw e;
  }
  if (gone()) return;
  const video = V.video;
  say($("#inputmsg"), "", true);
  const picks = sel && Array.isArray(sel.picks) ? sel.picks : [];  // in run order: view 0 first
  V.input = { picks, video, preview: p, placed: V.inputPlaced, imgs: [], btns: [], figs: [],
    head: h("div", { class: "viewshead" }), hint: h("p", { class: "hint" }) };
  const ratio = p ? `aspect-ratio: ${p.width} / ${p.height}` : null;  // keeps cells sized before lazy images load
  const figs = picks.map((pk, i) => {
    const text = viewText(pk, i), img = h("img", { src: viewSrc(i), loading: "lazy", alt: "", style: ratio });  // the caption says it
    const star = i === 0 ? h("b", { class: "chip", "aria-hidden": "true" }, "★") : null;
    const btn = video ? h("button", { type: "button", class: "pic", "aria-label": `${text}: show it in the video`,
      onclick: () => seekInput(i) }, img, star) : null;
    V.input.imgs.push(img);
    if (btn) V.input.btns.push(btn);
    return h("figure", {}, btn || h("div", { class: "pic" }, img, star), h("figcaption", {}, text));
  });
  V.input.figs = figs;
  fill($("#inputviews"), V.input.head, V.input.hint, h("div", { class: "views" }, figs));
  inputHead();
  if (job.lidar) await renderCloud(job, usedClouds(job, picks), gone);
  else {
    freeCloud();
    fill($("#inputlidar"));
  }
}
function inputHead() {  // the strip's heading and hint, with the mesh fit toggle once place has run
  const I = V.input;
  fill(I.head, h("h4", {}, `Views used (${I.picks.length})`), I.placed ? h("button", { type: "button", class: "small secondary",
    "aria-pressed": String(V.overlay === true), onclick: (e) => toggleOverlay(e.currentTarget) }, "Mesh fit") : null);
  inputHint();
}
// The strip's hint: view 0's mark, and the overlay's colors while the mesh fit shows.
const inputHint = () => fill(V.input.hint, "★ view 0" + (V.input.placed && V.overlay === true ? ". Red: mesh. Green: mask outline." : ""));
function toggleOverlay(btn) {  // each view's photo, or place's overlay of the posed mesh on it
  V.overlay = V.overlay !== true;
  btn.setAttribute("aria-pressed", String(V.overlay));
  V.input.imgs.forEach((img, i) => img.setAttribute("src", viewSrc(i)));
  inputHint();
}
function seekInput(i) {  // the preview at view i's frame, paused: it never starts playing by itself
  const v = V.input && V.input.video, pk = V.input && V.input.picks[i], p = V.input && V.input.preview;
  if (!v || !pk || !p) return;
  const spot = previewSpot(pk, p);
  v.pause();
  v.currentTime = spot.at + 0.001;  // 1 ms in, so that a browser that rounds the time shows this frame, not the one before
  V.input.figs.forEach((f, k) => f.classList.toggle("at", k === i));
  V.input.btns.forEach((b, k) => b.setAttribute("aria-current", String(k === i)));  // the view that the video shows
  // The frame that the video shows: the preview keeps only some frames, so it can be the kept one before the view's.
  say($("#inputmsg"), `View ${i}` + (spot.frame == null ? "" : `, frame ${spot.frame}`) + `, at ${fmt(spot.at)}`, true);
  v.scrollIntoView({ block: "nearest" });
}

// ---- 5c. the LiDAR points of the views (Inputs tab) ----

// The files of LiDAR points (lidar.cloud_glb's) that the viewer shows, each {id, label, path, name, save}: run/lidar/<i>.glb
// of view i, which select copies for the views used (picks give their names), else lidar/cloud/<frame>.glb of each view
// of the pool, which the server offers only while the scale has failed (server.pool_clouds).
const CLOUD = /^run\/lidar\/(\d{1,3})\.glb$/, POOL_CLOUD = /^lidar\/cloud\/(f\d{5,9})\.glb$/;
const usedClouds = (job, picks) => (job.files || []).map((f) => CLOUD.exec(f)).filter(Boolean)
  .sort((a, b) => a[1] - b[1]).map(([path, i]) => ({ id: i, label: "view " + i, path, name: ((picks || [])[i] || {}).name,
    save: `${job.id}_lidar_view${i}.glb` }));
const poolClouds = (job) => (job.files || []).map((f) => POOL_CLOUD.exec(f)).filter(Boolean).sort()
  .map(([path, n]) => ({ id: n, label: n, path, name: n, save: `${job.id}_lidar_${n}.glb` }));
const NO_CLOUD = "No LiDAR points to show.";
const NO_GL = "Cannot draw the points here: ";
const NO_COMPARE = "No comparison for this view.";
const DIFF_SPAN_MM = 15;  // full color at this many mm, in every view, so that the drift between views shows
// The viewer drawn now (renderCloud). It lives outside V, since openJob makes a new V before clearInput frees it: a
// page has few WebGL contexts, so each one goes before another job or selection draws.
let cloudNow = null;
function freeCloud() {
  if (cloudNow && cloudNow.gl) cloudNow.gl.free();
  cloudNow = null;
}
// lidar.cloud_glb's file: {scan, used, da3, diff}, each a Float32Array of x, y, z in glTF axes (a set that is not in it
// is empty), delta, the diff points' gaps in meters, and compare, the diff node's extras {s, median, used}, or null.
// Only the nodes named exactly so count, each a mesh whose first primitive is POINTS (mode 0) with a float VEC3 POSITION,
// tightly packed, inside the BIN chunk. A diff node also has a float SCALAR _DIFF of as many rows, and its extras: s
// over 0, used a count of its points, and median_m a number, null only when used is 0. Anything else throws.
function readCloud(buf) {
  const bad = () => { throw new Error("not a LiDAR points file"); };
  const at = (list, i) => Array.isArray(list) && Number.isInteger(i) && i >= 0 && i < list.length ? list[i] || {} : bad();
  const whole = (x) => Number.isInteger(x) && x >= 0;
  const none = () => new Float32Array(0), dv = new DataView(buf);
  const out = { scan: none(), used: none(), da3: none(), diff: none(), delta: none(), compare: null };
  if (buf.byteLength < 28 || dv.getUint32(0, true) !== 0x46546C67 || dv.getUint32(4, true) !== 2) bad();
  const jlen = dv.getUint32(12, true), bin = 28 + jlen;
  if (dv.getUint32(16, true) !== 0x4E4F534A || jlen % 4 || bin > buf.byteLength) bad();
  const blen = dv.getUint32(20 + jlen, true);
  if (dv.getUint32(24 + jlen, true) !== 0x004E4942 || bin + blen > buf.byteLength) bad();
  let doc;
  try { doc = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jlen))); } catch (e) { bad(); }
  const floats = (i, type, k) => {  // accessor i, k floats a row
    const acc = at(doc.accessors, i), view = at(doc.bufferViews, acc.bufferView);
    const vo = view.byteOffset || 0, ao = acc.byteOffset || 0, n = acc.count;
    if (acc.componentType !== 5126 || acc.type !== type || view.byteStride || (view.buffer || 0) !== 0
      || ![vo, ao, n, view.byteLength].every(whole) || (vo + ao) % 4 || ao + 4 * k * n > view.byteLength
      || vo + view.byteLength > blen) bad();
    return new Float32Array(buf, bin + vo + ao, k * n);
  };
  for (const node of doc && Array.isArray(doc.nodes) ? doc.nodes : []) {
    const name = node && node.name;
    if (!["scan", "used", "da3", "diff"].includes(name)) continue;
    const prim = at(at(doc.meshes, node.mesh).primitives, 0), attrs = prim.attributes || {};
    if (prim.mode !== 0) bad();
    out[name] = floats(attrs.POSITION, "VEC3", 3);
    if (name !== "diff") continue;
    const { s, median_m: median, used } = node.extras || {};
    out.delta = floats(attrs._DIFF, "SCALAR", 1);
    if (3 * out.delta.length !== out.diff.length || !(Number.isFinite(s) && s > 0) || !whole(used)
      || used > out.delta.length || !(used ? Number.isFinite(median) : median === null)) bad();
    out.compare = { s, median, used };
  }
  return out;
}
// Where a view starts: the eye at the origin, where the view's camera stood (so the first picture looks like the
// photo), looking at the median of the used points, else of the scan's points in front of the camera (z < 0 in glTF
// axes), else 1 m ahead. The eye is at target + dist (cos pitch sin yaw, sin pitch, cos pitch cos yaw).
const PITCH_MAX = 89 * Math.PI / 180;
function orbitStart(scan, used) {
  const median = (a, keep) => {
    const xyz = [[], [], []];
    for (let i = 0; i + 2 < a.length; i += 3) if (keep(a[i + 2])) for (let k = 0; k < 3; k++) xyz[k].push(a[i + k]);
    return xyz[0].length ? xyz.map((v) => v.sort((p, q) => p - q)[v.length >> 1]) : null;
  };
  const target = median(used, () => true) || median(scan, (z) => z < 0) || [0, 0, -1];
  const dist = Math.max(Math.hypot(...target), 0.05);
  const pitch = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, Math.asin(Math.max(-1, Math.min(1, -target[1] / dist)))));
  return { target, dist, yaw: Math.atan2(-target[0], -target[2]), pitch };
}
// 4x4 matrices, column-major as WebGL takes them.
const perspective = (fovy, aspect, near, far) => {
  const f = 1 / Math.tan(fovy / 2), r = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * r, -1, 0, 0, 2 * far * near * r, 0];
};
function lookAt(eye, at) {  // world to camera, y up
  const unit = (v) => { const n = Math.hypot(...v); return v.map((x) => x / n); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const z = unit(eye.map((x, k) => x - at[k])), x = unit(cross([0, 1, 0], z)), y = cross(z, x);
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
}
function matmul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[4 * c + r] += a[4 * k + r] * b[4 * c + k];
  return o;
}
// The camera of the view as line segments: from the origin to the 4 corners of a frame of that aspect ratio at depth d
// (down -z), and the frame's 4 sides. Up and down, the frame takes 3/4 of the viewer's field (0.35 of tan 25°), and
// fieldOf widens the field on a canvas narrower than the frame, so the first picture shows it whole at any shape.
const FRAME_HALF = 0.35, FIELD_HALF = Math.tan(25 * Math.PI / 180);
function pyramid(d, aspect) {
  const hh = FRAME_HALF * d, hw = hh * aspect, c = [[-hw, hh, -d], [hw, hh, -d], [hw, -hh, -d], [-hw, -hh, -d]];
  return new Float32Array(c.flatMap((p, i) => [0, 0, 0, ...p, ...p, ...c[(i + 1) % 4]]));
}
// The viewer's vertical field in radians: 50°, or wider when the canvas (canvasAspect) is narrower than the frame
// (frameAspect), so the frame fills as much of the field across as up and down.
const fieldOf = (frameAspect, canvasAspect) => 2 * Math.atan(FIELD_HALF * Math.max(1, frameAspect / canvasAspect));
// The WebGL viewer of a canvas: {show(data, aspect), reset(), compare(yes), draw(), free()}, or null without WebGL. on.sized() runs at
// the first size of the canvas that is not 0 (a hidden tab has none), and on.lost() when the browser takes the context
// back. The scan is drawn with the depth test, the camera's pyramid on it, and the used points last without it, so they
// always show. With compare(true), a view with a diff set shows the da3 set in the scan's place and the diff points in
// the used points': the shader takes each point's color from --fg toward --near or --far by its gap (delta) over
// DIFF_SPAN_MM. The colors are style.css's tokens, read at each draw. It draws only on a change, never in a loop.
function cloudGL(canvas, on) {
  const gl = canvas.getContext && canvas.getContext("webgl", { antialias: true });
  if (!gl) return null;
  const prog = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, "attribute vec3 p; attribute float d; uniform mat4 M; "
    + "uniform float size, span; uniform vec3 color, lo, hi; varying vec3 c; "
    + "void main() { gl_Position = M * vec4(p, 1.0); gl_PointSize = size; "
    + "float t = clamp(d / span, -1.0, 1.0); c = mix(color, t < 0.0 ? lo : hi, abs(t)); }"],
  [gl.FRAGMENT_SHADER, "precision mediump float; varying vec3 c; uniform bool dot; "
    + "void main() { if (dot && length(gl_PointCoord - 0.5) > 0.5) discard; gl_FragColor = vec4(c, 1.0); }"]]) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    gl.attachShader(prog, sh);
  }
  gl.bindAttribLocation(prog, 0, "p");  // the attribute that is always on
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
  const loc = Object.fromEntries(["M", "size", "span", "color", "lo", "hi", "dot"]
    .map((u) => [u, gl.getUniformLocation(prog, u)]));
  const attr = gl.getAttribLocation(prog, "p"), gap = gl.getAttribLocation(prog, "d");
  const range = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) || [1, 1];
  const bufs = Object.fromEntries(["scan", "used", "da3", "diff", "delta", "cam"].map((k) => [k, gl.createBuffer()]));
  const n = { scan: 0, used: 0, da3: 0, diff: 0, delta: 0, cam: 0 };
  let start = null, view = null, frame = 0, dead = false, freed = false, sized = false, frameAspect = 4 / 3;
  let comparing = false;
  const put = (k, a) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, bufs[k]);
    gl.bufferData(gl.ARRAY_BUFFER, a, gl.STATIC_DRAW);
    n[k] = a.length / 3;
  };
  const token = (name) => {  // a color token as [r, g, b] in 0..1
    const m = /^#([0-9a-f]{6})$/i.exec(getComputedStyle(document.documentElement).getPropertyValue(name).trim());
    return m ? [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16) / 255) : [0.5, 0.5, 0.5];
  };
  const render = () => {
    frame = 0;
    if (dead) return;
    const bg = token("--viewer-bg"), dpr = window.devicePixelRatio || 1;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!view) return;
    const { target: t, dist: d, yaw, pitch } = view, cp = Math.cos(pitch);
    const eye = [t[0] + d * cp * Math.sin(yaw), t[1] + d * Math.sin(pitch), t[2] + d * cp * Math.cos(yaw)];
    gl.useProgram(prog);
    gl.uniform1f(loc.span, DIFF_SPAN_MM / 1000);
    const aspect = canvas.width / Math.max(1, canvas.height);
    gl.uniformMatrix4fv(loc.M, false, matmul(perspective(fieldOf(frameAspect, aspect), aspect, 0.02 * d, 200 * d),
      lookAt(eye, t)));
    gl.enableVertexAttribArray(attr);
    const pass = (k, mode, px, color) => {
      if (!n[k]) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, bufs[k]);
      gl.vertexAttribPointer(attr, 3, gl.FLOAT, false, 0, 0);
      gl.uniform1f(loc.size, Math.min(Math.max(px * dpr, range[0]), range[1]));
      gl.uniform3fv(loc.color, token(color));
      gl.uniform1i(loc.dot, mode === gl.POINTS ? 1 : 0);
      gl.drawArrays(mode, 0, n[k]);
    };
    const both = comparing && n.diff > 0;  // a view without a diff set shows its scan and used points
    gl.enable(gl.DEPTH_TEST);
    pass(both ? "da3" : "scan", gl.POINTS, 2, "--line-strong");
    pass("cam", gl.LINES, 1, "--fg");
    gl.disable(gl.DEPTH_TEST);
    if (!both) {
      pass("used", gl.POINTS, 4, "--accent");
      return;
    }
    gl.uniform3fv(loc.lo, token("--near"));
    gl.uniform3fv(loc.hi, token("--far"));
    gl.bindBuffer(gl.ARRAY_BUFFER, bufs.delta);
    gl.enableVertexAttribArray(gap);
    gl.vertexAttribPointer(gap, 1, gl.FLOAT, false, 0, 0);
    pass("diff", gl.POINTS, 4, "--fg");
    gl.disableVertexAttribArray(gap);  // the other sets have no gaps: without an array, d is 0
  };
  const draw = () => {
    if (dead || frame) return;
    if (typeof requestAnimationFrame === "function") frame = requestAnimationFrame(render);
    else render();
  };
  const resize = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
    if (dead || !(w > 0 && h > 0)) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    draw();
    if (!sized) {
      sized = true;
      on.sized();
    }
  };
  const ro = typeof ResizeObserver === "function" ? new ResizeObserver(resize) : null;
  if (ro) ro.observe(canvas);
  else {
    window.addEventListener("resize", resize);
    setTimeout(resize);
  }
  const dark = typeof matchMedia === "function" ? matchMedia("(prefers-color-scheme: dark)") : null;
  if (dark && dark.addEventListener) dark.addEventListener("change", draw);
  // Drag turns, two fingers pinch, the wheel zooms; on the focused canvas the arrow keys turn and + and - zoom.
  const turn = (dyaw, dpitch) => {
    if (!view) return;
    view.yaw += dyaw;
    view.pitch = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, view.pitch + dpitch));
    draw();
  };
  const zoom = (k) => {
    if (!view) return;
    view.dist = Math.max(0.02, Math.min(200, view.dist * k));
    draw();
  };
  const down = new Map();  // the pointers down: id -> {x, y}
  const spread = () => { const [a, b] = down.values(); return Math.hypot(a.x - b.x, a.y - b.y); };
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    down.set(e.pointerId, { x: e.clientX, y: e.clientY });
  });
  canvas.addEventListener("pointermove", (e) => {
    const p = down.get(e.pointerId);
    if (!p) return;
    const before = down.size === 2 ? spread() : 0, dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    if (down.size === 1) turn(-0.01 * dx, 0.01 * dy);
    else if (down.size === 2 && before > 0 && spread() > 0) zoom(before / spread());
  });
  for (const type of ["pointerup", "pointercancel"]) canvas.addEventListener(type, (e) => down.delete(e.pointerId));
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoom(Math.exp(0.001 * e.deltaY * (e.deltaMode === 1 ? 16 : 1)));
  }, { passive: false });
  const KEYS = { ArrowLeft: [5, 0], ArrowRight: [-5, 0], ArrowUp: [0, -5], ArrowDown: [0, 5] };  // degrees
  canvas.addEventListener("keydown", (e) => {
    if (KEYS[e.key]) turn(KEYS[e.key][0] * Math.PI / 180, KEYS[e.key][1] * Math.PI / 180);
    else if (e.key === "+" || e.key === "=") zoom(1 / 1.2);
    else if (e.key === "-" || e.key === "_") zoom(1.2);
    else return;
    e.preventDefault();
  });
  canvas.addEventListener("webglcontextlost", () => {
    if (dead) return;
    dead = true;
    on.lost();
  });
  return {
    show(data, aspect) {
      if (dead) return;
      for (const k of ["scan", "used", "da3", "diff", "delta"]) put(k, data[k]);
      start = orbitStart(data.scan, data.used);
      frameAspect = aspect;
      put("cam", pyramid(0.1 * start.dist, aspect));
      view = { ...start };
      draw();
    },
    reset() {
      if (!start) return;
      view = { ...start };
      draw();
    },
    compare(yes) {  // keeps the view where it is
      comparing = yes;
      draw();
    },
    draw,
    free() {
      if (freed) return;
      freed = dead = true;
      if (frame && typeof cancelAnimationFrame === "function") cancelAnimationFrame(frame);
      if (ro) ro.disconnect();
      else window.removeEventListener("resize", resize);
      if (dark && dark.removeEventListener) dark.removeEventListener("change", draw);
      Object.values(bufs).forEach((b) => gl.deleteBuffer(b));
      gl.deleteProgram(prog);
      const lose = gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();
    },
  };
}
// The LiDAR points of the views (usedClouds or poolClouds; a job with LiDAR), under the views used: the view picker,
// Reset, Compare with DA3 once a view read has a diff set, the canvas and a legend with the counts and the view's scale
// from pool/lidar.json. A view's file is fetched once, and not before the canvas has a size, so a hidden tab loads
// nothing. gone: renderInput's guard against another page or selection that came meanwhile.
async function renderCloud(job, shown, gone) {
  freeCloud();
  const box = $("#inputlidar");
  const bar = (...kids) => h("div", { class: "cloudbar" }, h("h4", {}, "LiDAR points"), kids);
  if (!shown.length) {
    fill(box, bar(), h("p", { class: "hint" }, NO_CLOUD));
    return;
  }
  let rec = null;
  if (job.files.includes("pool/lidar.json")) {
    rec = await api(file("pool/lidar.json")).then(lidarCompact, () => null);  // without it, the legend has no scale
    if (gone()) return;
  }
  const first = shown[0].id;  // view 0 when it has points
  // The canvas takes the keys (application), and its legend describes it.
  const C = cloudNow = { job, shown: new Map(shown.map((x) => [x.id, x])), at: first, data: new Map(), sized: false,
    gl: null, onCanvas: false, compare: false, now: null,
    scale: new Map(((rec && rec.views) || []).map((v) => [v.name, v])), body: h("div", {}),
    canvas: h("canvas", { class: "cloud", role: "application", "aria-roledescription": "3D view", tabindex: "0",
      "aria-describedby": "cloudlegend" }), legend: h("p", { class: "legend", id: "cloudlegend" }) };
  const pick = h("select", { onchange: (e) => cloudShow(C, e.target.value) },
    shown.map((x) => h("option", { value: x.id, selected: x.id === first }, x.label)));
  C.box = h("label", {}, h("input", { type: "checkbox", onchange: (e) => cloudCompare(C, e.target.checked) }),
    "Compare with DA3");
  C.box.hidden = true;
  fill(box, bar(h("label", {}, "View ", pick), h("button", { type: "button", class: "small secondary",
    onclick: () => { if (C.gl) C.gl.reset(); } }, "Reset"), C.box), C.body);
  fill(C.body, C.canvas, C.legend);
  C.onCanvas = true;
  C.gl = cloudGL(C.canvas, {
    sized: () => { C.sized = true; cloudShow(C, C.at); },
    lost: () => { const gl = C.gl; C.gl = null; if (gl) gl.free(); cloudShow(C, C.at); },
  });
  await cloudShow(C, first);
}
// View i in the viewer: its points once read, with "loading" in the legend until then; without WebGL, or for a file
// that does not parse, a line that offers the file instead.
async function cloudShow(C, i) {
  C.at = i;
  C.now = null;
  const { path: p, save: name, label } = C.shown.get(i);
  const offer = () => {
    C.onCanvas = false;
    fill(C.body, h("p", { class: "msg" }, NO_GL, h("a", { href: file(p, name), download: name }, "download them"), "."));
  };
  if (!C.gl) return offer();
  C.canvas.setAttribute("aria-label", "LiDAR points of " + label);
  if (!C.onCanvas) {
    C.onCanvas = true;
    fill(C.body, C.canvas, C.legend);
  }
  cloudLegend(C, null);
  if (!C.sized) return;
  if (!C.data.has(i)) {
    C.data.set(i, T.fetch(file(p)).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.arrayBuffer();
    }).then(readCloud));
  }
  let data = null;
  try { data = await C.data.get(i); } catch (e) { C.data.delete(i); }  // fetched again when the view is picked again
  if (cloudNow !== C || C.at !== i || !C.gl) return;
  if (!data) return offer();
  const pv = C.job.preview;
  C.gl.show(data, pv && pv.width > 0 && pv.height > 0 ? pv.width / pv.height : 4 / 3);
  if (data.diff.length) C.box.hidden = false;
  cloudLegend(C, C.now = data);
}
function cloudCompare(C, on) {  // the box stays on across views
  C.compare = on;
  if (C.gl) C.gl.compare(on);
  cloudLegend(C, C.now);
}
// The legend: each set's square, name and count. The used points' count comes with the view's scale and status as the
// review gives them (lidarViewText: "×1.21 (9 pts, outlier)"), when pool/lidar.json has the view. While the comparison
// shows: the DA3 surface (a Stray capture has none), the colors and their fixed scale, the scale used and the used
// points' median. With the box on, a view without a diff set says so. "loading" until the file is read.
function cloudLegend(C, data) {
  if (!data) {
    fill(C.legend, "loading");
    return;
  }
  const key = (set, text) => h("span", { class: "key" }, h("span", { class: "sq " + set, "aria-hidden": "true" }), text);
  if (C.compare && data.diff.length) {
    const { s, median, used } = data.compare, mm = Math.round(1e4 * median) / 10;  // rounded first: never "-0.0"
    const mid = used ? `Median of the ${count(used)} used points: ${mm >= 0 ? "+" : ""}${mm.toFixed(1)} mm` : "No used points";
    fill(C.legend, data.da3.length ? key("da3", `DA3 surface (${count(data.da3.length / 3)})`) : null, " ",
      h("span", { class: "key long" }, h("span", { class: "sq diff", "aria-hidden": "true" }),
        `Blue: LiDAR nearer than DA3. Red: LiDAR farther. Full color at ±${DIFF_SPAN_MM} mm`), " ",
      h("span", { class: "key" }, `Scale ×${s.toFixed(3)}`), " ", h("span", { class: "key" }, mid));
    return;
  }
  const v = C.scale.get(C.shown.get(C.at).name);
  fill(C.legend, key("scan", `Whole scan (${count(data.scan.length / 3)})`), " ",
    key("used", "Used for the scale" + (v ? ": " + lidarViewText(v) : ` (${count(data.used.length / 3)})`)),
    C.compare ? [" ", h("span", { class: "key long" }, NO_COMPARE)] : null);
}

// ---- 5d. the videos of the object (Results tab) ----

// The videos above the viewers, side by side, each as [its slot in V, its file, its stage, its caption, its name for
// a screen reader]: the turntable stage's 360 video, when the job asked for one and its render worked, and the
// softvideo stage's video of the soft part pressed and let go, when the MuJoCo model has one. A slot holds the video,
// the key of its draw, its figure and whether it plays when the tab shows (V.spin, V.spinKey, V.spinFig, V.spinOn).
// A job stored on the site before the softvideo stage has no record of it.
const SPIN = "turntable/turntable.mp4", SOFT = "softvideo/softvideo.mp4";
const VIDEOS = [["spin", SPIN, "turntable", "360 video", (what) => `360 video of the ${what}`],
  ["soft", SOFT, "softvideo", "Soft-part video", (what) => `The soft part of the ${what}, pressed and let go`]];
function clearSpin() {  // another job opens, or a new build hides the results: the old videos go
  for (const [slot] of VIDEOS) V[slot] = V[slot + "Key"] = V[slot + "Fig"] = null;
  fill($("#spin"));
}
// One video: muted, looping, with controls that pause it; it does not start by itself when the system asks for reduced
// motion. It is drawn once per file, so a refresh or a new signed URL does not start it again. When it cannot play,
// the caption stays and a line offers the file, as for a viewer. Returns true when its figure changed.
function drawVideo(job, [slot, p, stage, title, label]) {
  if (!job.files.includes(p)) {
    const had = !!V[slot + "Fig"];
    V[slot] = V[slot + "Key"] = V[slot + "Fig"] = null;
    return had;
  }
  const key = JSON.stringify([job.id, (job.stages[stage] || {}).ended, (job.sizes || {})[p]]);
  if (key === V[slot + "Key"]) return false;
  V[slot + "Key"] = key;
  V[slot + "On"] = !reducedMotion();
  const name = `${job.id}_${p.split("/").pop()}`, cap = h("figcaption", {}, title);
  const fig = V[slot + "Fig"] = h("figure", { class: "spinfig" }, cap);
  const video = V[slot] = h("video", { src: file(p), autoplay: V[slot + "On"], muted: true, loop: true, playsinline: true,
    controls: true, preload: "auto", "aria-label": label(job.prompt || "object"),
    onerror: () => {
      if (V[slot] === video) V[slot] = null;
      fig.replaceChildren(cap, h("p", { class: "msg" }, "Could not play the video: ",
        h("a", { href: file(p, name), download: name }, "download it"), "."));
    } });
  video.muted = true;  // the attribute only sets the default
  if (V.tab && V.tab !== "results") video.pause();  // drawn while another tab shows: no start there
  fig.append(video);
  return true;
}
function renderSpin(job) {
  const changed = VIDEOS.map((v) => drawVideo(job, v));  // each one, so that neither misses its draw
  if (changed.some(Boolean)) fill($("#spin"), VIDEOS.map(([slot]) => V[slot + "Fig"]));
}
// The videos play only while the Results tab shows. Each stops when the tab hides, and plays again when the tab shows
// if it played when the tab hid (V.spinOn, V.softOn), or at first unless the system asks for reduced motion.
function spinTab(name) {
  if (name === V.tab) return;
  for (const [slot] of VIDEOS) {
    const v = V[slot];
    if (!v) continue;
    if (name === "results") {
      if (V[slot + "On"]) Promise.resolve(v.play()).catch(() => {});  // a browser may refuse it: the controls still start it
    } else {
      if (V.tab === "results") V[slot + "On"] = !v.paused;
      v.pause();
    }
  }
}

route();
