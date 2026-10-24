// HARPOR web app UI: jobs list and upload; a job page with the panels that wait for the user (camera topic, review of
// the views and of the part list, a LiDAR failure) above five tabs: Results (the viewers and the facts), Downloads (the
// files, by group), Inputs (the input's facts, its preview video and the views the build used), Settings (the build
// settings, with LiDAR for the metric scale) and Stages and log.
// Plain DOM code, no build step. The API key is only read from the password fields and sent once.
"use strict";

const $ = (sel) => document.querySelector(sel);
const ACTIVE = ["analyzing", "queued", "running"];
const BUILD = ["extract", "masks", "prep", "da3", "lidar", "select", "sam3d", "place", "propose", "parts", "agent", "align", "vomp",
  "mjcf", "usd", "urdf", "package"];
// job.json "agent" (steps.AGENTS): the key it takes and the placeholder of the key field.
const AGENTS = { astra: { key: "OpenAI API key", placeholder: "sk-..." },
  opus: { key: "Anthropic API key", placeholder: "sk-ant-..." }, none: { key: null, placeholder: "" } };
const RAN = { codex: "GPT-6-Astra", claude: "Claude Opus 5.5" };  // agent_result.agent, the controller's name
const VLM = { astra: "GPT-6-Astra", opus: "Claude Opus 5.5" };  // proposal.json "agent": the model that listed the parts
const agentOf = (a) => a === false ? "none" : a === true || a == null ? "astra" : a;  // older jobs hold true or false
// What each stage does, shown muted next to its name.
const STAGE_LABEL = { topics: "camera topics", frames: "frame analysis", extract: "extract frames", masks: "SAM 3 masks",
  prep: "prepare views", da3: "depth (Depth Anything 3)", lidar: "metric scale (LiDAR)", select: "view selection",
  sam3d: "mesh (MV-SAM3D)", place: "pose and scale", propose: "part list", parts: "split into parts", agent: "agent refinement", align: "align upright", vomp: "materials (VoMP)",
  mjcf: "MuJoCo model", usd: "USD", urdf: "URDF", package: "results.zip" };
// The agent's time budget (agent_step.BUDGET_S): it sets the joints of an articulated object in 900 s.
const AGENT_BUDGET_S = { refinement: 600, articulation: 900 };
// A stage's label; the agent's names its task and, unless the stage is skipped, its time budget.
const stageLabel = (name, articulated, skipped) => name !== "agent" ? STAGE_LABEL[name] || ""
  : (articulated ? "agent sets the joints" : STAGE_LABEL.agent) +
    (skipped ? "" : `, ${AGENT_BUDGET_S[articulated ? "articulation" : "refinement"] / 60} min budget`);
const PARTS_MODES = ["vlm", "vlm_edit", "user"];  // steps.PARTS_MODES
const JOINTS = ["revolute", "prismatic", "fixed", "unknown"];  // gator.parts.propose.JOINTS
// agent_result.verdict of an articulation (the agent sets joints); a refinement's verdicts show as they are.
const VERDICT = { articulated: "articulated: the parts have joints", unchanged: "unchanged: the parts have no joints",
  needs_review: "needs review" };
const LIST_STATUS = { review: "needs review", ready: "ready to start", choose_topic: "choose topic" };  // badge text in the jobs list
const NEXT = { choose_topic: "Pick the camera topic below.", ready: "Name the object, check the settings, then Start.",
  analyzing: "Reading the input. This takes a few seconds to a minute.",
  queued: "The job waits for the GPU. You can close this page; the job keeps going.",
  running: "This runs on the GPU node. You can close this page; the job keeps going.",
  review: "Check the picked views, then Build.",
  done: "Look at the meshes in the Results tab.",  // the Results tab's first line names the Downloads tab
  failed: "Read the log of the failed stage, then Retry.", cancelled: "Retry continues from the first stage that is not done.",
  parts: "Check the part list, edit it if it is wrong, then Build with these parts.",  // a review of the part list
  // The lidar stage failed (a rosbag); ... that has a calibration.
  lidar: "Upload a calibration in the Settings tab, then Retry. Or turn LiDAR off there and build again.",
  lidar_cal: "Replace the calibration in the Settings tab, then Retry. Or turn LiDAR off there and build again.",
  // A Stray capture takes no calibration, and a retry reads the same depth and masks again.
  lidar_stray: "Turn LiDAR off or change the views in the Settings tab, then build again. Retry helps only after a file " +
    "or disk error.",
  // A failed or cancelled job whose scale ran with another calibration (V.scaleStale).
  stale: "The calibration changed since the scale was found. Retry starts again from the lidar stage." };
const KIND_LABEL = { video: "video", rosbag: "rosbag", stray: "Stray capture" };  // job.json source kind, in the jobs list
const OFFLINE = "Cannot reach the server. Is the SSH tunnel up?";
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
// What the Settings and Downloads tabs and the views strip of the Inputs tab say while they have nothing to show.
const SETTINGS_LATER = "The settings appear here after the app reads the frames.";
const SETTINGS_BUSY = "You can change the settings when the job stops.";
const FILES_LATER = "The files appear here when the build has made them.";
const FILES_BUSY = "The files appear here when the build stops.";  // a build replaces the files as it runs
// What the Downloads tab says above its files when results.zip is not one of them: before the build (in review), and
// after a build that stopped.
const FILES_PARTIAL = "results.zip and the other files appear here when the build has made them.";
const FILES_STOPPED = "The build stopped before it made results.zip, so the tab lists only the files that it made.";
const RESULTS_UNREAD = "Could not read the results: ";  // a result file that does not parse (drawJob)
const NO_VIEWS = "The views used appear here after the app selects them.";
// The open job page: id, job, meta, picks, view0, sel, ov (pending overrides), es, state, scaleStale (the LiDAR scale
// ran with another calibration); tab (the tab shown), pick (the tab the user picked), tabPhase (the phase of the status
// that the tab follows) and hashTab (the hash's tab when the page opened); logAtEnd and logTop (where the log was
// left); and input, inputKey, videoKey, video, selected, inputPlaced, inputAsked, placeAsked and overlay (the Inputs
// tab: renderInput).
let V = {};
let listKey = null, upXhr = null;  // the jobs list as last drawn; the running upload

function h(tag, attrs, ...kids) {
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
function s(tag, attrs) {  // an SVG element
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  return e;
}
const fill = (el, ...kids) => el.replaceChildren(...kids.flat().filter((x) => x != null && x !== false));  // skips empty slots
const badge = (t, text) => h("span", { class: "badge " + t }, text || t.replace("_", " "));
const fmt = (sec) => sec < 60 ? sec.toFixed(1) + " s" : sec < 3600 ? Math.floor(sec / 60) + " min " + Math.floor(sec % 60) + " s"
  : Math.floor(sec / 3600) + " h " + Math.floor((sec % 3600) / 60) + " min";
const file = (p) => `/api/jobs/${V.id}/file/${p}`;
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
// [{name, parent, joint, count}] (parent and joint null when not given). gator.parts.propose.validate checks
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
  try { r = await fetch(path, opts); } catch (e) { online(false); throw new Error(OFFLINE); }
  online(true);
  const j = await r.json().catch(() => ({}));
  if (r.status === 401) throw new Error("not logged in: open the URL with ?token= that the server printed");
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
  tipClose();  // an open info note of the Results or Downloads tab goes with its page
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
      h("td", { colspan: 6 }, "No jobs yet. Upload a video, a rosbag or a Stray Scanner export of one object to start.")));
    return;
  }
  $("#jobs tbody").replaceChildren(...(rows || []).map((j) => h("tr", { class: j.status in LIST_STATUS || j.status === "failed" ? "act" : null },
    h("td", { class: "thumb" }, h("img", { src: `/api/jobs/${j.id}/thumb/0`, alt: "", loading: "lazy",
      onerror: (e) => e.target.replaceWith(h("span", { class: "noimg" })) })),
    h("td", { class: "input" }, h("a", { href: "#/job/" + j.id }, j.video),
      j.kind ? h("div", { class: "hint" }, KIND_LABEL[j.kind] || j.kind) : null),
    h("td", { class: "obj" }, j.prompt || ""),
    h("td", { class: "when", title: j.created.replace("T", " ") }, shortTime(j.created)),
    h("td", { class: "stat" }, badge(j.status, LIST_STATUS[j.status])),
    h("td", { class: "acts" }, ACTIVE.includes(j.status) || j.status === "uploading" ? ""
      : h("button", { class: "small danger", type: "button", "aria-label": "Delete " + j.video, onclick: () => remove(j) }, "Delete")))));
}
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
  if (!confirm(`Delete the job ${j.video}${j.prompt ? " (" + j.prompt + ")" : ""} from ${shortTime(j.created)} and all its files?`)) return;
  try { await api("/api/jobs/" + j.id, { method: "DELETE" }); } catch (e) { say($("#upmsg"), e.message); }
  showList();
}
const KINDS = /\.(mp4|mov|m4v|avi|mkv|webm|bag|mcap|db3|zip)$/i;  // jobs.VIDEOS, jobs.BAGS and jobs.STRAY
function upload(f) {
  if (!f || upXhr) return;
  if (!KINDS.test(f.name)) {
    say($("#upmsg"), `${f.name} is not a video (.mp4 .mov .m4v .avi .mkv .webm), a rosbag (.bag .mcap .db3) or a Stray ` +
      "Scanner export (.zip). For a ROS 2 bag folder, upload the .mcap or .db3 file in it. For a Stray Scanner " +
      "export folder, zip the folder first.");
    return;
  }
  const xhr = upXhr = new XMLHttpRequest(), prog = $("#upprog"), t0 = Date.now();
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
    $("#uprate").textContent = `${size(e.loaded)} of ${size(e.total)}, ${Math.round(100 * e.loaded / e.total)}%, ` +
      `${(rate / 1e6).toFixed(1)} MB/s, about ${fmt(Math.round(left))} left`;  // not a live region: it changes many times a second
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
  show("job");
  V = { id, picks: [], view0: null, hashTab: tab || null };
  // The last job's, until refresh draws this one's: the waiting panels, the tabs and what the tabs show.
  for (const x of ["review", "partsreview", "lidarfail", "topics", "tabs", ...Object.values(TABS), "resultsnone", "resultsbody",
    "downloadsnone", "downloadsbody", "inputnone", "inputbody", "framesnone", "framesbody"]) $("#" + x).hidden = true;
  clearInput();
  $("#log").textContent = "";
  $("#logname").textContent = "";
  $("#rawlog").hidden = true;
  await refresh();
}
async function refresh() {
  const job = V.job = await api(`/api/jobs/${V.id}`);
  $("#jtitle").textContent = `${job.video.name}${job.prompt ? " - " + job.prompt : ""}`;
  $("#jid").textContent = "job " + job.id;
  if (ACTIVE.includes(job.status)) openStream();
  // A scale found with another calibration: Retry then starts from lidar (retryFrom).
  V.scaleStale = ["failed", "cancelled"].includes(job.status) ? await scaleStale(job).catch(() => false) : false;
  renderState({ status: job.status, current: job.current, error: job.error, warnings: job.warnings, key_ok: job.key_ok,
    agent: job.agent, articulated: job.articulated, parts_mode: job.parts_mode, parts_review: job.parts_review,
    stages: job.progress }, false);
  // The tab follows the status at the end, once its panel is drawn; also when a panel could not be drawn.
  try { await drawJob(job); } finally { if (V.job === job) followStatus(job.status); }
}
async function drawJob(job) {  // refresh's panels: the waiting panels and what the tabs show
  const analyzed = job.stages.frames.status === "done", rosbag = job.source && job.source.kind === "rosbag";
  // A rosbag job chooses its camera topic when it has several, after a failed frames run, or on "Change the camera topic".
  const idle = !ACTIVE.includes(job.status) && !["review", "done"].includes(job.status);
  $("#topics").hidden = !rosbag || !job.stages.topics || job.stages.topics.status !== "done" || !(job.status === "choose_topic" ||
    (idle && (V.pickTopic || (["failed", "cancelled"].includes(job.status) && !analyzed))));
  if (!$("#topics").hidden) await loadTopics(job).catch((e) => { say($("#topicmsg"), e.message); });
  $("#tkeep").hidden = !(V.pickTopic && analyzed);
  // "Change the camera topic", in the Inputs tab (whose label names the topic), while the table does not show.
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
  // The lidar stage failed: its record (pool/lidar.json, ok false) with the reason and the check images.
  $("#lidarfail").hidden = !lidarFailed(job.status, job.progress);
  if (!$("#lidarfail").hidden) await renderLidarFail(job).catch((e) => fill($("#lidarfailbody"), h("p", { class: "msg" }, e.message)));
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
  else tipClose();  // its button hides with the old results
  $("#resultsbody").hidden = !results;
  $("#resultsnone").hidden = results;
  // The input's preview video, and the views used once they are selected (in review, during the build and after).
  V.inputAsked = false;
  V.selected = (job.stages.select || {}).status === "done";
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
  const again = job.status !== "ready";
  $("#framesum").textContent = !again ? "Frames and build settings" : job.status === "review" ? "Start over with other settings"
    : "Build again with other settings";
  $("#start").textContent = !again ? "Start" : job.status === "review" ? "Start over" : "Build again";
  $("#rebuildhint").hidden = !again;
  rebuildHint();
  await loadFrames();
}

// ---- the tabs (the WAI-ARIA tabs pattern) ----

// Show the tab `name` and hide the other panels. pick: the user chose it, so the hash keeps it (a reload shows it
// again) until the job's status changes. When the focus was in the panel that hides, or on a tab that is no longer
// selected (the status changed), the new tab takes it. The preview video stops when its tab hides.
function selectTab(name, pick) {
  const old = V.tab && V.tab !== name ? $("#" + TABS[V.tab]) : null, had = document.activeElement, tab = $("#tab-" + name);
  const lost = !!had && had !== tab && ((!!old && old.contains(had) === true) || $("#tabs").contains(had) === true);
  if (V.tab === "inputs" && name !== "inputs" && V.video) V.video.pause();
  if (V.tab !== name) tipClose();  // an open info note of the Results or Downloads tab
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
// "Download the files from the Downloads tab." at the top of the Results tab: the tab name opens that tab. The button
// hides with its panel, so the Downloads tab takes the focus (selectTab). selectTab does not scroll, so the tab comes
// into view here, below the sticky header (scroll-padding-top in style.css).
$("#todlbtn").onclick = () => {
  selectTab("downloads", true);
  $("#tab-downloads").scrollIntoView({ block: "nearest" });
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
function rebuildHint() {  // under "Build again": what Retry, at the top, does instead
  const from = $("#retry").hidden || !V.state ? null : retryFrom(V.state);
  $("#rebuildhint").textContent = "This runs every stage again with these settings." + (from === null ? ""
    : V.scaleStale === true && from === "lidar" ? " Retry, at the top, starts from the lidar stage instead, because the " +
      "calibration changed since the scale was found." : " Retry, at the top, continues from the stage that stopped instead.");
}

// ---- the camera topic of a rosbag (frames/topics.json) ----

const RAW_HINT = "MV-SAM3D expects undistorted images; pick a rectified topic if the bag has one";
async function loadTopics(job) {
  const r = await api(file("frames/topics.json")), cur = job.source.topic;
  // A listing from before LiDAR topics were listed (no "lidar"): choosing the current topic again lists them.
  const inUse = job.stages.frames.status === "done" && Array.isArray(r.lidar);
  const lens = (t) => t.rectified === true ? badge("rectified") : t.rectified === false
    ? [badge("raw", "raw lens distortion"), h("div", { class: "hint" }, RAW_HINT)]
    : [badge("unknown"), h("div", { class: "hint" }, "no camera_info next to it, so the lens distortion is not known. " + RAW_HINT)];
  $("#ttable tbody").replaceChildren(...r.topics.map((t) => h("tr", { class: t.topic === cur ? "current" : null, "aria-current": t.topic === cur ? "true" : null },
    h("td", {}, t.preview ? h("img", { src: file(t.preview), alt: "", loading: "lazy" }) : ""),
    h("td", { class: "topic" }, h("code", {}, t.topic.split("/").flatMap((x, i) => i > 1 ? [h("wbr"), "/" + x] : i ? ["/" + x] : [x]))),
    h("td", { "data-label": "Type" }, `${t.type} ${t.format || ""}`), h("td", { "data-label": "Messages" }, String(t.count)),
    h("td", { "data-label": "Rate" }, t.rate_hz != null ? t.rate_hz.toFixed(1) + " Hz" : "?"),
    h("td", { "data-label": "Size" }, t.width ? `${t.width}x${t.height}` : "?"),
    h("td", { class: "lens" }, lens(t)),
    h("td", {}, t.topic === cur && inUse ? h("span", { class: "hint" }, "in use")
      : h("button", { class: "small", type: "button", onclick: (e) => useTopic(t.topic, e.target) }, "Use this topic")))));
  $("#tskipped").textContent = r.skipped.length ? "Not usable: " + r.skipped.map((x) => `${x.topic} (${x.why})`).join(", ") : "";
}
async function useTopic(topic, btn) {
  await act([btn], $("#topicmsg"), "reading the frames of " + topic, () => post(`/api/jobs/${V.id}/topic`, { topic }), () => {
    V.pickTopic = false;
    V.meta = null;
    $("#log").textContent = "";
  });
}
const smooth = () => matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
$("#chtopic").onclick = async () => {  // the topic table, above the tabs, takes the focus
  V.pickTopic = true;
  await refresh().catch((e) => say($("#topicmsg"), e.message));
  if ($("#topics").hidden) return;
  $("#topicshead").focus({ preventScroll: true });
  $("#topics").scrollIntoView({ block: "start", behavior: smooth() });
};
$("#tkeep").onclick = async () => {  // back to "Change the camera topic", in the Inputs tab
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
  const cur = st.current, curText = cur ? cur + (stageLabel(cur, art) ? " (" + stageLabel(cur, art) + ")" : "") : "";
  const sum = { analyzing: cur === "topics" ? "Listing the camera topics of the bag" : "Reading the frames",
    choose_topic: "Choose the camera topic of the bag", ready: "Ready: choose the frames and start", queued: "Waiting for the GPU",
    running: "Running " + curText, review: partList ? "Waiting for you to check the part list" : "Waiting for your review",
    done: "Done", failed: "Failed", cancelled: "Cancelled", uploaded: "Uploaded" }[st.status] || st.status;
  const overall = ["queued", "running"].includes(st.status) ? h("div", { class: "overall" },
    h("progress", { max: build.length, value: nDone, "aria-label": "build progress" }),
    h("span", { class: "hint" }, `${nDone} of ${build.length} build stages done`)) : null;
  fill($("#jstatus"), badge(st.status), " ", sum, overall, st.error ? h("div", { class: "msg errbox" }, st.error) : null);
  const from = retryFrom(st);
  let next = ["failed", "cancelled"].includes(st.status) && from === null ? "Build again from the Settings tab."
    : partList ? NEXT.parts : lidarFailed(st.status, st.stages)
      ? NEXT[!V.job ? "lidar" : V.job.kind === "stray" ? "lidar_stray" : V.job.calibration ? "lidar_cal" : "lidar"]
      : V.scaleStale === true && from === "lidar" && ["failed", "cancelled"].includes(st.status) ? NEXT.stale
        : NEXT[st.status] || "";
  const ran = (name) => build.some((x) => x.name === name && x.status === "done");
  if (["queued", "running"].includes(st.status) && V.job && V.job.auto_continue === false && !ran("select"))
    next += " It stops after the view selection for your review.";
  if (["queued", "running"].includes(st.status) && art && st.parts_mode === "vlm_edit" && !ran("propose"))
    next += " It stops after the VLM lists the parts, so you can check the list.";
  $("#jnext").textContent = next;
  const said = sum + (st.error ? ". " + st.error : "");
  if (said !== V.said) { V.said = said; $("#announce").textContent = said; }
  $("#jwarn").replaceChildren(...(st.warnings || []).map((w) => h("li", {}, w)));
  const name = V.job ? V.job.video.name : V.id || "";
  $("#where").replaceChildren(h("span", { class: "crumb" }, "/ " + name), badge(st.status));
  document.title = ({ running: `${cur || "running"} ${Math.min(nDone + 1, build.length)}/${build.length}`, queued: "Queued",
    review: partList ? "Part list waiting" : "Review waiting", done: "Done", failed: "Failed", cancelled: "Cancelled", ready: "Ready", analyzing: "Reading",
    choose_topic: "Choose topic" }[st.status] || st.status) + (name ? " · " + name : "") + " · HARPOR";

  const started = st.stages.some((x) => x.started || x.status !== "pending");
  $("#runbody").hidden = !started;
  $("#runnone").hidden = started;
  const nFailed = st.stages.filter((x) => x.status === "failed").length;
  $("#runsum").textContent = `${nDone} of ${build.length} build stages done` + (nFailed ? `, ${nFailed} failed` : "") +
    (st.status === "running" && cur ? `, running ${cur}` : "");
  const now = Date.now() / 1000, live = !!V.es;
  const focused = document.activeElement && document.activeElement.dataset && document.activeElement.dataset.stage;
  $("#stages").replaceChildren(...st.stages.map((x) => h("li", { class: x.status },
    h("div", { class: "nm" }, live ? h("span", { class: "name" }, x.name)
      : h("button", { type: "button", class: "name link", "data-stage": x.name, "aria-controls": "log", "aria-pressed": String(x.name === V.logStage),
        onclick: () => showLog(x.name) }, x.name),
    h("span", { class: "label" }, stageLabel(x.name, art, x.status === "skipped"))),
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

  $("#cancel").hidden = !(ACTIVE.includes(st.status) || st.status === "review");
  $("#retry").hidden = !["failed", "cancelled"].includes(st.status) || from === null;
  if (!$("#retry").hidden) $("#retry").textContent = from === "frames" ? "Retry: read the input again"
    : from === "propose" && (st.stages.find((y) => y.name === "propose") || {}).status === "done" ? "Retry: check the part list"
      : "Retry from " + from;
  const agent = agentOf(st.agent);
  $("#keyrow").hidden = st.key_ok || agent === "none" || !["review", "failed", "cancelled"].includes(st.status) || from === "frames";
  $("#key2name").textContent = (AGENTS[agent] || AGENTS.astra).key || "API key";
  // The key goes next to the button that sends it.
  const home = st.status === "review" ? partList ? $("#partsbar") : $("#revbar") : $("#actions");
  if ($("#keyrow").parentElement !== home) home.prepend($("#keyrow"));
  if (follow !== false) followStatus(st.status);
}
function retryFrom(st) {  // "frames" analyzes the input again (no key needed); else the first build stage not done
  const stage = (name) => st.stages.find((y) => y.name === name);
  if (stage("frames").status !== "done") return "frames";
  const x = st.stages.find((x) => BUILD.includes(x.name) && !["done", "skipped"].includes(x.status));
  if (!x || stage("extract").status === "pending") return null;
  // A job stopped before its part list was confirmed goes back to that list (the server resumes from propose).
  const unconfirmed = st.articulated === true && st.parts_mode === "vlm_edit" && (stage("propose") || {}).status === "done"
    && !(V.job && V.job.parts && V.job.parts.confirmed === true);
  const name = x.name === "parts" && unconfirmed ? "propose" : x.name;
  // A scale found with another calibration (V.scaleStale): the server starts from lidar (Runner.resume).
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
function showRun() {  // the Stages and log tab opens: the log where it was left, the clocks current
  const log = $("#log");
  log.scrollTop = V.logAtEnd === false ? V.logTop || 0 : log.scrollHeight;
  tick();
}
function openStream() {
  if (V.es) return;
  const id = V.id, es = V.es = new EventSource(`/api/jobs/${id}/events`);
  $("#logname").textContent = "live log (connecting)";
  V.logStage = null;
  V.logAtEnd = true;
  $("#rawlog").hidden = true;
  es.onopen = () => { if (V.id === id) $("#logname").textContent = "live log"; };
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
  es.addEventListener("end", () => { closeStream(); if (V.id === id) refresh(); });
}
function closeStream() {
  if (V.es) V.es.close();
  V.es = null;
}
async function showLog(stage) {
  if (V.es) return;  // the live log is showing
  const r = await fetch(file(`logs/${stage}.log`));
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
  if (!confirm("Cancel this job? The running stage stops and the server forgets the API key. " +
    "Retry continues from the first stage that is not done.")) return;
  act([$("#cancel"), $("#retry")], $("#runmsg"), "cancelling", () => post(`/api/jobs/${V.id}/cancel`), null, true);
};
$("#retry").onclick = () => {
  const from = retryFrom(V.state);
  let r = null;
  return act([$("#retry"), $("#cancel")], $("#runmsg"), "retrying from " + from,
    async () => { r = await post(`/api/jobs/${V.id}/continue`, { from, api_key: $("#key2").value || null }); },
    () => {
      $("#key2").value = "";
      $("#log").textContent = "";
      // The server may start earlier than the page asked: say where the run starts.
      if (r && r.from && r.from !== from) say($("#runmsg"), r.from === "lidar"
        ? "The calibration changed since the scale was found, so the run starts from lidar." : `The run starts from ${r.from}.`, true);
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
    if (job.prompt) $("#prompt").value = job.prompt;
    $("#nviews").value = $("#nval").textContent = job.n_views || 6;
    $("#pause").checked = !job.auto_continue;
    for (const r of document.querySelectorAll("input[name=agent]")) r.checked = r.value === agentOf(job.agent);
    const parts = job.parts || {};  // older jobs lack it, and "articulated"
    $("#articulated").checked = job.articulated === true;
    partsPick = PARTS_MODES.includes(parts.mode) ? parts.mode : "vlm";
    $("#partlist").value = formatParts(parts.list);
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
  $("#facts").textContent = `${v.n_frames} frames, ${v.duration_s.toFixed(1)} s, ${v.fps} fps, ${v.width}x${v.height}, ` +
    `${v.codec}${v.bit_depth ? " " + v.bit_depth + "-bit" : ""}; ${m.candidates.length} candidate views`;
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
  $("#inputwarntext").textContent = nWarn === 1 ? "The input has a warning." : `The input has ${nWarn} warnings.`;
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
  return [a == null ? "No frame shows gravity, so the frames stay as stored."
    : (v.rotation ? `The frames are turned ${v.rotation} deg counterclockwise to stand upright.`
      : "The frames stand upright as stored.") + ` Gravity gives this turn in ${Math.round(100 * a)}% of the frames that show it.`,
  `${v.n_depth} of ${v.n_frames} frames have a depth image${d.length === 2 ? ` (${d[0]}x${d[1]})` : ""}.`,
  `${v.n_confidence} have a confidence image.`].join(" ");
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
function agentToggle() {  // the key field names the chosen agent's provider; without an agent no key is asked for or sent
  const a = AGENTS[chosenAgent()];
  $("#keylabel").hidden = !a.key;
  $("#key").required = !!a.key;
  $("#keyname").textContent = a.key || "";
  $("#key").placeholder = a.placeholder;
  partsToggle();
}
for (const r of document.querySelectorAll("input[name=agent]")) r.onchange = agentToggle;
let partsPick = "vlm";  // the parts mode the user chose; "No agent" shows "I type the parts" and keeps this choice
const chosenPartsMode = () => chosenAgent() === "none" ? "user" : partsPick;  // steps.parts_mode: no agent, no VLM
function partsToggle() {  // the parts settings of an articulated object; the typed list only for "I type the parts"
  const art = $("#articulated").checked === true, none = chosenAgent() === "none", pm = chosenPartsMode();
  for (const r of document.querySelectorAll("input[name=partsmode]")) {
    r.disabled = none && r.value !== "user";
    r.checked = r.value === pm;
  }
  $("#partsbox").hidden = !art;
  $("#partsnoagent").hidden = !none;
  $("#partspause").hidden = pm !== "vlm_edit";
  $("#partlistbox").hidden = pm !== "user";
  $("#partlist").required = art && pm === "user";
}
for (const r of document.querySelectorAll("input[name=partsmode]")) r.onchange = () => { partsPick = r.value; partsToggle(); };
$("#articulated").onchange = partsToggle;

// ---- 2b. LiDAR for the metric scale: a bag's LiDAR topic with its TF link or the calibration; a Stray capture's depth ----

const LIDAR_ON = { rosbag: "The LiDAR's points set the metric scale.", stray: "The iPhone's LiDAR depth sets the metric scale." };
const LIDAR_OFF = "Off: Depth Anything 3 estimates the scale.";
// topics.json from before LiDAR topics were listed (lidar.OLD_TOPICS); POST /topic lists them, but not in review or done.
const OLD_TOPICS = {
  idle: "This bag was read before the app listed LiDAR topics. To list them, use Change the camera topic in the Inputs tab, " +
    "and choose the same topic again.",
  late: "This bag was read before the app listed LiDAR topics. To use its LiDAR, upload the bag again." };
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
// The status line of the camera topic's link to a LiDAR topic: [text, true when the user must act].
function lidarLink(found, camTopic, lidarTopic, cal) {
  if (!lidarTopic) return ["choose a LiDAR topic to see its link to the camera", false];
  const cam = (found.topics || []).find((t) => t.topic === camTopic), lid = (found.lidar || []).find((t) => t.topic === lidarTopic);
  const how = cam && (cam.tf_links || {})[lidarTopic];
  if (!how) return [cal ? "no TF link: the calibration links them" : "no TF link: upload the calibration", !cal];
  const chain = (lid && tfChain(found.tf || [], cam.frame_id, lid.frame_id)) || [cam.frame_id, lid ? lid.frame_id : lidarTopic];
  return [`linked by ${how === "static" ? "/tf_static" : "/tf"}: ${chain.join(" <- ")}`, false];
}
// job.json "calibration" in short (server.calibration_summary).
const calText = (c) => [
  `Calibration ${c.name}: the LiDAR is ${(100 * c.translation_m).toFixed(1)} cm from the camera, turned ${Number(c.rotation_deg).toFixed(1)} deg.`,
  c.has_intrinsics ? "It gives K." : "K comes from the camera_info.",
  c.camera_axes === "body" ? "Its camera axes are body axes (x forward, y left, z up)." : null,
  c.time_offset_s ? `Its time offset is ${c.time_offset_s} s.` : null].filter(Boolean).join(" ");  // #calhelp says it replaces TF
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
  const list = Array.isArray(found.lidar) ? found.lidar : null, usable = stray || (!!list && list.length > 0);
  $("#uselidar").disabled = !usable;
  if (!usable) $("#uselidar").checked = false;
  const on = $("#uselidar").checked === true;
  const skipped = (found.lidar_skipped || []).map((x) => `${x.topic} (${x.why})`).join(", ");
  $("#lidarhint").textContent = usable ? on ? LIDAR_ON[kind] : LIDAR_OFF
    : found.error ? "The bag's topics cannot be read: " + found.error
      : !list ? OLD_TOPICS[["review", "done"].includes(job.status) ? "late" : "idle"]
        : "This bag has no LiDAR topic (PointCloud2 or Livox)." + (skipped ? " Not usable: " + skipped + "." : "");
  // The topic, its link and the calibration: a bag's only. No "required" on the select: lidarChoice says what to do.
  $("#lidaropts").hidden = stray || !usable || !on;
  if ($("#lidaropts").hidden) return;
  const cal = job.calibration || null;
  const [link, act] = lidarLink(found, job.source && job.source.topic, $("#lidartopic").value, cal);
  if ($("#lidarlink").textContent !== link) $("#lidarlink").textContent = link;  // a live region: only a change is read out
  $("#lidarlink").className = act ? "warn" : "hint";
  $("#lidarskipped").textContent = skipped ? "Not usable: " + skipped : "";
  $("#calsum").hidden = !cal;
  $("#calsum").textContent = cal ? calText(cal) : "";
  $("#calbtn").textContent = cal ? "Replace the calibration" : "Upload a calibration";
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
// Whether a rosbag job's done lidar stage ran with another calibration than the job's now (steps.done("lidar") is false):
// the server then retries from lidar. /selection says so (calibration_changed); without a selection, pool/lidar.json's
// calibration file is compared with the job's by name.
async function scaleStale(job) {
  const st = (job.stages || {}).lidar, files = job.files || [];
  if (job.kind !== "rosbag" || !job.lidar || !st || st.status !== "done") return false;
  if (files.includes("selection.json")) {
    const lid = (await api(`/api/jobs/${V.id}/selection`)).lidar;
    return !!lid && lid.calibration_changed === true;
  }
  if (!files.includes("pool/lidar.json")) return false;
  const ext = (await api(file("pool/lidar.json"))).extrinsic || {}, cal = job.calibration;
  return cal ? ext.source !== "calibration" || ext.file !== cal.name : ext.source === "calibration";
}
// Upload a calibration file at once (PUT: the server checks and summarises it), or remove it (f null). The summary or
// the server's reason shows in the box. A review shows again whether the scale is current; a failed or cancelled job
// says when Retry starts from lidar.
async function calibrate(f) {
  const focus = document.activeElement;
  // Retry and Build wait for the upload: they would run with the old calibration.
  const btns = [$("#calremove"), $("#start"), $("#calfile"), $("#retry"), $("#build")], was = btns.map((b) => b.disabled);
  btns.forEach((b) => { b.disabled = true; });
  say($("#calmsg"), f ? "checking " + f.name : "removing the calibration", true);
  const url = `/api/jobs/${V.id}/calibration`;
  let done = null;
  try {
    const r = await (f ? api(url + "?name=" + encodeURIComponent(f.name), { method: "PUT", body: f })
      : api(url, { method: "DELETE" }));
    if (V.job) V.job.calibration = f ? r : null;
    done = f ? `The calibration ${r.name} is saved.` : "The calibration is removed.";
  } catch (e) { say($("#calmsg"), e.message); }
  if (done !== null && V.sel && $("#review").hidden === false) {
    try { V.sel.lidar = (await api(`/api/jobs/${V.id}/selection`)).lidar; renderSelLidar(); } catch (e) { /* the old summary stays */ }
  } else if (done !== null && V.job && ["failed", "cancelled"].includes(V.job.status)) {
    V.scaleStale = await scaleStale(V.job).catch(() => false);
    if (V.state) renderState(V.state);  // the Retry button names the stage it starts from
    rebuildHint();
    if (V.scaleStale) done += " The scale was found with the old calibration, so Retry starts from the lidar stage.";
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
    const kids = [h("img", { src: `/api/jobs/${V.id}/thumb/${i}`, loading: "lazy", alt: manual ? "" : "frame " + i, style: ratio }),
      picked ? h("b", {}, String(k + 1) + (V.picks[k] === V.view0 ? " ★" : "")) : null, h("span", {}, at)];
    const cls = "cell" + (cand.has(i) ? " cand" : "") + (picked ? " pick" : "");
    return manual ? h("button", { type: "button", class: cls, "data-i": i, "aria-pressed": String(picked), "aria-label": `frame ${i} at ${at}` }, ...kids)
      : h("div", { class: cls, "data-i": i }, ...kids);
  }));
  if (typeof fi === "string") {  // the grid was drawn again under a focused frame button
    const b = document.querySelector('#grid [data-i="' + fi + '"]');
    if (b) b.focus();
  }
  $("#pickcount").textContent = `${V.picks.length} picked (2 to 12)` + (V.picks.length && V.view0 == null ? "; view 0 is the first pick" : "");
  $("#picks").replaceChildren(...V.picks.map((p, k) => h("li", {},
    `${k + 1}. frame ${snap(p)}` + (snap(p) !== p ? ` (sharper than ${p})` : "") + ` at ${m.times[snap(p)].toFixed(2)} s `,
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
    $("#pickcount").textContent = "12 picked, the most allowed: remove one first";
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
  // An articulated object sends who lists its parts, and the typed list for "I type the parts".
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
      !confirm("Build again? This replaces the current results of this job (meshes, physics and results.zip).")) return;
  const agent = chosenAgent(), keyed = agent !== "none";
  $("#start").disabled = true;
  say($("#startmsg"), keyed ? "checking the key" : "", true);
  try {
    await post(`/api/jobs/${V.id}/start`, { mode: mode(), prompt: $("#prompt").value, n_views: +$("#nviews").value,
      picks: manual ? V.picks : [], view0: manual ? V.view0 : null, auto_continue: !$("#pause").checked,
      agent, api_key: keyed ? $("#key").value : null, articulated, ...(articulated ? { parts_mode: pm, parts } : {}), lidar });
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
  V.ov = { locked: [...sel.overrides.locked], excluded: [...sel.overrides.excluded], instance: { ...sel.overrides.instance }, view0 };
  V.dirty = false;
  const pick = (x) => x == null ? "?" : x;
  $("#selsum").textContent = sel.ok
    ? `${sel.picks.length} views (asked for ${sel.n_requested}); every candidate is within ${pick(sel.coverage_deg)} deg of a pick; the closest picks are ${pick(sel.min_spacing_deg)} deg apart.`
    : "The automatic selection needs help: lock or exclude views and select again, or start over in manual mode.";
  renderSelLidar();
  $("#selwarn").replaceChildren(...sel.warnings.filter((w) => !V.state.warnings.includes(w)).map((w) => h("li", {}, w)));
  $("#build").disabled = !sel.ok;
  $("#reselect").disabled = true;
  $("#discard").hidden = true;
  say($("#revmsg"), "", true);
  $("#ring").setAttribute("aria-label", `camera positions seen from above: ${sel.picks.length} picked of ${sel.candidates.length} candidates`);
  ring(sel);
  renderCards();
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
// One view's scale and point count: "x1.05 (280 pts)", "x1.21 (90 pts, outlier)", "few points (12 pts)", "no points".
function lidarViewText(v) {
  const st = v.status in LIDAR_STATUS ? LIDAR_STATUS[v.status] : String(v.status || "").replace(/_/g, " ");
  const pts = count(v.points_used) + " pts";
  return v.s != null ? `x${Number(v.s).toFixed(2)} (${pts}${st ? ", " + st : ""})`
    : (st || "no scale") + (v.points_used ? ` (${pts})` : "");
}
// The scale (or why there is none), a changed calibration, the warnings, and the check images folded.
function lidarView(lid) {
  const c = lid.counts || {}, views = lid.views || [], dir = lid.check_dir || "lidar/check";
  return [lid.ok ? h("p", { class: "lidarhead" }, `Metric scale from LiDAR: x${Number(lid.s).toFixed(3)} ` +
      `(${c.views_used} of ${c.views} views, spread ${pct(lid.spread_rel)})`)
    : h("p", { class: "msg" }, "No metric scale from LiDAR: " + (lid.reason || "no reason given")),
  lid.calibration_changed ? h("p", { class: "warn" }, "The calibration changed after this scale was found. " +
      "Build finds the scale again first, then stops here.") : null,
  (lid.warnings || []).length ? h("ul", { class: "warnings" }, lid.warnings.map((w) => h("li", {}, w))) : null,
  views.length ? h("details", { class: "sub", open: V.lidarOpen === true, ontoggle: (e) => { V.lidarOpen = e.target.open; } },
    h("summary", {}, `LiDAR check images (${views.length})`),
    // lidar_scale.check_image: r / s against the combined s; without one, each view's own s, and white without that.
    h("p", { class: "hint" }, "The green line is the mask outline. Blue points are nearer than the scaled depth, and red " +
      "points are farther. White points agree with it." + (lid.ok ? "" : " Each view uses its own scale here. A view " +
      "without one shows all its points white.") + " Open an image to see it larger."),
    h("div", { class: "checks" }, views.map((v) => h("figure", {},
      h("a", { href: file(`${dir}/${v.name}.jpg`), target: "_blank", rel: "noopener" },
        h("img", { src: file(`${dir}/${v.name}.jpg`), loading: "lazy", alt: "LiDAR points in " + v.name })),
      h("figcaption", {}, `${v.name}: ${lidarViewText(v)}`))))) : null];
}
function renderSelLidar() {  // the review header: LiDAR's scale, or DA3's own
  const lid = V.sel.lidar, job = V.job || {};
  fill($("#sellidar"), lid ? lidarView(lid) : h("p", {}, job.lidar
    ? "Metric scale from LiDAR: not found yet. Build finds it first, then stops here." : NO_LIDAR),
  // A bag's calibration can change in review, in the Settings tab, without a start over.
  job.kind === "rosbag" && job.lidar ? h("p", {}, h("button", { class: "small secondary", type: "button",
    onclick: toLidarSettings }, "Change the calibration")) : null);
}
// The job failed at the lidar stage (stages: job.progress or the state's stages).
const lidarFailed = (status, stages) => status === "failed"
  && ((stages || []).find((x) => x.name === "lidar") || {}).status === "failed";
async function renderLidarFail(job) {
  const rec = job.files.includes("pool/lidar.json") ? lidarCompact(await api(file("pool/lidar.json"))) : null;
  fill($("#lidarfailbody"), rec ? lidarView(rec)
    : h("p", { class: "hint" }, "The stage stopped before it wrote its record. The reason is above, and its log is in the " +
      "Stages and log tab."),
  h("p", {}, h("button", { class: "small secondary", type: "button", onclick: toLidarSettings }, "Go to the LiDAR settings")));
}
function toLidarSettings() {  // the LiDAR box in the Settings tab: the calibration upload, or LiDAR off
  selectTab("settings", true);
  $("#lidarbox").scrollIntoView({ block: "center", behavior: smooth() });
  ($("#lidaropts").hidden ? $("#uselidar") : $("#calfile")).focus({ preventScroll: true });
}

function ring(sel) {
  const svg = $("#ring"), c = sel.ring.center, pts = sel.ring.points;
  svg.replaceChildren();
  if (!c) return;
  const r = 1.2 * Math.max(1e-3, ...pts.filter(Boolean).map((p) => Math.hypot(p[0] - c[0], p[1] - c[1])));
  svg.setAttribute("viewBox", `${c[0] - r} ${-c[1] - r} ${2 * r} ${2 * r}`);
  const d = r / 30, order = new Map(sel.picks.map((p, k) => [p.name, k]));
  svg.append(s("path", { d: `M${c[0] - d} ${-c[1]}h${2 * d}M${c[0]} ${-c[1] - d}v${2 * d}`, stroke: "#333", "stroke-width": d / 3 }));
  sel.candidates.forEach((cand, i) => {
    const p = pts[i];
    if (!p) return;
    const [x, y] = [p[0], -p[1]], picked = order.has(cand.name), color = picked ? "#1a66d6" : "#999";
    let mark;
    if (cand.name === sel.anchor) {
      const star = [...Array(10)].map((_, k) => { const a = Math.PI * k / 5 - Math.PI / 2, rr = k % 2 ? d : 2.4 * d; return `${x + rr * Math.cos(a)},${y + rr * Math.sin(a)}`; });
      mark = s("polygon", { points: star.join(" "), fill: "#ffb300", stroke: "#333", "stroke-width": d / 4 });
    } else mark = s("circle", { class: picked ? "pick" : "cand", cx: x, cy: y, r: 1.2 * d, fill: picked ? color : "#fff", stroke: color, "stroke-width": d / 3 });
    const title = s("title");
    title.textContent = `${cand.name} (${cand.status})`;
    mark.append(title);
    svg.append(mark);
    if (picked) {
      const label = s("text", { x, y: y - 2.6 * d, "font-size": 2.4 * d, "text-anchor": "middle" });
      label.textContent = String(order.get(cand.name));
      svg.append(label);
    }
  });
}
function renderCards() {
  const sel = V.sel, ov = V.ov, order = new Map(sel.picks.map((p, k) => [p.name, k]));
  const lviews = new Map(((sel.lidar && sel.lidar.views) || []).map((v) => [v.name, v]));  // each view's LiDAR scale
  const card = (c) => {
    const sam = sel.sam3[c.name] || { instances: [] }, k = order.get(c.name), lv = lviews.get(c.name);
    const use = ov.locked.includes(c.name) ? "in" : ov.excluded.includes(c.name) ? "out" : "auto";
    const inst = ov.instance[c.name] || "";
    return h("div", { class: "card " + c.status, role: "group", "aria-labelledby": "card-" + c.name },
      h("div", { class: "cimg" }, h("img", { src: file(`${sel.check_dir}/${c.name}.jpg`), loading: "lazy", alt: "" }),
        k != null ? h("b", { class: "chip" }, `view ${k}${c.name === sel.anchor ? " ★" : ""}`) : null),
      h("div", { class: "meta", id: "card-" + c.name }, `${c.name} at ${c.time_s.toFixed(2)} s`),
      h("div", {}, badge(c.status), c.q != null ? ` q ${c.q.toFixed(2)}` : "",
        c.consistency != null ? ` agree ${c.consistency.toFixed(2)}` : "",
        c.angle_to_nearest_pick != null && k == null ? ` ${c.angle_to_nearest_pick} deg from a pick` : ""),
      lv ? h("div", { class: "lidar" + (lv.status === "used" ? "" : " off") }, (lv.s != null ? "LiDAR " : "LiDAR: ") + lidarViewText(lv)) : null,
      h("select", { "aria-label": c.name + ": use", onchange: (e) => edit(c.name, "use", e.target.value, e.target) },
        h("option", { value: "auto", selected: use === "auto" }, "auto"),
        h("option", { value: "in", selected: use === "in" }, "always use"),
        h("option", { value: "out", selected: use === "out" }, "never use")),
      h("select", { "aria-label": c.name + ": mask", onchange: (e) => edit(c.name, "instance", e.target.value, e.target) },
        h("option", { value: "", selected: !inst }, "mask: top score"),
        ...sam.instances.map((x) => h("option", { value: String(x.index), selected: inst === String(x.index) }, `mask: instance ${x.index} (${x.score.toFixed(2)})`)),
        h("option", { value: "none", selected: inst === "none" }, "not in this frame")),
      h("label", { class: "check" }, h("input", { type: "radio", name: "view0", checked: (ov.view0 || sel.anchor) === c.name,
        onchange: (e) => edit(c.name, "view0", null, e.target) }), " view 0"));
  };
  const picked = sel.candidates.filter((c) => order.has(c.name)).sort((a, b) => order.get(a.name) - order.get(b.name));
  const others = sel.candidates.filter((c) => !order.has(c.name)).sort((a, b) => a.frame - b.frame);
  fill($("#cards"), h("h4", {}, `Picked views (${picked.length})`), h("div", { class: "cards" }, ...picked.map(card)),
    others.length ? h("h4", {}, `Other candidates (${others.length}): set "always use" to add one`) : null,
    others.length ? h("div", { class: "cards" }, ...others.map(card)) : null);
}
function edit(name, what, value, el) {
  const ov = V.ov;
  if (what === "use") {
    ov.locked = ov.locked.filter((n) => n !== name);
    ov.excluded = ov.excluded.filter((n) => n !== name);
    if (value === "in") ov.locked.push(name);
    if (value === "out") ov.excluded.push(name);
  } else if (what === "instance") {
    if (value) ov.instance[name] = value; else delete ov.instance[name];
  } else ov.view0 = name;
  V.dirty = true;
  if (el && el.closest) el.closest(".card").classList.add("edited");
  $("#reselect").disabled = false;
  $("#build").disabled = true;
  $("#discard").hidden = false;
  say($("#revmsg"), "select again to apply your changes before the build", true);
}
$("#discard").onclick = () => loadReview().catch((e) => say($("#revmsg"), e.message));
$("#manualstart").onclick = async () => {  // the Settings tab in manual mode, with the frames open
  V.manual = true;
  V.meta = null;
  await refresh().catch((e) => say($("#revmsg"), e.message));
  selectTab("settings", true);
  $("#gridbox").open = true;
  $("#framesum").focus({ preventScroll: true });
  $("#tabs").scrollIntoView({ block: "start", behavior: smooth() });
};
$("#reselect").onclick = () => act([$("#reselect"), $("#build")], $("#revmsg"), "selecting the views again",
  () => post(`/api/jobs/${V.id}/overrides`, { ...V.ov, api_key: $("#key2").value || null }),
  () => { $("#key2").value = ""; $("#log").textContent = ""; });
$("#build").onclick = () => act([$("#build"), $("#reselect")], $("#revmsg"), "checking the key and queuing the build",
  () => post(`/api/jobs/${V.id}/continue`, { api_key: $("#key2").value || null }),
  () => { $("#key2").value = ""; $("#log").textContent = ""; });

// ---- 4b. the part list ("The VLM proposes, I edit the list": the job waits after propose) ----

async function loadParts() {
  const r = V.parts = await api(`/api/jobs/${V.id}/parts`), p = r.proposal || { parts: [], warnings: [] };
  const moving = p.parts.filter((x) => x.parent), n = moving.length;
  const who = p.source === "vlm" ? VLM[p.agent] || "The VLM" : "You";
  fill($("#partssum"), !p.parts.length ? "There is no part list yet." : n
    ? `${who} listed ${n} moving ${n === 1 ? "part" : "parts"} of the ${r.noun}. The base, which does not move, is ${p.base}.`
    : `${who} found no moving part. Add the parts that move, one per line under the base, ${p.base}.`);
  $("#partswarn").replaceChildren(...(p.warnings || []).map((w) => h("li", {}, w)));
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
  return act([$("#partsbuild"), $("#partsreset")], $("#partsmsg"), "checking the parts and queuing the build",
    () => same ? post(`/api/jobs/${V.id}/continue`, { api_key }) : post(`/api/jobs/${V.id}/parts`, { parts: keepHints(parts, proposed), api_key }),
    () => { $("#key2").value = ""; $("#log").textContent = ""; V.partsKey = null; });
};

// ---- 5. results ----

// What each item of the Results and Downloads tabs is and how it was found, shown by its info button (info()): the rows
// of the facts list, the viewers and the files to download (dl_). One to three plain sentences each. Sources:
// build_plan.md notes 4b, 5, 9, 11 and 12, and the code they name (mvsam3d.place, lidar_scale, views, steps,
// agent_step, controller, vomp_run, mjcf, urdf, parts.stage, jobs.package).
const INFO = {
  metric_scale: "With LiDAR, each view's scale is the median, over the LiDAR points inside the object's mask, of the " +
    "LiDAR depth divided by Depth Anything 3's depth at the same pixel. The number shown is the weighted median of " +
    "the views that agree, and select multiplies DA3's depth and camera positions by it before MV-SAM3D, so later " +
    "sizes are in meters. Without LiDAR, the sizes come from DA3's own depth estimate, which is not measured.",
  mask_iou: "For each view, the app draws the posed mesh from that view's camera and compares it with the SAM 3 mask: " +
    "the shared area divided by the area of both together, from 0 to 1. This number is the mean over the views used. " +
    "Below 0.5 the agent is skipped, and physics uses the Stage A mesh or its split parts.",
  parts_lost: "Each entry is a group of SAM 3 masks of a listed name that did not become a part, so a name can show " +
    "here and still have a part. \"not found\" means that no mask of the name was left in any view, because SAM 3 " +
    "found none with a score of at least 0.4 or the vote dropped them all. Otherwise the face vote gives the reason: " +
    "the group covers less than 0.3% of the surface, or fewer than half of the views that see it well and found the " +
    "name hold it.",
  agent: "GPT-6-Astra (through the Codex CLI) or Claude Opus 5.5 (through the Anthropic API), both at xhigh effort. It " +
    "has 600 s to edit the Stage A mesh in Blender, or 900 s to set the joints of an articulated object's split " +
    "parts, and time spent exporting GLBs does not count, up to 600 s. It is skipped when it is turned off or the " +
    "mean mask IoU is below 0.5, and physics then uses the Stage A mesh or the split parts without joints.",
  agent_verdict: "The agent's verdict when it called finish: complete, unchanged or needs_review for a refined mesh, " +
    "or articulated, unchanged or needs review for joints. In brackets is where the result came from: finish, " +
    "checkpoint, export or fallback_original. A checkpoint or an export means that the session ended without a " +
    "valid finish, so the verdict is needs review; fallback_original means that the agent left no valid result, so " +
    "its input was kept.",
  agent_reason: "The agent's own words when it called finish. When the session ended without a valid finish, it is " +
    "the defect noted at the last checkpoint, or the app's note on the export or the fallback. When the agent's " +
    "result held your API key, the reason and the limitations are left out.",
  limitation: "A flaw that the agent says is still in its result, written when it called finish. The agent can list " +
    "at most 3.",
  joints: "The number of joints in the GLB that the agent delivered, fixed joints included. With 0, physics welds the " +
    "parts into one body. A ? means that the app could not read the GLB's joints.",
  joint_changes: "The joint changes (set_joint calls that set or removed a joint) that the app accepted during the " +
    "session. A change that added a problem to the joint tree was refused and undone, and it is not counted.",
  agent_tokens: "The tokens that the Claude agent used over the whole session, from its log: new input, output, " +
    "cache reads and cache writes. A turn is one reply of the model.",
  agent_cost: "An estimate from the token counts at Claude Opus 5.5's prices per million tokens: $4 input, $20 " +
    "output, $0.20 cache read and $5 cache write. In a turn with a refusal fallback, both tries count, each at its " +
    "own model's prices, but a try that failed and was sent again does not, so the estimate can be low. A ? means " +
    "that the log gave no cost; your Anthropic bill gives the real cost.",
  links: "The bodies of the MuJoCo model (mjcf.json), one for each part of the GLB that physics used. Each link " +
    "has its own mass, which the Link masses row lists.",
  links_welded: "The GLB that physics used has no joints, so the MuJoCo, USD and URDF models hold the whole object as " +
    "one rigid body. This happens when the agent is off or skipped, or when it delivered no joints.",
  base: "The link that the other links hang from. It is fixed to the world when the agent gave it a fixed joint to " +
    "the world, and otherwise it can move freely.",
  joint: "How this part moves on its parent, from the joint that the agent set: it turns about an axis (revolute), " +
    "slides along one (prismatic) or is fixed. The range runs from the lower to the upper limit, where 0 is the " +
    "part's pose in the mesh.",
  joint_welded: "The agent gave this part no joint, so physics welds it to the named link: the nearest part above " +
    "it in the GLB's node tree, or else the base.",
  mass: "VoMP predicts a density for every occupied voxel of the aligned mesh on a 64^3 grid, and the mass is the sum " +
    "of density times voxel volume. An articulated object adds up its links, each on its own grid; a link with no " +
    "VoMP result counts with its collision hulls filled at the median density when the parts have joints, and is " +
    "left out when they have none. Mass grows with the size cubed, so a 5% size error gives about a 16% mass error.",
  com: "The mass-weighted mean of the voxel centers (and of any link filled from its collision hulls), as x, y and z " +
    "in meters. The frame is that of the physics models: z up, x along the long side, and the origin at the center " +
    "of the footprint at the bottom of the object.",
  link_masses: "Each link's mass from VoMP, each link on its own 64^3 grid. A link with no VoMP result, such as a " +
    "thin lid with no occupied voxel, gets its collision hulls filled at the object's median density instead.",
  mass_note: "VoMP gave no result for these links, for example a thin lid with no occupied voxel on its 64^3 grid. " +
    "The mass and center of mass above leave them out, and so do the MuJoCo, USD and URDF models, which still keep " +
    "their shapes.",
  youngs: "Stiffness: the median of VoMP's predicted Young's modulus over the object's voxels, in pascals (1 GPa is " +
    "1e9 Pa). For an articulated object, the voxels of all links count, each by its volume. In our tests VoMP v1.1 " +
    "gave about 4 GPa to most objects, steel and glass included.",
  poisson: "How much the material bulges sideways when it is squeezed, with no unit (usually 0 to 0.5). It is the " +
    "median of VoMP's prediction over the object's voxels; for an articulated object, each voxel counts by its volume.",
  density: "The median of VoMP's predicted density over the object's voxels, in kg/m^3; for an articulated object, " +
    "each voxel counts by its volume. The mass uses each voxel's own density, not this median. In our tests VoMP " +
    "v1.1 gave about 1,000 kg/m^3 to most objects.",
  view_posed: "MV-SAM3D's mesh from all the views used, with its size, position and a small turn fitted to the SAM 3 " +
    "masks. It is in view 0's camera frame, where the viewer's up is the camera's forward axis, so the object looks " +
    "tipped over; aligned.glb is the upright copy. The size is the box around it along view 0's camera axes, in cm, " +
    "so a tilted object gives a box larger than the object.",
  view_parts: "The posed mesh cut into parts: SAM 3 masked each part's name in the views, the masks voted on every " +
    "face, and Blender cut the mesh and filled the holes. The parts have no joints yet, and they are in the same " +
    "frame as the Stage A mesh. The size is the box around them along view 0's camera axes, in cm.",
  view_refined: "The agent's GLB, in the same frame as the Stage A mesh: that mesh after the agent edited it in " +
    "Blender, or as it was when the verdict is unchanged. Physics uses it. The size is the box around it along view " +
    "0's camera axes, in cm.",
  view_rigged: "The split parts as the agent delivered them, in one GLB in the Stage A mesh's frame, with the joints " +
    "it set (none when the caption says without joints). Physics makes each part a link and each joint a joint, " +
    "welds a part without a joint to the part above it, and makes one body when there are no joints. The size is " +
    "the box around it along view 0's camera axes, in cm.",
  dl_zip: "The last stage (package) writes it. A file that holds your API key is left out, such as the agent's " +
    "result when the key was found in it.",
  dl_posed: "The Stage A mesh in view 0's camera frame, in meters, with the scale from LiDAR when LiDAR is on, else " +
    "from Depth Anything 3's estimate.",
  dl_parts: "The split parts in one GLB, one node per part, without joints, in the same frame as posed.glb.",
  dl_refined: "The agent's GLB, in the same frame as posed.glb. For an articulated object, each joint that the agent " +
    "set is in its child part's node extras (gator_joint); a part without one has none.",
  dl_aligned: "The GLB that physics used, turned upright, with the bottom at 0, the footprint centered on the origin " +
    "and the long side on x (z up in MuJoCo, y up in glTF). Up is the object's own axis closest to the up of the " +
    "table or floor under it in view 0, or to the cameras' mean up when no plane fits there. Materials that let " +
    "light through are made opaque, because VoMP rejects them.",
  dl_urdf: "A URDF of the object, built from the MuJoCo model's meshes and masses: one link when the GLB has no " +
    "joints, else one link per part. Each part but the base gets the agent's joint to its parent, or a fixed one to " +
    "the link that physics welds it to when the agent gave it none, and a fixed base adds a world link.",
  dl_placement: "The record of the place step: the pose that puts the MV-SAM3D mesh in view 0's camera frame, the " +
    "scale and pose fitted to the SAM 3 masks, and the metric scale. Its check gives each view's mask IoU and how " +
    "close view 0's DA3 points lie to the mesh.",
  dl_summary: "VoMP's summary of the aligned mesh: the voxel counts, the median and the 5th to 95th percentile of " +
    "Young's modulus, Poisson ratio and density, and the mass, center of mass and inertia of the rigid body. For an " +
    "articulated object it adds each link's numbers and the links that VoMP could not voxelize.",
  dl_lidar: "The record of the lidar stage: the metric scale, each view's scale, point count and status, how the " +
    "views were combined, the LiDAR source, and the link to the camera (for a rosbag the TF chain or the uploaded " +
    "calibration; a Stray capture's depth is already aligned with the color camera). When no scale was found, ok " +
    "is false and reason says why.",
  dl_selection: "The view selection: every candidate frame with its quality and status, such as picked or excluded, " +
    "the picks in run order with view 0 first, the cameras seen from above, the warnings and the metric scale that " +
    "select applied (null without LiDAR). ok is false when the selection failed, and the warnings say why.",
  dl_parts_json: "The record of the parts stage: each part's name, parent, guessed joint, share of the surface and " +
    "box, then the parts that were not split off, with the reason. It also gives the seams between parts that touch, " +
    "with up to 4 fitted lines that may be hinge axes.",
  dl_mjcf: "The record of the MuJoCo model: its inputs, versions, counts and start height. For an object with joints " +
    "it also lists the links with their masses, the joints, the base, the welded parts and the pairs that do not " +
    "collide. The model itself (the XML and its meshes) comes only in results.zip.",
  part_masks: "SAM 3 ran on each view used with each part's name as the prompt, the base included, and every mask " +
    "with a score of at least 0.4 is shown. The face vote starts from all of them, then drops the masks that lie " +
    "outside the object, cover the whole object, are the union of two or more others, or repeat a mask with a higher " +
    "score.",
};

// The info button of an item: a small "i" after its label, and its note (INFO[key]) that shows on hover (a mouse or a
// pen) and on keyboard focus. A click, a tap, Enter or Space toggles it: it shows and stays open, or it hides when it
// showed. Escape, a click elsewhere and focus that moves on close it. One note shows at a time. The note is fixed to
// the window, so it moves no layout and is not clipped by its figure; it stays TIP_EDGE px inside the window's edges.
// row: for a facts row, a function that gives its dd; the note goes under that value or above the row's label, from
// the value's left edge, so the value and the column of info buttons stay in view. Other notes go under or above
// their button.
const TIP_EDGE = 12, TIP_GAP = 6, TIP_DELAY = 150;  // px from the window's edge, px from the button, ms to reach the note
let tipN = 0, tip = null, tipTimer = null;  // the notes' ids; the open note {t, pin, hover, focus}; the hover's close
function info(key, label, row) {
  const text = INFO[key];
  if (!text) return null;
  const id = "note-" + ++tipN, t = { row: row || null, down: null };
  const hover = (e) => !e || e.pointerType !== "touch";  // a tap is a click, not a hover
  t.note = h("span", { class: "infonote", id, role: "tooltip", hidden: true }, text);
  t.btn = h("button", { type: "button", class: "infobtn", "aria-label": "About " + label, "aria-describedby": id,
    onpointerdown: () => { t.down = !!tip && tip.t === t; },  // shown before the press, which may focus the button
    onclick: (e) => tipClick(t, e), onfocus: () => tipOn(t, "focus"), onblur: () => tipOff(t, "focus") }, "i");
  t.wrap = h("span", { class: "info", onpointerenter: (e) => { if (hover(e)) tipOn(t, "hover"); },
    onpointerleave: (e) => { if (hover(e)) tipOff(t, "hover"); } }, t.btn, t.note);
  return t.wrap;
}
function tipWait(fn) {  // fn after TIP_DELAY ms (the hover's close); tipWait() cancels it
  if (tipTimer !== null) clearTimeout(tipTimer);
  tipTimer = fn ? setTimeout(() => { tipTimer = null; fn(); }, TIP_DELAY) : null;
}
function tipOn(t, why) {  // why: "hover", "focus" or "pin" (a click)
  tipWait();
  if (!tip || tip.t !== t) {
    tipClose();
    tip = { t, pin: false, hover: false, focus: false };
    t.note.hidden = false;
    t.wrap.classList.toggle("on", true);
  }
  tip[why] = true;
  tipPlace();
}
function tipOff(t, why) {
  if (!tip || tip.t !== t) return;
  tip[why] = false;
  if (why === "focus") tip.pin = false;  // the focus moved on: a note kept open by a click closes too
  if (tip.pin || tip.hover || tip.focus) return;
  if (why !== "hover") tipClose();
  else tipWait(() => { if (tip && tip.t === t && !tip.pin && !tip.hover && !tip.focus) tipClose(); });  // time to reach the note
}
// A click of a pointer (detail 1 or more) goes by the note as it was at pointerdown, before the press focused the
// button (a tap in Chrome shows the note by focus first); a click of a key (detail 0) by the note as it is.
function tipClick(t, e) {
  const shown = e && e.detail > 0 && t.down !== null ? t.down : !!tip && tip.t === t;
  t.down = null;
  if (shown) tipClose();
  else tipOn(t, "pin");
}
function tipClose() {
  tipWait();
  if (!tip) return;
  tip.t.note.hidden = true;
  tip.t.wrap.classList.toggle("on", false);
  tip = null;
}
function tipPlace() {  // under its value or button, or above when only that fits; inside the window, below the header
  if (!tip) return;
  const { btn, note, wrap } = tip.t, row = tip.t.row && tip.t.row();
  if (btn.isConnected === false) { tipClose(); return; }
  const vw = document.documentElement.clientWidth || window.innerWidth, vh = window.innerHeight;
  const head = $("header") ? $("header").getBoundingClientRect().bottom : 0, b = btn.getBoundingClientRect();
  if (b.bottom <= head || b.top >= vh) { tipClose(); return; }  // the button went under the sticky header or off the window
  note.style.left = note.style.top = "0px";  // measured at the corner, at its own width
  const n = note.getBoundingClientRect(), r = row ? row.getBoundingClientRect() : b;
  const label = row && wrap.parentElement ? wrap.parentElement.getBoundingClientRect() : b;  // a row's dt
  const x = row ? r.left : b.left + b.width / 2 - n.width / 2;
  const left = Math.max(TIP_EDGE, Math.min(x, vw - TIP_EDGE - n.width));
  const above = Math.min(r.top, label.top, b.top) - TIP_GAP - n.height, below = Math.max(r.bottom, b.bottom) + TIP_GAP;
  const top = below + n.height > vh - TIP_EDGE && above >= Math.max(TIP_EDGE, head + TIP_GAP) ? above : below;
  note.style.left = Math.round(left) + "px";
  note.style.top = Math.round(top) + "px";
}
function tipKey(e) {
  if (e.key === "Escape" && tip) tipClose();
}
function tipOutside(e) {  // a click or a tap anywhere but the open note and its button
  if (tip && !(e.target && tip.t.wrap.contains(e.target))) tipClose();
}
document.addEventListener("keydown", tipKey);
document.addEventListener("pointerdown", tipOutside, true);
window.addEventListener("scroll", tipPlace, { capture: true, passive: true });  // any box that scrolls moves the button
window.addEventListener("resize", tipPlace);

async function renderResults(job) {
  tipClose();  // its button is about to go
  const has = (p) => job.files.includes(p), json = async (p) => has(p) ? (await fetch(file(p))).json() : null;
  const cm = (x) => (100 * x).toFixed(0);
  const viewer = (p, title, key) => {
    const name = h("span", {}, title), cap = h("figcaption", {}, name, info(key, title)), fig = h("figure", {}, cap);
    fig.append(h("model-viewer", { src: file(p), "camera-controls": true, "environment-image": "neutral", "shadow-intensity": "1",
      "interaction-prompt": "none", alt: title,
      onload: (e) => {  // the size of the model, in the GLB's meters
        const d = e.target.getDimensions && e.target.getDimensions();
        if (d) name.textContent = `${title}: ${cm(d.x)} x ${cm(d.y)} x ${cm(d.z)} cm`;
      },
      onerror: () => fig.replaceChildren(cap, h("p", { class: "msg" }, `Could not show ${p.split("/").pop()}: `,
        h("a", { href: file(p), download: p.split("/").pop() }, "download it"), ".")) }));
    return fig;
  };
  const off = agentOf(job.agent) === "none", a = job.agent_result, art = job.articulated === true;
  const ran = a && RAN[a.agent];  // results from before the Opus agent do not name theirs
  const rigged = !!a && a.mode === "articulation";  // the agent set the joints of the split parts
  const by = ran ? ", by " + ran : "";
  const notRefined = art ? "no joints (agent turned off), so physics welds the parts into one body" : "not refined (agent turned off)";
  const refined = has("refined.glb") ? viewer("refined.glb", !rigged ? (ran ? "Refined by " + ran : "Refined by the agent")
    : a.joints_delivered > 0 ? "Parts with joints" + by : "Agent result without joints" + by, rigged ? "view_rigged" : "view_refined")
    : null;
  const split = has("run/parts.glb") ? viewer("run/parts.glb", "Parts (split, before rigging)", "view_parts") : null;
  $("#viewers").classList.toggle("single", !refined && !split);
  $("#resultsbody").classList.toggle("onefig", !refined && !split);  // a wide screen puts the facts beside it
  // With the agent off there is no agent viewer; the Agent row of the facts says so.
  fill($("#viewers"), viewer("run/posed.glb", "Stage A (MV-SAM3D, posed)", "view_posed"), split, refined);
  const [place, phys, mj, parts, pr, lid] = await Promise.all([json("run/placement.json"), json("physics/summary.json"),
    art ? json("physics/mjcf/mjcf.json") : null, art ? json("run/parts.json") : null,
    art && has("run/parts.json") ? api(`/api/jobs/${V.id}/parts`).catch(() => null) : null, json("pool/lidar.json")]);
  // [label, value, the key of its note in INFO]
  const rows = [], num = (x, d = 3) => x == null ? "?" : Number(x).toPrecision(d);
  if (place) rows.push(["Metric scale", metricScale(place, lid, job), "metric_scale"]);
  if (place) rows.push(["Stage A mean mask IoU", num(place.check && place.check.mean_mask_iou), "mask_iou"]);
  const lost = parts ? (parts.parts || []).filter((x) => x.status !== "found") : [];  // not_found, dropped_small, unconfirmed
  if (lost.length) rows.push(["Parts not split off", lost.map((x) => `${x.prompt} (${x.why || x.status.replace("_", " ")})`).join("; "), "parts_lost"]);
  if (job.stages.agent.status === "skipped") rows.push(["Agent", off ? notRefined : "skipped: " + job.stages.agent.note, "agent"]);
  else if (a) {
    if (a.model) rows.push(["Agent", `${ran || a.agent} (${a.model})`, "agent"]);
    rows.push(["Agent verdict", `${rigged && VERDICT[a.verdict] || a.verdict} (${a.source})`, "agent_verdict"],
      ["Agent reason", a.reason || "", "agent_reason"]);
    for (const l of a.limitations || []) rows.push(["Limitation", l, "limitation"]);
    if (rigged) rows.push(["Joints", a.joints_delivered != null ? String(a.joints_delivered) : "?", "joints"]);
    if (rigged && a.counts && a.counts.joints != null) rows.push(["Joint changes by the agent", String(a.counts.joints), "joint_changes"]);
    if (a.tokens) {
      const t = a.tokens;
      rows.push(["Agent tokens", `${count(t.input_tokens)} input, ${count(t.output_tokens)} output, ` +
        `${count(t.cache_read_input_tokens)} cache read, ${count(t.cache_creation_input_tokens)} cache write` +
        (a.turns != null ? ` in ${a.turns} turns` : ""), "agent_tokens"],
        ["Agent cost (estimate)", a.cost_usd != null ? "about $" + Number(a.cost_usd).toFixed(2) : "?", "agent_cost"]);
    }
  }
  if (mj && mj.links) {  // the MJCF of an object with joints: a body per link
    const names = Object.keys(mj.links), deg = (x) => (x * 180 / Math.PI).toFixed(0), cmr = (x) => (100 * x).toFixed(1);
    rows.push(["Links", `${names.length}: ${names.join(", ")}`, "links"],
      ["Base", `${mj.root}, ${mj.fixed_base ? "fixed to the world" : "free to move"}`, "base"]);
    for (const j of mj.joints || []) rows.push([`Joint of ${j.child}`, j.type === "revolute"
      ? `turns on ${j.parent}, ${deg(j.range[0])} to ${deg(j.range[1])} deg` : j.type === "prismatic"
        ? `slides on ${j.parent}, ${cmr(j.range[0])} to ${cmr(j.range[1])} cm` : `fixed to ${j.parent}`, "joint"]);
    for (const [k, to] of Object.entries(mj.welded || {})) rows.push([`Joint of ${k}`, `none, so it is welded to ${to}`, "joint_welded"]);
  } else if (mj && art) rows.push(["Links", "1: the parts have no joints, so physics welds them into one body", "links_welded"]);
  if (phys) {
    const whole = mj && mj.links ? mj.inertial : null;  // every link, also one that VoMP could not voxelize
    const rb = whole || phys.rigid_body || {}, md = phys.median || {}, skipped = Object.keys(phys.links_skipped || {});
    rows.push(["Mass", num(rb.mass) + " kg", "mass"], ["Center of mass", (rb.com || []).map((x) => num(x)).join(", ") + " m", "com"]);
    if (whole) rows.push(["Link masses", Object.entries(mj.links).map(([k, l]) => `${k} ${num(l.inertial.mass)} kg` +
      (l.source === "hull" ? " (from its collision hulls)" : "")).join(", "), "link_masses"]);
    else if (skipped.length) rows.push(["Mass note", `VoMP could not voxelize ${skipped.join(", ")}, so the mass covers only ` +
      `${(phys.rigid_body_covers || []).join(", ")}.`, "mass_note"]);
    rows.push(["Young's modulus (median)", num(md.youngs_modulus_pa) + " Pa", "youngs"],
      ["Poisson ratio (median)", num(md.poisson_ratio), "poisson"], ["Density (median)", num(md.density_kg_m3) + " kg/m3", "density"]);
  }
  // The label and its info button in one dt, then the value; the note goes by the value (the dd, made after the note).
  $("#facts2").replaceChildren(h("dl", {}, ...rows.flatMap(([k, v, key]) => {
    let dd = null;
    const dt = h("dt", {}, k, info(key, k, () => dd));
    return [dt, dd = h("dd", {}, v)];
  })));
  const checks = (pr && pr.checks) || [];  // run/parts/sam3/check/<view>.jpg
  fill($("#partchecks"), checks.length ? h("details", { class: "sub" }, h("summary", {}, `Part masks in the views (${checks.length})`),
    h("p", { class: "hint" }, "Each part's SAM 3 masks in its own color. Open one to see it larger.",
      info("part_masks", "Part masks in the views")),
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
  return `x${m.toFixed(3)} from LiDAR` + (shown.length ? ` (${shown.join(", ")})` : "");
}

// ---- 5a. the Downloads tab: the job's files by group ----

// The groups in order, each with its files as [path in the job folder, the name it downloads as, the key of its note
// in INFO]. They are server.OUTPUTS, which server.ALLOW serves; the tab lists those that exist (job.files). The MJCF
// and USD models, the URDF's meshes and the agent's result are not served alone: they come only in results.zip.
const DOWNLOADS = [
  ["Everything", [["results.zip", "results.zip", "dl_zip"]]],
  ["Meshes", [["run/posed.glb", "posed.glb", "dl_posed"], ["run/parts.glb", "parts.glb", "dl_parts"],
    ["refined.glb", "refined.glb", "dl_refined"], ["physics/aligned.glb", "aligned.glb", "dl_aligned"]]],
  ["Simulation", [["physics/urdf/model.urdf", "model.urdf", "dl_urdf"]]],
  ["Records", [["run/placement.json", "placement.json", "dl_placement"], ["physics/summary.json", "summary.json", "dl_summary"],
    ["pool/lidar.json", "lidar.json", "dl_lidar"], ["selection.json", "selection.json", "dl_selection"],
    ["run/parts.json", "parts.json", "dl_parts_json"], ["physics/mjcf/mjcf.json", "mjcf.json", "dl_mjcf"]]]];
// What the Simulation group says above its one file.
const SIM_TEXT = "The URDF's meshes come only in results.zip, in urdf/meshes, so model.urdf does not load alone.";
// A file's size: "512 B", "3.4 KB", "35 KB", "1.2 MB", "100 MB", "2.50 GB" (1 KB is 1000 bytes, as in size()).
const fileSize = (b) => b < 1e3 ? b + " B" : b < 999.5e3 ? (b / 1e3).toFixed(b < 9.95e3 ? 1 : 0) + " KB" : size(b);
// agent_result.reason when the agent's result held the API key (server.get_job): jobs.package left it out of the zip.
const KEYED_RESULT = "left out: the agent's result held the API key";
// What results.zip holds (jobs.package): the URDF (a job from before the URDF stage has none), the LiDAR record, the
// agent's result and the split parts only when the job has them. The MJCF and USD models are not served alone.
function zipText(job, has) {
  const a = job.agent_result;
  const models = has("physics/urdf/model.urdf") ? "the MuJoCo (MJCF), USD and URDF models" : "the MuJoCo (MJCF) and USD models";
  const parts = [models + " with their meshes", "the GLB meshes", "the physics summary",
    "the placement", "the view selection", has("pool/lidar.json") ? "the LiDAR scale (lidar.json)" : null,
    a && a.reason !== KEYED_RESULT ? "the agent's result" : null,
    has("run/parts.glb") ? "the split parts (parts.glb, parts.json)" : null].filter(Boolean);
  return `results.zip holds ${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}. ` +
    "The MJCF and USD models come only in it.";
}
// What the tab says above its files when results.zip is not one of them: before a build ends (FILES_PARTIAL), or
// after a build that stopped (FILES_STOPPED). A done job has results.zip (package is the last stage).
const filesNote = (status) => ["failed", "cancelled"].includes(status) ? FILES_STOPPED : status === "done" ? null : FILES_PARTIAL;
// The files of the job that exist, by group: each a download link (named <job id>_<file>, so files of two jobs do not
// clash), its size (job.sizes) and its info button. While a build runs the tab lists none, since the build replaces
// them as it goes (jobs.DERIVED); it says one sentence then, and when there is no file yet.
function renderDownloads(job) {
  tipClose();  // the button of an open note is about to go
  const files = ACTIVE.includes(job.status) ? [] : job.files || [], sizes = job.sizes || {};
  const has = (p) => files.includes(p);
  const groups = DOWNLOADS.map(([title, rows]) => [title, rows.filter(([p]) => has(p))]).filter(([, rows]) => rows.length);
  const row = ([p, name, key]) => h("li", { class: "dlrow" },
    h("a", { href: file(p), download: `${job.id}_${name}`, class: p === "results.zip" ? "dl primary" : "dl" }, name),
    typeof sizes[p] === "number" ? h("span", { class: "dlsize" }, fileSize(sizes[p])) : null, info(key, name));
  const note = groups.length && !has("results.zip") ? filesNote(job.status) : null;
  fill($("#downloadsbody"), note ? h("p", { class: "dlnote" }, note) : null, groups.map(([title, rows]) => {
    const id = "dl-" + title.toLowerCase(), text = title === "Everything" ? zipText(job, has) : title === "Simulation" ? SIM_TEXT : null;
    return h("div", { class: "dlgroup" }, h("h3", { id }, title), text ? h("p", { class: "hint" }, text) : null,
      h("ul", { class: "dlist", "aria-labelledby": id }, rows.map(row)));
  }));
  $("#downloadsbody").hidden = !groups.length;
  $("#downloadsnone").hidden = groups.length > 0;
  $("#downloadsnone").textContent = ["queued", "running"].includes(job.status) ? FILES_BUSY : FILES_LATER;
}

// ---- 5b. the input: its preview video and the views the build used ----

// job.preview (server.preview_of): the record of frames/preview.mp4 (video.Preview) with "seek", where each view is in
// it; {error} when the frames stage could not write it; {missing} when the file is gone since; or null for a job read
// before the app made previews.
const NO_PREVIEW = "This job has no preview video, because its input was read before the app made previews.";
const MISSING_PREVIEW = "The preview video of this job is missing on the server.";
// The input's name and kind: "table.MOV (video)", "run.bag (rosbag, camera topic /cam/image_rect)".
function inputLabel(job) {
  const src = job.source || {}, kind = job.kind || src.kind || "video";
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
function clearInput() {  // the frames are read again, or another job opens: the old video and views go
  V.input = V.inputKey = V.videoKey = V.video = null;
  fill($("#inputvideo"));
  fill($("#inputviews"));
  say($("#inputmsg"), "", true);
}
// The Inputs tab once the frames are read: the preview video, and the views used once they are selected.
async function renderInput(job) {
  const p = job.preview && job.preview.file ? job.preview : null;
  // The video is drawn once per preview file, so a refresh or a new selection does not stop it. Its "seek" follows the
  // selection, so it does not count.
  const vkey = JSON.stringify([job.id, job.preview && { ...job.preview, seek: null }]);
  if (vkey !== V.videoKey) {
    V.videoKey = vkey;
    V.inputKey = null;  // the views are buttons only next to a video
    const video = V.video = p ? h("video", { src: file("frames/preview.mp4"), controls: true, preload: "metadata",
      playsinline: true, muted: true, width: p.width, height: p.height, style: `--ar: ${p.width} / ${p.height}`,
      "aria-label": "Preview of " + job.video.name }) : null;
    if (video) video.muted = true;  // the attribute only sets the default
    const fps = p && p.fps ? `, ${Number(Number(p.fps).toFixed(1))} fps` : "";
    const why = !job.preview ? NO_PREVIEW : job.preview.missing ? MISSING_PREVIEW
      : `The app could not make a preview video: ${job.preview.error}`;
    fill($("#inputvideo"), video ? h("figure", { class: "inputfig" }, video, h("figcaption", {}, inputLabel(job),
      h("span", { class: "hint" }, ` Preview at ${p.width}x${p.height}${fps}.`)))
      : [h("p", {}, inputLabel(job)), h("p", { class: "hint" }, why)]);
  }
  V.inputPlaced = job.files.includes("run/placement.json");  // place drew run/overlay_<i>.png for every view first
  if ((job.stages.select || {}).status !== "done") {  // no views yet, or a new selection runs: the old views go
    if (V.inputKey !== "none") {
      V.input = null;
      V.inputKey = "none";
      fill($("#inputviews"), h("p", { class: "hint" }, NO_VIEWS));
      say($("#inputmsg"), "", true);
    }
    return;
  }
  // The views are drawn once per selection. A place run that ends or goes changes only the heading (its mesh fit toggle).
  const key = JSON.stringify([job.id, (job.stages.select || {}).ended, vkey]);
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
}
function inputHead() {  // the strip's heading and hint, with the mesh fit toggle once place has run
  const I = V.input;
  fill(I.head, h("h4", {}, `Views used (${I.picks.length})`), I.placed ? h("button", { type: "button", class: "small secondary",
    "aria-pressed": String(V.overlay === true), onclick: (e) => toggleOverlay(e.currentTarget) }, "Show the mesh fit") : null);
  fill(I.hint, (I.video ? "Click a view to see its frame in the video. " : "") + "★ marks view 0." +
    (I.placed ? " The mesh fit shows the posed mesh in red and the outline of the SAM 3 mask in green." : ""));
}
function toggleOverlay(btn) {  // each view's photo, or place's overlay of the posed mesh on it
  V.overlay = V.overlay !== true;
  btn.setAttribute("aria-pressed", String(V.overlay));
  V.input.imgs.forEach((img, i) => img.setAttribute("src", viewSrc(i)));
}
function seekInput(i) {  // the preview at view i's frame, paused: it never starts playing by itself
  const v = V.input && V.input.video, pk = V.input && V.input.picks[i], p = V.input && V.input.preview;
  if (!v || !pk || !p) return;
  const spot = previewSpot(pk, p);
  v.pause();
  v.currentTime = spot.at + 0.001;  // 1 ms in, so that a browser that rounds the time shows this frame, not the one before
  V.input.figs.forEach((f, k) => f.classList.toggle("at", k === i));
  V.input.btns.forEach((b, k) => b.setAttribute("aria-current", String(k === i)));  // the view that the video shows
  say($("#inputmsg"), `The video shows view ${i} at ${fmt(spot.at)}` + (spot.frame == null ? "." : spot.frame === pk.frame
    ? `: frame ${spot.frame}.` : `: frame ${spot.frame}, since the preview has no frame ${pk.frame}.`), true);
  v.scrollIntoView({ block: "nearest" });
}

route();
