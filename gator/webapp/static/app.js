// HARPOR web app UI: jobs list and upload, camera topic, frames and build settings (with LiDAR for the metric
// scale), stages and log, review of the views and of the part list (an articulated object), results.
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
  review: "Check the picked views, then Build.", done: "Download results.zip below, or look at the meshes.",
  failed: "Read the log of the failed stage, then Retry.", cancelled: "Retry continues from the first stage that is not done.",
  parts: "Check the part list, edit it if it is wrong, then Build with these parts.",  // a review of the part list
  lidar: "Upload a calibration, then Retry; or Build again with LiDAR off.",  // the lidar stage failed (a rosbag)
  lidar_cal: "Replace the calibration, then Retry; or Build again with LiDAR off.",  // ... that has a calibration
  // A Stray capture takes no calibration, and a retry reads the same depth and masks again.
  lidar_stray: "Build again with LiDAR off, or with other views. Retry helps only after a file or disk error.",
  // A failed or cancelled job whose scale ran with another calibration (V.scaleStale).
  stale: "The calibration changed since the scale was found. Retry starts again from the lidar stage." };
const KIND_LABEL = { video: "video", rosbag: "rosbag", stray: "Stray capture" };  // job.json source kind, in the jobs list
const OFFLINE = "Cannot reach the server. Is the SSH tunnel up?";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
// The open job page: id, job, meta, picks, view0, sel, ov (pending overrides), es, state, scaleStale (the LiDAR scale
// ran with another calibration) and lidarEdit (the build settings show in review for the calibration).
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
const size = (b) => b >= 1e9 ? (b / 1e9).toFixed(2) + " GB" : (b / 1e6).toFixed(b >= 1e8 ? 0 : 1) + " MB";
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
}

function show(id) {
  for (const x of ["list", "job"]) $("#" + x).hidden = x !== id;
}
function route() {
  closeStream();
  const m = location.hash.match(/^#\/job\/([\w-]+)$/);
  if (m) openJob(m[1]).then(() => { if (V.id === m[1]) $("#jtitle").focus({ preventScroll: true }); })
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

async function openJob(id) {
  show("job");
  V = { id, picks: [], view0: null };
  // The last job's, until refresh draws this one's.
  $("#review").hidden = $("#partsreview").hidden = $("#results").hidden = $("#lidarfail").hidden = true;
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
    stages: job.progress });
  // After a failure the stages and log come first; else the section to act on does.
  const runFirst = ["failed", "cancelled"].includes(job.status);
  if (runFirst !== V.runFirst) {
    V.runFirst = runFirst;
    if (runFirst) $("#topics").before($("#run")); else $("#job").append($("#run"));
    $("#log").scrollTop = $("#log").scrollHeight;  // a moved box starts at the top; the error lines are at the end
  }
  const analyzed = job.stages.frames.status === "done", rosbag = job.source && job.source.kind === "rosbag";
  // A rosbag job chooses its camera topic when it has several, after a failed frames run, or on "Change topic".
  const idle = !ACTIVE.includes(job.status) && !["review", "done"].includes(job.status);
  $("#topics").hidden = !rosbag || !job.stages.topics || job.stages.topics.status !== "done" || !(job.status === "choose_topic" ||
    (idle && (V.pickTopic || (["failed", "cancelled"].includes(job.status) && !analyzed))));
  if (!$("#topics").hidden) await loadTopics(job).catch((e) => { say($("#topicmsg"), e.message); });
  $("#tkeep").hidden = !(V.pickTopic && analyzed);
  // In review the settings show only after "Start over in manual mode" or "Change the calibration".
  $("#frames").hidden = !analyzed || ACTIVE.includes(job.status) || (job.status === "review" && !V.manual && !V.lidarEdit);
  if ($("#frames").hidden) V.framesFor = null;
  $("#topicrow").hidden = !rosbag;
  $("#topicname").textContent = rosbag ? job.source.topic : "";
  $("#chtopic").hidden = !idle || !$("#topics").hidden;
  if (!$("#frames").hidden) await settings(job);
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
  $("#results").hidden = ACTIVE.includes(job.status) || !job.files.includes("run/posed.glb");
  if (!$("#results").hidden) await renderResults(job);
  if (job.status === "failed" && !V.es && !$("#log").textContent) {
    const failed = job.progress.find((x) => x.status === "failed");
    if (failed) await showLog(failed.name);
  }
}
async function settings(job) {  // the frames and build settings, once they show
  const again = job.status !== "ready";
  $("#framesum").textContent = !again ? "Frames and build settings" : job.status === "review" ? "Start over with other settings"
    : "Build again with other settings";
  $("#start").textContent = !again ? "Start" : job.status === "review" ? "Start over" : "Build again";
  $("#rebuildhint").hidden = !again;
  rebuildHint();
  if (V.framesFor !== job.status) $("#frames").open = job.status === "ready" || !!V.manual;  // keep the user's toggle
  V.framesFor = job.status;
  await loadFrames();
}
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
$("#chtopic").onclick = () => { V.pickTopic = true; refresh(); };
$("#tkeep").onclick = () => { V.pickTopic = false; refresh(); };

// ---- 3. stages, log and the event stream ----

function renderState(st) {
  const changed = !V.state || V.state.status !== st.status;
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
  let next = ["failed", "cancelled"].includes(st.status) && from === null ? "Build again with the settings below."
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

  $("#run").hidden = !st.stages.some((x) => x.started || x.status !== "pending");
  if (changed) $("#run").open = ACTIVE.includes(st.status) || ["failed", "cancelled"].includes(st.status);
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
    // The agent without one, and the part stages of a rigid object.
    x.status === "skipped" && /^turned off/.test(x.note || "") ? badge("skipped", "skipped (turned off)") : badge(x.status),
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
setInterval(() => {  // the clocks of running stages, without drawing the list again
  if (!V.state || !ACTIVE.includes(V.state.status)) return;
  const now = Date.now() / 1000;
  for (const e of document.querySelectorAll("#stages .secs[data-started]:not([data-ended])"))
    e.textContent = fmt(Math.max(0, now - Number(e.dataset.started)));
}, 1000);

const quiet = (text) => text.replace(/^PROGRESS [\d.]+\r?\n/gm, "");  // the stage list shows progress already
function appendLog(text) {
  const log = $("#log"), atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  log.textContent = (log.textContent + quiet(text)).slice(-200000);
  if (atEnd) log.scrollTop = log.scrollHeight;
}
function openStream() {
  if (V.es) return;
  const id = V.id, es = V.es = new EventSource(`/api/jobs/${id}/events`);
  $("#logname").textContent = "live log (connecting)";
  V.logStage = null;
  $("#rawlog").hidden = true;
  es.onopen = () => { if (V.id === id) $("#logname").textContent = "live log"; };
  es.onerror = () => {
    if (V.id === id) $("#logname").textContent = es.readyState === 2 ? "live log lost: reload the page" : "live log (reconnecting)";
  };
  es.addEventListener("state", (e) => { if (V.id === id) renderState(JSON.parse(e.data)); });
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
  lidarToggle();
  sparkline(m);
  renderFrames();
}
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
  idle: "This bag was read before the app listed LiDAR topics. To list them, choose its camera topic again with Change topic.",
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
    V.manual = V.lidarEdit = false;
    await refresh();
  } catch (err) { say($("#startmsg"), err.message); }
  $("#start").disabled = false;
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
  // A bag's calibration can change in review: the build settings show for it, without a start over.
  job.kind === "rosbag" && job.lidar ? h("p", {}, h("button", { class: "small secondary", type: "button",
    onclick: editCalibration }, "Change the calibration")) : null);
}
async function editCalibration() {
  V.lidarEdit = true;
  $("#frames").hidden = false;
  await settings(V.job).catch((e) => say($("#revmsg"), e.message));
  toLidarSettings();
}
// The job failed at the lidar stage (stages: job.progress or the state's stages).
const lidarFailed = (status, stages) => status === "failed"
  && ((stages || []).find((x) => x.name === "lidar") || {}).status === "failed";
async function renderLidarFail(job) {
  const rec = job.files.includes("pool/lidar.json") ? lidarCompact(await api(file("pool/lidar.json"))) : null;
  fill($("#lidarfailbody"), rec ? lidarView(rec)
    : h("p", { class: "hint" }, "The stage stopped before it wrote its record. The reason is above, and its log is below."),
  h("p", {}, h("button", { class: "small secondary", type: "button", onclick: toLidarSettings }, "Go to the LiDAR settings")));
}
function toLidarSettings() {  // the LiDAR box in the build settings: the calibration upload, or LiDAR off
  $("#frames").open = true;
  $("#lidarbox").scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
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
$("#manualstart").onclick = async () => {
  V.manual = true;
  V.meta = null;
  await refresh().catch((e) => say($("#revmsg"), e.message));
  $("#frames").open = $("#gridbox").open = true;
  $("#frames > summary").focus({ preventScroll: true });
  $("#frames").scrollIntoView({ block: "start", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
};
$("#reselect").onclick = () => act([$("#reselect"), $("#build")], $("#revmsg"), "selecting the views again",
  () => post(`/api/jobs/${V.id}/overrides`, { ...V.ov, api_key: $("#key2").value || null }),
  () => { $("#key2").value = ""; $("#log").textContent = ""; });
$("#build").onclick = () => act([$("#build"), $("#reselect")], $("#revmsg"), "checking the key and queuing the build",
  () => post(`/api/jobs/${V.id}/continue`, { api_key: $("#key2").value || null }),
  () => { $("#key2").value = ""; $("#log").textContent = ""; V.lidarEdit = false; });

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

async function renderResults(job) {
  const has = (p) => job.files.includes(p), json = async (p) => has(p) ? (await fetch(file(p))).json() : null;
  const cm = (x) => (100 * x).toFixed(0);
  const viewer = (p, title) => {
    const cap = h("figcaption", {}, title), fig = h("figure", {}, cap);
    fig.append(h("model-viewer", { src: file(p), "camera-controls": true, "environment-image": "neutral", "shadow-intensity": "1",
      "interaction-prompt": "none", alt: title,
      onload: (e) => {  // the size of the model, in the GLB's meters
        const d = e.target.getDimensions && e.target.getDimensions();
        if (d) cap.textContent = `${title}: ${cm(d.x)} x ${cm(d.y)} x ${cm(d.z)} cm`;
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
    : a.joints_delivered > 0 ? "Parts with joints" + by : "Agent result without joints" + by) : null;
  const split = has("run/parts.glb") ? viewer("run/parts.glb", "Parts (split, before rigging)") : null;
  $("#viewers").classList.toggle("single", !refined && !split);
  fill($("#viewers"), viewer("run/posed.glb", "Stage A (MV-SAM3D, posed)"), split, refined,
    !refined && off ? h("p", { class: "hint" }, notRefined) : null);
  const [place, phys, mj, parts, pr, lid] = await Promise.all([json("run/placement.json"), json("physics/summary.json"),
    art ? json("physics/mjcf/mjcf.json") : null, art ? json("run/parts.json") : null,
    art && has("run/parts.json") ? api(`/api/jobs/${V.id}/parts`).catch(() => null) : null, json("pool/lidar.json")]);
  const rows = [], num = (x, d = 3) => x == null ? "?" : Number(x).toPrecision(d);
  if (place) rows.push(["Metric scale", metricScale(place, lid, job)]);
  if (place) rows.push(["Stage A mean mask IoU", num(place.check && place.check.mean_mask_iou)]);
  const lost = parts ? (parts.parts || []).filter((x) => x.status !== "found") : [];  // not_found, dropped_small, unconfirmed
  if (lost.length) rows.push(["Parts not split off", lost.map((x) => `${x.prompt} (${x.why || x.status.replace("_", " ")})`).join("; ")]);
  if (job.stages.agent.status === "skipped") rows.push(["Agent", off ? notRefined : "skipped: " + job.stages.agent.note]);
  else if (a) {
    if (a.model) rows.push(["Agent", `${ran || a.agent} (${a.model})`]);
    rows.push(["Agent verdict", `${rigged && VERDICT[a.verdict] || a.verdict} (${a.source})`], ["Agent reason", a.reason || ""]);
    for (const l of a.limitations || []) rows.push(["Limitation", l]);
    if (rigged) rows.push(["Joints", a.joints_delivered != null ? String(a.joints_delivered) : "?"]);
    if (rigged && a.counts && a.counts.joints != null) rows.push(["Joint changes by the agent", String(a.counts.joints)]);
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
      ? `turns on ${j.parent}, ${deg(j.range[0])} to ${deg(j.range[1])} deg` : j.type === "prismatic"
        ? `slides on ${j.parent}, ${cmr(j.range[0])} to ${cmr(j.range[1])} cm` : `fixed to ${j.parent}`]);
    for (const [k, to] of Object.entries(mj.welded || {})) rows.push([`Joint of ${k}`, `none, so it is welded to ${to}`]);
  } else if (mj && art) rows.push(["Links", "1: the parts have no joints, so physics welds them into one body"]);
  if (phys) {
    const whole = mj && mj.links ? mj.inertial : null;  // every link, also one that VoMP could not voxelize
    const rb = whole || phys.rigid_body || {}, md = phys.median || {}, skipped = Object.keys(phys.links_skipped || {});
    rows.push(["Mass", num(rb.mass) + " kg"], ["Center of mass", (rb.com || []).map((x) => num(x)).join(", ") + " m"]);
    if (whole) rows.push(["Link masses", Object.entries(mj.links).map(([k, l]) => `${k} ${num(l.inertial.mass)} kg` +
      (l.source === "hull" ? " (from its collision hulls)" : "")).join(", ")]);
    else if (skipped.length) rows.push(["Mass note", `VoMP could not voxelize ${skipped.join(", ")}, so the mass covers only ` +
      `${(phys.rigid_body_covers || []).join(", ")}.`]);
    rows.push(["Young's modulus (median)", num(md.youngs_modulus_pa) + " Pa"], ["Poisson ratio (median)", num(md.poisson_ratio)],
      ["Density (median)", num(md.density_kg_m3) + " kg/m3"]);
  }
  $("#facts2").replaceChildren(h("dl", {}, ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])));
  const checks = (pr && pr.checks) || [];  // run/parts/sam3/check/<view>.jpg
  fill($("#partchecks"), checks.length ? h("details", { class: "sub" }, h("summary", {}, `Part masks in the views (${checks.length})`),
    h("p", { class: "hint" }, "Each part's SAM 3 masks in its own color. Open one to see it larger."),
    h("div", { class: "checks" }, ...checks.map((c) => h("a", { href: file(c), target: "_blank", rel: "noopener" },
      h("img", { src: file(c), loading: "lazy", alt: "part masks in view " + /(\d+)\.jpg$/.exec(c)[1] }))))) : null);
  const link = (p, label, name, primary) => has(p) ? [h("a", { href: file(p), download: name, class: primary ? "dl primary" : "dl" }, label)] : [];
  fill($("#downloads"), ...link("results.zip", "results.zip", `${job.id}_results.zip`, true),
    ...link("run/posed.glb", "posed.glb", "posed.glb"), ...link("run/parts.glb", "parts.glb", "parts.glb"),
    ...link("refined.glb", "refined.glb", "refined.glb"), ...link("physics/aligned.glb", "aligned.glb", "aligned.glb"),
    ...link("physics/urdf/model.urdf", "model.urdf", "model.urdf"),
    has("results.zip") ? h("p", { class: "hint" }, "results.zip holds the MJCF, USD and URDF models, the GLB meshes, the physics summary, " +
      "the placement, the view selection" + (has("pool/lidar.json") ? ", the LiDAR scale (lidar.json)" : "") + " and the agent result" +
      (has("run/parts.glb") ? ", and the split parts (parts.glb, parts.json)" : "") +
      ". The URDF's meshes are only in results.zip, in urdf/meshes.") : null);
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

route();
