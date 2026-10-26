// The demos page (demos.html): a gallery of cards, each a link to its demo's section (#demo-<name>), which style.css
// shows by :target. When a demo opens, on load or by its link, its 360 video plays muted and looping unless the system
// asks for reduced motion, as the job page's 360 video does (app.js renderSpin); the other videos pause, the demo's
// heading takes the focus, and the tab's title names the demo. Back in the gallery, the card of the demo just left
// takes it. The first pass waits for pageshow: until load the browser has not set :target, so the section is still
// hidden and its heading cannot take the focus; pageshow also comes back from the back/forward cache. Without this
// script the cards still open the demos, and a video's controls play it.
"use strict";

const TITLE = document.title;  // the gallery's
let shown = null;  // the demo open before this change

function showDemo() {
  const id = location.hash.slice(1);
  const t = id ? document.getElementById(id) : null;
  const open = t && t.classList.contains("demo") ? t : null;
  const still = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  for (const v of document.querySelectorAll(".demo video")) {
    if (open && open.contains(v) && !still) {
      v.muted = true;
      v.loop = true;
      const p = v.play();
      if (p && typeof p.catch === "function") p.catch(() => {});  // a refused play leaves the controls to play it
    } else {
      v.pause();
    }
  }
  if (open) {
    const h = open.querySelector("h2");
    document.title = `${h.textContent} · ${TITLE}`;
    h.focus();
  } else {
    document.title = TITLE;
    const back = shown && document.querySelector(`.gallery a[href="#${shown.id}"]`);
    if (back) back.focus();
  }
  shown = open;
}

addEventListener("pageshow", showDemo);
addEventListener("hashchange", showDemo);
