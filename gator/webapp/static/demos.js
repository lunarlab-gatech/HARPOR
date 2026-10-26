// The demos page (demos.html): its videos start by themselves unless the system asks for reduced motion, as the job
// page's 360 video does (app.js renderSpin). Their controls still play them.
"use strict";

if (typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches) {
  for (const v of document.querySelectorAll("video[autoplay]")) {
    v.autoplay = false;
    v.pause();
  }
}
