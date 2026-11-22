# Work folder and helper commands

This folder is `/work`, the only folder you can write to.

## Layout

- `input/manifest.json`: the frame W, the glTF axis mapping, the units, and one entry per input view
  with its camera (`K`, the OpenCV world-to-camera matrix `w2c`, `width`, `height`) and its files
  under `input/` (`image` and `mask`).
- `original/original.glb`: the asset to refine. `original/previews/` has renders of it under each input camera
  (`view00_original.png`, ...) and from 6 orbit views, and a `sheet.png` of all of them.
- `candidate.blend`: the scene that the helpers read. It starts as `original/original.glb` loaded
  with Blender's glTF importer.
- `exports/`, `reviews/` and `checkpoints/`: written by the helpers.
- `run_budget.json`: the deadline as Unix seconds (`deadline`, compare with `date +%s`) and as ISO
  time. Time spent in `export_glb` does not count, and the deadline moves by that time, up to 600 s
  in total. In a session with rounds it also gives the `round` (1 to 3) and its
  `kind` (`build` or `repair`).
- `evidence/round_NN.json` (sessions with rounds): the host's check of the candidate of round NN.

Shell commands run in `/work`. Blender 5.2 runs headless:
`blender -b candidate.blend --python-exit-code 1 --python edit.py` (the exit-code flag must come
before `--python`, or a failing script still exits 0). Save the scene with
`bpy.ops.wm.save_as_mainfile(filepath="/work/candidate.blend")`.

## Rounds

A session with rounds runs up to three of them, each with a new agent that knows only its prompt
and this folder. The first round builds. After each build the host compares the candidate's surface
with the original's and checks the hand-off. While it finds
something to fix, a repair round follows (at most two). Each round has its own budget and
deadline. `candidate.blend` and the other files carry over from one round to the next. The host
keeps the hand-off of every build and repair and delivers the best of them, so a worse repair does
not replace an earlier one; a hand-off whose round did not pass the host's check is delivered as
`needs_review`.

## Helper commands

- `export_glb [--blend PATH]`: export the visible objects of `candidate.blend` (or of a .blend
  inside `/work`) to `exports/NNN.glb` and re-import it to check it. Prints the GLB path, its
  sha256, mesh, material, image and triangle counts, the deadline and any warnings. An unchanged
  scene returns the previous export,
  unless its file in `exports/` changed. The host keeps its own copy of each export, which the other
  helpers read; `checkpoint` and `finish` refuse a GLB whose file in `exports/` changed.
- `review [--clay] [--detail X Y Z R DX DY DZ ...]`: export the scene, then render the original and
  the candidate with the same cameras and lights: every input view, cropped to the object, and 6
  orbit views. Prints a receipt ID, `sheet.png` (one row per view: photo, original, candidate,
  overlay) and `review.json` (bounding boxes, center shift, extent ratio, and mask IoU and crop
  window per input view). Single renders are in the same folder. Look at the PNGs with `view_image`.
  `--clay` renders every surface opaque gray. `--detail` adds a close-up camera on the sphere
  with center X Y Z and radius R, looking along DX DY DZ, all in Blender world coordinates; give
  it up to 4 times.
- `checkpoint --receipt ID --defect TEXT`: keep a read-only copy of a reviewed GLB in
  `checkpoints/`, labeled `needs_review` with the open defect. The receipt comes from `review`
  (`r...`).
- `finish --receipt ID --verdict VERDICT --reason TEXT [--limitation TEXT ...]`: hand off the
  reviewed GLB and end the session, or the round. Verdicts: `complete`, `unchanged` or
  `needs_review`. Give `--limitation` up to 3 times. The host keeps the reason, each limitation and a checkpoint's
  defect as one line of plain text: line breaks become spaces, and control characters are dropped.

Every helper but `finish` ends its output with the seconds left before the deadline, also when it
fails, but not after a usage error.

## LiDAR geometry (only when `input/lidar/manifest.json` exists)

A job with measured LiDAR returns has `input/lidar/manifest.json` (its `mode` is `inspect` or `sculpt`) and
`input/lidar/fitting.npz` (the fitting returns: `origin` and `end` in W, meters, and the photo `pixel`). The host keeps
other views back and checks every hand-off on them; you never see those.

- `original/previews/lidar/summary.json`: the repair regions (`L1`, `L2`, ...) with `status` (`target`, `conflict`
  or `observation`), `seed`, `normal` and `bounds` in W, the median signed error in meters, `direction` and
  `suggested_op`; the surfaces without returns; any alignment notice. `lidar_viewNN.png` draws the returns on the
  photos: red, the mesh sticks out toward the sensor; blue, it lies behind the measured surface; a cross, a miss.
- Every `review` adds `lidar.json` and `lidar_viewNN.png` to its folder, with each region's current fit, and prints a
  line per region.
- `lidar_sculpt --receipt ID --patch L1 --op plane|push [--strength S]` (mode `sculpt` only): one bounded edit of
  `/work/candidate.blend` from that review's correspondences. `plane` flattens a flat surface onto the measured
  plane; `push` moves a curved surface along its normals by the measured offset. The rim, seams, UVs, materials and
  object transforms stay. It refuses a scene that changed since the review, a region that is not a `target`, an
  ambiguous or too coarse patch, and a result that flips faces. It keeps the reviewed scene in
  `sculpt/<receipt>-before.blend`; copy that back over `candidate.blend` to undo.

Example: `review` (receipt `r002-...`, "lidar L1 target ... median signed -9.8 mm; suggested plane"), then
`lidar_sculpt --receipt r002-... --patch L1 --op plane`, `review` again (L1 near 0 mm), `checkpoint`, and `finish`
with a reason that names the measured repair (L1) apart from the repairs inferred from the photos. Do not sculpt a
`conflict` or `observation` region, or for an alignment notice. In mode `inspect` the helper is off and the LiDAR
evidence only informs your judgment.
