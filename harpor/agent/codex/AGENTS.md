# Work folder and helper commands

This folder is `/work`, the only folder you can write to.

## Layout

- `input/manifest.json`: the frame W, the glTF axis mapping, the units, and one entry per input view
  with its camera (`K`, the OpenCV world-to-camera matrix `w2c`, `width`, `height`) and its files
  under `input/` (`image` and `mask`).
- `input/physics.json` (sessions with physics supervision): VoMP's material and mass estimates for
  the original, made before refinement. They are estimates, not ground truth.
- `original/original.glb`: the asset to refine. `original/previews/` has renders of it under each input camera
  (`view00_original.png`, ...) and from 6 orbit views, and a `sheet.png` of all of them.
- `candidate.blend`: the scene that the helpers read. It starts as `original/original.glb` loaded
  with Blender's glTF importer.
- `exports/`, `reviews/` and `checkpoints/`: written by the helpers.
- `run_budget.json`: the deadline as Unix seconds (`deadline`, compare with `date +%s`) and as ISO
  time. Time spent in `export_glb` does not count, and the deadline moves by that time, up to 600 s
  in total. In a session with rounds it also gives the `round` (1 to 4) and its
  `kind` (`build`, `repair` or `replan`).
- `evidence/round_NN.json` (sessions with rounds): the host's check of the candidate of round NN,
  with the physics judgment. `evidence/physics_<sha>/` holds the MuJoCo frames it judged.
- `plan.md`: the plan that a planning round writes.

Shell commands run in `/work`. Blender 5.2 runs headless:
`blender -b candidate.blend --python-exit-code 1 --python edit.py` (the exit-code flag must come
before `--python`, or a failing script still exits 0). Save the scene with
`bpy.ops.wm.save_as_mainfile(filepath="/work/candidate.blend")`.

## Rounds

A session with physics supervision runs in rounds, up to four, each with a new agent that knows
only its prompt and this folder. The first round builds. After each build the host compares the
candidate's surface with the original's, lets it settle under gravity in MuJoCo with the VoMP
materials, and asks a vision model to judge the frames against the photos. While it finds
something to fix, a repair round follows (at most two), or once a planning round (`replan`), which
only writes `plan.md` and in which every helper is refused. Each round has its own budget and
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
