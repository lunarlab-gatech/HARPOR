# Work folder, tools and helper commands

You work in the folder `/work`. It is the only folder you can write to. The task is in the first user
message.

## Tools

- `bash`: one shell that stays open for the whole session. It starts in `/work`. A directory change
  or a variable you set in one command still holds in the next command. The output combines stdout
  and stderr. A non-zero exit code is shown at the end of the output. Output longer than 30000
  characters is cut in the middle. One command may run for at most 300 s. After 300 s the shell is
  killed and a new shell starts in `/work`. Commands get no input, so do not start programs that wait
  for input. The shell has no network: downloads and package installs fail. `{"restart": true}`
  starts a new shell.
- `str_replace_based_edit_tool`: view, create and edit text files in `/work`. Give a path as
  `/work/...` or relative to `/work`. `view` shows a file with line numbers (`view_range` picks
  lines) or lists a folder two levels deep. `create` writes a whole file. `str_replace` replaces text
  that occurs exactly once in the file. `insert` adds text after line `insert_line` (0 is the top of
  the file). This tool does not open binary files.
- `view_image`: look at a PNG or JPEG file in `/work`, for example a review sheet, a single render or
  an input photo. An image larger than 2576 px on its long side is shown downscaled. Use it for every
  image you need to judge. The other tools cannot show images.

## Layout

- `input/manifest.json`: the mode (`refinement`, `from_scratch` or `articulation`), the frame W, the
  glTF axis mapping, the units, and one entry per input view with its camera (`K`, the OpenCV
  world-to-camera matrix `w2c`, `width`, `height`) and its files under `input/` (`image`, `mask`, and
  `depth` and `confidence` when present).
- `input/parts.json` (articulation only): the parts of the split object, each with its parent, joint
  guess and boxes, and the seams where parts touch, with straight lines fitted to them.
- `original/original.glb`: the asset to refine (refinement), or the object split into one mesh object
  per part (articulation). `original/previews/` has renders of it under each input camera
  (`view00_original.png`, ...) and from 6 orbit views, and a `sheet.png` of all of them.
- `candidate.blend`: the scene that the helpers read. In refinement and articulation it starts as
  `original/original.glb` loaded with Blender's glTF importer. From scratch, you create it.
- `exports/`, `reviews/`, `reviews_articulation/`, `set_joint/` and `checkpoints/`: written by the
  helpers. `joints.json`: the joints of `candidate.blend` after the last `set_joint`.
- `run_budget.json`: the deadline of the session, as Unix seconds (`deadline`, compare it with
  `date +%s`) and as ISO time. Time spent in `export_glb` does not count, and the deadline moves by
  that time, up to 600 s in total. Read the file again after an export.

Blender 5.2 runs headless in the shell:
`blender -b candidate.blend --python-exit-code 1 --python edit.py`. The exit-code flag must come
before `--python`, or a failing script still exits 0. Save the scene with
`bpy.ops.wm.save_as_mainfile(filepath="/work/candidate.blend")`.

## Helper commands

Run these in the bash tool by name.

- `export_glb [--blend PATH]`: export the visible objects of `candidate.blend` (or of a .blend
  inside `/work`) to `exports/NNN.glb` and re-import it to check it. Prints the GLB path, its
  sha256, mesh, material, image and triangle counts, the deadline and any warnings; in articulation
  also the mesh node names and the number of joints. An unchanged scene returns the previous export.
- `review [--clay] [--detail X Y Z R DX DY DZ ...]`: export the scene, then render the original and
  the candidate with the same cameras and lights: every input view, cropped to the object, and 6
  orbit views. Prints a receipt ID, `sheet.png` (one row per view: photo, original, candidate,
  overlay) and `review.json` (bounding boxes, center shift, extent ratio, and mask IoU and crop
  window per input view). Single renders are in the same folder. Look at the PNGs with
  `view_image`. `--clay` renders every surface opaque gray. `--detail` adds a close-up camera on the
  sphere with center X Y Z and radius R, looking along DX DY DZ, all in Blender world coordinates;
  give it up to 4 times.
- `set_joint --child NAME --parent NAME --type revolute|prismatic --origin X Y Z --axis X Y Z --lower L
  --upper U`, `set_joint --child NAME --parent NAME|world --type fixed` and
  `set_joint --child NAME --remove` (articulation only): set or remove the joint of one part object in
  `candidate.blend`, in one Blender run that saves the scene. Coordinates are Blender world
  coordinates (W). The axis is a unit vector, and `lower <= 0 <= upper` with `lower < upper`, in
  radians or meters. Only a fixed joint takes the parent `world`. The helper then checks the joints
  of the whole scene and undoes a change that adds a problem, such as a cycle or a second root.
  Prints the joints (also in `joints.json`), the root and the parts without a joint.
- `review_articulation [--states K] [--clay]` (articulation only): export the scene, then move every
  joint from lower to upper in K steps (default 3, from 2 to 5), all joints at once. Renders each
  state under every input view, cropped to the object, and 2 orbit views, and checks each moving
  part for penetration. Prints a receipt ID (`a...`), `sheet.png` (one row per view: each input
  view with the photo, then the states; the 2 orbit views with the states) and `report.json` (per
  joint and state: the share of the moving part's surface samples that lie inside another part,
  the largest depth among them, and the part that holds the most of them). It needs a revolute or
  prismatic joint; when no joint moves, `review` gives the receipt.
- `checkpoint --receipt ID --defect TEXT`: keep a read-only copy of a reviewed GLB in
  `checkpoints/`, labeled `needs_review` with the open defect. The receipt comes from `review`
  (`r...`) or from `review_articulation` (`a...`).
- `finish --receipt ID --verdict VERDICT --reason TEXT [--limitation TEXT ...]`: hand off the
  reviewed GLB and end the session. Verdicts for refinement: `complete`, `unchanged` or
  `needs_review`. Verdicts from scratch: `reconstruction_complete` or `needs_review`. Verdicts for
  articulation: `articulated`, `unchanged` or `needs_review`; `articulated` needs a GLB with joints
  that `review_articulation` reviewed. Give `--limitation` up to 3 times. The session ends right after
  a successful `finish`.

Every helper but `finish` ends its output with the seconds left before the deadline, also when it
fails, but not after a usage error.

## Measured times

From 6 earlier Claude Opus 5.5 sessions of this task (a round table and an office chair) on a PACE
ICE node with an L40S GPU, with a 600 s budget:

- A Blender script that inspects or edits the scene: 2 to 5 s.
- `export_glb`: about 5 s, which does not count against the budget.
- `review`: its export, then 27 to 30 s of rendering (median 29 s), which counts.
- One model turn, thinking included: median 4 s. 9 in 10 turns took less than 15 s.
- From the end of one review to the end of the next, with the edits, the checks and the next review:
  47 to 74 s, median 51 s.

The articulation helpers count against the budget, except the export inside `review_articulation`,
which does not count (as in `review`). These times come from one Claude Opus 5.5 session of the
articulation mode on PACE ICE: an office chair with 5 input views, with a 900 s budget.

- `set_joint`: 4.9 s, one Blender run that opens and saves the scene. A refused change takes a second
  run to undo it.
- `review_articulation --states 5`: 53 s in all, for 35 renders (5 states under the 5 input views and
  2 orbit views). Its export took 5.4 s of the 53 s, and that export does not count.
