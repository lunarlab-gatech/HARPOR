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

- `input/manifest.json`: the mode (`refinement` or `articulation`), the frame W, the glTF axis
  mapping, the units, and one entry per input view with its camera (`K`, the OpenCV world-to-camera
  matrix `w2c`, `width`, `height`) and its files under `input/` (`image` and `mask`).
- `input/parts.json` (articulation only): the parts of the split object, each with its parent, joint
  guess and boxes, and the seams where parts touch, with straight lines fitted to them. Its
  `requirements`, when it has them, say what the parts must do; the host checks the candidate
  against them, and they cannot be changed.
- `original/original.glb`: the asset to refine (refinement), or the object split into one mesh object
  per part (articulation). `original/previews/` has renders of it under each input camera
  (`view00_original.png`, ...) and from 6 orbit views, and a `sheet.png` of all of them.
- `candidate.blend`: the scene that the helpers read. It starts as `original/original.glb` loaded
  with Blender's glTF importer.
- `exports/`, `reviews/`, `reviews_articulation/`, `set_joint/`, `simulate/` and `checkpoints/`:
  written by the helpers. `joints.json`: the joints of `candidate.blend` after the last `set_joint`.
- `run_budget.json`: the deadline of the session, or of the round, as Unix seconds (`deadline`,
  compare it with `date +%s`) and as ISO time. Time spent in `export_glb` and `simulate` does not
  count: the deadline moves by that time, up to 600 s in total. Read the file again after an export.
  In a session with rounds it also gives the `round` (1 to 4) and its `kind` (`build`, `repair` or
  `replan`).
- `evidence/round_NN.json` (sessions with rounds): the host's check of the candidate of round NN.
- `plan.md`: the plan that a planning round writes.

Blender 5.2 runs headless in the shell:
`blender -b candidate.blend --python-exit-code 1 --python edit.py`. The exit-code flag must come
before `--python`, or a failing script still exits 0. Save the scene with
`bpy.ops.wm.save_as_mainfile(filepath="/work/candidate.blend")`.

## Rounds

An articulation whose `input/parts.json` has requirements of parts that the split found runs in
rounds, up to four, each with a new agent that knows only its prompt and this folder. The first
round builds. After each build the host checks the candidate, and while it finds something to
fix, a repair round follows (at most two), or once a planning round (`replan`), which only writes
`plan.md` and in which every helper is refused. Each round has its own budget and deadline.
`candidate.blend` and the other files carry over from one round to the next. The host keeps the
hand-off of every build and repair and delivers the best of them, so a worse repair does not
replace an earlier one; only a hand-off whose round passed the host's check is delivered as
`articulated`.

## Helper commands

Run these in the bash tool by name.

- `export_glb [--blend PATH]`: export the visible objects of `candidate.blend` (or of a .blend
  inside `/work`) to `exports/NNN.glb` and re-import it to check it. Prints the GLB path, its
  sha256, mesh, material, image and triangle counts, the deadline and any warnings; in articulation
  also the mesh node names and the number of joints. An unchanged scene returns the previous export,
  unless its file in `exports/` changed. The host keeps its own copy of each export, which the other
  helpers read; `checkpoint` and `finish` refuse a GLB whose file in `exports/` changed.
- `review [--clay] [--tint LINK ...] [--detail X Y Z R DX DY DZ ...]`: export the scene, then render the original and
  the candidate with the same cameras and lights: every input view, cropped to the object, and 6
  orbit views. Prints a receipt ID, `sheet.png` (one row per view: photo, original, candidate,
  overlay) and `review.json` (bounding boxes, center shift, extent ratio, and mask IoU and crop
  window per input view). Single renders are in the same folder. Look at the PNGs with
  `view_image`. `--clay` renders every surface opaque gray. `--detail` adds a close-up camera on the
  sphere with center X Y Z and radius R, looking along DX DY DZ, all in Blender world coordinates;
  give it up to 4 times.
  `--tint LINK` renders that mesh object of the candidate in one flat color, so that a part shows
  against the rest; give it up to 4 times, without `--clay`.
- `set_joint --child NAME --parent NAME --type revolute|prismatic --origin X Y Z --axis X Y Z --lower L
  --upper U`, `set_joint --child NAME --parent NAME|world --type fixed`,
  `set_joint --child NAME --parent NAME --type soft --material foam|gel|rubber` and
  `set_joint --child NAME --remove` (articulation only): set or remove the joint of one part object in
  `candidate.blend`, in one Blender run that saves the scene. Coordinates are Blender world
  coordinates (W). The axis is a unit vector, and `lower <= 0 <= upper` with `lower < upper`, in
  radians or meters. Only a fixed joint takes the parent `world`. A soft joint is for a part that
  dents under a press and springs back, and no part can hang from it. The helper then checks the
  joints of the whole scene and undoes a change that adds a problem, such as a cycle or a second root.
  Prints the joints (also in `joints.json`), the root and the parts without a joint.
- `add_part --name RESERVE [--source OBJECT] --faces FILE` and
  `add_part --name RESERVE [--source OBJECT] --plane PX PY PZ NX NY NZ [--plane ...] --seed X Y Z`
  (articulation only): create a part that the split did not find, by cutting its faces out of a part
  object in `candidate.blend`, in one Blender run that saves the scene. RESERVE is the `reserve` of a
  requirement in `input/parts.json` whose part is null; no other name is taken. `--source` defaults to
  that requirement's parent. `--faces` is a `.npy` file under `/work` with the polygon indices of the
  source object as the saved scene holds them (or one bool per polygon). `--plane`, up to 6 times,
  with `--seed`, takes the faces on the normal's side of every plane, and of those the connected piece
  nearest the seed; all in W. The new object keeps the UVs and materials, and the cut openings of both
  objects are closed. Prints the face counts, the part's share of the source's and of the object's
  area, its connected pieces and its box in W. It refuses an empty selection and one that holds the
  whole source. The scene from before the cut is kept as `add_part/NNN_before.blend`; copy it over
  `candidate.blend` to take the cut back.
- `move_faces --from HOLDER --to OWNER --faces FILE` and
  `move_faces --from HOLDER --to OWNER --plane PX PY PZ NX NY NZ [--plane ...] --seed X Y Z`
  (articulation only): move faces that the split put in the wrong part object to the object they
  belong to, in one Blender run that saves the scene. Both are part objects of `candidate.blend`, and
  either may be the body. The selection is `add_part`'s: a `.npy` file of polygon indices of HOLDER, or
  planes with a seed, in W. The two objects are cut and closed again as one surface, and both keep
  their names and joints. Prints the faces moved and, per object, its faces and connected pieces. It
  refuses an empty selection and one that holds all of HOLDER's surface. The scene from before is
  kept as `move_faces/NNN_before.blend`; copy it over `candidate.blend` to take the move back.
- `review_articulation [--states K] [--clay]` (articulation only): export the scene, then move every
  joint from lower to upper in K steps (default 3, from 2 to 5), all joints at once. Renders each
  state under every input view, cropped to the object, and 2 orbit views, and checks each moving
  part for penetration. Prints a receipt ID (`a...`), `sheet.png` (one row per view: each input
  view with the photo, then the states; the 2 orbit views with the states) and `report.json` (per
  joint and state: the share of the moving part's surface samples that lie inside another part,
  the largest depth among them, and the part that holds the most of them). It needs a revolute,
  prismatic or soft joint; when no joint moves and none is soft, `review` gives the receipt.
- `simulate` (articulation only): export the scene, then run the host's check of a candidate on it:
  the joints, the surface against `original/original.glb`, each hinge and slide swept through its
  range for clearance, and MuJoCo, which drives each hinge and slide through the range that passed
  and presses each soft part. Prints the outcome and scope, one line per requirement (its outcome,
  the range it achieved and the codes of its findings), the surface shares and the top findings;
  the whole report is in `simulate/sNNN-<sha>/report.json`. It runs outside the time budget, like
  `export_glb`, for at most 270 s with its export.
- `checkpoint --receipt ID --defect TEXT`: keep a read-only copy of a reviewed GLB in
  `checkpoints/`, labeled `needs_review` with the open defect. The receipt comes from `review`
  (`r...`) or from `review_articulation` (`a...`).
- `finish --receipt ID --verdict VERDICT --reason TEXT [--limitation TEXT ...]`: hand off the
  reviewed GLB and end the session, or the round. Verdicts for refinement: `complete`, `unchanged`
  or `needs_review`. Verdicts for articulation: `articulated`, `unchanged` or `needs_review`;
  `articulated` needs a GLB with joints that `review_articulation` reviewed and, when
  `input/parts.json` has requirements, a surface that the host's check does not block. Give
  `--limitation` up to 3 times. The session, or the round, ends right after a successful `finish`.
  The host keeps the reason, each limitation and a checkpoint's defect as one line of plain text:
  line breaks become spaces, and control characters are dropped.

Every helper but `finish` ends its output with the seconds left before the deadline, also when it
fails, but not after a usage error.

## Measured times

From 6 earlier sessions of this task on one data-center GPU, with a 600 s budget. Your machine may
differ.

- A Blender script that inspects or edits the scene: 2 to 5 s.
- `export_glb`: about 5 s, which does not count against the budget.
- `review`: its export, then 27 to 30 s of rendering (median 29 s), which counts.
- One model turn, thinking included: median 4 s. 9 in 10 turns took less than 15 s.
- From the end of one review to the end of the next, with the edits, the checks and the next review:
  47 to 74 s, median 51 s.

The articulation helpers count against the budget, except the export inside `review_articulation`,
which does not count (as in `review`), and `simulate`, which does not count at all. These times come
from one session of the articulation mode on the same GPU: an office chair with 5 input views, with
a 900 s budget.

- `set_joint`: 4.9 s, one Blender run that opens and saves the scene. A refused change takes a second
  run to undo it.
- `review_articulation --states 5`: 53 s in all, for 35 renders (5 states under the 5 input views and
  2 orbit views). Its export took 5.4 s of the 53 s, and that export does not count.
