# Work folder and helper commands

This folder is `/work`, the only folder you can write to.

## Layout

- `input/manifest.json`: the mode (`refinement` or `articulation`), the frame W, the glTF axis
  mapping, the units, and one entry per input view with its camera (`K`, the OpenCV world-to-camera
  matrix `w2c`, `width`, `height`) and its files under `input/` (`image` and `mask`).
- `input/parts.json` (articulation only): the parts of the split object, each with its parent, joint
  guess and boxes, and the seams where parts touch, with straight lines fitted to them.
- `original/original.glb`: the asset to refine (refinement), or the object split into one mesh object
  per part (articulation). `original/previews/` has renders of it under each input camera
  (`view00_original.png`, ...) and from 6 orbit views, and a `sheet.png` of all of them.
- `candidate.blend`: the scene that the helpers read. It starts as `original/original.glb` loaded
  with Blender's glTF importer.
- `exports/`, `reviews/`, `reviews_articulation/`, `set_joint/` and `checkpoints/`: written by the
  helpers. `joints.json`: the joints of `candidate.blend` after the last `set_joint`.
- `run_budget.json`: the deadline as Unix seconds (`deadline`, compare with `date +%s`) and as ISO
  time. Time spent in `export_glb` does not count, and the deadline moves by that time, up to
  600 s in total.

Shell commands run in `/work`. Blender 5.2 runs headless:
`blender -b candidate.blend --python-exit-code 1 --python edit.py` (the exit-code flag must come
before `--python`, or a failing script still exits 0). Save the scene with
`bpy.ops.wm.save_as_mainfile(filepath="/work/candidate.blend")`.

## Helper commands

- `export_glb [--blend PATH]`: export the visible objects of `candidate.blend` (or of a .blend
  inside `/work`) to `exports/NNN.glb` and re-import it to check it. Prints the GLB path, its
  sha256, mesh, material, image and triangle counts, the deadline and any warnings; in articulation
  also the mesh node names and the number of joints. An unchanged scene returns the previous export.
- `review [--clay] [--detail X Y Z R DX DY DZ ...]`: export the scene, then render the original and
  the candidate with the same cameras and lights: every input view, cropped to the object, and 6
  orbit views. Prints a receipt ID, `sheet.png` (one row per view: photo, original, candidate,
  overlay) and `review.json` (bounding boxes, center shift, extent ratio, and mask IoU and crop
  window per input view). Single renders are in the same folder. Look at the PNGs with `view_image`.
  `--clay` renders every surface opaque gray. `--detail` adds a close-up camera on the sphere
  with center X Y Z and radius R, looking along DX DY DZ, all in Blender world coordinates; give
  it up to 4 times.
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
  `needs_review`. Verdicts for articulation: `articulated`, `unchanged` or `needs_review`;
  `articulated` needs a GLB with joints that `review_articulation` reviewed. Give `--limitation` up
  to 3 times.

Every helper but `finish` ends its output with the seconds left before the deadline, also when it
fails, but not after a usage error.
