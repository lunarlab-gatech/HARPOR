# Articulation: give the moving parts their joints, then hand off

The object in original/original.glb is already split into parts. Each part is one mesh object in
candidate.blend, named after the part. Give every part that can move a joint: its type, its axis and
origin, and its range. Make sure each moving part can move through its range without passing through
the other parts. Then hand off the reviewed GLB. The deliverable is that GLB, with the joints inside it.
It is not a report. Finish when the joints are reviewed and correct, not when the clock runs out.

## Inputs

- input/parts.json describes the split. parts has one entry per part: name (the object's name),
  parent (the part it is attached to; the base comes first and has none), joint_guess (revolute,
  prismatic, fixed or unknown, from the part list), axis_hint, faces, confidence, and two boxes in W:
  aabb (min, max, center, extent) and obb (center, axes as rows with the longest first, extents along
  those axes, and the 8 corners). The base's joint_guess is always fixed; it does not mean a fixed
  base. Only entries with status found have an object. The others have name null and come last.
  seams has one entry per pair of parts that touch: the edge count, length and center of the cut, up
  to 4 straight lines fitted to it (p0, p1, a unit direction, length, inlier_share and the mean normal
  of each side along that line), and the mean normal of each side along the whole seam. The seam lines
  are the candidates for hinge axes.
- input/manifest.json, the photos and masks under input/, and the previews of the split object in
  original/previews/.
- candidate.blend starts as original/original.glb loaded with Blender's glTF importer. Blender world
  coordinates are frame W: x right, y down and z forward, as seen from the camera of view 0. Every
  coordinate in parts.json is in W, and set_joint takes W too.

Read /work/run_budget.json at launch. The budget is 900 seconds, and the recorded deadline is
authoritative. It is a ceiling, not a target. The budget counts reasoning, edits, set_joint and the
review renders; export_glb does not count. Save a packed .blend after each geometry change and let the
helpers export it.

## Rules carried over from refinement

Keep the native origin, orientation and numeric scale. Never move, rotate, scale or rename a part
object: the joints name the objects, and the parts as they are now are the rest pose (every joint at 0)
that the photos show. Keep every part visible, because hidden objects are not exported. Batch
compatible edits. Before another revision of a reviewed candidate, call checkpoint on its receipt with
the open defect. If a change makes the result worse, go back to the earlier reviewed candidate. Do not
write new validation, rendering or report scripts; use the helpers.

## Joints

Use set_joint for every joint. It stores the joint in candidate.blend as the object's custom property
gator_joint, and export_glb puts it in the GLB. Do not write that property yourself. When you join
objects, make the part's object the active one, so that its name and its joint stay. A copy of a part
keeps the part's gator_joint; remove it from the copy with set_joint --child COPY --remove.

Your Blender scripts open candidate.blend and save it again. To go back to a reviewed candidate, start
an empty scene (bpy.ops.wm.read_factory_settings(use_empty=True)), import its GLB from checkpoints/ or
exports/, pack the images (bpy.ops.file.pack_all()) and save the scene as /work/candidate.blend. The
importer brings back each part's gator_joint; check the joint count with export_glb. Never import a GLB
into a scene that still holds the parts: the new objects get names such as lid.001, and the joints name
the old ones.

- Compute every origin and axis in Python from input/parts.json: the seam lines and the boxes. Never
  guess a coordinate and never read one off an image.
- A door turns about the seam line on its hinged side: origin = that line's p0, axis = its direction.
  The hinge is on the side away from the handle; the photos show which side that is.
- A lid turns about the seam line along its back edge. The back is the side away from the latch and
  from the front of the object; the photos show which side that is. A laptop screen turns about the
  seam line where it meets the base.
- A drawer front is a thin panel. It slides along the axis of its obb with the smallest extent (the
  last row of axes), pointing out of the body (away from the body's center). Its upper limit is the
  depth it can travel, at most about 0.8 of the body's extent along that axis. Do not take the slide
  axis from the seam's mean normals: along a closed seam loop they cancel out, and what is left can
  point anywhere. The normals of a single seam line are a better check.
- Choose the sign so that the part opens away from the body. For a revolute joint with origin o and
  axis a, let r be the part's center minus o, and n the unit vector from the body's center toward the
  part's center. If (a x r) . n < 0, use -a. If the photos show the part closed, the range is
  [0, upper]. If they show it open, as with an open laptop, the way back to closed is negative: set
  lower to minus the angle (or distance) from the photographed pose to closed, and upper to how much
  further it opens (0 if it cannot open further).
- The range must hold 0: lower <= 0 <= upper, in radians (revolute) or meters (prismatic), with
  lower < upper. A door or lid usually opens 1.5 to 2 rad; use what the photos and the object allow.
- A part that cannot move gets a fixed joint to the part it is attached to. Give the base
  --parent world --type fixed only when the object is fixed in place (built in, or fixed to a wall or a
  floor). A free-standing object keeps a free base: leave the base without a joint.
- Set the joints from the base outward: first the parts attached to the base, then the parts attached
  to those parts. A joint whose parent is neither the base nor a part with a joint makes that parent a
  second root. set_joint refuses that once another joint is set, and it prints the root, which must be
  the base. It also refuses a cycle. A refused change is undone.

## Geometry for the joints

Change the geometry only where the joints need it. A drawer is only its front panel after the split,
and the body has a flat cap where the drawer was cut out: a fan of faces inside the drawer's seam
loop. To let the drawer slide as a real drawer, cut a cavity into the body and keep the body closed.
In bmesh: delete the body's cap faces inside the drawer's seam loop, extrude the boundary edges of the
hole (bmesh.ops.extrude_edge_only), move the new vertices into the body, against the slide axis, by the
depth the drawer travels, close the back with bmesh.ops.holes_fill, and recalculate the normals
(bmesh.ops.recalc_face_normals). Then build a tray with five faces (open at the top) behind the
drawer front, a little smaller than the opening, and join it to the drawer object. A new object
parented to a part's object also moves with that part in every export, so the tray may be its own
object parented to the drawer, without a joint. Give new faces the material of the part they belong
to. Faces that the split put in the wrong part may be moved to the right part's object. Do not refine
textures or shapes elsewhere.

## Review

Run review_articulation after the joints are set. It needs at least one revolute or prismatic joint.
It exports the scene and moves every joint from lower to upper (all joints at once), renders each
state under every input camera and 2 orbit views, and reports, per moving part and state, the share of
its surface samples that lie inside another part, the largest depth among them, and the part that
holds the most of them. Penetration shallower than 0.5% of the object's size (or a quarter of a thin
part's thickness) is not counted, so touching cut faces do not show up. A part that passes through a
thin wall of a larger part may show only in the renders. The check needs closed meshes:
review_articulation warns when a part is not closed, and the numbers of the parts that move into it
are then uncertain. So keep the body closed. Inspect the sheet with the photos. Fix any penetration: a
hinge on the wrong side, a sign that swings the part into the body, a range that is too large, or a
drawer without a cavity. Review again after the fix.

## Hand-off

Call finish with a review receipt, a verdict, one short reason and at most three short limitations.
That is the final action. Verdicts:

- articulated: every part that can move has a joint, and review_articulation showed it moving
  correctly, without penetration. Use the receipt of that review (a...). finish refuses articulated for
  a GLB without joints or without an articulation review.
- unchanged: no part can move, or the split is unusable. The split object is delivered without joints.
- needs_review: some joints are missing or wrong. Name the open defect.

When no joint moves (unchanged, or needs_review without a moving joint), review_articulation has
nothing to render: run review and finish with its receipt (r...).

Do not mark unresolved work as articulated. If the session ends without finish, the controller
delivers your newest checkpoint, or else your newest export, as needs_review.
