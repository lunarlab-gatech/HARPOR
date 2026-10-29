# Asset-guided refinement: complete the important repairs, then hand off

Improve this reconstructed GLB against its supplied photographs. The deliverable
is the selected GLB, not a presentation, report, gallery or exhaustive audit.
Work efficiently:
finish when the important supported work is done, not when the clock runs out.

## Scope before scheduling

Inspect the useful photographs and supplied original previews. Identify the
highest-value geometry, texture, articulation and material improvements. Allocate
effort according to supported fidelity gains, not a fixed number of objectives.
A few concise notes suffice; do not write a lengthy plan,
sketch document, progress narrative or repeated interpretation of the photographs.
The existing mesh already gives proportions, orientation and attachment anchors.

Read work/run_budget.json at launch. The full ten-minute (600-second) work
ceiling is available immediately. The recorded deadline is authoritative. It is
an emergency ceiling, not a target duration, quality score, or invitation to
fill every available minute. There are no prescribed inspection/edit/freeze time
fractions. Finish early when the required repairs are resolved. Unused allowance
is not a reason to find more defects, add detail or perform another correction.

The budget counts reasoning, modeling, review renders and visual inspection.
GLB serialization through the controller helper, its import checks, and final
publication are outside this budget. Save a packed .blend after each material
improvement, then use the supplied export helper instead of exporting GLB inline
in a modeling script. The helper only serializes that frozen scene; it does not
perform extra modeling. Refresh run_budget.json after export since the wall-clock
deadline moves by the measured export duration. Export has a separate operational
timeout; do not keep issuing exports of unchanged scenes.

Batch compatible repairs. After a material candidate, use the supplied review
tool and inspect its images. There is no fixed limit on revisions or correction
passes. Autonomously choose edits, pose/appearance comparisons and detail work
according to expected fidelity gain, evidence, uncertainty and remaining time.
Small supported defects remain eligible for repair. If a correction regresses
the result, retain the earlier reviewed candidate and reassess the next action.

Stop immediately after selecting and inspecting the best available candidate if:

- Supported geometry, articulation and appearance goals are satisfied, with no
  worthwhile evidenced improvement left within the remaining work budget; or
- Remaining uncertainty cannot be resolved from the evidence, or another local
  attempt is unlikely to yield a worthwhile improvement. Use needs_review and
  state the actual remaining defect in this case.

Do not stop at a trivial shader change while a required structural repair is
feasible, or lower the quality target just to finish early. Decide how much
supported fine detail to recover within the budget; do not invent unreadable
lettering or unsupported features simply to consume unused time. If the original is
already appropriate, deliver it as unchanged after inspection; never fabricate
an improvement or silently label a failed repair as complete.

## Geometry: preserve reliable information, repair defective parts

Use the existing asset as the scaffold. Keep its native origin, orientation and
numeric scale, and preserve reliable geometry and appearance. Do not replace an
entire distinctive object with generic primitives, globally fit/align it, resize
it or floor-snap it. Blender's glTF axis conversion must be applied exactly once.

Complete missing external legs, supports, handles, rotors and appendages. Infer
occluded external structure from surviving contours, attachment sites, repetition,
symmetry and simple construction priors. An occluded part is not necessarily
absent; visible masks may omit it. Keep inferred details simple and mention the
uncertainty in the short handoff reason/limitations. Do not invent internals.

For a desk missing legs, retain its tabletop and tilted pose; build supports from
its underside anchors. Texture cleanup alone is insufficient. New parts may
expand its bounds. The review helper frames the union of original and candidate.

Replace a damaged local surface/component when patching cannot recover it.
Remove competing damaged faces, attach cleanly, and restore UVs/materials. Avoid
floating pieces, overlapping covers, closure of functional holes and flattening
whole components to hide damage. Preserve distinctive asymmetry where supported.

Strainers, whisks and baskets require actual wire gaps/perforations, not alpha
paint on a solid shell. Reuse reliable rims/handles/profile, remove erroneous
surfaces, and batch procedural wires at a useful coarse resolution. Use review
--clay to verify openings with fully opaque material. Do not remove the sieve.

## Texture and material repairs

Diagnose geometry, UV, base color, normal, roughness/metalness and baked-lighting
errors separately. A material scalar change does not restore recoverable detail.
If the atlas contains the detail, repair UV mapping. Otherwise use clean local
RGB correspondence to restore visible labels, color boundaries and patterns.
Check orientation and stretching. Do not invent unreadable text or brands.

Remove neighbor/background contamination locally, avoiding cloned repeated marks
or uniform paint replacing a distinctive surface. Preserve appropriate PBR maps;
give new parts connected compatible materials rather than white defaults.
Use supplied estimated cameras/depth as noisy evidence, not exact geometry.
Check camera conventions and use local anchors as needed, allocating comparison
effort by its expected benefit while preserving the original native frame.
Retain an earlier good repair if a later texture projection regresses.

## Bounded review and GLB handoff

Use the prebuilt review tool; it snapshots the candidate, checks import/materials,
reports native-frame anchor evidence, and produces matched diagnostic views and
a comparison image. Do not write new validation, rendering, panel or report
scripts. Inspect that comparison and relevant detail views. If a small important
repair is invisible at whole-object scale, supply a compact shared detail rig.
Anchor statistics are evidence, not automatic proof of pose or reconstruction
quality. Resolve concerns using unchanged regions and the supplied photographs.

Check remaining time before another correction. There is no fixed freeze window:
estimate the edit and required inspection cost, retain a valid checkpoint, and
leave enough time to select a candidate and commit the short handoff. Use
needs_review if required work remains. Do not reserve time for a report or an
extra render of unchanged bytes. Never claim inspection not done.

Once the chosen candidate passes your inspection, call finish with its review
receipt, a verdict, one short reason and at most three short limitations. That
is the final action. A needs_review handoff is preferable to exhausting the
clock writing documentation. Do not re-render unchanged files, inspect the same
image again, revise reports, emit long prose or perform further edits afterward.
The controller terminates the session, independently validates the exact selected
GLB, assigns the object ID, and publishes it. No agent-generated report or gallery
is required. Never mark unresolved required work complete.

Before another revision, call checkpoint on the inspected review with
needs_review and the outstanding defect. Keep that snapshot intact. If the worker
is interrupted, the controller will retain a valid checkpoint rather than lose
the object. A recovered uninspected export is explicitly needs_review. If no
edited candidate validates, an unchanged original is only a labeled fallback,
never a successful refinement. Make an edited, reviewable checkpoint early.
