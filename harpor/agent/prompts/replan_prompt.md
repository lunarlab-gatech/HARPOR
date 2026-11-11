# Planning round: write a plan, then stop

The object in original/original.glb is split into parts, one mesh object per part in candidate.blend,
named after the part. This session runs in rounds, and each round is a new agent. An earlier round
gave the parts joints in candidate.blend and handed off a candidate, and the host checked it against
the requirements in input/parts.json. A finding of that check needs a new plan, not a small repair:
a part is missing, a joint has the wrong type, a part hangs on the wrong part, the joints are not a
valid tree, a moving part lies in another part at rest or cannot move at all, a soft part has no
support, does not dent or does not spring back, or most of the original surface is gone. The findings are below, and {evidence} holds
the details.

This round only plans, and every helper command is refused in it. candidate.blend is your work from
the last round: leave it as it is, since the next round builds on it. Read the findings,
input/parts.json (its seams and boxes, and its requirements), input/manifest.json, the photos and
the previews in original/previews/. Inspect candidate.blend with a Blender script if you need to. Then
write /work/plan.md and stop: for each finding, its cause and the change that fixes it, with the part,
its parent, the joint type and, for a hinge or slide, the origin and axis in W (computed from
input/parts.json) and the range. A new agent carries out the plan in the next round.

The requirements cannot be changed. A span target (min_span) is the part list's estimate of the
part's full travel, made from one photo. Set each range to what the part can really do: it must be
clear in the sweep and reachable in MuJoCo, so narrow a range that is blocked. A range that reaches
less than its target still meets the requirement, and is recorded as reduced, while it keeps a
fifth of the target and 10 degrees. Do not widen a range to reach a target through other parts. A joint must have the requirement's type.

Read /work/run_budget.json at launch. The budget is {budget_s:.0f} seconds, and the recorded deadline
is authoritative. The round ends when you stop or at the deadline.

{best}

The last check:

{feedback}
