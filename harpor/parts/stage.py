"""The steps of the parts stage of an articulated job (build_plan.md note 11).

Runs in envs/sam3d from the repo root; J is the job folder:

    python -m harpor.parts.stage propose --job J [--key-file F] [--agent astra|opus]
    python -m harpor.parts.stage vote --job J
    python -m harpor.parts.stage split --job J --sif SIF [--blender CMD]
    python -m harpor.parts.stage finish --job J

  propose  harpor.parts.propose: run/proposal.json. The key file (OPENAI_API_KEY or ANTHROPIC_API_KEY)
           is needed only when the VLM lists the parts. harpor.stage_a.part_masks then runs in
           envs/sam3seg and writes run/parts/sam3.
  vote     harpor.parts.vote: run/parts/faces.npz and run/parts/vote.json.
  split    blender/split_parts.py in the agent's SIF: run/parts.glb and run/parts/split_report.json, log
           run/parts/split.log, at most 600 s. The container sees only J/run and the Blender scripts
           (as controller.blender_command sees only the session's work folder and two folders of
           its control/). --blender CMD
           (split like a shell) replaces the container and Blender, as GATOR_BLENDER does for the agent's
           helpers; the tests pass the fake Blender of harpor.agent.fake_agent.
  finish   run/parts.json, the stage's done marker, written last. parts.glb must pass controller.check_glb,
           and its mesh nodes must be the found parts, flat, without transforms. Per part: name, role,
           prompt, parent, joint_guess, axis_hint, faces, area_share, confidence, status, aabb and obb
           in W, visibility per view, views, cap_faces; parts that were not found or were dropped come
           last, with no name (status not_found, dropped_small or unconfirmed, and why, from vote.json).
           Per pair of adjacent parts the seam (geometry.seams): edges, length, center, up to 4 fitted
           lines (hinge-axis candidates) and the normals of both sides. The requirements, see below.

A part's parent is the part that its proposal parent names. When that name was found more than once
(two drawers), the one with the longest seam to the part is taken; when it was not found, its own
parent, up to the base.

The requirements of parts.json are the proposal's (harpor.parts.propose; a proposal older than them
gets propose.default_requirements) in the fields of paper Eq. 2: part and parent are U, action a,
observe O, test psi and scope D. A proposal requirement makes max(count, found) records, r1 on: one per
found link of its part, in the vote's order, then status not_found with part None for each one missing.
The first count are required; a found link beyond the count is not, so that an agent that joins a
fragment to its part does not miss a required link. A cluster that was not found makes no record. Each
record holds id, part, prompt, parent (a link; the proposal's parent name when not found), kind (joint
or soft), joint (unknown takes either moving joint), material (soft), text, action, observe, test (empty
without a span target), scope, source, status (found or not_found) and required. A revolute part's
range becomes its test min_span, in radians. A prismatic part's goes only into text, since a length
depends on the scale, which is an estimate without LiDAR.
"""

import argparse
import json
import shlex
import signal
import time
from pathlib import Path

import numpy as np

from harpor.fileio import write_json

SPLIT_TIMEOUT_S = 600.0
MIN_RESPONSE = 0.001  # of D: a soft part's smallest dent, harpor.physics.sim_check.RESPONSE


def _run_dir(job):
    return Path(job) / "run"


def propose(job, key_file=None, agent=None):
    from harpor.parts import propose as P

    return P.run(job, key_file, agent)


def vote(job):
    from harpor.parts import vote as V

    rec = V.run(_run_dir(job))
    found = [c for c in rec["clusters"] if c["status"] == "found"]
    print(f"vote: {rec['counts']['faces']} faces, {len(found)} parts ({', '.join(c['name'] for c in found)}) "
          f"in {rec['seconds']:.1f} s")
    for c in found:
        print(f"  {c['name']}: {c['faces']} faces, {c['area_share']:.1%} of the surface, confidence {c['confidence']:.2f}")
    lost = {}
    for c in rec["clusters"]:
        if c["status"] != "found":
            lost.setdefault(c["prompt"], []).append(c["status"])
    for prompt, statuses in lost.items():  # one line per prompt; vote.json has every cluster and why
        print(f"  {prompt!r}: " + ", ".join(f"{statuses.count(s)} {s}" for s in dict.fromkeys(statuses)))
    for w in rec["warnings"]:
        print(f"  warning: {w}")
    return rec


def split_command(job, sif):
    """Blender in the agent's SIF in a container that sees only J/run (at its host path) and the
    Blender scripts, read-only (controller.sif_blender)."""
    from harpor.agent import controller

    return controller.sif_blender(sif, rw=[_run_dir(job).resolve()])


def split(job, sif=None, blender=None, timeout=SPLIT_TIMEOUT_S):
    """Run blender/split_parts.py on posed.glb and faces.npz. blender: the command that stands for
    Blender (a list or a shell string), else split_command(job, sif)."""
    from harpor.agent import tools

    R = _run_dir(job).resolve()
    if blender is None and sif is None:
        raise SystemExit("split needs --sif or --blender")
    blender = shlex.split(blender) if isinstance(blender, str) else blender or split_command(job, sif)
    for f in (R / "parts.glb", R / "parts.json", R / "parts" / "split_report.json"):
        f.unlink(missing_ok=True)
    cmd = tools.blender_argv(blender, "split_parts.py", R / "posed.glb", R / "parts" / "faces.npz", R / "parts.glb",
                             R / "parts" / "split_report.json")
    t = time.time()
    with open(R / "parts" / "split.log", "w") as log:
        try:
            code = tools._run(cmd, log, timeout)
        except Exception as e:  # the timeout, or a Blender that does not start
            raise SystemExit(f"split failed: {e}; see {R / 'parts' / 'split.log'}") from None
    if code != 0 or not (R / "parts" / "split_report.json").is_file():
        raise SystemExit(f"split failed (exit {code}); see {R / 'parts' / 'split.log'}")
    rep = json.loads((R / "parts" / "split_report.json").read_text())
    print(f"split: {len(rep['names_again'])} parts in {time.time() - t:.0f} s, faces matched by {rep['mapping']}")
    for p in rep["parts"]:
        print(f"  {p['name']}: {p['faces']} faces, {p['cap_faces']} cap faces in {p['holes']} holes"
              + (f", {p['open_seam_edges']} seam edges left open" if p["open_seam_edges"] else ""))
    for w in rep["warnings"]:
        print(f"  warning: {w}")
    return rep


def mesh_nodes(glb):
    """[(name, node dict, whether it is a root node)] of the mesh nodes of a GLB."""
    from harpor.physics.align import read_glb

    doc, _ = read_glb(glb)
    scene = doc["scenes"][doc.get("scene", 0)]
    roots = set(scene.get("nodes", []))
    return [(n.get("name"), n, i in roots) for i, n in enumerate(doc.get("nodes", [])) if "mesh" in n]


def _identity(node):
    if "matrix" in node and not np.allclose(node["matrix"], np.eye(4).reshape(-1)):
        return False
    return (np.allclose(node.get("translation", [0, 0, 0]), 0) and np.allclose(node.get("rotation", [0, 0, 0, 1]), [0, 0, 0, 1])
            and np.allclose(node.get("scale", [1, 1, 1]), 1))


def parents(clusters, proposal, names, seams):
    """{part name: parent part name (None for the base)}; see the module docstring."""
    from harpor.stage_a.sam3_masks import slug

    by_prompt = {p["name"]: p for p in proposal["parts"]}
    length = {}
    for s in seams:
        a, b = s["parts"]
        length[a, b] = length[b, a] = s["length"]
    out = {}
    for name in names:
        c = clusters[name]
        if c["role"] == "base":
            out[name] = None
            continue
        p, out[name] = by_prompt.get(c["prompt"], {}).get("parent"), names[0]
        while p is not None:
            if p == proposal["base"] or slug(p) == names[0]:
                break
            cands = [n for n in names[1:] if clusters[n]["prompt"] == p and n != name]
            if cands:
                out[name] = max(cands, key=lambda n: (length.get((name, n), 0.0), -names.index(n)))
                break
            p = by_prompt.get(p, {}).get("parent")
    return out


def eq2_fields(req):
    """kind, joint, material (soft), text, action, observe, test and scope of a proposal requirement."""
    if req["joint"] == "soft":
        return {"kind": "soft", "joint": "soft", "material": req["material"], "text": req["text"],
                "action": {"type": "press", "toward": "support"},
                "observe": {"type": "node_and_visible_residual", "after": "rigid alignment"},
                "test": {"min_response_diag": MIN_RESPONSE, "recovery": True},
                "scope": {"start": "mesh pose", "base": "fixed", "support": "parent", "gravity": False}}
    span, text, test = req.get("range"), req["text"], {}
    if span and req["joint"] == "revolute":
        test["min_span"] = round(float(np.radians(span)), 3)
    elif span and req["joint"] == "prismatic":
        text += f" (full travel about {span:g} cm)"
    return {"kind": "joint", "joint": req["joint"], "text": text,
            "action": {"type": "sweep", "directions": ["lower", "upper"]},
            "observe": {"type": "joint_range", "relative_to": "parent"}, "test": test,
            "scope": {"start": "mesh pose, every joint at 0", "base": "fixed", "gravity": False}}


def requirements(proposal, clusters, names, parent):
    """The requirements of parts.json (see the module docstring). clusters: the found clusters by name;
    names: the found parts, the base first; parent: parents()."""
    from harpor.parts import propose as P

    by_prompt = {p["name"]: p for p in proposal["parts"]}
    reqs = proposal.get("requirements")
    out = []
    for r in P.default_requirements(proposal["parts"]) if reqs is None else reqs:
        links = [n for n in names[1:] if clusters[n]["prompt"] == r["part"]]
        count = by_prompt.get(r["part"], {}).get("count", 1)
        for k in range(max(count, len(links))):
            link = links[k] if k < len(links) else None
            out.append({"id": f"r{len(out) + 1}", "part": link, "prompt": r["part"],
                        "parent": parent[link] if link else by_prompt.get(r["part"], {}).get("parent"),
                        **eq2_fields(r), "source": r["source"], "status": "found" if link else "not_found",
                        "required": k < count})
    return out


def finish(job):
    """Write run/parts.json (see the module docstring). Returns it."""
    from harpor.agent import controller, tools
    from harpor.parts import geometry
    from harpor.parts import vote as V

    R = _run_dir(job)
    glb = R / "parts.glb"
    vote_rec = json.loads((R / "parts" / "vote.json").read_text())
    proposal = json.loads((R / "proposal.json").read_text())
    split_rec = json.loads((R / "parts" / "split_report.json").read_text())
    with np.load(R / "parts" / "faces.npz") as data:
        label, names = data["part"].astype(int), [str(n) for n in data["names"]]
    (R / "parts.json").unlink(missing_ok=True)
    why = controller.check_glb(glb)
    if why:
        raise SystemExit(f"parts.glb: {why}")
    nodes = mesh_nodes(glb)
    found = [n for k, n in enumerate(names) if (label == k).any()]
    have = sorted(str(n) for n, _, _ in nodes)  # an unnamed node is "None"
    if have != sorted(found):
        raise SystemExit(f"parts.glb has the mesh nodes {have}, not the parts {sorted(found)}")
    bad = [n for n, node, root in nodes if not root or not _identity(node)]
    if bad:
        raise SystemExit(f"parts.glb: the nodes {bad} are not flat root nodes without a transform")
    x_w, faces = V.load_mesh(R / "posed.glb")
    if len(faces) != len(label):
        raise SystemExit(f"faces.npz has {len(label)} faces, posed.glb {len(faces)}; run vote again")

    seams = geometry.seams(geometry.topology(x_w, faces), label, names)
    clusters = {c["name"]: c for c in vote_rec["clusters"] if c["status"] == "found"}
    by_prompt = {p["name"]: p for p in proposal["parts"]}
    caps = {p["name"]: p for p in split_rec.get("parts", [])}
    parent = parents(clusters, proposal, names, seams)
    parts = []
    for k, name in enumerate(names):
        c, f = clusters[name], label == k
        p = by_prompt.get(c["prompt"], {})
        pts = x_w[faces[f]].reshape(-1, 3)
        parts.append({"name": name, "role": c["role"], "prompt": c["prompt"], "parent": parent[name],
                      "joint_guess": "fixed" if c["role"] == "base" else p.get("joint", "unknown"),
                      "axis_hint": p.get("axis_hint"), "faces": int(f.sum()), "area_share": c["area_share"],
                      "confidence": c["confidence"], "status": "found",
                      "aabb": geometry.aabb(pts) if len(pts) else None, "obb": geometry.obb(pts) if len(pts) else None,
                      "visibility": c["visibility"], "views": c.get("views", []),
                      "cap_faces": caps.get(name, {}).get("cap_faces"),
                      "open_seam_edges": caps.get(name, {}).get("open_seam_edges")})
    for c in vote_rec["clusters"]:
        if c["status"] != "found":
            p = by_prompt.get(c["prompt"], {})
            parts.append({"name": None, "role": "part", "prompt": c["prompt"], "parent": p.get("parent"),
                          "joint_guess": p.get("joint", "unknown"), "axis_hint": p.get("axis_hint"), "faces": 0,
                          "status": c["status"], "why": c.get("why")})
    reqs = requirements(proposal, clusters, names, parent)
    doc = {"version": 1, "frame": "W", "glb": "parts.glb", "glb_sha256": tools.sha256(glb), "noun": proposal.get("noun"),
           "base": names[0], "source": proposal.get("source"), "agent": proposal.get("agent"),
           "model": proposal.get("model"), "parts": parts, "requirements": reqs, "seams": seams,
           "split": {k: split_rec.get(k) for k in ("mapping", "unmatched", "welded_vertices", "custom_normals_cleared")},
           "warnings": [*proposal.get("warnings", []), *vote_rec.get("warnings", []), *split_rec.get("warnings", [])]}
    write_json(R / "parts.json", doc)
    print(f"parts.json: {len(names)} parts ({', '.join(names)}), {len(seams)} seams")
    for p in parts:
        if p["status"] == "found" and p["parent"]:
            print(f"  {p['name']} > {p['parent']} : {p['joint_guess']}, confidence {p['confidence']:.2f}")
    for r in reqs:
        print(f"  {r['id']} {r['part'] or r['prompt']!r}: {r['kind']} {r['joint']}, {r['status']}"
              + ("" if r["required"] else ", not required")
              + (f", min span {r['test']['min_span']} rad" if "min_span" in r["test"] else ""))
    for s in seams:
        top = s["lines"][0] if s["lines"] else None
        print(f"  seam {s['parts'][0]}|{s['parts'][1]}: {s['edges']} edges, {s['length']:.3f} long, {len(s['lines'])} lines"
              + (f", the longest {top['length']:.3f} ({top['inlier_share']:.0%})" if top else ""))
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--job", required=True)
    p.add_argument("--key-file", help="env file with OPENAI_API_KEY=... (astra) or ANTHROPIC_API_KEY=... (opus)")
    p.add_argument("--agent", choices=("astra", "opus", "none"), help="default: the job's agent")
    v = sub.add_parser("vote")
    v.add_argument("--job", required=True)
    s = sub.add_parser("split")
    s.add_argument("--job", required=True)
    s.add_argument("--sif", help="the agent's SIF, with Blender")
    s.add_argument("--blender", help="a command that stands for Blender instead of the SIF (tests)")
    f = sub.add_parser("finish")
    f.add_argument("--job", required=True)
    a = ap.parse_args(argv)
    if a.step == "propose":
        propose(a.job, a.key_file, a.agent)
    elif a.step == "vote":
        vote(a.job)
    elif a.step == "split":
        from harpor.agent import tools

        signal.signal(signal.SIGTERM, tools._on_sigterm)  # Cancel kills Blender's process group too
        split(a.job, a.sif, a.blender)
    else:
        finish(a.job)


if __name__ == "__main__":
    main()
