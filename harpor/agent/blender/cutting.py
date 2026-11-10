"""The cut-and-cap code shared by split_parts.py and add_part.py (build_plan.md note 11).

The scripts import it after putting their own folder first on sys.path, as they import render_common. A
cut edge is an edge with 1 in an int edge layer (gator_seam): an edge between faces of different parts.
After the faces of the other parts are deleted, the cut edges are boundary edges, and cap() closes the
holes they bound. A cap face has material 0, flat shading and one UV for all its corners, which is how
harpor.agent.surface tells a cap from the object's own surface. Only bmesh and numpy.
"""
# Untested: needs Blender 5.2, which the test environment does not have.

import bmesh
import numpy as np


def orient(bm, f, caps):
    """Reverse f when it runs along a shared edge in the same direction as its non-cap neighbour."""
    for loop in f.loops:
        for g in loop.edge.link_faces:
            if g is not f and g not in caps:
                other = next(x for x in g.loops if x.edge == loop.edge)
                if other.vert == loop.vert:
                    bmesh.ops.reverse_faces(bm, faces=[f])
                return


def loops_of(edges):
    """The closed chains of vertices that the given boundary edges form; a chain that does not close is left out."""
    nbr = {}
    for e in edges:
        a, b = e.verts
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    seen, loops = set(), []
    for start in nbr:
        if start in seen or len(nbr[start]) != 2:
            continue
        chain, prev, v = [start], None, start
        while True:
            seen.add(v)
            nxt = [w for w in nbr[v] if w is not prev]
            if not nxt or len(nbr[v]) != 2:
                chain = None
                break
            prev, v = v, nxt[0]
            if v is start:
                break
            chain.append(v)
        if chain and len(chain) >= 3:
            loops.append(chain)
    return loops


def fill(bm, sl):
    """Close every hole bounded by cut edges. Blender 5.2's holes_fill and edgenet_fill leave some simple loops
    open that edgeloop_fill closes (seen on a real MV-SAM3D chair), so edgeloop_fill comes first, then
    holes_fill, then a fan of triangles from a new vertex at the loop's centre. Returns (faces, methods)."""
    faces, methods = [], []
    for name, op in (("edgeloop_fill", lambda es: bmesh.ops.edgeloop_fill(bm, edges=es)["faces"]),
                     ("holes_fill", lambda es: bmesh.ops.holes_fill(bm, edges=es, sides=0)["faces"])):
        es = [e for e in bm.edges if e.is_boundary and e[sl]]
        if not es:
            break
        try:
            new = op(es)
        except Exception:  # an edge set that the operator cannot take; the next one may
            new = []
        if new:
            faces += new
            methods.append(name)
    for chain in loops_of([e for e in bm.edges if e.is_boundary and e[sl]]):
        c = bm.verts.new(np.mean([v.co for v in chain], 0).tolist())
        for a, b in zip(chain, chain[1:] + chain[:1]):
            try:
                faces.append(bm.faces.new((a, b, c)))
            except ValueError:  # the face exists already
                pass
        methods.append("fan")
    return faces, methods


def mean_uv(f, uv, caps):
    acc = [tuple(lp[uv].uv) for v in f.verts for lp in v.link_loops if lp.face is not f and lp.face not in caps]
    if acc:
        m = np.mean(acc, 0).tolist()
        for lp in f.loops:
            lp[uv].uv = m


def cap(bm, sl):
    """Close the holes that the cut edges of bm bound and make the new faces caps: wound against their
    neighbours, material 0, flat, one mean UV each, and a fan from the centre of every cap with more than 3
    corners (BEAUTY triangulation of a bent cut loop left zero-area slivers). Returns (holes, methods): the
    faces that fill made, before the fans, and the ways that made them."""
    caps, methods = fill(bm, sl)
    capset = set(caps)
    uv = bm.loops.layers.uv.active
    for f in caps:
        orient(bm, f, capset)
        f.material_index, f.smooth = 0, False
        if uv is not None:
            mean_uv(f, uv, capset)
    ngons = [f for f in caps if len(f.verts) > 3]
    if ngons:
        bmesh.ops.poke(bm, faces=ngons, center_mode="MEAN")
    return len(caps), methods
