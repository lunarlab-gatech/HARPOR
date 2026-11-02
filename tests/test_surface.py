"""harpor.agent.surface on synthetic GLBs, and the refinement session's surface report.

The original is a flat sheet in two links, body (40 x 40 cells) and lid (40 x 10 cells beyond it), so every face
holds 1/4000 of the native area. Candidates are copies with some faces deleted, moved, re-cut, flipped, given other
UVs or another material.
"""

import copy
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageFilter

from harpor import frames
from harpor.agent import controller as C
from harpor.agent import surface as S
from harpor.agent.fake_agent import write_glb
from harpor.physics.align import read_glb, write_glb as write_doc

REPO = Path(__file__).resolve().parents[1]
FAKE_BLENDER = [sys.executable, "-m", "harpor.agent.fake_agent", "--blender"]
N = 40  # cells per side of body
FACE = 1 / 4000  # the native share of one face
D = float(np.hypot(1.0, 1.25))  # the sheet's diagonal


def pattern(side=64, phase=0.0):
    """A smooth image with detail at a 16-pixel period: JPEG keeps it, a blur does not."""
    y, x = np.mgrid[:side, :side]
    rgb = [128 + 100 * np.sin(2 * np.pi * x / 16 + phase), 128 + 100 * np.cos(2 * np.pi * y / 16),
           128 + 60 * np.sin(2 * np.pi * (x + y) / 32)]
    return Image.fromarray(np.stack(rgb, 2).round().astype(np.uint8))


def paint(name="paint", **changes):
    material = {"name": name, "baseColorTexture": pattern(), "normalTexture": pattern(phase=1.0),
                "metallicRoughnessTexture": pattern(32), "metallicFactor": 0.0, "roughnessFactor": 0.6}
    return trimesh.visual.material.PBRMaterial(**{**material, **changes})


def sheet(rows, z0, material, uv=True):
    """A flat grid of N x rows cells of side 1/N at glTF height 0, from z0, two faces per cell in row order, with
    UVs (x, z / 1.25)."""
    x, z = np.meshgrid(np.linspace(0, 1, N + 1), z0 + np.arange(rows + 1) / N)
    v = np.c_[x.ravel(), np.zeros(x.size), z.ravel()]
    k = np.arange(rows * N)
    a = k // N * (N + 1) + k % N
    f = np.stack([np.c_[a, a + N + 1, a + 1], np.c_[a + 1, a + N + 1, a + N + 2]], 1).reshape(-1, 3)
    return {"v": v, "f": f, "uv": np.c_[x.ravel(), z.ravel() / 1.25] if uv else None, "material": material}


def original(uv=True, material=None):
    material = material or paint()
    return {"body": sheet(N, 0.0, material, uv), "lid": sheet(N // 4, 1.0, material, uv)}


def write(path, parts):
    scene = trimesh.Scene()
    for name, p in parts.items():
        mesh = trimesh.Trimesh(p["v"], p["f"], process=False)
        mesh.visual = trimesh.visual.TextureVisuals(uv=p["uv"], material=p["material"])
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    scene.export(path, file_type="glb")
    return path


def first(part, k):
    """The mask of a part's first k faces: a strip along x from its first row."""
    mask = np.zeros(len(part["f"]), bool)
    mask[:k] = True
    return mask


def detach(part, mask, move=(0.0, 0.0, 0.0), uv_shift=0.0):
    """The part with the masked faces on their own copies of their vertices, moved and with shifted UVs."""
    used = np.unique(part["f"][mask])
    new = np.full(len(part["v"]), -1)
    new[used] = len(part["v"]) + np.arange(len(used))
    f = part["f"].copy()
    f[mask] = new[f[mask]]
    uv = None if part["uv"] is None else np.r_[part["uv"], part["uv"][used] + uv_shift]
    return {**part, "v": np.r_[part["v"], part["v"][used] + move], "f": f, "uv": uv}


def split(part, mask):
    """The part with each masked face cut into 3 at its centroid: the same surface on other faces."""
    f = part["f"][mask]
    c = len(part["v"]) + np.arange(len(f))
    v = np.r_[part["v"], part["v"][f].mean(1)]
    uv = None if part["uv"] is None else np.r_[part["uv"], part["uv"][f].mean(1)]
    cut = np.r_[np.c_[f[:, 0], f[:, 1], c], np.c_[f[:, 1], f[:, 2], c], np.c_[f[:, 2], f[:, 0], c]]
    return {**part, "v": v, "f": np.r_[part["f"][~mask], cut], "uv": uv}


def compare(tmp_path, cand, orig=None):
    o = write(tmp_path / "original.glb", orig or original())
    r = S.compare(write(tmp_path / "candidate.glb", cand), o)
    assert "error" not in r, r
    json.dumps(r, allow_nan=False)
    return r


def codes(r):
    return [f["code"] for f in r["findings"]]


def test_a_copy_retains_everything(tmp_path):
    r = compare(tmp_path, original())
    assert r["shares"] == {"retained": 1.0, "modified": 0.0, "uv_changed": 0.0, "material_changed": 0.0,
                           "flipped": 0.0, "deleted": 0.0, "unresolved": 0.0, "added": 0.0, "present": 1.0}
    assert r["native_area"] == pytest.approx(1.25) and r["caps_area"] == 0.0 and r["diagonal"] == pytest.approx(D)
    assert r["links"] == {"body": {"native": 1.0, "kept": 1.0}, "lid": {"native": 0.25, "kept": 0.25}}
    assert (r["findings"], r["lost_regions"], r["materials"], r["blocks"]) == ([], [], [], False)


def test_deleted_faces_and_their_regions_in_w(tmp_path):
    """Two strips go: 80 faces of body's first row and 40 of lid's last; boxes in W, largest first."""
    cand = original()
    body, lid = cand["body"], cand["lid"]
    body["f"] = body["f"][80:]
    lid["f"] = lid["f"][:-40]
    r = compare(tmp_path, cand)
    assert r["shares"]["deleted"] == pytest.approx(120 * FACE, abs=8 * FACE)
    assert r["shares"]["retained"] == pytest.approx(1 - 120 * FACE, abs=1e-6)
    assert codes(r) == ["surface_deleted"] and r["blocks"] is False  # ENFORCE is off
    big, small = r["lost_regions"]
    assert big["share"] > small["share"] > 0
    w = frames.W_FROM_GLTF  # x stays, glTF z is -y in W
    assert np.allclose(big["min"], w @ [0, 0, 1 / N], atol=1e-4) and np.allclose(big["max"], w @ [1, 0, 0], atol=1e-4)
    assert np.allclose(small["min"], w @ [0.5, 0, 1.25], atol=1e-4)
    assert r["links"]["body"]["kept"] == pytest.approx(1 - 80 / 3200, abs=8 / 3200)


def test_faces_cut_on_the_same_surface_are_modified_not_deleted(tmp_path):
    """The near-surface fallback: re-cut faces, or faces moved by 0.002 D, are still there; moved by 0.05 D they
    are gone."""
    cand = original()
    cand["body"] = split(cand["body"], first(cand["body"], 400))
    r = compare(tmp_path, cand)
    assert r["shares"]["modified"] == pytest.approx(400 * FACE) and r["shares"]["deleted"] == 0.0
    assert r["shares"]["added"] == pytest.approx(400 * FACE)
    assert r["shares"]["retained"] == pytest.approx(1 - 400 * FACE)
    assert codes(r) == [] and r["links"]["body"]["kept"] == pytest.approx(1.0)
    for lift, kind in ((0.002, "modified"), (0.05, "deleted")):
        cand = original()
        cand["body"] = detach(cand["body"], first(cand["body"], 400), move=(0.0, lift * D, 0.0))
        r = compare(tmp_path, cand)
        assert r["shares"][kind] == pytest.approx(400 * FACE, abs=4 * FACE), lift


@pytest.mark.parametrize("changes, code, share", [
    ({"uv": None}, "surface_uv", "uv_changed"),
    ({"material": paint(metallicFactor=1.0)}, "surface_material", "material_changed")])
def test_re_cut_faces_without_uvs_or_with_another_material_have_changed(tmp_path, changes, code, share):
    """The re-cut faces of body's first 400 as their own link, without UVs or with a metallic copy of the paint."""
    cand = original()
    body = split(cand["body"], first(cand["body"], 400))
    cand["body"], cand["cut"] = {**body, "f": body["f"][:-1200]}, {**body, "f": body["f"][-1200:], **changes}
    r = compare(tmp_path, cand)
    assert r["shares"]["modified"] == pytest.approx(400 * FACE) and r["shares"]["deleted"] == 0.0
    assert r["shares"][share] == pytest.approx(400 * FACE) and codes(r) == [code]


def behind(part, depth):
    """The part's far side: a copy depth below it that faces the other way, as the back of a thin panel."""
    return {**part, "v": part["v"] - [0.0, depth, 0.0], "f": part["f"][:, ::-1]}


def test_the_back_of_a_thin_panel_does_not_stand_in_for_its_front(tmp_path):
    """body is a panel 4 mm thick, under NEAR D (8 mm): its front deleted is deleted, though its back lies within
    NEAR D of it; its front re-cut is still there."""
    orig = original()
    orig["back"] = behind(orig["body"], 0.004)
    r = compare(tmp_path, {"back": orig["back"], "lid": orig["lid"]}, orig)
    assert r["shares"]["deleted"] == pytest.approx(1 / 2.25, abs=4 * FACE) and codes(r) == ["surface_deleted"]
    r = compare(tmp_path, {**orig, "body": split(orig["body"], first(orig["body"], 3200))}, orig)
    assert r["shares"]["deleted"] == 0.0 and r["shares"]["modified"] == pytest.approx(1 / 2.25, abs=1e-5)
    assert codes(r) == []


def test_uv_changes_on_matched_faces(tmp_path):
    cand = original()
    cand["body"] = detach(cand["body"], first(cand["body"], 80), uv_shift=0.01)
    r = compare(tmp_path, cand)
    assert r["shares"]["uv_changed"] == pytest.approx(80 * FACE) and r["shares"]["modified"] == pytest.approx(80 * FACE)
    assert codes(r) == ["surface_uv"]
    cand["body"] = detach(original()["body"], first(cand["body"], 80), uv_shift=5e-5)  # within UV_TOL
    assert compare(tmp_path, cand)["shares"]["retained"] == 1.0


def test_uvs_on_the_original_and_none_on_the_candidate_have_changed(tmp_path):
    cand = original()
    cand["lid"]["uv"] = None
    r = compare(tmp_path, cand)
    assert r["shares"]["uv_changed"] == pytest.approx(0.2) and r["shares"]["retained"] == pytest.approx(0.8)
    assert codes(r) == ["surface_uv"]


def test_flipped_faces(tmp_path):
    cand = original()
    cand["body"]["f"][:80] = cand["body"]["f"][:80, ::-1]
    r = compare(tmp_path, cand)
    assert r["shares"]["flipped"] == pytest.approx(80 * FACE)
    assert r["shares"]["retained"] == pytest.approx(1 - 80 * FACE)
    assert codes(r) == ["surface_flipped"]


def test_another_material_on_matched_faces(tmp_path):
    """The first row of body as its own link with a metallic copy of the paint."""
    cand = original()
    body = cand["body"]
    mask = first(body, 80)
    cand["row"] = {**body, "f": body["f"][mask], "material": paint(metallicFactor=1.0)}
    body["f"] = body["f"][~mask]
    r = compare(tmp_path, cand)
    assert r["shares"]["material_changed"] == pytest.approx(80 * FACE) and r["shares"]["added"] == 0.0
    assert codes(r) == ["surface_material"]
    assert r["materials"] == [{"original": "paint", "candidate": "paint", "why": ["metallicFactor [0.0] to [1.0]"],
                               "share": pytest.approx(80 * FACE)}]
    assert r["links"]["body"]["kept"] == pytest.approx(1 - 80 / 3200)  # its first row went to the link row


def patched(image):
    out = image.copy()
    out.paste((0, 0, 0), (20, 20, 29, 29))  # 2% of the image
    return out


@pytest.mark.parametrize("key, edit, why", [
    ("baseColorTexture", patched, "differs by"),
    ("baseColorTexture", lambda im: im.filter(ImageFilter.GaussianBlur(3)), "differs by"),
    ("baseColorTexture", lambda im: im.resize((32, 32)), "resized from 64x64 to 32x32"),
    ("normalTexture", patched, "differs by"),
    ("metallicRoughnessTexture", lambda im: im.resize((64, 64)), "resized from 32x32 to 64x64"),
    ("normalTexture", lambda im: None, "removed"),
])
def test_a_changed_image_changes_the_material(tmp_path, key, edit, why):
    material = paint()
    changed = paint(**{key: edit(getattr(material, key))})
    r = compare(tmp_path, original(material=changed), original(material=material))
    assert r["shares"]["material_changed"] == 1.0 and codes(r) == ["surface_material"]
    assert [w for m in r["materials"] for w in m["why"] if w.startswith(key)][0].startswith(f"{key} {why}")


def reencoded(path, image, fmt, **save):
    """Swap every image of a GLB for image encoded as fmt, under a new name."""
    doc, binary = read_glb(path)
    data = io.BytesIO()
    image.save(data, fmt, **save)
    for k in range(len(doc["images"])):
        binary += b"\0" * (-len(binary) % 4)
        doc["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data.getvalue())})
        binary += data.getvalue()
        doc["images"][k] = {"bufferView": len(doc["bufferViews"]) - 1, "mimeType": f"image/{fmt.lower()}",
                            "name": f"renamed_{k}"}
    doc["buffers"][0]["byteLength"] = len(binary)
    write_doc(path, doc, binary)
    return path


@pytest.mark.parametrize("fmt, save", [("JPEG", {"quality": 95}), ("PNG", {"compress_level": 1})])
def test_a_renamed_or_reencoded_texture_is_the_same_image(tmp_path, fmt, save):
    image = pattern()
    material = paint(baseColorTexture=image, normalTexture=image, metallicRoughnessTexture=image)
    o = write(tmp_path / "original.glb", original(material=material))
    c = reencoded(write(tmp_path / "candidate.glb", original(material=paint("paint.001", baseColorTexture=image))),
                  image, fmt, **save)
    loaded = S.read_surface(c).materials[0]
    assert loaded.name == "paint.001" and {i["mimeType"] for i in read_glb(c)[0]["images"]} == {f"image/{fmt.lower()}"}
    assert (np.asarray(loaded.baseColorTexture) != np.asarray(image)).any() == (fmt == "JPEG")
    r = S.compare(c, o)
    assert r["shares"]["retained"] == 1.0 and r["materials"] == [] and codes(r) == []
    if fmt == "JPEG":
        assert 0 < S.tile_difference(image, loaded.baseColorTexture) < S.IMAGE_TOL


def caps(part, k=4):
    """The part with a fan of k faces at its x = 1 edge, standing up, every corner at one UV (as split_parts
    gives its caps), two corners off by float noise."""
    c = len(part["v"])
    fan = np.array([[1.0, 0.0, 0.5], [1.0, 0.0, 0.6], [1.0, 0.1, 0.6], [1.0, 0.1, 0.5], [1.0, 0.05, 0.55]])
    f = np.array([[c + i, c + (i + 1) % 4, c + 4] for i in range(k)])
    uv = np.full((5, 2), 0.3)
    uv[1, 0] += 1e-7
    uv[4, 1] += 1e-7
    return {**part, "v": np.r_[part["v"], fan], "f": np.r_[part["f"], f], "uv": np.r_[part["uv"], uv]}


def test_caps_are_not_native(tmp_path):
    orig = original()
    orig["body"] = caps(orig["body"])
    r = compare(tmp_path, orig, orig)
    assert r["caps_area"] == pytest.approx(0.01) and r["native_area"] == pytest.approx(1.25)
    assert r["shares"]["retained"] == 1.0 and r["links"]["body"]["native"] == pytest.approx(1.0)
    cand = copy.deepcopy(orig)
    cand["body"]["f"] = cand["body"]["f"][:-4]  # the caps go
    assert compare(tmp_path, cand, orig)["shares"] == compare(tmp_path, orig, orig)["shares"]
    cand = copy.deepcopy(orig)
    cand["body"]["uv"][-5:] += 0.2  # and new UVs on them change nothing either
    assert compare(tmp_path, cand, orig)["shares"]["retained"] == 1.0
    cand["body"]["uv"][-6] += 0.2  # a native corner does
    assert compare(tmp_path, cand, orig)["shares"]["uv_changed"] > 0


def test_a_glb_without_uvs_has_no_caps_and_its_matched_faces_are_unresolved(tmp_path):
    material = trimesh.visual.material.PBRMaterial(name="plain", baseColorFactor=[200, 120, 60, 255])
    orig = original(uv=False, material=material)
    orig["body"] = {**caps({**orig["body"], "uv": np.zeros((len(orig["body"]["v"]), 2))}), "uv": None}
    r = compare(tmp_path, orig, orig)
    assert r["caps_area"] == 0.0 and r["native_area"] == pytest.approx(1.26)
    assert r["shares"]["unresolved"] == 1.0 and r["shares"]["retained"] == 0.0 and codes(r) == []
    unit = 1 / 3200 / 1.26  # the share of a cell's face
    cand = copy.deepcopy(orig)
    cand["body"]["f"] = cand["body"]["f"][120:]
    cand["lid"]["f"][:80] = cand["lid"]["f"][:80, ::-1]
    r = compare(tmp_path, cand, orig)
    assert r["shares"]["deleted"] == pytest.approx(120 * unit, abs=8 * unit)
    assert r["shares"]["flipped"] == pytest.approx(80 * unit, abs=1e-5)
    assert codes(r) == ["surface_deleted", "surface_flipped"]
    assert r["shares"]["unresolved"] == pytest.approx(1 - 200 * unit, abs=8 * unit)


def test_a_primitive_whose_uvs_are_one_point_has_no_uvs(tmp_path):
    """Blender's join gives an object without UVs a UV of 0 at every corner: lid's faces are native, not caps, so
    deleting lid is deleting a fifth of the surface."""
    orig = original()
    plain = trimesh.visual.material.PBRMaterial(name="plain", baseColorFactor=[200, 120, 60, 255])
    orig["lid"] = {**orig["lid"], "uv": np.zeros((len(orig["lid"]["v"]), 2)), "material": plain}
    r = compare(tmp_path, orig, orig)
    assert r["caps_area"] == 0.0 and r["native_area"] == pytest.approx(1.25)
    assert r["shares"]["unresolved"] == pytest.approx(0.2) and r["shares"]["retained"] == pytest.approx(0.8)
    r = compare(tmp_path, {"body": orig["body"]}, orig)
    assert r["shares"]["deleted"] == pytest.approx(0.2) and codes(r) == ["surface_deleted"]


def test_a_tray_added_as_its_own_link_does_not_count(tmp_path):
    tray = sheet(4, 0.4, paint("tray", baseColorFactor=[10, 200, 10, 255]))
    tray["v"] = tray["v"] + [0.0, -0.05, 0.0]
    r = compare(tmp_path, {**original(), "tray": tray})
    assert r["shares"]["added"] == pytest.approx(0.1 / 1.25) and r["shares"]["retained"] == 1.0
    assert codes(r) == [] and set(r["links"]) == {"body", "lid"}
    assert all(a["native"] == a["kept"] for a in r["links"].values())


def test_faces_moved_into_another_link_are_not_kept_there(tmp_path):
    cand = original()
    lid = cand["lid"]
    cand["body"] = {**cand["body"], "v": np.r_[cand["body"]["v"], lid["v"]], "uv": np.r_[cand["body"]["uv"], lid["uv"]],
                    "f": np.r_[cand["body"]["f"], lid["f"][:400] + len(cand["body"]["v"])]}
    lid["f"] = lid["f"][400:]
    r = compare(tmp_path, cand)
    assert r["shares"]["retained"] == 1.0 and codes(r) == []
    assert r["links"]["lid"] == {"native": 0.25, "kept": pytest.approx(0.125)}
    assert r["links"]["body"] == {"native": 1.0, "kept": 1.0}


def delete(cand, k):
    cand["body"]["f"] = cand["body"]["f"][k:]


def shift_uv(cand, k):
    cand["body"] = detach(cand["body"], first(cand["body"], k), uv_shift=0.01)


def recolor(cand, k):
    body = cand["body"]
    cand["row"] = {**body, "f": body["f"][:k], "material": paint(baseColorFactor=[255, 0, 0, 255])}
    body["f"] = body["f"][k:]


def flip(cand, k):
    cand["body"]["f"][:k] = cand["body"]["f"][:k, ::-1]


@pytest.mark.parametrize("edit, code, under, over", [
    (delete, "surface_deleted", 60, 120), (shift_uv, "surface_uv", 20, 80), (recolor, "surface_material", 20, 80),
    (flip, "surface_flipped", 20, 80), (delete, "surface_identity", 1800, 2200)])
def test_each_limit(tmp_path, edit, code, under, over):
    """Just under and well over each limit (one face is 1/4000)."""
    for k, found in ((under, False), (over, True)):
        cand = original()
        edit(cand, k)
        assert (code in codes(compare(tmp_path, cand))) == found, k


def test_any_failure_is_an_error_report(tmp_path, monkeypatch):
    o = write(tmp_path / "original.glb", original())
    (tmp_path / "garbage.glb").write_bytes(b"this is not a GLB file")
    for glb, orig in ((tmp_path / "garbage.glb", o), (tmp_path / "missing.glb", o), (o, tmp_path / "garbage.glb")):
        r = S.compare(glb, orig)
        assert set(r) == {"error"} and isinstance(r["error"], str), glb
    only_caps = {"body": caps({**sheet(1, 0.0, paint()), "f": np.zeros((0, 3), int)})}
    r = S.compare(o, write(tmp_path / "caps.glb", only_caps))
    assert r == {"error": "ValueError: the original has no native area"}
    monkeypatch.setattr(S, "read_surface", lambda glb: 1 / 0)
    assert S.compare(o, o) == {"error": "ZeroDivisionError: division by zero"}


def test_primitives_keep_their_images_undecoded_and_a_large_one_is_refused(tmp_path, monkeypatch):
    """load_s gives each primitive its material, not a copy, since a copy decodes the material's images again, once for
    every primitive and every node that uses them: a node that repeats body's mesh holds body's images, and none is
    decoded yet. An image of more than MAX_IMAGE_PIXELS pixels, read from its header, is an error of compare, and
    check_glb refuses its GLB."""
    from harpor.physics.mjcf import load_s

    o = write(tmp_path / "original.glb", original())
    doc, binary = read_glb(o)
    body = next(n for n in doc["nodes"] if n["name"] == "body")
    doc["nodes"].append({"name": "again", "mesh": body["mesh"], "translation": [0.0, 0.0, 5.0]})
    doc["scenes"][doc.get("scene", 0)]["nodes"].append(len(doc["nodes"]) - 1)
    write_doc(tmp_path / "twice.glb", doc, binary)
    loaded = {link: m.visual.material for link, _, m in load_s(tmp_path / "twice.glb", by_link=True)}
    assert sorted(loaded) == ["again", "body", "lid"] and loaded["again"] is loaded["body"]
    images = [getattr(mat, k) for mat in loaded.values() for k in S.IMAGES]
    assert all(im is not None and im.tile for im in images)  # PIL's tiles are still to decode
    assert C.check_glb(o) is None and S.large_image([]) is None
    monkeypatch.setattr(S, "MAX_IMAGE_PIXELS", 64 * 64 - 1)
    why = "its baseColorTexture is 64x64 pixels, over 4,095"
    assert S.compare(o, o) == {"error": f"ValueError: original.glb: {why}"}
    assert C.check_glb(o) == f"image too large: {why}"


def run_session(tmp_path, mode):
    """A session whose agent does nothing, so the original is delivered."""
    original = tmp_path / "in" / "original.glb"
    original.parent.mkdir(parents=True)
    write_glb(original, "parts" if mode == "articulation" else "good", 1.0)
    parts = None
    if mode == "articulation":
        parts = tmp_path / "in" / "parts.json"
        parts.write_text(json.dumps({"version": 1, "frame": "W", "glb": "parts.glb", "base": "body", "seams": [],
                                     "parts": []}))
    C.prepare_session(tmp_path / "s", mode, "p", [], source="s", metric=True, original=original, parts=parts)
    with tempfile.TemporaryDirectory(prefix="gator-") as sock_dir:
        r = C.run_session(tmp_path / "s", [sys.executable, "-c", "pass"], blender=FAKE_BLENDER, sock_dir=sock_dir,
                          budget_s=20.0, poll_s=0.02, kill_after_s=0.5, finish_wait_s=0.5)
    assert r == json.loads((tmp_path / "s" / "result.json").read_text())
    return r


def test_a_refinement_session_reports_the_surface(tmp_path, monkeypatch):
    """The fake original is a box without UVs, delivered unchanged; articulation has no surface key yet."""
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    r = run_session(tmp_path / "a", "refinement")
    assert r["source"] == "fallback_original" and r["surface"]["shares"]["unresolved"] == 1.0
    assert r["surface"]["findings"] == [] and r["surface"]["blocks"] is False
    assert "surface" not in run_session(tmp_path / "b", "articulation")


def test_a_failed_comparison_does_not_stop_the_session(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.setattr(S, "read_surface", lambda glb: 1 / 0)
    r = run_session(tmp_path, "refinement")
    assert r["surface"] == {"error": "ZeroDivisionError: division by zero"} and r["glb"] == "original.glb"
