"""gator/webapp/demos.py: the demos folder, its checks and private guard, the demos page, add from a finished fake
job, and the command line."""
import json
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("av")
pytest.importorskip("PIL")
from fastapi.testclient import TestClient  # noqa: E402
from test_webapp_server import ARTICULATED, TOKEN, login, run_app_js, uploaded, wait, write_video  # noqa: E402

from gator.webapp import demos, server, steps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OPUS = "sk-ant-test-" + secrets.token_hex(16)
PARTS = [{"name": "chair base"}, {"name": "seat", "parent": "chair base", "joint": "revolute"}]
SRC = (server.STATIC / "demos.html").read_text()


def glb(gltf, data=b""):
    """A GLB of the glTF JSON gltf and the binary chunk data."""
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js
    if data:
        data += b"\0" * (-len(data) % 4)
        body += struct.pack("<I4s", len(data), b"BIN\0") + data
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


TRIANGLE = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 36}],
            "bufferViews": [{"buffer": 0, "byteLength": 36}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0],
                           "max": [1, 1, 0]}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}], "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}], "scene": 0}
POINTS = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)


def hand_demo(folder, name="office-chair", title="Office chair", order=0, **spec):
    """A demo written by hand from small files: a short H.264 video, its poster, a one-triangle GLB and a zip with a
    text and a binary member. spec overrides demo.json's keys. Returns the demo's folder."""
    from PIL import Image

    d = Path(folder) / name
    d.mkdir(parents=True)
    write_video(d / "turntable.mp4", n=12, size=(64, 64))
    Image.new("RGB", (64, 64), (90, 90, 90)).save(d / "poster.jpg", "JPEG")
    (d / "aligned.glb").write_bytes(glb(TRIANGLE, POINTS))
    with zipfile.ZipFile(d / "results.zip", "w") as z:
        z.writestr("summary.json", json.dumps({"rigid_body": {"mass": 2.0}}))
        z.writestr("aligned.glb", glb(TRIANGLE, POINTS))
    j = {"version": 1, "title": title, "order": order, "video": "turntable.mp4", "poster": "poster.jpg",
         "model": "aligned.glb", "facts": [["Input", "video, 5 views"], ["Mass", "2.00 kg"]],
         "downloads": ["results.zip", "aligned.glb", "turntable.mp4"], **spec}
    (d / "demo.json").write_text(json.dumps(j))
    return d


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    """A finished fake job: an articulated office chair with the agent opus (so it has links and a joint) and the 360
    video. (job folder, the job as GET /api/jobs/{id} gives it, replies for app.js: its JSON files by URL)."""
    tmp = tmp_path_factory.mktemp("demojob")
    cfg = steps.Config(G=tmp / "G", runs=tmp / "runs", fake=True)
    with TestClient(server.make_app(cfg, TOKEN, key_dir=tmp / "keys")) as c:
        login(c)
        job_id = uploaded(c, tmp)
        r = c.post(f"/api/jobs/{job_id}/start", json={**ARTICULATED, "prompt": "office chair", "agent": "opus",
                                                      "api_key": OPUS, "parts_mode": "user", "parts": PARTS,
                                                      "turntable": True})
        assert r.status_code == 200, r.text
        view = wait(c, job_id, "done", "failed", timeout=180)
        assert view["status"] == "done", view["error"]
        replies = {f"/api/jobs/{job_id}/file/{p}": c.get(f"/api/jobs/{job_id}/file/{p}").json()
                   for p in view["files"] if p.endswith(".json")}
        replies[f"/api/jobs/{job_id}/parts"] = c.get(f"/api/jobs/{job_id}/parts").json()
    return cfg.runs / job_id, view, replies


@pytest.fixture
def made(job, tmp_path):
    """The demos folder with the demo that add made from the job."""
    out = tmp_path / "demos"
    demos.add(job[0], out)
    return out


def test_add_from_a_finished_job(job, made, tmp_path):
    """add writes demo.json, the 360 video, its first frame as the poster, aligned.glb and results.zip, named after the
    prompt, with the facts of the Results tab. No path of the machine is left in demo.json or in results.zip's text,
    and load takes the demo. A second add needs --force."""
    J, view, _ = job
    d = made / "office-chair"
    assert sorted(p.name for p in d.iterdir()) == ["aligned.glb", "demo.json", "poster.jpg", "results.zip", "turntable.mp4"]
    spec = json.loads((d / "demo.json").read_text())
    assert spec["title"] == "Office chair" and spec["video"] == "turntable.mp4" and spec["model"] == "aligned.glb"
    assert spec["downloads"] == ["results.zip", "aligned.glb", "turntable.mp4"]
    n = len(json.loads((J / "selection.json").read_text())["picks"])
    assert spec["facts"] == [["Input", f"video, {n} views"], ["Size", spec["facts"][1][1]], ["Mass", spec["facts"][2][1]],
                             ["Links", "2: chair_base, seat"], ["Joint of seat", spec["facts"][4][1]],
                             ["Agent", "Claude Opus 5.5"]]
    assert re.fullmatch(r"\d+ x \d+ x \d+ cm", spec["facts"][1][1]) and spec["facts"][2][1].endswith(" kg")
    assert spec["facts"][4][1].startswith("turns on chair_base, ")
    assert (d / "turntable.mp4").read_bytes() == (J / "turntable" / "turntable.mp4").read_bytes()
    assert (d / "aligned.glb").read_bytes() == (J / "physics" / "aligned.glb").read_bytes()
    assert (d / "poster.jpg").read_bytes()[:3] == b"\xff\xd8\xff"
    with zipfile.ZipFile(J / "results.zip") as a, zipfile.ZipFile(d / "results.zip") as b:
        assert a.namelist() == b.namelist()
        roots = {str(r).encode() for r in (J.parents[1], J.parents[1].resolve(), tmp_path, tmp_path.resolve())}
        assert any(r in a.read(n) for n in a.namelist() for r in roots)  # the job's records hold its paths
        for n in b.namelist():
            text = b.read(n)
            assert not any(r in text for r in roots), n
            assert not demos.MACHINE_BYTES.search(text) and OPUS.encode() not in text, n
    assert not demos.MACHINE_PATH.search((d / "demo.json").read_text()) and "clip" not in (d / "demo.json").read_text()
    [x] = demos.load(made)
    assert x.name == "office-chair" and x.sizes["results.zip"] == (d / "results.zip").stat().st_size
    with pytest.raises(demos.DemoError, match="exists; add --force"):
        demos.add(J, made)
    demos.add(J, made, title="Desk chair", facts=[("Mass", "about 7 kg"), ("Agent", ""), ("Note", "fake")], order=2,
              force=True)
    [x] = demos.load(made)
    assert x.title == "Desk chair" and x.order == 2 and ("Mass", "about 7 kg") in x.facts and ("Note", "fake") in x.facts
    assert "Agent" not in dict(x.facts) and sorted(p.name for p in made.iterdir()) == ["office-chair"]


def test_demo_facts_are_rows_of_the_results_tab(job):
    """app.js renderResults on the same job: each Mass, Links and Joint fact is one of its rows, and the Agent row
    starts with the Agent fact (it adds the model)."""
    J, view, replies = job
    out = run_app_js(replies, """
        (async () => {
          V = { id: """ + json.dumps(view["id"]) + """ };
          await renderResults(""" + json.dumps(view) + """);
          console.log(JSON.stringify(texts));
        })()
        """)
    rows = {out[i + 1]: out[i + 3] for i in range(len(out) - 3) if out[i] == "dt" and out[i + 2] == "dd"}
    facts = dict(demos.facts_of(J, json.loads((J / "job.json").read_text())))
    for label in ("Mass", "Links", "Joint of seat"):
        assert rows[label] == facts[label], label
    assert rows["Agent"].startswith(facts["Agent"] + " (")


def test_add_refuses_private_data(job, tmp_path):
    """A job whose input is table.MOV, a prompt that names a private scene, a job under a private folder, and a rosbag
    job without --public-bag are refused with the rule's reason, and nothing is written. A demo.json edited by hand to
    name a private scene stops load."""
    J = job[0]
    out = tmp_path / "demos"

    def copy(where, **edit):
        dst = tmp_path / where
        shutil.copytree(J, dst, symlinks=True)
        spec = json.loads((dst / "job.json").read_text())
        for k, v in edit.items():
            spec[k] = {**spec[k], **v} if isinstance(v, dict) else v
        (dst / "job.json").write_text(json.dumps(spec))
        return dst

    for src, why in [(copy("a", video={"name": "table.MOV"}), "the input's name: looks private (matches 'table.mov')"),
                     (copy("b", prompt="Pallet"), "the prompt: looks private (matches 'pallet')"),
                     (copy("gator_private/c"), "the source: looks private (matches 'gator_private')"),
                     (copy("d", source={"kind": "rosbag"}), "a rosbag job"),
                     (copy("e", status="review"), "the job is review, not done")]:
        with pytest.raises(demos.DemoError, match=re.escape(why)):
            demos.add(src, out)
        assert not out.exists() or not list(out.iterdir()), why
    with pytest.raises(demos.DemoError, match=re.escape("the prompt: looks private (matches 'cage')")):
        demos.add(copy("f", source={"kind": "rosbag"}, prompt="cage"), out, public_bag=True)  # names still count
    demos.add(copy("g", source={"kind": "rosbag"}), out, public_bag=True)
    spec = json.loads((out / "office-chair" / "demo.json").read_text())
    (out / "office-chair" / "demo.json").write_text(json.dumps({**spec, "title": "Cage"}))
    with pytest.raises(demos.DemoError, match=re.escape("office-chair/demo.json: looks private (matches 'cage')")):
        demos.load(out)
    shutil.rmtree((novideo := copy("h")) / "turntable")
    with pytest.raises(demos.DemoError, match="the job has no 360 video; start it with Make a 360 video on"):
        demos.add(novideo, tmp_path / "x")


def test_add_from_a_run_folder(job, tmp_path):
    """A run folder (no job.json) needs --video and --glb; the facts come from the records that are there."""
    J = job[0]
    run = tmp_path / "chair_real"
    shutil.copytree(J / "physics", run / "physics")
    shutil.copy(J / "turntable" / "turntable.mp4", run / "spin.mp4")
    with pytest.raises(demos.DemoError, match="has no job.json: give --video and --glb"):
        demos.add(run, tmp_path / "demos", video="spin.mp4")
    (run / "selection.json").write_text(json.dumps({"input": "table.MOV"}))  # the records count too
    with pytest.raises(demos.DemoError, match=re.escape("selection.json: looks private (matches 'table.mov')")):
        demos.add(run, tmp_path / "demos", video="spin.mp4", glb="physics/aligned.glb")
    (run / "selection.json").unlink()
    x = demos.add(run, tmp_path / "demos", video="spin.mp4", glb=str(run / "physics" / "aligned.glb"))
    assert (x.name, x.title, x.downloads) == ("chair-real", "Chair real", ("aligned.glb", "turntable.mp4"))
    assert dict(x.facts)["Size"].endswith(" cm") and "Input" not in dict(x.facts)
    with pytest.raises(demos.DemoError, match="cannot read"):
        demos.add(run, tmp_path / "demos", video="physics/aligned.glb", glb="physics/aligned.glb", name="bad")
    import av

    with av.open(str(run / "old.mp4"), "w", format="mp4") as c:  # MPEG-4 Part 2, which not every browser plays
        s = c.add_stream("mpeg4", rate=24)
        s.width, s.height, s.pix_fmt = 64, 64, "yuv420p"
        for i in range(4):
            for pk in s.encode(av.VideoFrame.from_ndarray(np.full((64, 64, 3), 40 * i, np.uint8), format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    with pytest.raises(demos.DemoError, match=r"old.mp4 is mpeg4 in yuv420p; a demo's video is h264 in yuv420p"):
        demos.add(run, tmp_path / "demos", video="old.mp4", glb="physics/aligned.glb", name="bad")
    assert sorted(p.name for p in (tmp_path / "demos").iterdir()) == ["chair-real"]


def test_load_rules(tmp_path, monkeypatch):
    """Each rule of a demo, one case each; .DS_Store files and no folder are fine, and a missing folder is an error."""
    assert demos.load(None) == []
    with pytest.raises(demos.DemoError, match=re.escape(f"no demos folder {tmp_path / 'none'}")):
        demos.load(tmp_path / "none")
    good = hand_demo(tmp_path / "base")
    (good.parent / ".DS_Store").write_bytes(b"\0")
    (good / ".DS_Store").write_bytes(b"\0")
    (good.parent / demos.PLACEHOLDERS).write_text('{"version": 1, "placeholders": ["Laptop"]}')  # not a demo
    [x] = demos.load(good.parent)
    assert x.files == ("turntable.mp4", "aligned.glb", "poster.jpg", "results.zip")
    n = 0

    def case(why, change=None, name="office-chair", **spec):
        nonlocal n
        n += 1
        folder = tmp_path / f"c{n}"
        d = hand_demo(folder, name=name, **spec)
        if change:
            change(d)
        with pytest.raises(demos.DemoError, match=re.escape(why)):
            demos.load(folder)

    def put(f, data):
        return lambda d: (d / f).write_bytes(data)

    def zipped(member, data):
        def change(d):
            with zipfile.ZipFile(d / "results.zip", "w") as z:
                z.writestr(member, data)
        return change

    def link(d):
        (d / "poster.jpg").unlink()
        (d / "poster.jpg").symlink_to(d / "turntable.mp4")

    case("Office_Chair: a demo's name is a-z, 0-9 and -", name="Office_Chair")
    case("office-chair: title is text of 1 to 60 characters", title="x" * 61)
    case("facts is a list of at most 8", facts=[["a", "b"]] * 9)
    case("unknown keys: color", color="red")
    case("office-chair/notes.zip is not listed in demo.json", put("notes.zip", b"PK\x03\x04"))
    case("office-chair/poster.jpg is missing, or not a regular file", link)
    case("office-chair/page.html: a demo file is named", put("page.html", b"<p>"), downloads=["page.html"])
    case("office-chair/poster.jpg is not a .jpg file inside", put("poster.jpg", b"GIF89a"))
    case("office-chair/turntable.mp4 is missing", lambda d: (d / "turntable.mp4").unlink())
    case("office-chair: video is the name of a .mp4 file", video=None)
    case("office-chair/aligned.glb uses KHR_draco_mesh_compression",
         put("aligned.glb", glb({**TRIANGLE, "extensionsUsed": ["KHR_draco_mesh_compression"]}, POINTS)))
    case("office-chair/aligned.glb points to files outside it (uri)",
         put("aligned.glb", glb({**TRIANGLE, "buffers": [{"byteLength": 36, "uri": "data.bin"}]})))
    case("office-chair/results.zip's a.bin holds a key shape", zipped("a.bin", b"\0sk-" + b"A" * 40 + b"\0"))
    case("office-chair/results.zip's mjcf.json holds a key shape or a machine path",
         zipped("mjcf.json", json.dumps({"mesh": "/storage/ice1/0/1/ali497/x.obj"})))
    case("office-chair/results.zip's mjcf.json: looks private (matches 'pallet')", zipped("mjcf.json", '{"name": "pallet"}'))
    case("office-chair/results.zip's mjcf.json holds a key shape or a machine path",
         zipped("mjcf.json", json.dumps({"mesh": "/storage/coda1/p-abc/0/ali497/x.obj"})))
    case("office-chair/aligned.glb: looks private (matches 'cage')",
         put("aligned.glb", glb({**TRIANGLE, "nodes": [{"mesh": 0, "name": "Cage_scan"}]}, POINTS)))
    case("office-chair/demo.json holds a key shape or a machine path", title="/Users/ana/chair.glb")
    case("office-chair/demo.json: looks private (matches 'cage')",  # as read: a JSON escape hides no word
         lambda d: (d / "demo.json").write_text((d / "demo.json").read_text().replace('"Office chair"', '"\\u0043age 3"')))
    monkeypatch.setattr(demos, "MAX_FILE", 1000)
    case("office-chair/turntable.mp4 is ", None)


def test_page(tmp_path):
    """The demos in place of the marker, every value escaped: a gallery with one card per demo, each one link to its
    section with its poster, title and two short facts, then each demo's section: a link back to the gallery, its
    heading for the focus, the video muted and looping with its controls and poster but no autoplay and nothing
    preloaded, the facts beside it, the 3D view folded with a lazy model-viewer, and the files as the Downloads tab lists
    them (results.zip first and primary, named <demo>_<file>, with sizes). Without demos the page says so."""
    hand_demo(tmp_path, name="b-chair", title="<b>Chair</b> & co", order=1,
              facts=[["Input", "video, 5 views"], ["Mass", "2.00 kg"], ["Size", "69 x 57 x 96 cm"]])
    hand_demo(tmp_path, name="a-box", title="Box", order=1, poster=None, downloads=[],
              facts=[["Links", "2: lid, a body with a long name"], ["Agent", "off"], ["Input", "video"], ["Note", "x"]])
    (tmp_path / "a-box" / "poster.jpg").unlink()
    (tmp_path / "a-box" / "results.zip").unlink()
    hand_demo(tmp_path, name="z-first", title="First", order=0, facts=[])
    shown = demos.load(tmp_path)
    assert [x.name for x in shown] == ["z-first", "a-box", "b-chair"]
    out = demos.page(SRC, shown)
    assert "<b>Chair" not in out and "&lt;b&gt;Chair&lt;/b&gt; &amp; co" in out and demos.MARKER not in out
    ids = re.findall(r'id="([^"]+)"', out)
    assert len(ids) == len(set(ids)) and {"demo-z-first", "demo-a-box", "demo-b-chair"} <= set(ids)
    gallery = re.search(r'<ul class="gallery">\n(.*?)\n</ul>', out, re.S)[1].split("\n")
    assert out.index('<ul class="gallery">') < out.index("<section")  # the gallery first, then the sections
    assert gallery == [
        '<li><a class="democard" href="#demo-z-first"><img class="dcimg" src="demos/z-first/poster.jpg" alt="" '
        'loading="lazy"><span class="dctitle">First</span></a></li>',
        '<li><a class="democard" href="#demo-a-box"><span class="dcimg"></span><span class="dctitle">Box</span>'
        "<dl><dt>Agent</dt><dd>off</dd><dt>Input</dt><dd>video</dd></dl></a></li>",  # no Size or Mass: short ones
        '<li><a class="democard" href="#demo-b-chair"><img class="dcimg" src="demos/b-chair/poster.jpg" alt="" '
        'loading="lazy"><span class="dctitle">&lt;b&gt;Chair&lt;/b&gt; &amp; co</span>'
        "<dl><dt>Size</dt><dd>69 x 57 x 96 cm</dd><dt>Mass</dt><dd>2.00 kg</dd></dl></a></li>"]
    for href in re.findall(r'<a class="democard" href="#([^"]+)"', out):
        assert out.count(f'<section class="demo" id="{href}">') == 1, href
    chair = out[out.index('id="demo-b-chair"'):]
    assert chair.startswith('id="demo-b-chair">\n<p class="back"><a href="#">All demos</a></p>\n'
                            '<h2 tabindex="-1">&lt;b&gt;Chair&lt;/b&gt; &amp; co</h2>\n')
    assert ('<video src="demos/b-chair/turntable.mp4" poster="demos/b-chair/poster.jpg" muted loop playsinline '
            'controls preload="none" aria-label="360 video of &lt;b&gt;Chair&lt;/b&gt; &amp; co"></video>') in chair
    assert "autoplay" not in out
    assert ("<dl><dt>Input</dt><dd>video, 5 views</dd><dt>Mass</dt><dd>2.00 kg</dd><dt>Size</dt><dd>69 x 57 x 96 cm</dd>"
            "</dl>") in chair
    assert re.search(r'<details class="sub"><summary>3D view</summary>\s*<model-viewer src="demos/b-chair/aligned.glb" '
                     r'loading="lazy" camera-controls[^>]*></model-viewer>\s*</details>', chair)
    rows = re.findall(r'<a class="([^"]+)" href="([^"]+)" download="([^"]+)">([^<]+)</a><span class="dlsize">([^<]+)</span>',
                      chair)
    sizes = {f: demos.size_text((tmp_path / "b-chair" / f).stat().st_size) for f in ("results.zip", "aligned.glb", "turntable.mp4")}
    assert rows == [("dl primary", "demos/b-chair/results.zip", "b-chair_results.zip", "results.zip", sizes["results.zip"]),
                    ("dl", "demos/b-chair/aligned.glb", "b-chair_aligned.glb", "aligned.glb", sizes["aligned.glb"]),
                    ("dl", "demos/b-chair/turntable.mp4", "b-chair_turntable.mp4", "turntable.mp4", sizes["turntable.mp4"])]
    box = out[out.index('id="demo-a-box"'):out.index('id="demo-b-chair"')]
    assert "poster=" not in box and "dlist" not in box
    assert '<p class="emptytab">No demos yet.</p>' in demos.page(SRC, []) and "<section" not in demos.page(SRC, [])
    assert "gallery" not in demos.page(SRC, [])
    for bad in (SRC.replace(demos.MARKER, ""), SRC.replace(demos.MARKER, demos.MARKER * 2)):
        with pytest.raises(demos.DemoError, match="copies of <!-- demos -->"):
            demos.page(bad, [])
    assert demos.files(tmp_path, shown)["demos/b-chair/poster.jpg"] == tmp_path / "b-chair" / "poster.jpg"


def test_page_with_placeholders(tmp_path):
    """The placeholders' cards follow the demos' cards, in the file's order: the same frame with an empty picture, the
    title escaped and "Coming soon", and no link, tabindex or section, so nothing opens. With placeholders only, the
    page is a gallery of them and does not say "No demos yet."."""
    hand_demo(tmp_path, name="chair", title="Chair")
    soon = ["Drawer cabinet", "<i>Lamp</i>"]
    out = demos.page(SRC, demos.load(tmp_path), soon)
    cards = re.search(r'<ul class="gallery">\n(.*?)\n</ul>', out, re.S)[1].split("\n")
    assert len(cards) == 3 and cards[0].startswith('<li><a class="democard" href="#demo-chair">')
    assert cards[1:] == ['<li><div class="democard soon"><span class="dcimg"></span><span class="dctitle">Drawer cabinet'
                         '</span><span class="dcnote">Coming soon</span></div></li>',
                         '<li><div class="democard soon"><span class="dcimg"></span><span class="dctitle">&lt;i&gt;Lamp'
                         '&lt;/i&gt;</span><span class="dcnote">Coming soon</span></div></li>']
    assert not any(x in "".join(cards[1:]) for x in ("<a", "href", "tabindex", "role=", "<button"))
    assert out.count("<section") == 1
    only = demos.page(SRC, [], soon)
    assert only.count('class="democard soon"') == 2 and "<section" not in only and "No demos yet." not in only
    assert "<a class=\"democard\"" not in only


def test_placeholders_rules(tmp_path):
    """placeholders.json: none for no folder or no file; its titles in order, but those that a demo has; and each
    rule, one case each, also through a JSON escape."""
    assert demos.placeholders(None) == [] and demos.placeholders(tmp_path) == []
    with pytest.raises(demos.DemoError, match=re.escape(f"no demos folder {tmp_path / 'none'}")):
        demos.placeholders(tmp_path / "none")
    p = tmp_path / demos.PLACEHOLDERS
    p.write_text(json.dumps({"version": 1, "placeholders": ["Drawer cabinet", "Laptop", "x" * 60]}))
    assert demos.placeholders(tmp_path) == ["Drawer cabinet", "Laptop", "x" * 60]

    def case(why, text):
        p.write_text(text if isinstance(text, str) else json.dumps(text))
        with pytest.raises(demos.DemoError, match=re.escape(why)):
            demos.placeholders(tmp_path)

    case("placeholders.json is not JSON", "{")
    case("placeholders.json is not a JSON object", ["Laptop"])
    case("placeholders.json has unknown keys: note", {"version": 1, "placeholders": ["Laptop"], "note": "x"})
    case("placeholders.json: version is 1", {"placeholders": ["Laptop"]})
    case("placeholders.json: placeholders is a list of 1 to 12 titles; delete the file for none",
         {"version": 1, "placeholders": []})
    case("placeholders.json: placeholders is a list of 1 to 12", {"version": 1, "placeholders": ["a"] * 13})
    case("placeholders.json: placeholders is a list of 1 to 12", {"version": 1, "placeholders": "Laptop"})
    case("placeholders.json: placeholder 2 is text of 1 to 60 characters", {"version": 1, "placeholders": ["a", ""]})
    case("placeholders.json: placeholder 1 is text of 1 to 60", {"version": 1, "placeholders": ["x" * 61]})
    case("placeholders.json: placeholder 1 is text of 1 to 60", {"version": 1, "placeholders": [3]})
    case("placeholders.json: placeholder 1 is plain text on one line, with no space at either end",
         {"version": 1, "placeholders": ["Desk\nlamp"]})
    for bad in ("Desk\u2028lamp", "Desk\x85lamp", "\u202eDoor", "   "):
        case("placeholders.json: placeholder 2 is plain text on one line", {"version": 1, "placeholders": ["Door", bad]})
    case("placeholders.json: placeholder 1 is plain text on one line, with no space", {"version": 1, "placeholders": ["Door "]})
    case("placeholders.json: version is 1", {"version": True, "placeholders": ["Laptop"]})
    case("placeholders.json: version is 1", {"version": 1.0, "placeholders": ["Laptop"]})
    case("placeholders.json: each title once", {"version": 1, "placeholders": ["Door", "Door"]})
    case("placeholders.json: each title once", {"version": 1, "placeholders": ["Door", "door"]})
    case("placeholders.json: looks private (matches 'pallet')", {"version": 1, "placeholders": ["Pallet"]})
    case("placeholders.json holds a key shape or a machine path", {"version": 1, "placeholders": ["/Users/ana/x"]})
    case("placeholders.json: looks private (matches 'pallet')", '{"version": 1, "placeholders": ["\\u0050allet B"]}')
    case("placeholders.json holds a key shape or a machine path",
         '{"version": 1, "placeholders": ["\\/Users\\/ana\\/x"]}')
    p.write_text(json.dumps({"version": 1, "placeholders": ["Drawer cabinet", "Laptop"]}))
    hand_demo(tmp_path / "in", name="laptop", title="laptop")  # its demo has come in, so its card goes
    assert demos.placeholders(tmp_path, demos.load(tmp_path / "in")) == ["Drawer cabinet"]
    p.unlink()
    p.symlink_to(tmp_path / "elsewhere.json")
    with pytest.raises(demos.DemoError, match="placeholders.json is not a regular file"):
        demos.placeholders(tmp_path)
    p.unlink()
    p.mkdir()
    with pytest.raises(demos.DemoError, match="placeholders.json is not a regular file"):
        demos.placeholders(tmp_path)


def test_demos_template_is_a_plain_page():
    """static/demos.html: the head that site.py rewrites (each anchor once), the CDN model-viewer as index.html has it,
    demos.js as its one script of its own, no inline style, and one marker."""
    from gator.webapp import site

    for once in ('<meta charset="utf-8">', 'href="/static/style.css"', 'href="/"', demos.MARKER, site.MODEL_VIEWER):
        assert SRC.count(once) == 1, once
    assert SRC.count("<script") == 2 and '<script src="/static/demos.js" defer></script>' in SRC and "style=" not in SRC
    assert '<a id="helplink" href="help.html#demos">Help</a>' in SRC and "<title>Demos · HARPOR</title>" in SRC


def test_demos_js_plays_the_open_demo():
    """demos.js, on pageshow and on each hashchange: the targeted demo's video plays muted and looping (not under
    reduced motion), every other video pauses, the demo's heading takes the focus and the tab's title names the demo.
    Back in the gallery (no target, or one that is not a demo), every video pauses, the title is the gallery's again
    and the card of the demo just left takes the focus. Nothing runs before pageshow: until load the browser has not
    set :target, so the section is hidden and its heading cannot take the focus."""
    if shutil.which("node") is None:
        pytest.skip("no node")
    code = """
        const log = [];
        let loaded = false;  // the mock's :target: before load nothing shows, so nothing can take the focus
        const el = (name, extra) => Object.assign({ name, focus() { if (loaded) log.push("focus " + name); } }, extra);
        const video = (name) => ({ name, muted: false, loop: false,
          play() { log.push("play " + name + (this.muted && this.loop ? " muted loop" : "")); return Promise.reject(); },
          pause() { log.push("pause " + name); } });
        const va = video("a"), vb = video("b");
        const demo = (id, v) => ({ id, classList: { contains: (c) => c === "demo" }, contains: (x) => x === v,
                                   querySelector: (q) => q === "h2" ? el("h2 " + id, { textContent: "Title " + id }) : null });
        const ids = { "demo-a": demo("demo-a", va), "demo-b": demo("demo-b", vb),
                      other: { id: "other", classList: { contains: () => false } } };
        const cards = { '.gallery a[href="#demo-a"]': el("card a"), '.gallery a[href="#demo-b"]': el("card b") };
        let reduce = false;
        const on = {};
        globalThis.location = { hash: process.argv[2] };
        globalThis.document = { title: "Demos · HARPOR", getElementById: (id) => ids[id] || null,
                                querySelector: (q) => cards[q] || null,
                                querySelectorAll: (q) => q === ".demo video" ? [va, vb] : [] };
        globalThis.matchMedia = (q) => ({ matches: reduce && q === "(prefers-reduced-motion: reduce)" });
        globalThis.addEventListener = (type, fn) => { on[type] = fn; };
        require("vm").runInThisContext(require("fs").readFileSync(process.argv[1], "utf8"));
        const steps = [log.splice(0)];
        loaded = true;
        on.pageshow();
        steps.push([...log.splice(0), document.title]);
        for (const [hash, still] of [["#demo-b", false], ["", false], ["#demo-a", true], ["#other", false]]) {
          location.hash = hash;
          reduce = still;
          on.hashchange();
          steps.push([...log.splice(0), document.title]);
        }
        console.log(JSON.stringify(steps));
        """
    def run(hash):
        p = subprocess.run(["node", "-e", code, str(server.STATIC / "demos.js"), hash], capture_output=True, text=True,
                           timeout=30)
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout)

    gallery, a, b = "Demos · HARPOR", "Title demo-a · Demos · HARPOR", "Title demo-b · Demos · HARPOR"
    assert run("#demo-a") == [[],  # the script alone: it waits for pageshow
                              ["play a muted loop", "pause b", "focus h2 demo-a", a],  # opened by its URL
                              ["pause a", "play b muted loop", "focus h2 demo-b", b],
                              ["pause a", "pause b", "focus card b", gallery],  # back to the gallery
                              ["pause a", "pause b", "focus h2 demo-a", a],  # reduced motion: the video stays still
                              ["pause a", "pause b", "focus card a", gallery]]
    assert run("")[:2] == [[], ["pause a", "pause b", gallery]]  # the gallery: nothing plays, the focus stays


def test_mirrors_of_app_js():
    """KIND and RAN are app.js's KIND_LABEL and RAN, and size_text is its fileSize."""
    app = (server.STATIC / "app.js").read_text()
    for name, mirror in (("KIND_LABEL", demos.KIND), ("RAN", demos.RAN)):
        body = re.search(rf"const {name} = \{{(.*?)\}};", app)[1]
        assert dict(re.findall(r'(\w+): "([^"]*)"', body)) == mirror, name
    if shutil.which("node") is None:
        pytest.skip("no node")
    code = "\n".join(re.search(rf"^const {n} = .*;$", app, re.M)[0] for n in ("size", "fileSize"))
    values = [0, 999, 1000, 9949, 9950, 35_000, 999_499, 999_500, 1_234_567, 99_949_999, 99_950_000, 999_499_999,
              999_500_000, 2_500_000_000]
    p = subprocess.run(["node", "-e", code + f"\nconsole.log(JSON.stringify({values}.map(fileSize)));"],
                       capture_output=True, text=True, timeout=30)
    assert json.loads(p.stdout) == [demos.size_text(v) for v in values]


def test_command_line(job, tmp_path):
    """add and check through python -m gator.webapp.demos: WEBAPP_DEMOS is the default folder, errors print one
    "error: " line and exit 1."""
    def run(*args, env=None):
        e = {k: v for k, v in os.environ.items() if k != "WEBAPP_DEMOS"}
        return subprocess.run([sys.executable, "-m", "gator.webapp.demos", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=120, env={**e, **(env or {})})
    out = tmp_path / "demos"
    r = run("add", str(job[0]))
    assert r.returncode == 1 and r.stderr == "error: give --out DIR or set WEBAPP_DEMOS\n"
    r = run("add", str(job[0]), "--fact", "Note=fake", "--fact", "Agent=", env={"WEBAPP_DEMOS": str(out)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith(f"wrote {out / 'office-chair'}: turntable.mp4 ") and "results.zip" in r.stdout
    r = run("add", str(job[0]), "--out", str(out), "--fact", "oops")
    assert r.returncode == 1 and r.stderr == "error: --fact 'oops' is LABEL=VALUE\n"
    r = run("add", str(job[0]), "--out", str(out))
    assert r.returncode == 1 and r.stderr.startswith("error: ") and "add --force" in r.stderr
    r = run("check", env={"WEBAPP_DEMOS": str(out)})
    assert r.returncode == 0 and re.fullmatch(r"1 demo, [\d.]+ KB, ok\n", r.stdout), r.stdout
    (out / demos.PLACEHOLDERS).write_text(json.dumps({"version": 1, "placeholders": ["Laptop", "Door"]}))
    r = run("check", str(out))
    assert r.returncode == 0 and re.fullmatch(r"1 demo, 2 placeholders, [\d.]+ KB, ok\n", r.stdout), r.stdout
    (out / demos.PLACEHOLDERS).write_text(json.dumps({"version": 1, "placeholders": []}))
    r = run("check", str(out))
    assert r.returncode == 1 and r.stderr == ("error: placeholders.json: placeholders is a list of 1 to 12 titles; "
                                              "delete the file for none\n")
    (out / demos.PLACEHOLDERS).unlink()
    (out / "notes.txt").write_text("x")
    r = run("check", str(out))
    assert r.returncode == 1 and r.stderr == "error: notes.txt is not a demo folder (a demo is a folder, not a link)\n"
    r = run("check", str(tmp_path / "none"))
    assert r.returncode == 1 and r.stderr == f"error: no demos folder {tmp_path / 'none'}\n"
    assert run("check").returncode == 1 and run("bogus").returncode == 2
