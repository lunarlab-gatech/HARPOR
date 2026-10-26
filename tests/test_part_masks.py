import json

import numpy as np
import pytest
from PIL import Image

from gator.stage_a import part_masks as PM
from gator.stage_a.sam3d import load_mask
from test_sam3_masks import FakeSegmenter

H, W = 90, 120


def rect(r0, r1, c0, c1):
    m = np.zeros((H, W), bool)
    m[r0:r1, c0:c1] = True
    return m


def photos(tmp_path, names=("10", "2", "1", "0"), found=None):
    d = tmp_path / "run" / "input" / "images"
    d.mkdir(parents=True)
    seg = FakeSegmenter(found or {})
    seg.images = {}
    for i, n in enumerate(names):
        a = np.full((H, W, 3), (i * 40, 100, 200), np.uint8)
        Image.fromarray(a).save(d / f"{n}.png")
        seg.images[n] = a
    (d / "._0.png").write_bytes(b"\0\5\26\7")  # macOS AppleDouble copy
    return d, seg


def test_every_instance_of_every_part_is_written(tmp_path):
    top, bottom, lid = rect(10, 30, 20, 60), rect(50, 70, 20, 60), rect(0, 8, 0, 120)
    found = {"0": {"drawer": [(0.6, bottom), (0.9, top)], "lid": [(0.8, lid)]},
             "1": {"drawer": [(0.7, top), (0.3, bottom)]}}
    images, seg = photos(tmp_path, found=found)
    out = tmp_path / "run" / "parts" / "sam3"
    rec = PM.run(images, ["drawer", "lid", "Glass door"], out, segmenter=seg)
    assert [n for n, _ in seg.calls] == ["0", "1", "2", "10"]  # one call per photo, natural order
    assert all(p == ["drawer", "lid", "Glass door"] for _, p in seg.calls)
    assert (load_mask(out / "drawer" / "0_0.png") == top).all() and (load_mask(out / "drawer" / "0_1.png") == bottom).all()
    assert (load_mask(out / "drawer" / "1_0.png") == top).all() and not (out / "drawer" / "1_1.png").exists()  # 0.3 < 0.4
    assert sorted(p.name for p in (out / "check").iterdir()) == ["0.jpg", "1.jpg", "10.jpg", "2.jpg"]
    saved = json.loads((out / "part_masks.json").read_text())
    assert saved == rec and saved["not_found"] == ["Glass door"] and saved["views"] == ["0", "1", "2", "10"]
    drawer = saved["parts"][0]
    assert drawer["slug"] == "drawer" and drawer["instances"] == 3 and drawer["views"]["2"] == []
    assert drawer["views"]["0"] == [
        {"index": 0, "score": 0.9, "box_xyxy": [20.0, 10.0, 60.0, 30.0], "mask_px": 800, "file": "drawer/0_0.png"},
        {"index": 1, "score": 0.6, "box_xyxy": [20.0, 50.0, 60.0, 70.0], "mask_px": 800, "file": "drawer/0_1.png"}]
    assert saved["parts"][2]["slug"] == "glass_door" and (out / "glass_door").is_dir()
    assert saved["size_wh"]["0"] == [W, H] and saved["threshold"] == 0.4


def test_a_rerun_replaces_the_whole_folder(tmp_path):
    images, seg = photos(tmp_path, names=("0", "1"), found={"0": {"lid": [(0.9, rect(0, 8, 0, 120))]}})
    out = tmp_path / "run" / "parts" / "sam3"
    PM.run(images, ["lid", "door"], out, segmenter=seg)
    assert (out / "lid" / "0_0.png").exists() and (out / "door").is_dir()
    seg.found = {"1": {"lid": [(0.9, rect(0, 8, 0, 60))]}}
    PM.run(images, ["lid"], out, segmenter=seg)
    assert not (out / "lid" / "0_0.png").exists() and (out / "lid" / "1_0.png").exists()
    assert not (out / "door").exists() and not (out.parent / ".tmp_sam3").exists()


def test_a_base_only_list_loads_no_model(tmp_path):
    images, _ = photos(tmp_path, names=("0",))
    rec = PM.run(images, [], tmp_path / "run" / "parts" / "sam3", segmenter=None)
    assert rec["parts"] == [] and rec["not_found"] == [] and rec["model"] is None
    assert (tmp_path / "run" / "parts" / "sam3" / "check" / "0.jpg").exists()


def test_bad_input_is_refused(tmp_path):
    images, seg = photos(tmp_path, names=("0",))
    with pytest.raises(SystemExit, match="same folder"):
        PM.run(images, ["top drawer", "Top-drawer"], tmp_path / "out", segmenter=seg)
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no PNG or JPG"):
        PM.run(tmp_path / "empty", ["lid"], tmp_path / "out", segmenter=seg)


def test_the_cli_segments_the_base_first_then_every_part(tmp_path, monkeypatch):
    images, seg = photos(tmp_path, names=("0",), found={"0": {"lid": [(0.9, rect(0, 8, 0, 120))]}})
    proposal = {"base": "box body", "parts": [{"name": "box body", "parent": None}, {"name": "lid", "parent": "box body"}]}
    (tmp_path / "run" / "proposal.json").write_text(json.dumps(proposal))
    monkeypatch.setattr(PM.Segmenter, "load", classmethod(lambda cls: seg))
    PM.main(["--run", str(tmp_path / "run"), "--proposal", str(tmp_path / "run" / "proposal.json"),
             "--out", str(tmp_path / "run" / "parts" / "sam3")])
    assert seg.calls == [("0", ["box body", "lid"])]
    rec = json.loads((tmp_path / "run" / "parts" / "sam3" / "part_masks.json").read_text())
    assert [(p["name"], p["instances"]) for p in rec["parts"]] == [("box body", 0), ("lid", 1)]
    assert PM.part_prompts({"base": "box body", "parts": [{"name": "box body"}]}) == []
