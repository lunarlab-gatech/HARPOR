import json

import numpy as np
import pytest
from PIL import Image

from gator.stage_a import mvsam3d as M
from gator.stage_a import sam3_masks as S
from gator.stage_a.sam3d import load_mask

H, W = 120, 160


def rect(r0, r1, c0, c1):
    m = np.zeros((H, W), bool)
    m[r0:r1, c0:c1] = True
    return m


class FakeSegmenter:
    """Stands in for SAM 3: found[name][prompt] is a list of (score, mask), in any order."""

    info = {"model": "fake", "device": "cpu"}

    def __init__(self, found):
        self.found, self.calls = found, []

    def __call__(self, image, prompts, threshold=0.5, mask_threshold=0.5):
        name = next(n for n, im in self.images.items() if np.array_equal(im, image))
        self.calls.append((name, list(prompts)))
        out = {}
        for p in prompts:
            inst = sorted((sm for sm in self.found.get(name, {}).get(p, []) if sm[0] > threshold), key=lambda sm: -sm[0])
            masks = np.array([m for _, m in inst], bool).reshape(-1, *image.shape[:2])
            boxes = np.array([[m.nonzero()[1].min(), m.nonzero()[0].min(), m.nonzero()[1].max() + 1,
                               m.nonzero()[0].max() + 1] for m in masks], float).reshape(-1, 4)
            out[p] = (np.array([s for s, _ in inst], float), boxes, masks)
        return out

    def peak_gib(self):
        return None


def scene(tmp_path, names=("10", "2", "1"), found=None):
    (tmp_path / "scene" / "images").mkdir(parents=True)
    seg = FakeSegmenter(found or {})
    seg.images = {}
    for i, n in enumerate(names):
        a = np.full((H, W, 3), (i * 40, 100, 200), np.uint8)
        Image.fromarray(a).save(tmp_path / "scene" / "images" / f"{n}.png")
        seg.images[n] = a
    (tmp_path / "scene" / "images" / ".DS_Store").write_bytes(b"")
    (tmp_path / "scene" / "images" / "._1.png").write_bytes(b"\0\5\26\7")  # macOS AppleDouble copy
    (tmp_path / "scene" / "images" / "old.png").mkdir()
    return tmp_path / "scene", seg


def test_masks_are_the_top_instance_and_prep_reads_them(tmp_path):
    big, small = rect(20, 100, 30, 130), rect(40, 60, 50, 70)
    found = {n: {"toy": [(0.6, small), (0.9, big)]} for n in ("1", "2", "10")}
    sc, seg = scene(tmp_path, found=found)
    S.run(sc, ["toy"], segmenter=seg)
    assert [n for n, _ in seg.calls] == ["1", "2", "10"]
    for n in ("1", "2", "10"):
        assert Image.open(sc / "sam3_toy" / f"{n}.png").mode == "L"
        assert set(np.unique(np.array(Image.open(sc / "sam3_toy" / f"{n}.png")))) == {0, 255}
        assert (load_mask(sc / "sam3_toy" / f"{n}.png") == big).all()
        assert (sc / "sam3_toy" / "check" / f"{n}.jpg").exists()
    rec = json.loads((sc / "sam3_toy" / "sam3.json").read_text())
    v = rec["views"][0]
    assert rec["prompt"] == "toy" and v["name"] == "1" and v["picked"] == [0] and v["pick"] == "top_score"
    assert [i["score"] for i in v["instances"]] == [0.9, 0.6] and v["mask_px"] == big.sum()
    assert v["instances"][0]["box_xyxy"] == [30, 20, 130, 100]
    M.prep(sc, "sam3_toy", tmp_path / "run", max_side=200)
    assert M.view_names(tmp_path / "run") == ["1", "2", "10"]
    assert (np.array(Image.open(tmp_path / "run" / "masks" / "1.png")) > 0).sum() == big.sum()


def test_pick_takes_one_instance_or_a_union(tmp_path):
    a, b = rect(10, 40, 10, 40), rect(60, 90, 90, 120)
    found = {n: {"toy": [(0.9, a), (0.7, b)]} for n in ("1", "2")}
    sc, seg = scene(tmp_path, names=("1", "2"), found=found)
    S.run(sc, ["toy"], picks=S.parse_picks(["1=1", "2=0,1"]), segmenter=seg)
    assert (load_mask(sc / "sam3_toy" / "1.png") == b).all()
    assert (load_mask(sc / "sam3_toy" / "2.png") == (a | b)).all()
    views = json.loads((sc / "sam3_toy" / "sam3.json").read_text())["views"]
    assert [(v["picked"], v["pick"]) for v in views] == [([1], "user"), ([0, 1], "user")]


def test_bad_picks_are_refused(tmp_path):
    found = {n: {"toy": [(0.9, rect(10, 40, 10, 40))]} for n in ("1", "2")}
    sc, seg = scene(tmp_path, names=("1", "2"), found=found)
    for picks, prompts, msg in (({"1": [1]}, ["toy"], "found 1 instances"),
                                ({"3": [0]}, ["toy"], "not in"),
                                ({"1": [0]}, ["toy", "cup"], "one --prompt")):
        with pytest.raises(SystemExit, match=msg):
            S.run(sc, prompts, picks=picks, segmenter=seg, overwrite=True)
    for bad in ("1", "1=", "=2", "1=a", "1=1,", "1=None"):
        with pytest.raises(SystemExit):
            S.parse_picks([bad])
    assert S.parse_picks(["img 2=2,0,2", "3=none"]) == {"img 2": [0, 2], "3": []}


def test_a_photo_without_an_instance_gets_no_mask(tmp_path):
    found = {"1": {"toy": [(0.8, rect(10, 40, 10, 40))]}}
    sc, seg = scene(tmp_path, names=("1", "2"), found=found)
    S.run(sc, ["toy"], segmenter=seg)
    assert (sc / "sam3_toy" / "1.png").exists() and not (sc / "sam3_toy" / "2.png").exists()
    assert (sc / "sam3_toy" / "check" / "2.jpg").exists()
    v = json.loads((sc / "sam3_toy" / "sam3.json").read_text())["views"][1]
    assert v["mask"] is None and v["instances"] == [] and "no instance" in v["warnings"][0]
    with pytest.raises(SystemExit, match="no mask for \\['cup'\\]"):
        S.run(sc, ["cup"], segmenter=seg)
    assert json.loads((sc / "sam3_cup" / "sam3.json").read_text())["views"][0]["mask"] is None


def test_an_earlier_run_is_kept_unless_overwrite(tmp_path):
    found = {n: {"toy": [(0.8, rect(10, 40, 10, 40))]} for n in ("1", "2")}
    sc, seg = scene(tmp_path, names=("1", "2"), found=found)
    S.run(sc, ["toy"], segmenter=seg)
    with pytest.raises(SystemExit, match="--overwrite"):
        S.run(sc, ["toy"], segmenter=seg)
    del found["2"]  # the rerun finds nothing in photo 2, so its old mask must go
    (sc / "sam3_toy" / "notes.txt").write_text("mine")
    (sc / "sam3_toy" / "1_mask.png").write_bytes((sc / "sam3_toy" / "1.png").read_bytes())  # a hand-made mask
    S.run(sc, ["toy"], segmenter=seg, overwrite=True)
    assert (sc / "sam3_toy" / "1.png").exists() and not (sc / "sam3_toy" / "2.png").exists()
    assert (sc / "sam3_toy" / "notes.txt").read_text() == "mine" and (sc / "sam3_toy" / "1_mask.png").exists()


def test_a_failed_rerun_leaves_the_earlier_run_as_it_was(tmp_path):
    found = {n: {"toy": [(0.9, rect(10, 40, 10, 40)), (0.7, rect(60, 90, 90, 120))]} for n in ("1", "2", "10")}
    sc, seg = scene(tmp_path, found=found)
    S.run(sc, ["toy"], segmenter=seg)
    before = {f.relative_to(sc): f.read_bytes() for f in (sc / "sam3_toy").rglob("*") if f.is_file()}
    with pytest.raises(SystemExit, match="found 2 instances"):  # photo 10 comes last
        S.run(sc, ["toy"], picks={"10": [2]}, segmenter=seg, overwrite=True)

    class Broken:
        @classmethod
        def load(cls):
            raise OSError("gated repo")

    S.Segmenter, real = Broken, S.Segmenter
    try:
        with pytest.raises(OSError):
            S.run(sc, ["toy"], overwrite=True)
    finally:
        S.Segmenter = real
    assert {f.relative_to(sc): f.read_bytes() for f in (sc / "sam3_toy").rglob("*") if f.is_file()} == before


def test_a_rerun_keeps_earlier_picks(tmp_path):
    a, b = rect(10, 40, 10, 40), rect(60, 90, 90, 120)
    found = {n: {"toy": [(0.9, a), (0.7, b)]} for n in ("1", "2", "3")}
    sc, seg = scene(tmp_path, names=("1", "2", "3"), found=found)
    S.run(sc, ["toy"], picks={"1": [1]}, segmenter=seg)
    S.run(sc, ["toy"], picks={"2": [1], "3": []}, segmenter=seg, overwrite=True)
    assert (load_mask(sc / "sam3_toy" / "1.png") == b).all() and (load_mask(sc / "sam3_toy" / "2.png") == b).all()
    assert not (sc / "sam3_toy" / "3.png").exists()  # 3=none: the object is not in photo 3
    S.run(sc, ["toy"], picks={"1": [0]}, segmenter=seg, overwrite=True)  # NAME=0 goes back to the top score
    assert (load_mask(sc / "sam3_toy" / "1.png") == a).all() and (load_mask(sc / "sam3_toy" / "2.png") == b).all()
    assert not (sc / "sam3_toy" / "3.png").exists()
    views = json.loads((sc / "sam3_toy" / "sam3.json").read_text())["views"]
    assert [(v["pick"], v["picked"]) for v in views] == [("top_score", [0]), ("user", [1]), ("user", [])]
    assert views[2]["warnings"] == ["left out by --pick 3=none"]
    S.run(sc, ["toy"], threshold=0.8, segmenter=seg, overwrite=True)  # instance 1 of photo 2 is gone
    v = json.loads((sc / "sam3_toy" / "sam3.json").read_text())["views"][1]
    assert (load_mask(sc / "sam3_toy" / "2.png") == a).all() and v["pick"] == "top_score"
    assert "earlier pick 2=1 is gone" in v["warnings"][0]
    with pytest.raises(SystemExit, match="--pick 2=1: SAM 3 found 1 instances"):
        S.run(sc, ["toy"], picks={"2": [1]}, threshold=0.8, segmenter=seg, overwrite=True)


def test_a_rerun_stopped_while_writing_keeps_the_old_record(tmp_path, monkeypatch):
    found = {n: {"toy": [(0.9, rect(10, 40, 10, 40)), (0.7, rect(60, 90, 90, 120))]} for n in ("1", "2")}
    sc, seg = scene(tmp_path, names=("1", "2"), found=found)
    S.run(sc, ["toy"], picks={"1": [1]}, segmenter=seg)
    before = (sc / "sam3_toy" / "sam3.json").read_text()
    calls, real = [], S.replace

    def flaky(path, save, *args, **kw):
        calls.append(path)
        if len(calls) == 3:
            raise KeyboardInterrupt
        real(path, save, *args, **kw)

    monkeypatch.setattr(S, "replace", flaky)
    with pytest.raises(KeyboardInterrupt):
        S.run(sc, ["toy"], picks={"2": [1]}, segmenter=seg, overwrite=True)
    assert (sc / "sam3_toy" / "sam3.json").read_text() == before
    assert (sc / "sam3_toy" / "1.png").exists() and (sc / "sam3_toy" / "2.png").exists()


def test_two_prompts_share_one_encoding_and_overlaps_are_flagged(tmp_path):
    cup, plate = rect(10, 40, 10, 40), rect(50, 110, 20, 150)
    found = {"1": {"cup": [(0.9, cup)], "glass": [(0.8, cup)], "plate": [(0.95, plate)]}}
    sc, seg = scene(tmp_path, names=("1",), found=found)
    S.run(sc, ["cup", "glass", "plate"], segmenter=seg)
    assert seg.calls == [("1", ["cup", "glass", "plate"])]
    warn = {p: json.loads((sc / f"sam3_{p}" / "sam3.json").read_text())["views"][0]["warnings"]
            for p in ("cup", "glass", "plate")}
    assert warn["cup"] == ["mask IoU 1.00 with the mask of prompt 'glass'"] and len(warn["glass"]) == 1
    assert warn["plate"] == []


def test_prompt_folders(tmp_path):
    assert S.slug("Coffee cup") == "coffee_cup" and S.slug(" red/blue  box ") == "red_blue_box"
    sc, seg = scene(tmp_path, names=("1",), found={"1": {"cup": [(0.9, rect(10, 40, 10, 40))]}})
    for prompts, msg in ((["cup", "Cup"], "same folder twice"), (["cup", "cup"], "same folder twice"),
                         (["!!"], "no letters or digits")):
        with pytest.raises(SystemExit, match=msg):
            S.run(sc, prompts, segmenter=seg)
    assert not (sc / "sam3_cup").exists()


def test_segmenter_sorts_instances_and_adds_the_cls_box():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers", minversion="5.0")
    from types import SimpleNamespace

    from transformers import BatchFeature, Sam3ImageProcessor

    low = torch.full((1, 3, 8, 12), -8.0)
    low[0, 0, :, :6] = 8.0  # query 0: left half
    low[0, 1, :, 6:] = 8.0  # query 1: right half

    class Model:
        def __init__(self):
            self.kwargs = []

        def get_vision_features(self, pixel_values):
            return "vision"

        def __call__(self, vision_embeds, input_ids, attention_mask, **boxes):
            self.kwargs.append(boxes)
            return SimpleNamespace(pred_logits=torch.tensor([[1.0, 3.0, -5.0]]), presence_logits=torch.tensor([[10.0]]),
                                   pred_boxes=torch.tensor([[[0, 0, 0.5, 1], [0.5, 0, 1, 1], [0, 0, 1, 1]]]),
                                   pred_masks=low)

    class Processor:
        post_process_instance_segmentation = Sam3ImageProcessor().post_process_instance_segmentation

        def __call__(self, images=None, text=None, return_tensors=None):
            if images is not None:
                return {"pixel_values": torch.zeros(1, 3, 8, 8)}
            return BatchFeature({"input_ids": torch.ones(1, 32, dtype=torch.long), "attention_mask": torch.ones(1, 32)})

    seg = S.Segmenter(Model(), Processor(), "cpu")
    image = np.zeros((40, 60, 3), np.uint8)
    scores, boxes, masks = seg(image, ["toy"])["toy"]
    assert np.allclose(scores, [0.9526, 0.7311], atol=1e-4)  # query order was 0.73, 0.95
    assert np.allclose(boxes, [[30, 0, 60, 40], [0, 0, 30, 40]]) and masks.dtype == bool
    assert masks[0][:, 30:].all() and not masks[0][:, :30].any() and masks[1][:, :30].all()
    box = seg.model.kwargs[-1]
    assert box["input_boxes_labels"].tolist() == [[-10]] and box["input_boxes"].shape == (1, 1, 4)
    seg(image, ["toy"], cls_token=False)
    assert seg.model.kwargs[-1] == {}
    scores, boxes, masks = seg(image, ["toy"], threshold=0.99)["toy"]
    assert scores.shape == (0,) and boxes.shape == (0, 4) and masks.shape == (0, 40, 60)
