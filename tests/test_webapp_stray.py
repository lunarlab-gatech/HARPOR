"""Stray Scanner captures (harpor/webapp/stray.py): unpacking the zip safely, the turn from gravity, the
upright frames and the depth that lands on them. Every capture is made here from fake_steps' scene: a
box filmed by a camera that circles it, stored as a phone's sensor stores it."""

import io
import json
import shutil
import stat
import struct
import zipfile
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

pytest.importorskip("av")
from harpor.stage_a import lidar_scale  # noqa: E402
import fake_steps as F  # noqa: E402
from harpor.webapp import stray  # noqa: E402
from harpor.webapp import video as V  # noqa: E402

SIZE = (320, 240)  # the video as stored: sensor landscape
DEPTH = (256, 192)
TOP = "capture"
FLIP = np.diag([1.0, -1.0, -1.0])  # fake_steps' world (y down) to ARKit's (+y up)
# Stored camera axes to the axes of the image turned once counterclockwise (np.rot90): x' = y, y' = -x.
TURN = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
HEADER = ("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, "
          "distortion_center_y")
MARK_MM = 1234
DOWN = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # an OpenCV camera looking straight down
FILES = ["camera_matrix.csv", "imu.csv", "odometry.csv", "rgb.mp4"]


def upright_size(size, k):
    return tuple(size) if k % 2 == 0 else tuple(size[::-1])


def picture(i, size):
    """Frame i as it looks upright: 4x4 cells and a bright bar that moves right, so every frame is sharp."""
    w, h = size
    rng = np.random.default_rng(1)
    a = (rng.random((h // 4, w // 4, 3)) * 180 + 30).astype(np.uint8).repeat(4, 0).repeat(4, 1)
    x = (8 + 3 * i) % (w // 2)
    a[h // 3:h // 2, x:x + w // 5] = 240
    return a


def stored(a, k):
    """An upright image as the sensor stores it: k quarter turns clockwise, so k counterclockwise undo it."""
    return np.ascontiguousarray(np.rot90(a, -k))


def rgb_of_depth(u_d, v_d, size=SIZE):
    """The stored video's pixel at the centre of depth pixel (u_d, v_d)."""
    return (u_d + 0.5) * size[0] / DEPTH[0] - 0.5, (v_d + 0.5) * size[1] / DEPTH[1] - 0.5


def quat(R):
    return Rotation.from_matrix(R).as_quat()  # qx, qy, qz, qw


def export(d, n=8, size=SIZE, portrait=0, n_depth=None, odometry=True, mark=None, rate=10, without_depth=(),
           without_confidence=()):
    """A Stray Scanner export folder d/capture of fake_steps' scene, as write_stray describes; returns it."""
    import av

    k, (W, H) = portrait, size
    up_rgb, up_depth = upright_size(size, k), upright_size(DEPTH, k)
    root = Path(d) / TOP
    for f in ("depth", "confidence"):
        (root / f).mkdir(parents=True)
    n_depth = n if n_depth is None else n_depth
    rows = []
    with av.open(str(root / "rgb.mp4"), "w") as c:
        s = c.add_stream("libx264", rate=rate, options={"crf": "12"})
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.time_base = Fraction(1, rate)
        for i in range(n):
            angle = F.SWEEP * i / max(n - 1, 1)
            rgb = stored(picture(i, up_rgb), k)
            mm = np.round(stored(F.render(*up_depth, angle)[0], k) * 1000).astype(np.uint16)
            if mark is not None:  # a white dot on black where depth pixel mark is, and its own depth
                x, y = (round(c) for c in rgb_of_depth(*mark, size))
                rgb[y - 6:y + 7, x - 6:x + 7] = 0
                rgb[y - 2:y + 3, x - 2:x + 3] = 255
                mm[mark[1], mark[0]] = MARK_MM
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = i - 1, Fraction(1, rate)  # as Stray writes it: frame 0 before the start
            for p in s.encode(frame):
                c.mux(p)
            if i < n_depth and i not in without_depth:
                Image.fromarray(mm).save(root / "depth" / f"{i:06d}.png")
                if i not in without_confidence:
                    Image.fromarray(np.full(mm.shape, 2, np.uint8)).save(root / "confidence" / f"{i:06d}.png")
            R, C = F.camera(angle)  # world to the upright camera, fake_steps' world
            Rwc = FLIP @ (np.linalg.matrix_power(TURN, k).T @ R).T  # the stored camera to ARKit's world
            x, y, z = FLIP @ C
            q, f = quat(Rwc), F.FOCAL * up_rgb[0]
            rows.append(f"{1000 + i / rate:.6f}, {i}, {x:.6f}, {y:.6f}, {z:.6f}, {q[0]:.9f}, {q[1]:.9f}, {q[2]:.9f}, "
                        f"{q[3]:.9f}, {f:g}, {f:g}, {(W - 1) / 2:g}, {(H - 1) / 2:g}, {(W - 1) / 2:g}, {(H - 1) / 2:g}")
        for p in s.encode():
            c.mux(p)
    if odometry:
        (root / "odometry.csv").write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    f = F.FOCAL * up_rgb[0]
    (root / "camera_matrix.csv").write_text(f"{f:g},0.0,{(W - 1) / 2:g}\n0.0,{f:g},{(H - 1) / 2:g}\n0.0,0.0,1.0\n")
    (root / "imu.csv").write_text("timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n1000.0, 0, -9.81, 0, 0, 0, 0\n")
    return root


def zip_export(root, out, top=TOP, junk=True):
    """root zipped as Finder does: one top folder, its directory entries and __MACOSX junk."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        if top:
            z.writestr(f"{top}/", "")
        for p in sorted(Path(root).rglob("*")):
            arc = (f"{top}/" if top else "") + p.relative_to(root).as_posix()
            if p.is_dir():
                z.writestr(arc + "/", "")
            else:
                z.write(p, arc)
        if junk:
            z.writestr(f"{top}/.DS_Store", b"\0\1" * 32)
            z.writestr(f"{top}/notes.txt", "not part of the export")
            z.writestr(f"{top}/depth/preview.jpg", b"not a depth image")
            z.writestr(f"__MACOSX/{top}/._rgb.mp4", b"\0\1" * 32)
            z.writestr(f"__MACOSX/._{top}", b"\0\1" * 32)
    return out


def write_stray(d, n=8, size=SIZE, portrait=0, **kw):
    """A zipped Stray Scanner export of fake_steps' scene, d/capture.zip (the folder is d/capture): n
    frames of rgb.mp4 (libx264; frame i at the angle fake_steps.frame_angle gives it) stored as the sensor
    sees them at size (landscape), turned portrait quarter turns clockwise from upright (0: a landscape
    capture; 1 or 3: a portrait one, which the app turns back by that many counterclockwise turns),
    depth/NNNNNN.png with the scene's true z-depth in mm at 256x192 (the first n_depth frames but those
    in without_depth), confidence 2 everywhere (but for without_confidence), and odometry.csv with each
    frame's camera-to-world pose in OpenCV axes in ARKit's +y up world and its intrinsics at the video's
    size. As Stray's VideoEncoder does, frame 0 is written one tick before the start, so the mp4's edit
    list hides it from a reader that follows it. mark=(u_d, v_d): a white dot on black in every frame
    where that depth pixel is, and the depth MARK_MM there."""
    root = export(d, n, size, portrait, **kw)
    return zip_export(root, Path(d) / "capture.zip")


def files_under(root):
    return sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file())


def refused(z, dest, match):
    with pytest.raises(SystemExit) as e:
        stray.unpack(z, dest)
    assert match in str(e.value), str(e.value)
    assert not (dest / "export").exists() and not (dest / ".tmp").exists() and not (dest / ".unpacked.json").exists()
    return str(e.value)


def test_unpack_takes_only_the_export_and_once(tmp_path, monkeypatch):
    src = tmp_path / "src"
    z = write_stray(src, n=4)
    dest = tmp_path / "job" / "video" / "stray"
    root = stray.unpack(z, dest)
    assert root == dest / "export"
    frames = [f"{d}/{i:06d}.png" for d in ("confidence", "depth") for i in range(4)]
    assert files_under(root) == sorted(FILES + frames)  # no .DS_Store, notes, preview.jpg or __MACOSX
    assert all(stat.S_IMODE((root / f).stat().st_mode) == 0o600 for f in files_under(root))
    assert (root / "rgb.mp4").read_bytes() == (src / TOP / "rgb.mp4").read_bytes()
    assert sorted(p.name for p in dest.iterdir()) == [".unpacked.json", "export"]
    rec = json.loads((dest / ".unpacked.json").read_text())
    assert rec == {"bytes": z.stat().st_size, "mtime_ns": z.stat().st_mtime_ns, "root": "export",
                   "members": sorted(FILES + frames)}

    # The same zip again: nothing is read. A missing member or another zip unpacks it again.
    with monkeypatch.context() as m:
        m.setattr(stray.zipfile, "ZipFile", None)
        assert stray.unpack(z, dest) == root
    (root / "depth" / "000002.png").unlink()
    (root / "stale.txt").write_text("from the earlier unpack")
    assert stray.unpack(z, dest) == root and files_under(root) == sorted(FILES + frames)
    flat = zip_export(src / TOP, tmp_path / "flat.zip", top="", junk=False)  # zipped without the top folder
    assert stray.unpack(flat, dest) == root and files_under(root) == sorted(FILES + frames)
    assert json.loads((dest / ".unpacked.json").read_text())["bytes"] == flat.stat().st_size


BASE = ("rgb.mp4", "odometry.csv", "depth/000000.png")


def bad_zip(path, *entries, base=True):
    """A small export with the extra entries ((name or ZipInfo, data), ...)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        if base:
            for name in BASE:
                z.writestr(f"{TOP}/{name}", f"stand-in for {name}")
        for name, data in entries:
            z.writestr(name, data)
    return path


def test_unpack_refuses_unsafe_and_odd_zips(tmp_path, monkeypatch):
    dest = tmp_path / "job" / "video" / "stray"
    for i, (entry, why) in enumerate([
            (f"{TOP}/../evil.txt", '".."'), ("/tmp/evil.txt", "an absolute path"),
            (f"{TOP}\\depth\\000001.png", "a backslash"), ("C:/evil.txt", "a drive letter"),
            ("__MACOSX/../../evil.txt", '".."')]):
        z = bad_zip(tmp_path / f"slip{i}.zip", (entry, "x"))
        assert entry in [x.filename for x in zipfile.ZipFile(z).infolist()]  # zipfile kept the name as given
        assert refused(z, dest, f"the zip has an entry with {why}").endswith("; zip the export folder again")
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "job" / "video" / "evil.txt").exists()

    link = zipfile.ZipInfo(f"{TOP}/depth/000001.png")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    refused(bad_zip(tmp_path / "link.zip", (link, "/etc/passwd")), dest, "the zip has a symlink")

    bomb = bad_zip(tmp_path / "bomb.zip", (f"{TOP}/imu.csv", b"\0" * (10 << 20)))
    refused(bomb, dest, f"{TOP}/imu.csv unpacks to more than 200 times its compressed size")
    junk = bad_zip(tmp_path / "junk.zip", (f"{TOP}/notes.bin", b"\0" * (10 << 20)))  # not extracted: no limit
    assert files_under(stray.unpack(junk, dest)) == ["depth/000000.png", "odometry.csv", "rgb.mp4"]
    monkeypatch.setattr(stray, "TOTAL_MAX", 50)
    big = bad_zip(tmp_path / "big.zip")
    assert refused(big, dest, "unpacked, the export would take ") == (
        f"unpacked, the export would take {sum(len(f'stand-in for {n}') for n in BASE)} bytes, more than 50 bytes for "
        f"a {big.stat().st_size} bytes zip; this is "
        "not a Stray Scanner export")
    monkeypatch.undo()

    assert refused(bad_zip(tmp_path / "two.zip", ("other/rgb.mp4", "x")), dest, "the zip has 2 rgb.mp4 files") == \
        f"the zip has 2 rgb.mp4 files ({TOP}/rgb.mp4, other/rgb.mp4); zip one Stray Scanner export"
    refused(bad_zip(tmp_path / "none.zip", (f"{TOP}/depth/000000.png", "x"), base=False), dest,
            "the zip has no rgb.mp4; zip the Stray Scanner export folder")
    with pytest.warns(UserWarning, match="Duplicate name"):
        twice = bad_zip(tmp_path / "twice.zip", (f"{TOP}/odometry.csv", "again"))
    refused(twice, dest, f"the zip holds {TOP}/odometry.csv twice")
    (tmp_path / "junk.bin").write_bytes(b"PK not really a zip")
    refused(tmp_path / "junk.bin", dest, "junk.bin is not a readable zip")

    # Digits other than 0 to 9 do not make a frame image's name.
    odd = bad_zip(tmp_path / "digits.zip", (f"{TOP}/depth/\u0661\u0662\u0663\u0664\u0665\u0666.png", "x"),
                  (f"{TOP}/depth/\uff10\uff10\uff10\uff10\uff10\uff11.png", "x"))
    root = stray.unpack(odd, dest)
    assert files_under(root) == ["depth/000000.png", "odometry.csv", "rgb.mp4"] and stray._indices(root, "depth") == {0}


def central(path, name):
    """The offset of name's central directory record in the zip at path."""
    data = path.read_bytes()
    pos = 0
    while True:
        pos = data.index(b"PK\x01\x02", pos)
        n = struct.unpack_from("<H", data, pos + 28)[0]
        if data[pos + 46:pos + 46 + n] == name.encode():
            return pos
        pos += 4


def patched(path, at, fmt, value):
    data = bytearray(path.read_bytes())
    struct.pack_into(fmt, data, at, value)
    path.write_bytes(bytes(data))
    return path


def test_unpack_counts_the_entries_before_zipfile_reads_them(tmp_path, monkeypatch):
    """zipfile builds an object for every entry of the central directory, so the entries are counted in
    it first, whatever the zip's end record says."""
    dest = tmp_path / "job" / "video" / "stray"
    many = bad_zip(tmp_path / "many.zip", ("other.txt", "x"))
    lying = shutil.copy(many, tmp_path / "lying.zip")
    end = Path(lying).read_bytes().rindex(b"PK\x05\x06")
    patched(patched(Path(lying), end + 8, "<H", 1), end + 10, "<H", 1)  # the end record says 1 entry
    assert len(zipfile.ZipFile(lying).infolist()) == 4
    monkeypatch.setattr(stray, "MAX_ENTRIES", 3)
    with monkeypatch.context() as m:
        m.setattr(stray.zipfile, "ZipFile", None)  # never reached
        for z in (many, Path(lying)):
            assert refused(z, dest, "more than 3 entries") == \
                "the zip has more than 3 entries; zip one Stray Scanner export"
    with pytest.raises(SystemExit, match="the zip has more than 3 entries"):  # members() checks them too
        stray.members(zipfile.ZipFile(many).infolist(), many.stat().st_size)
    monkeypatch.undo()
    with open(many, "rb") as f:
        assert stray._entries(f) == 4
    monkeypatch.setattr(stray, "DIR_MAX", 100)
    size = len(many.read_bytes()) - many.read_bytes().index(b"PK\x01\x02") - 22
    assert refused(many, dest, "the zip's list of entries") == (
        f"the zip's list of entries takes {size} bytes, more than 100 bytes; zip one Stray Scanner export")


def test_unpack_refuses_other_compression_and_unreadable_zips(tmp_path, monkeypatch):
    """bzip2 and lzma members are refused before they are read (zipfile does not limit their memory); a
    member that zipfile cannot read, or a full disk, is a plain reason."""
    dest = tmp_path / "job" / "video" / "stray"
    for method in (zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA):
        z = tmp_path / f"m{method}.zip"
        with zipfile.ZipFile(bad_zip(z), "a") as f:
            f.writestr(f"{TOP}/imu.csv", b"0, 0, -9.81\n" * 100, compress_type=method)
        assert refused(z, dest, f"{TOP}/imu.csv uses compression method {method}") == (
            f"{TOP}/imu.csv uses compression method {method}, not deflate; zip the export again with the usual "
            "(deflate) compression")
    with zipfile.ZipFile(bad_zip(tmp_path / "junk.zip"), "a") as f:  # members that are not extracted are never read
        f.writestr(f"{TOP}/notes.txt", b"x" * 100, compress_type=zipfile.ZIP_BZIP2)
        f.writestr(f"__MACOSX/{TOP}/._imu.csv", b"x" * 100, compress_type=zipfile.ZIP_LZMA)
    assert files_under(stray.unpack(tmp_path / "junk.zip", dest)) == ["depth/000000.png", "odometry.csv", "rgb.mp4"]

    z = bad_zip(tmp_path / "version.zip")
    patched(z, central(z, f"{TOP}/odometry.csv") + 6, "<H", 69)  # needs zip version 6.9
    refused(z, dest, "version.zip is not a readable zip: zip file version 6.9")
    z = bad_zip(tmp_path / "utf8.zip")
    at = central(z, f"{TOP}/odometry.csv")
    patched(patched(z, at + 8, "<H", 0x800), at + 46 + len(TOP) + 1, "<B", 0xFF)  # a name that is not UTF-8
    refused(z, dest, "utf8.zip is not a readable zip: 'utf-8' codec can't decode")

    def full(*a):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(stray, "_extract", full)
    assert refused(bad_zip(tmp_path / "ok.zip"), dest, "cannot unpack") == \
        "cannot unpack ok.zip: [Errno 28] No space left on device"


def test_unpack_counts_the_real_bytes(tmp_path):
    """A member whose data is longer than its header says, or than the zip allows, stops the unpacking,
    whatever zipfile would do with it."""
    class Lying:
        def __init__(self, data):
            self.data = data

        def open(self, info):
            return io.BytesIO(self.data)

    info = zipfile.ZipInfo(f"{TOP}/odometry.csv")
    info.file_size = 10
    with pytest.raises(SystemExit, match="unpacks to more bytes than the zip says"):
        stray._extract(Lying(b"x" * 11), info, tmp_path / "a" / "odometry.csv", 1000)
    info.file_size = 100
    with pytest.raises(SystemExit, match="unpacks to more bytes than the zip allows"):
        stray._extract(Lying(b"x" * 50), info, tmp_path / "b" / "odometry.csv", 40)
    assert stray._extract(Lying(b"x" * 50), info, tmp_path / "c" / "odometry.csv", 50) == 50
    assert stat.S_IMODE((tmp_path / "c" / "odometry.csv").stat().st_mode) == 0o600


def test_upright_turns_from_gravity():
    upright = np.diag([1.0, -1.0, -1.0])  # OpenCV camera, level, looking along ARKit's -z
    for k in range(4):
        q = quat(upright @ np.linalg.matrix_power(TURN, k))
        assert stray.upright_turns([q] * 3) == (k, 1.0)
    tilted = Rotation.from_euler("x", -40, degrees=True).as_matrix()  # looking down at an object
    assert stray.upright_turns([quat(tilted @ upright @ TURN)]) == (1, 1.0)
    mixed = [quat(upright @ TURN)] * 7 + [quat(upright)] * 3  # the phone turned for 3 of the 10 frames
    k, agree = stray.upright_turns(mixed)
    assert k == 1 and agree == pytest.approx(0.7)
    down = DOWN  # straight down: no vote
    assert stray.upright_turns([quat(down)] * 2) == (0, None)
    assert stray.upright_turns([quat(down)] * 5 + [quat(upright @ TURN.T)]) == (3, 1.0)
    assert stray.upright_turns([[0.0, 0.0, 0.0, 0.0], quat(upright)]) == (0, 1.0)  # an empty pose does not vote
    assert stray.upright_turns([[0.0, 0.0, 0.0, 0.0]]) == (0, None)


@pytest.mark.parametrize("k", [0, 1, 3])
def test_probe_and_upright_frames(tmp_path, k):
    """A landscape capture is left as it is; a portrait one, stored turned, is turned back by gravity."""
    root = export(tmp_path, n=6, portrait=k)
    info = stray.probe(root)
    W, H = upright_size(SIZE, k)
    assert (info["kind"], info["rotation"], info["rotation_source"], info["container_rotation"]) == ("stray", 90 * k,
                                                                                                    "gravity", 0)
    assert (info["width"], info["height"], info["n_frames"], info["codec"]) == (W, H, 6, "h264")
    assert (info["n_depth"], info["n_confidence"], info["n_odometry"], info["depth_size"]) == (6, 6, 6, list(DEPTH))
    assert info["upright_agreement"] == 1.0 and info["warnings"] == [] and stray.turns(info) == k
    got = list(stray.frames(root, info, keep={0, 4}))
    assert [i for i, _, _ in got] == [0, 4] and got[1][1] - got[0][1] == pytest.approx(0.4, abs=1e-3)
    for i, _, a in got:
        assert a.shape == (H, W, 3) and np.abs(a.astype(float) - picture(i, (W, H))).mean() < 8
    _, _, small = next(stray.frames(root, info, width=120))
    assert small.shape == (round(120 * H / W), 120, 3)


def test_read_odometry(tmp_path):
    root = export(tmp_path, n=3)
    odo = stray.read_odometry(root)
    assert odo["frame"].tolist() == [0, 1, 2] and odo["timestamp"] == pytest.approx([1000, 1000.1, 1000.2])
    assert odo["quat"].shape == (3, 4) and odo["position"].shape == (3, 3)
    assert odo["intrinsics"][0] == pytest.approx([F.FOCAL * SIZE[0], F.FOCAL * SIZE[0], 159.5, 119.5])
    # An older export without the intrinsics columns.
    lines = (root / "odometry.csv").read_text().splitlines()
    (root / "odometry.csv").write_text("\n".join(", ".join(x.split(", ")[:9]) for x in lines) + "\n")
    assert stray.read_odometry(root)["intrinsics"] is None and len(stray.read_odometry(root)["frame"]) == 3
    (root / "odometry.csv").write_text("timestamp, frame, x, y, z\n1.0, 0, 0, 0, 0\n")
    with pytest.raises(SystemExit, match="odometry.csv has no qx, qy, qz, qw column; this export is not supported"):
        stray.read_odometry(root)
    (root / "odometry.csv").write_text(lines[0] + "\n" + lines[1].replace("1000.000000", "soon") + "\n")
    with pytest.raises(SystemExit, match="odometry.csv has a row that is not all numbers"):
        stray.read_odometry(root)


def test_odometry_that_is_not_text_or_csv(tmp_path):
    """The frames stage (probe) gives a plain reason for an odometry.csv that is not UTF-8 text or has a field longer
    than the csv module reads (128 KB). Both escaped as a traceback before."""
    root = export(tmp_path, n=4)
    good = (root / "odometry.csv").read_bytes()
    (root / "odometry.csv").write_bytes(good.replace(b"timestamp", b"timestamp\xe9", 1))
    with pytest.raises(SystemExit, match=r"^odometry.csv is not UTF-8 text; this export is not supported$"):
        stray.probe(root)
    (root / "odometry.csv").write_bytes(good + b"1," + b"9" * 200_000 + b"\n")
    with pytest.raises(SystemExit) as e:
        stray.probe(root)
    assert str(e.value) == ("odometry.csv cannot be read: field larger than field limit (131072); this export is not "
                            "supported")
    assert stray.intrinsics(root, 4)[1] == "camera_matrix.csv"  # the lidar stage's K falls back as for a bad row


@pytest.mark.parametrize("k", [0, 1, 3])
def test_a_depth_pixel_lands_on_its_rgb_pixel(tmp_path, k):
    """The marked depth pixel and the dot drawn at its colour pixel come out at the same place in the upright
    frame; and every depth sample of the scene lands where the upright camera sees that depth. With
    odometry.csv's K, depth_points gives the same samples and their camera points row for row, in the upright
    frame's axes: the mark's point projects onto the mark through the upright frame's K."""
    mark = (100, 60)
    root = export(tmp_path, n=4, portrait=k, mark=mark)
    info = stray.probe(root)
    W, H = info["width"], info["height"]
    only = np.zeros(DEPTH[::-1], np.uint8)  # confidence 2 at the mark only
    only[mark[1], mark[0]] = 2
    Image.fromarray(only).save(root / "confidence" / "000000.png")
    uvz = stray.depth_points(root, 0, (W, H), k, None)[0]
    assert uvz.shape == (1, 3) and uvz.dtype == np.float32 and uvz[0, 2] == pytest.approx(MARK_MM / 1000)
    _, _, a = next(stray.frames(root, info, keep={0}))
    u, v = uvz[0, :2]
    x0, y0 = int(round(u)) - 5, int(round(v)) - 5  # inside the black square
    g = V.gray(a[y0:y0 + 11, x0:x0 + 11])
    ys, xs = np.nonzero(g > 128)
    assert 9 <= len(xs) <= 40  # the white dot, inside its black square
    assert abs(xs.mean() + x0 - u) < 0.75 and abs(ys.mean() + y0 - v) < 0.75
    expect = lidar_scale.rot90_points(*rgb_of_depth(*mark), SIZE, k)
    assert (u, v) == pytest.approx(expect, abs=1e-4)
    K, source = stray.intrinsics(root, 4)
    assert source == "odometry.csv" and K.shape == (4, 3, 3)
    Ku = F.intrinsics(W, H)  # the upright frame's: f stays, the centre turns with the frame

    def pixels(xyz):
        return np.c_[Ku[0, 0] * xyz[:, 0] / xyz[:, 2] + Ku[0, 2], Ku[1, 1] * xyz[:, 1] / xyz[:, 2] + Ku[1, 2]]

    same, xyz, scan = stray.depth_points(root, 0, (W, H), k, K[0])
    assert np.array_equal(same, uvz) and xyz.dtype == scan.dtype == np.float32 and xyz.shape == (1, 3)
    assert pixels(xyz)[0] == pytest.approx((u, v), abs=1e-3) and xyz[0, 2] == pytest.approx(MARK_MM / 1000)
    assert len(scan) == DEPTH[0] * DEPTH[1]  # every pixel with a depth, at any confidence
    samples, xyz, _ = stray.depth_points(root, 3, (W, H), k, K[3])
    assert np.array_equal(samples, stray.depth_points(root, 3, (W, H), k, None)[0]) and len(xyz) == len(samples)
    assert np.allclose(pixels(xyz), samples[:, :2], atol=1e-3) and np.allclose(xyz[:, 2], samples[:, 2])
    assert stray.depth_points(root, 3, (W, H), k, None)[1:] == (None, None)

    # Frame 1: every pixel (confidence 2, z within 0.1 to 5 m; the empty view is 4 m) is a sample at the
    # upright pixel where the scene has that depth.
    uvz = stray.depth_points(root, 1, (W, H), k, None)[0]
    assert len(uvz) == DEPTH[0] * DEPTH[1]
    assert 0 <= uvz[:, 0].min() and uvz[:, 0].max() <= W - 1 and 0 <= uvz[:, 1].min() and uvz[:, 1].max() <= H - 1
    depth = F.render(W, H, F.SWEEP / 3)[0]
    near = uvz[:, 2] < F.FAR - 0.01
    err = np.abs(depth[lidar_scale.pixel(uvz[near, 1]), lidar_scale.pixel(uvz[near, 0])] / uvz[near, 2] - 1)
    assert np.median(err) < 2e-3 and np.mean(err < 0.01) > 0.97
    within = stray.depth_points(root, 1, (W, H), k, None, z_range=(0.1, 3.5))[0]
    assert len(within) == (uvz[:, 2] <= 3.5).sum() < len(uvz)
    half = np.full(DEPTH[::-1], 2, np.uint8)
    half[:, :DEPTH[0] // 2] = 1
    Image.fromarray(half).save(root / "confidence" / "000001.png")
    assert len(stray.depth_points(root, 1, (W, H), k, None)[0]) == DEPTH[0] * DEPTH[1] // 2
    assert len(stray.depth_points(root, 1, (W, H), k, None, min_conf=1)[0]) == DEPTH[0] * DEPTH[1]
    (root / "confidence" / "000001.png").unlink()
    with pytest.raises(ValueError, match="depth/000001.png has no confidence/000001.png"):
        stray.depth_points(root, 1, (W, H), k, None)
    assert len(stray.depth_points(root, 1, (W, H), k, None, min_conf=0)[0]) == DEPTH[0] * DEPTH[1]
    Image.fromarray(half).save(root / "depth" / "000002.png")  # 8-bit, not depth
    with pytest.raises(ValueError, match="depth/000002.png is a L image, not 16-bit depth"):
        stray.depth_points(root, 2, (W, H), k, None)


def test_intrinsics_from_odometry_else_camera_matrix(tmp_path):
    """odometry.csv's fx, fy, cx, cy per frame first; camera_matrix.csv for every frame without them; a
    camera_matrix.csv that is not one gives none, without a crash."""
    root = export(tmp_path, n=3)
    K, source = stray.intrinsics(root, 3)
    f = F.FOCAL * SIZE[0]
    want = [[f, 0, 159.5], [0, f, 119.5], [0, 0, 1]]
    assert source == "odometry.csv" and np.allclose(K, want)
    lines = (root / "odometry.csv").read_text().splitlines()
    (root / "odometry.csv").write_text("\n".join(", ".join(x.split(", ")[:9]) for x in lines) + "\n")
    K, source = stray.intrinsics(root, 3)
    assert source == "camera_matrix.csv" and K.shape == (3, 3, 3) and np.allclose(K, want)
    for bad in ("1,0,0\n0,1,0\n", "1,0,0\n0,1,0\n0,0,2\n", "-1,0,0\n0,1,0\n0,0,1\n", "nan,0,0\n0,1,0\n0,0,1\n",
                "a,b,c\n0,1,0\n0,0,1\n", "1,0,0,4\n0,1,0\n0,0,1\n", "1,0,0\n0,1,0\n0,0,1\n" + " " * 5000):
        (root / "camera_matrix.csv").write_text(bad)
        assert stray.intrinsics(root, 3) == (None, None), bad
    (root / "camera_matrix.csv").write_bytes(b"\xff\xfe\x00")
    assert stray.intrinsics(root, 3) == (None, None)
    (root / "camera_matrix.csv").unlink()
    assert stray.intrinsics(root, 3) == (None, None)


def which(a):
    """Which frame the upright frame a is, from the left edge of picture()'s bright bar (3 px a frame)."""
    h = a.shape[0]
    return round((np.flatnonzero(V.gray(a[h // 3 + 2:h // 2 - 2]).mean(0) > 225)[0] - 8) / 3)


def test_rgb_mp4_is_read_with_the_frame_its_edit_list_hides(tmp_path):
    """Stray writes frame 0 one tick before the mp4's start, so a reader that follows the edit list sees
    frame 1 first and one frame fewer. Read with stray.OPEN, frame i of rgb.mp4 is frame i of depth/ and
    odometry.csv."""
    root = export(tmp_path, n=8, portrait=1)
    assert V.probe(root / "rgb.mp4")["n_frames"] == 7  # the plain reading: frame 0 is hidden
    assert [which(a) for _, _, a in V.frames(root / "rgb.mp4", turns=1)] == list(range(1, 8))
    info = stray.probe(root)
    assert info["n_frames"] == info["n_odometry"] == info["n_depth"] == 8 and info["warnings"] == []
    assert [which(a) for _, _, a in stray.frames(root, info)] == list(range(8))


def test_frames_without_depth_and_counts_that_do_not_match(tmp_path):
    """Depth images are paired by name: frames that ARKit gave no depth (or no confidence) only warn. An
    export whose frames and odometry rows differ in number, or that names depth past the last frame, is
    refused, since its frames cannot be paired."""
    root = export(tmp_path / "a", n=8, without_depth=(0, 1, 2), without_confidence=(5,))
    info = stray.probe(root)
    assert (info["n_frames"], info["n_depth"], info["n_confidence"], info["n_odometry"]) == (8, 5, 4, 8)
    assert info["warnings"] == ["3 of the 8 frames have no depth image, so they have no LiDAR samples",
                                "1 of the 5 depth images have no confidence image, so those frames have no LiDAR "
                                "samples"]

    def why(root):
        with pytest.raises(SystemExit) as e:
            stray.probe(root)
        return str(e.value)

    shutil.copy(root / "depth" / "000003.png", root / "depth" / "000008.png")
    assert why(root) == "depth/000008.png is past the last of rgb.mp4's 8 frames; this export is not supported"
    (root / "depth" / "000008.png").unlink()
    lines = (root / "odometry.csv").read_text().splitlines()
    (root / "odometry.csv").write_text("\n".join([lines[0]] + lines[:0:-1]) + "\n")  # rows in another order
    assert stray.probe(root)["n_odometry"] == 8
    (root / "odometry.csv").write_text("\n".join(lines[:8]) + "\n")
    assert why(root) == "rgb.mp4 has 8 frames and odometry.csv has 7 rows; this export is not supported"
    (root / "odometry.csv").write_text("\n".join(lines + [lines[-1]]) + "\n")
    assert why(root) == "rgb.mp4 has 8 frames and odometry.csv has 9 rows; this export is not supported"
    (root / "odometry.csv").write_text("\n".join(lines[:8] + [lines[8].replace(", 7, ", ", 6, ", 1)]) + "\n")
    assert why(root) == ("odometry.csv's frame column is not 0 to 7, one row for each frame; this export is not "
                         "supported")
    assert why(export(tmp_path / "b", n=8, n_depth=3)) == (
        "only 3 of rgb.mp4's 8 frames have a depth image; this export is not supported")
    assert why(export(tmp_path / "d", n=2, n_depth=0)) == (
        "the export has no depth images (depth/000000.png, ...); this export is not supported")
    assert why(export(tmp_path / "c", n=4, odometry=False)) == (
        "the export has no odometry.csv, so its frames cannot be turned upright; this export is not supported")


def test_depth_images_larger_than_a_phones(tmp_path, monkeypatch):
    """A depth or confidence image is refused by its size before it is decoded."""
    root = export(tmp_path, n=4)
    big = np.zeros((1000, 1100), np.uint16)
    Image.fromarray(big).save(root / "depth" / "000002.png")
    with pytest.raises(ValueError, match="depth/000002.png is 1100x1000, larger than a phone's depth images"):
        stray.depth_points(root, 2, SIZE, 0, None)
    Image.fromarray(big.astype(np.uint8)).save(root / "confidence" / "000001.png")
    with pytest.raises(ValueError, match="confidence/000001.png is 1100x1000, larger than a phone's depth images"):
        stray.depth_points(root, 1, SIZE, 0, None)
    assert len(stray.depth_points(root, 1, SIZE, 0, None, min_conf=0)[0]) == DEPTH[0] * DEPTH[1]
    monkeypatch.setattr(stray, "MAX_DEPTH_PIXELS", 1000)
    with pytest.raises(SystemExit, match="depth/000000.png is 256x192, larger than a phone's depth images; this "
                                         "export is not supported"):
        stray.probe(root)


def broken_chunk(a):
    """a as a PNG whose first IDAT chunk says it is 4 bytes longer than it is. Pillow then reads the CRC and the next
    chunk's length as data, finds no chunk type after them and raises SyntaxError, not OSError. a must be noisy, so
    that its data takes more than one IDAT chunk."""
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    b, pos = bytearray(buf.getvalue()), 8
    while b[pos + 4:pos + 8] != b"IDAT":
        pos += 12 + struct.unpack_from(">I", b, pos)[0]
    struct.pack_into(">I", b, pos, struct.unpack_from(">I", b, pos)[0] + 4)
    return bytes(b)


def noisy_depth():
    return (np.random.default_rng(0).random(DEPTH[::-1]) * 3000 + 200).astype(np.uint16)


def short_phys(a):
    """a as a PNG with a pHYs chunk of 4 bytes, where 9 belong: Pillow raises ValueError for it, not OSError."""
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    b, data = buf.getvalue(), b"pHYs\0\0\0\1"
    return b[:33] + struct.pack(">I", 4) + data + struct.pack(">I", zlib.crc32(data)) + b[33:]  # after IHDR


def test_depth_and_confidence_images_that_cannot_be_used(tmp_path):
    """A depth or confidence image that Pillow cannot decode is a ValueError that names the file, also for a broken
    chunk (Pillow's SyntaxError) or a short pHYs chunk (Pillow's ValueError), and without the server's path. A
    confidence image must have one channel: an RGB or LA one of the right size said "256x192 but depth is
    256x192" before. The frames stage names a damaged first depth image without the server's path too."""
    root = export(tmp_path, n=4)
    (root / "depth" / "000001.png").write_bytes(broken_chunk(noisy_depth()))
    with pytest.raises(ValueError, match=r"^depth/000001.png cannot be read: broken PNG file \(chunk "):
        stray.depth_points(root, 1, SIZE, 0, None)
    good = (root / "depth" / "000002.png").read_bytes()
    (root / "depth" / "000002.png").write_bytes(good[:len(good) // 2])
    with pytest.raises(ValueError, match=r"^depth/000002.png cannot be read: image file is truncated"):
        stray.depth_points(root, 2, SIZE, 0, None)
    (root / "confidence" / "000003.png").write_bytes(b"not a png")
    with pytest.raises(ValueError, match=r"^confidence/000003.png cannot be read: it is not an image, or its header is damaged$"):
        stray.depth_points(root, 3, SIZE, 0, None)
    (root / "depth" / "000002.png").write_bytes(short_phys(noisy_depth()))
    with pytest.raises(ValueError, match=r"^depth/000002.png cannot be read: Truncated pHYs chunk$"):
        stray.depth_points(root, 2, SIZE, 0, None)
    first = root / "depth" / "000000.png"
    keep = first.read_bytes()
    first.write_bytes(keep[:8] + b"\0" * 40)  # the PNG signature, then no header
    with pytest.raises(SystemExit) as e:
        stray.probe(root)
    assert str(e.value) == ("depth/000000.png cannot be read (it is not an image, or its header is damaged); this "
                            "export is not supported")
    first.write_bytes(keep)
    conf = root / "confidence" / "000000.png"
    grey = Image.open(conf)
    for mode in ("RGB", "LA"):
        grey.convert(mode).save(conf)
        with pytest.raises(ValueError, match=rf"^confidence/000000.png is a {mode} image, not one-channel confidence$"):
            stray.depth_points(root, 0, SIZE, 0, None)
    Image.fromarray(np.asarray(grey).astype(np.uint16)).save(conf)  # 16-bit grey is one channel too
    assert len(stray.depth_points(root, 0, SIZE, 0, None)[0]) == DEPTH[0] * DEPTH[1]


def test_a_damaged_frame_costs_only_that_frame(tmp_path):
    """The lidar stage on a capture with a depth image that has a broken chunk and a confidence image that is not
    an image: those frames get no samples, with a warning that names them, and the others give the scale. Before,
    the first failed the stage with a traceback and no record, and the second failed it with the server's path.
    Such frames are not cached, so a rerun reads them again. When they leave fewer than 3 views, the stage fails
    with a plain reason."""
    pytest.importorskip("rosbags")
    from test_webapp_lidar import failed, make_stray_job, scaled

    from harpor.stage_a import mvsam3d

    J = make_stray_job(tmp_path, portrait=1)
    names = mvsam3d.view_names(J / "pool")
    root = J / "video" / "stray" / "export"
    depth, conf = root / "depth" / f"{int(names[3][1:]):06d}.png", root / "confidence" / f"{int(names[5][1:]):06d}.png"
    good = {p: p.read_bytes() for p in (depth, conf)}
    depth.write_bytes(broken_chunk(noisy_depth()))
    conf.write_bytes(b"not a png")
    rec = scaled(J)
    assert rec["ok"] and rec["s"] == pytest.approx(1 / F.FAKE_DA3_SCALE, rel=0.01), rec["reason"]
    assert rec["warnings"][0].startswith(
        f"frames {names[3]}, {names[5]} have depth or confidence images that cannot be used, so they have no samples "
        f"(depth/{depth.name} cannot be read: broken PNG file (chunk ")
    assert rec["warnings"][1:] == [f"no LiDAR samples in 2 of {len(names)} views: {names[3]}, {names[5]}"]
    assert {v["name"] for v in rec["views"] if v["status"] == "no_samples"} == {names[3], names[5]}
    assert not (J / "lidar" / "samples.json").exists()
    for p, b in good.items():
        p.write_bytes(b)
    again = scaled(J)
    assert again["ok"] and again["warnings"] == [] and again["source"]["cache"] == "miss"
    assert (J / "lidar" / "samples.json").exists()

    (J / "lidar" / "samples.json").unlink()
    for n in names[2:]:  # only two views keep a depth image that can be read
        p = root / "depth" / f"{int(n[1:]):06d}.png"
        p.write_bytes(p.read_bytes()[:500])
    first = f"depth/{int(names[2][1:]):06d}.png"
    assert failed(J) == (f"only 2 of the {len(names)} views have depth and confidence images that can be used (at least 3 are "
                         f"needed); {first} cannot be read: image file is truncated")
    rec = json.loads((J / "pool" / "lidar.json").read_text())
    assert rec["warnings"] == [f"frames {', '.join(names[2:7])} and {len(names) - 7} more have depth or confidence "
                               f"images that cannot be used, so they have no samples ({first} cannot be read: image "
                               "file is truncated)"]

    # An export without confidence images: the reason names the depth images only.
    J = make_stray_job(tmp_path / "plain", portrait=1, without_confidence=range(24))
    names, root = mvsam3d.view_names(J / "pool"), J / "video" / "stray" / "export"
    for n in names[2:]:
        p = root / "depth" / f"{int(n[1:]):06d}.png"
        p.write_bytes(p.read_bytes()[:500])
    first = f"depth/{int(names[2][1:]):06d}.png"
    assert failed(J) == (f"only 2 of the {len(names)} views have depth images that can be used (at least 3 are needed); "
                         f"{first} cannot be read: image file is truncated")


def test_the_reason_of_a_stray_capture_names_its_causes(tmp_path):
    """A Stray capture's depth is aligned with the color camera, so the reason names no time window and no
    T_cam_lidar: with every depth pixel of low confidence, it names the confidence, the object's size and the
    masks."""
    pytest.importorskip("rosbags")
    from test_webapp_lidar import failed, make_stray_job

    from harpor.stage_a import mvsam3d

    J = make_stray_job(tmp_path, portrait=1)
    names = mvsam3d.view_names(J / "pool")
    for p in (J / "video" / "stray" / "export" / "confidence").glob("*.png"):
        Image.fromarray(np.ones(DEPTH[::-1], np.uint8)).save(p)
    assert failed(J) == (f"LiDAR points cover the object in only 0 of {len(names)} views (at least 3 are needed); "
                         "the phone's depth may have low confidence on the object, the object may be small or far "
                         "away, or the masks may miss it (see the check images)")


def test_a_phone_that_turned_and_no_confidence(tmp_path):
    root = export(tmp_path, n=8)
    lines = (root / "odometry.csv").read_text().splitlines()
    rows = [x.split(", ") for x in lines[1:]]
    for r in rows[:3]:  # 3 of the 8 frames held a quarter turn the other way
        R = Rotation.from_quat([float(x) for x in r[5:9]]).as_matrix() @ TURN
        r[5:9] = [f"{x:.9f}" for x in quat(R)]
    (root / "odometry.csv").write_text("\n".join([lines[0]] + [", ".join(r) for r in rows]) + "\n")
    for p in (root / "confidence").iterdir():
        p.unlink()
    info = stray.probe(root)
    assert info["rotation"] == 0 and info["upright_agreement"] == pytest.approx(5 / 8) and info["n_confidence"] == 0
    assert info["warnings"] == [
        "the export has no confidence/ images, so the LiDAR scale uses every depth pixel",
        f"only {5 / 8:.0%} of the frames are upright when turned 0 degrees (gravity in the image); the phone turned "
        "during the capture, so the other frames come out sideways"]

    # A capture filmed straight down: no frame shows which way is up, and the warning says so.
    down = [f"{x:.9f}" for x in quat(DOWN)]
    (root / "odometry.csv").write_text("\n".join([lines[0]] + [", ".join(r[:5] + down + r[9:]) for r in rows]) + "\n")
    info = stray.probe(root)
    assert info["rotation"] == 0 and info["upright_agreement"] is None
    assert info["warnings"][1] == ("the camera looked straight up or down in every frame, so gravity gives no turn; "
                                   "the frames are left as stored")
