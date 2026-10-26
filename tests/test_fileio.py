"""The shared file helpers (gator/fileio.py): JSON read and write, replace, and the private files."""

import os
import stat

import pytest

from gator import fileio


def test_write_json_is_atomic_with_plain_or_private_permissions(tmp_path):
    fileio.write_json(tmp_path / "a.json", {"x": 1})
    (tmp_path / "plain.json").write_text("{}")
    assert stat.S_IMODE((tmp_path / "a.json").stat().st_mode) == stat.S_IMODE((tmp_path / "plain.json").stat().st_mode)
    assert fileio.read_json(tmp_path / "a.json") == {"x": 1}
    fileio.write_json(tmp_path / "a.json", {"x": 2}, private=True)
    assert stat.S_IMODE((tmp_path / "a.json").stat().st_mode) == 0o600
    assert fileio.read_json(tmp_path / "a.json") == {"x": 2}
    with pytest.raises(TypeError):  # nothing to write: the old file stays whole
        fileio.write_json(tmp_path / "a.json", {"x": object()})
    assert fileio.read_json(tmp_path / "a.json") == {"x": 2}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "plain.json"]  # no temp file left
    (tmp_path / "bad.json").write_text("{")
    assert fileio.read_json(tmp_path / "bad.json") is None and fileio.read_json(tmp_path / "missing.json") is None


def test_replace_keeps_the_old_file_when_save_stops(tmp_path):
    f = tmp_path / "a.txt"
    fileio.replace(f, lambda p, text: p.write_text(text), "old")

    def stops(p):
        p.write_text("half")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fileio.replace(f, stops)
    assert f.read_text() == "old"


def test_write_private_makes_a_0600_file_in_a_0700_folder(tmp_path):
    d = tmp_path / "cfg" / "gator"
    f = d / "harpor_session.json"
    fileio.write_private(f, b'{"a": 1}')
    assert oct(os.stat(d).st_mode & 0o777) == "0o700" and oct(os.stat(f).st_mode & 0o777) == "0o600"
    assert fileio.read_private(f) == b'{"a": 1}'
    inode = os.stat(f).st_ino
    fileio.write_private(f, b'{"a": 2}')
    assert fileio.read_private(f) == b'{"a": 2}' and os.stat(f).st_ino != inode  # a new file, put in place at once
    assert sorted(p.name for p in d.iterdir()) == ["harpor_session.json"]  # no temp file left
    target = tmp_path / "elsewhere.json"
    target.write_text("keep")
    f.unlink()
    f.symlink_to(target)
    fileio.write_private(f, b"new")  # the link is replaced, never written through
    assert target.read_text() == "keep" and not f.is_symlink() and fileio.read_private(f) == b"new"
    os.chmod(d, 0o755)
    with pytest.raises(SystemExit, match="chmod 700"):
        fileio.write_private(f, b"x")
    os.chmod(d, 0o700)
    os.mkfifo(d / "fifo")
    with pytest.raises(SystemExit, match="chmod 600"):  # refused at once, without blocking on the FIFO
        fileio.read_private(d / "fifo")
