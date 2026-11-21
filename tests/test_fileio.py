"""The shared file helpers (harpor/fileio.py): JSON read and write, replace, and the private files."""

import stat

import pytest

from harpor import fileio


def test_config_dir_is_the_harpor_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert fileio.config_dir() == tmp_path / ".config" / "harpor"


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
    (tmp_path / "deep.json").write_text("[" * 100_000 + "]" * 100_000)  # json raises RecursionError, not ValueError
    assert fileio.read_json(tmp_path / "deep.json") is None


def test_replace_keeps_the_old_file_when_save_stops(tmp_path):
    f = tmp_path / "a.txt"
    fileio.replace(f, lambda p, text: p.write_text(text), "old")

    def stops(p):
        p.write_text("half")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fileio.replace(f, stops)
    assert f.read_text() == "old"
