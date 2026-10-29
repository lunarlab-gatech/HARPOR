"""File helpers that the stages, the web app and the agent's helpers share: JSON read and write, an atomic
replace, and the private files of ~/.config/gator. Standard library only, since the stages run in different
envs.
"""

import contextlib
import json
import os
import stat
import tempfile
from pathlib import Path


def read_json(path):
    """The JSON in path, or None when the file is missing or does not parse."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, RecursionError):  # RecursionError: nested too deep for json
        return None


def write_json(path, obj, private=False, indent=2):
    """Write a temp file in the same folder, then os.replace it, so readers never see a partial file.

    The temp file is created exclusively, and os.replace swaps the name itself, so neither follows a
    symlink. It gets the usual umask permissions (not mkstemp's 0600), so the controller and the
    agent can read each other's files when they run as different users. private: mode 0600. indent:
    json.dumps's (the web app's files use 1).
    """
    path = Path(path)
    text = json.dumps(obj, indent=indent)
    tmp = path.with_name(f".{path.name}.{os.urandom(6).hex()}")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o666), "w") as f:
        f.write(text)
    os.replace(tmp, path)


def replace(path, save, /, *args, **kw):
    """save(tmp, ...) under a temporary name, then rename it to path, so a run that stops halfway
    leaves the old file whole."""
    tmp = path.with_name(".tmp_" + path.name)
    save(tmp, *args, **kw)
    os.replace(tmp, path)


# ---- private files: the session file, the key files, webapp.json ----

def private_folder(folder, make=False):
    """Refuses (SystemExit) a folder that is not this user's, or that group or others may open. make: make it
    with mode 0700 when it is missing."""
    folder = Path(folder)
    if make and not os.path.lexists(folder):
        folder.mkdir(mode=0o700, parents=True)
    try:
        st = os.lstat(folder)
    except FileNotFoundError:
        raise SystemExit(f"{folder} is missing (mkdir -m 700 {folder})") from None
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise SystemExit(f"{folder} must be a folder of this user with mode 0700 (chmod 700 {folder})")


def read_private(path):
    """The bytes of a file that only this user may read (mode 0600), in such a folder. Refuses (SystemExit)
    anything else, a symlink too. The check and the read use one file descriptor, so the file checked is
    the file read."""
    path = Path(path)
    private_folder(path.parent)
    bad = f"{path} must be a file of this user with mode 0600 (chmod 600 {path})"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)  # O_NONBLOCK: a FIFO does not hang
    except FileNotFoundError:
        raise SystemExit(f"{path} is missing") from None
    except OSError:  # a symlink
        raise SystemExit(bad) from None
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise SystemExit(bad)
        return f.read()


def write_private(path, data):
    """data (bytes) into path with mode 0600: a temp file in the same folder, then os.replace, so a reader
    never sees half a file. Makes the folder with mode 0700 when it is missing, and refuses one that others
    may open."""
    path = Path(path)
    private_folder(path.parent, make=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")  # mode 0600
    try:
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
