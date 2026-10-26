"""A personal worker's session with the public HARPOR site (docs/public_site.md, Run a personal worker).

    python -m gator.webapp.worker login [--provider github|google] [--name NAME] [--port 8767] [--no-browser]
                                        [--site-config PATH]
    python -m gator.webapp.worker status | logout | add PATH [--name NAME]

login signs in as the user with PKCE and a loopback redirect (supa.sign_in), adds this machine on the site once
the owner approved the account (own_machine_bind), and writes ~/.config/gator/harpor_session.json, mode 0600 in
a folder of mode 0700. The file holds the site's URL and publishable key (from gator/webapp/site.json), the
user, the machine and the refresh token. The server with --worker opens it with Session.

Supabase takes each refresh token once, and a second use after its 10 s window ends the whole session. So only
one process uses the file at a time: Session, login, status and logout take the lock file next to it
(.harpor_session.lock, a flock with the holder's pid in it), and Session writes each new refresh token into the
file before the new access token is used (supa.Tokens). The access token lives only in memory.

status prints the account and this machine. logout ends the session at Supabase (scope local: this machine
only) and takes the refresh token out of the file; the file keeps the machine's id, so the next login on this
machine binds the same machine again instead of using up one of the user's machines. add streams a local file
to the server on this machine (PUT /api/upload with webapp.json's login cookie), and the worker copies the new
job to the site.

No command prints a token, a code, a verifier or a key. Exit codes: 0 for success, 1 for an error with a
one-line reason, 2 for wrong arguments.
"""

import contextlib
import fcntl
import functools
import json
import os
import re
import socket
import webbrowser
from pathlib import Path

import anyio
import httpx

from gator.webapp import jobs, steps, supa

FILE = Path(".config") / "gator" / "harpor_session.json"  # under the home folder
LOCK = ".harpor_session.lock"  # next to the session file
VERSION = 1
FIELDS = ("url", "publishable_key", "user_id", "machine_name", "mirror_since")  # what login writes
MACHINE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,59}")  # machines.name
KEY_FILES = {"astra": "openai.env", "opus": "anthropic.env"}  # in ~/.config/gator, as the HARPOR CLI has them
LOGIN = "python -m gator.webapp.worker login"
START = "python -m gator.webapp --worker"
WAITS = ("Your account waits for approval. HARPOR adds this machine when the server starts after the owner "
         "approves you.")
ADD_TIMEOUT = httpx.Timeout(600, connect=10)  # the server answers once the whole file is in


def path_of(path=None):
    return Path(path) if path else Path.home() / FILE


def machine_name(host=None):
    """This machine's name on the site by default: the host name's first label, cut to the allowed characters."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", (host or socket.gethostname()).split(".")[0]).lstrip("_.-")[:60]
    return name or "machine"


def read(path=None):
    """The session file. Refuses (SystemExit) a file that is not this user's with mode 0600 in a folder of mode
    0700, and one that login did not write."""
    path = path_of(path)
    if not os.path.lexists(path):
        raise SystemExit(f"this machine has no HARPOR session ({path}); run {LOGIN}")
    bad = f"{path} is not a HARPOR session file; run {LOGIN}"
    try:
        data = json.loads(supa.read_private(path))
    except ValueError:
        raise SystemExit(bad) from None
    if not isinstance(data, dict) or data.get("version") != VERSION or not all(
            isinstance(data.get(k), str) and data[k] for k in FIELDS):
        raise SystemExit(bad)
    if data["publishable_key"].startswith("sb_secret_"):
        raise SystemExit(bad)
    return data


def write(data, path=None):
    """The session file: a 0600 temp file, then os.replace; makes the folder with mode 0700 when it is missing."""
    supa.write_private(path_of(path), (json.dumps(data, indent=1) + "\n").encode())


class Lock:
    """The session file's lock: a flock on .harpor_session.lock, which holds the pid of its holder."""

    def __init__(self, path=None):
        self.session = path_of(path)
        self.path, self.fd = self.session.with_name(LOCK), None

    def __repr__(self):
        return f"Lock({self.path}, {'held' if self.fd is not None else 'free'})"

    def _open(self):
        supa.private_folder(self.path.parent, make=True)
        try:
            return os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        except OSError:  # a symlink, or a folder in its place
            raise SystemExit(f"{self.path} must be a file of this user (delete it)") from None

    def pid(self):
        """The pid that the lock file names, or None."""
        with contextlib.suppress(OSError, ValueError):
            return int(self.path.read_text().strip()) or None
        return None

    def holder(self):
        """The pid of another process that holds the lock (None when that pid is unknown), or False when it is free."""
        if self.fd is not None:
            return False
        fd = self._open()
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return self.pid()
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    def take(self):
        """Take the lock and write this pid into it; SystemExit while another process holds it."""
        if self.fd is not None:
            return
        fd = self._open()
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise SystemExit(f"another HARPOR process holds {self.session} (pid {self.pid() or 'unknown'}): the "
                             f"server with --worker, or a login; stop it first") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self.fd = fd

    def release(self):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


class Session:
    """The session of a personal worker: the session file, read under its lock, which this object holds until
    close(). bearer() gives an access token with at least 5 minutes left and renews it first when needed; each
    renewal writes the new refresh token into the file before anything uses the new access token. supa.SignedOut
    when Supabase refuses the refresh token. transport: an httpx transport for tests."""

    def __init__(self, path=None, transport=None):
        self.path = path_of(path)
        self.lock = Lock(self.path)
        self.lock.take()
        try:
            data = read(self.path)  # under the lock: a login may have written it a moment ago
            if not data.get("refresh_token"):
                raise SystemExit(f"this machine is signed out of the HARPOR site; run {LOGIN}")
        except BaseException:
            self.lock.release()
            raise
        self.data = data
        self.url, self.publishable_key = data["url"], data["publishable_key"]
        self.user_id, self.email, self.provider = data["user_id"], data.get("email"), data.get("provider")
        self.machine_id, self.machine_name = data.get("machine_id"), data["machine_name"]
        self.mirror_since = data["mirror_since"]
        self.tokens = supa.Tokens(supa.Auth(self.url, self.publishable_key, transport), data["refresh_token"],
                                  self._save)

    def __repr__(self):
        return f"Session({self.email or self.user_id}, {self.machine_name})"

    async def bearer(self):
        return await self.tokens.bearer()

    async def refresh(self):
        await self.tokens.refresh()

    def _save(self, refresh_token):
        self.data.update(refresh_token=refresh_token, saved_at=supa.now())
        write(self.data, self.path)

    def set_machine(self, machine_id):
        """This machine's id on the site, kept in the file (a login before the approval had none)."""
        self.machine_id = self.data["machine_id"] = machine_id
        write(self.data, self.path)

    def close(self):
        self.lock.release()


def local_key(agent, folder=None):
    """The machine's own API key for agent (--local-keys), read now: the OPENAI_API_KEY= line of
    ~/.config/gator/openai.env (astra) or the ANTHROPIC_API_KEY= line of anthropic.env (opus). The file must be
    this user's with mode 0600 in a folder of mode 0700. An error names the file and the line, never the text."""
    if agent not in KEY_FILES:
        raise ValueError(f"no key file for the agent {agent!r}")
    var = steps.KEY_VAR[agent]
    path = (Path(folder) if folder else Path.home() / ".config" / "gator") / KEY_FILES[agent]
    for line in supa.read_private(path).decode("utf-8", errors="replace").splitlines():
        k, sep, v = line.strip().removeprefix("export ").partition("=")
        if sep and k.strip() == var:
            key = v.strip().strip("\"'")
            if not jobs.KEY.fullmatch(key):
                raise ValueError(f"the {var} line of {path} is not one word of letters, digits, - and _")
            return key
    raise ValueError(f"{path} has no {var}= line")


# ---- the commands ----

async def _bind(supa_, user_id, machine_id, name):
    """(approved, machine id) for a new sign-in: own_machine_bind when the account is approved, with the machine id
    of an earlier session (a new machine when the owner removed that one); (False, None) before the approval."""
    store = supa.Store(supa_, user_id, machine_id)
    acc = await store.access()
    if not (acc or {}).get("approved"):
        return False, None
    try:
        return True, await store.bind(name)
    except supa.Error as e:
        if machine_id is None or not (e.message or "").startswith("no such machine"):
            raise
    store.machine_id = None
    return True, await store.bind(name)


def _opener(browser):
    if callable(browser):
        return browser
    return webbrowser.open if browser else None


def login(provider="github", name=None, port=8767, browser=True, site_config=None, *, path=None, transport=None,
          timeout=supa.LOGIN_TIMEOUT):
    """`worker login`. browser: open the sign-in page with webbrowser (True), only print it (False), or a function
    of the page's URL (tests). A login over an earlier session of the same account and project keeps its machine
    id and mirror_since. Returns the exit code."""
    from gator.webapp import site

    name = machine_name() if name is None else name
    if not isinstance(name, str) or not MACHINE.fullmatch(name):
        print(f"the machine name has 1 to 60 letters, digits, _ . and -, and starts with a letter or digit: {name!r}")
        return 2
    try:
        cfg = site.load_config(site_config or site.CONFIG)
    except site.SiteError as e:
        print(f"cannot read the site's config: {e}")
        return 1
    path = path_of(path)
    lock = Lock(path)
    try:
        lock.take()
        try:
            old = read(path) if os.path.lexists(path) else {}
        except SystemExit:
            old = {}  # a file that is not a session file: this login replaces it
        grant = anyio.run(functools.partial(supa.sign_in, cfg["url"], cfg["publishableKey"], provider, port,
                                            _opener(browser), print, timeout, transport))
        if (old.get("url"), old.get("user_id")) != (cfg["url"], grant.user_id):
            old = {}  # another account or project: this machine is new there

        async def bearer():
            return grant.access_token

        async def bind():
            async with supa.Supa(cfg["url"], cfg["publishableKey"], bearer=bearer, transport=transport) as s:
                return await _bind(s, grant.user_id, old.get("machine_id"), name)

        approved, machine_id = anyio.run(bind)
        write({"version": VERSION, "url": cfg["url"], "publishable_key": cfg["publishableKey"],
               "user_id": grant.user_id, "email": grant.email, "provider": grant.provider or provider,
               "machine_id": machine_id, "machine_name": name, "mirror_since": old.get("mirror_since") or jobs.stamp(),
               "refresh_token": grant.refresh_token, "saved_at": supa.now()}, path)
    except SystemExit as e:
        print(e)
        return 1
    except supa.Error as e:
        print(f"the sign-in did not finish: {e}")
        return 1
    finally:
        lock.release()
    if not approved:
        print(WAITS)
    print(f"signed in as {grant.email or grant.user_id}; this machine is {name} on the site. Start HARPOR with: "
          f"{START}")
    return 0


def status(*, path=None, transport=None):
    """`worker status`: the account and this machine; while the server holds the session, only what the file says."""
    path = path_of(path)
    try:
        data = read(path)
        held = Lock(path).holder()
    except SystemExit as e:
        print(e)
        return 1
    print(f"account: {data.get('email') or data['user_id']} ({data.get('provider') or 'unknown provider'})")
    print(f"machine: {data['machine_name']} ({data.get('machine_id') or 'not on the site yet'})")
    if held is not False:
        print(f"the HARPOR server on this machine holds the session (pid {held or 'unknown'})")
        return 0
    if not data.get("refresh_token"):
        print(f"signed out on this machine; run {LOGIN}")
        return 1

    async def ask(s):
        store = supa.Store.of(s, transport=transport)
        try:
            return await store.access(), await store.storage_used()
        finally:
            await store.close()

    try:
        s = Session(path, transport)
    except SystemExit as e:
        print(e)
        return 1
    try:
        acc, used = anyio.run(ask, s)
    except supa.Error as e:
        print(e if isinstance(e, supa.SignedOut) else f"the HARPOR site did not answer: {e}")
        return 1
    finally:
        s.close()
    acc = acc or {}
    yes = {True: "yes", False: "no"}
    print(f"approved: {yes[bool(acc.get('approved'))]}; agents: {', '.join(acc.get('agents') or []) or 'none'}")
    print(f"storage on the site: {used / 1e6:.0f} of {acc.get('max_result_bytes', 0) / 1e6:.0f} MB")
    print(f"the worker does not run; start it with {START}")
    return 0


def logout(*, path=None, transport=None):
    """`worker logout`: ends the session at Supabase (scope local) and takes the refresh token out of the file. The
    machine and its jobs stay on the site, and the file keeps the machine's id for the next login."""
    path = path_of(path)
    try:
        if not read(path).get("refresh_token"):
            print("this machine is signed out already")
            return 0
        s = Session(path, transport)
    except SystemExit as e:
        print(e)
        return 1

    async def end():
        await s.tokens.auth.logout(await s.bearer())  # the refresh first writes the new token: no token is lost

    try:
        anyio.run(end)
    except supa.SignedOut:
        pass  # the session had ended already
    except supa.Error as e:
        print(f"the HARPOR site did not answer ({e}); its copy of the session ends when it expires")
    try:
        s.data.update(refresh_token=None, saved_at=supa.now())
        write(s.data, path)
    except SystemExit as e:
        print(e)
        return 1
    finally:
        s.close()
    print("signed out on this machine; the machine and its jobs stay on the site")
    return 0


def add(path, name=None):
    """`worker add PATH`: stream a local file to the server on this machine, which runs all its upload checks, with
    the login cookie of ~/.config/gator/webapp.json (never printed). The worker copies the new job to the site."""
    from gator.webapp import server

    src = Path(path)
    if not src.is_file():
        print(f"{src} is not a file")
        return 1
    info_path = Path.home() / ".config" / "gator" / "webapp.json"
    if not info_path.exists():
        print(f"no HARPOR server runs on this machine ({info_path} is missing); start it: {START}")
        return 1
    try:
        info = json.loads(supa.read_private(info_path))
        port, token = int(info["port"]), str(info["token"])
    except SystemExit as e:
        print(f"{e}; start the server first: {START}")
        return 1
    except (ValueError, KeyError, TypeError):
        print(f"{info_path} names no server; start it: {START}")
        return 1
    own = info.get("worker") is True
    if not own:
        print("the server runs without --worker, so this job stays on this machine")
    try:
        with open(src, "rb") as f:
            r = httpx.put(f"http://127.0.0.1:{port}/api/upload", params={"name": name or src.name}, content=f,
                          headers={"Content-Length": str(src.stat().st_size), "Cookie": f"{server.COOKIE}={token}"},
                          timeout=ADD_TIMEOUT)
    except httpx.HTTPError as e:
        print(f"no HARPOR server answers on 127.0.0.1:{port} ({type(e).__name__}); start it with {START}")
        return 1
    try:
        out = r.json()
    except ValueError:
        out = {}
    if r.status_code != 200 or not isinstance(out, dict) or not isinstance(out.get("id"), str):
        detail = out.get("detail") if isinstance(out, dict) else None
        print(f"the server refused the file: {detail or r.status_code}")
        return 1
    print(f"job {out['id']} made" + ("; the worker copies it to the site within 10 s" if own else ""))
    return 0
