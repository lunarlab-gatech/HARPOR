"""The personal worker's session and commands (harpor/webapp/session.py): login against the fake Supabase's Auth
through a real loopback on 127.0.0.1, the 0600 session file and its lock, the token renewal that saves first,
status, logout, add, local keys, the server's real --worker start, and a real --worker server process that copies
a job made with `worker add` to the fake site."""

import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import anyio
import pytest

import fake_supabase as fs
from harpor import fileio
from harpor.webapp import server, session, supa

ROOT = Path(__file__).resolve().parents[1]
URL = "https://abcdefghijklmnop.supabase.co"  # site.json's: the fake answers on any host
SITE = {"url": URL, "publishableKey": fs.PUBLISHABLE}


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    return h


@pytest.fixture
def fake():
    return fs.FakeSupabase()


def site_json(tmp_path, **fields):
    p = tmp_path / "site.json"
    p.write_text(json.dumps({**SITE, **fields}))
    return p


def browser(fake, user, answer=None, seen=None):
    """A browser that signs user in: the provider and Supabase send it back to the loopback with a code for the
    page's challenge (or with answer instead), from a thread, as a real browser would."""
    def open_page(page):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(page).query))
        if seen is not None:
            seen.append(q)
        back = q["redirect_to"] + (answer or "?code=" + fake.authorize(user, q["code_challenge"]))

        def go():
            time.sleep(0.1)
            try:
                urllib.request.urlopen(back, timeout=10).read()
            except OSError:
                pass
        threading.Thread(target=go, daemon=True).start()
    return open_page


def login(fake, user, home, tmp_path, name="lab-box", **kw):
    kw.setdefault("timeout", 20)
    return session.login("github", name, 0, kw.pop("opener", None) or browser(fake, user), site_json(tmp_path),
                         transport=fake.transport, **kw)


def session_file(home):
    return home / ".config" / "harpor" / "harpor_session.json"


def hold_lock(home):
    """Another process that holds the session's lock, as a running server does; it prints ready, then waits."""
    lock = home / ".config" / "harpor" / ".harpor_session.lock"
    code = ("import fcntl, os, sys, time\n"
            f"fd = os.open({str(lock)!r}, os.O_RDWR | os.O_CREAT, 0o600)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\nos.ftruncate(fd, 0)\nos.write(fd, f'{os.getpid()}\\n'.encode())\n"
            "print('ready', flush=True)\ntime.sleep(60)\n")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.parent.chmod(0o700)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    return p


def secrets_of(fake):
    """Every token and code the fake handed out."""
    return [*fake.tokens, *fake.refresh_tokens, *fake.codes]


def test_login_binds_the_machine_and_writes_a_private_file(fake, home, tmp_path, capsys):
    c = fake.add_user("c@example.org", approved=True)
    seen = []
    assert login(fake, c, home, tmp_path, opener=browser(fake, c, seen=seen)) == 0
    f = session_file(home)
    assert stat.S_IMODE(f.stat().st_mode) == 0o600 and stat.S_IMODE(f.parent.stat().st_mode) == 0o700
    data = json.loads(f.read_text())
    mid = fake.row("machines", user_id=c)["id"]
    assert set(data) == {"version", "url", "publishable_key", "user_id", "email", "provider", "machine_id",
                         "machine_name", "mirror_since", "refresh_token", "saved_at"}
    assert data == {**data, "version": 1, "url": URL, "publishable_key": fs.PUBLISHABLE, "user_id": c,
                    "email": "c@example.org", "provider": "github", "machine_id": mid, "machine_name": "lab-box"}
    assert data["refresh_token"] in fake.refresh_tokens and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d",
                                                                         data["mirror_since"])
    assert seen[0]["provider"] == "github" and seen[0]["code_challenge_method"] == "s256"
    assert re.fullmatch(r"http://127\.0\.0\.1:\d+/auth/callback", seen[0]["redirect_to"])
    out = capsys.readouterr().out
    assert "signed in as c@example.org; this machine is lab-box on the site. Start HARPOR with: python -m " \
           "harpor.webapp --worker\n" in out
    assert not any(s in out for s in secrets_of(fake))

    # a login again keeps the machine and mirror_since (and may rename it); another account gets its own machine
    assert login(fake, c, home, tmp_path, name="lab-box-2") == 0
    again = json.loads(f.read_text())
    assert (again["machine_id"], again["mirror_since"]) == (mid, data["mirror_since"])
    assert again["refresh_token"] != data["refresh_token"] and fake.row("machines", id=mid)["name"] == "lab-box-2"
    d = fake.add_user("d@example.org", approved=True)
    assert login(fake, d, home, tmp_path) == 0
    other = json.loads(f.read_text())
    assert other["user_id"] == d and other["machine_id"] == fake.row("machines", user_id=d)["id"] != mid
    assert len(fake.rows("machines")) == 2

    # an account that waits for approval: no machine yet (the server binds it once the owner approves)
    b = fake.add_user("b@example.org")
    assert login(fake, b, home, tmp_path) == 0
    assert json.loads(f.read_text())["machine_id"] is None and session.WAITS in capsys.readouterr().out
    # the owner removed the machine: the next login makes a new one
    assert login(fake, c, home, tmp_path) == 0
    fake.t["machines"].clear()
    assert login(fake, c, home, tmp_path) == 0
    assert json.loads(f.read_text())["machine_id"] == fake.row("machines", user_id=c)["id"]


def test_login_refusals(fake, home, tmp_path, capsys):
    c = fake.add_user("c@example.org", approved=True)
    assert login(fake, c, home, tmp_path, name="-box") == 2
    assert "the machine name has 1 to 60 letters" in capsys.readouterr().out
    assert session.login("github", "box", 0, False, tmp_path / "none.json") == 1
    assert "cannot read the site's config: no " in capsys.readouterr().out
    assert session.login("github", "box", 0, False, site_json(tmp_path, publishableKey=fs.SECRET)) == 1
    assert "never the secret key" in capsys.readouterr().out
    assert login(fake, c, home, tmp_path, opener=browser(fake, c, "?error=access_denied&error_description=no+way")) == 1
    assert "the sign-in did not finish: the sign-in failed: no way" in capsys.readouterr().out
    assert login(fake, c, home, tmp_path, opener=lambda page: None, timeout=0.5) == 1
    assert "did not come back within 0.5 s" in capsys.readouterr().out
    assert not session_file(home).exists() and fake.rows("machines") == []

    p = hold_lock(home)  # the server holds the session
    try:
        assert login(fake, c, home, tmp_path) == 1
        assert f"another HARPOR process holds {session_file(home)} (pid {p.pid})" in capsys.readouterr().out
    finally:
        p.kill()
        p.wait()
    conf = home / ".config" / "harpor"
    conf.chmod(0o755)  # others may open the folder
    assert login(fake, c, home, tmp_path) == 1 and "chmod 700" in capsys.readouterr().out
    conf.chmod(0o700)
    session_file(home).write_text("not json")
    session_file(home).chmod(0o600)
    assert login(fake, c, home, tmp_path) == 0  # a broken file is replaced


def write_session(fake, home, user, machine_id=None, **fields):
    data = {"version": 1, "url": fs.URL, "publishable_key": fs.PUBLISHABLE, "user_id": user,
            "email": fake.users[user], "provider": "github", "machine_id": machine_id,
            "machine_name": "lab-box", "mirror_since": "2000-01-01T00:00:00",
            "refresh_token": fake.grant(user)["refresh_token"], "saved_at": supa.now(), **fields}
    session.write(data, session_file(home))
    return data


def test_session_holds_the_lock_and_saves_each_new_token_first(fake, home, monkeypatch):
    c = fake.add_user("c@example.org", approved=True)
    data = write_session(fake, home, c)
    s = session.Session(transport=fake.transport)
    try:
        assert (s.url, s.publishable_key, s.user_id, s.email, s.machine_id, s.machine_name, s.mirror_since) == (
            fs.URL, fs.PUBLISHABLE, c, "c@example.org", None, "lab-box", "2000-01-01T00:00:00")
        lock = session_file(home).with_name(".harpor_session.lock")
        assert lock.read_text().strip() == str(os.getpid()) and stat.S_IMODE(lock.stat().st_mode) == 0o600
        with pytest.raises(SystemExit, match=rf"another HARPOR process holds .* \(pid {os.getpid()}\)"):
            session.Session(transport=fake.transport)
        p = subprocess.run([sys.executable, "-c", "from harpor.webapp import session; session.Session()"],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
        assert p.returncode == 1 and "another HARPOR process holds" in p.stderr

        saved, real = [], session.write
        monkeypatch.setattr(session, "write", lambda d, path=None: (saved.append(d["refresh_token"]), real(d, path)))
        token = anyio.run(s.bearer)
        monkeypatch.setattr(session, "write", real)
        new = json.loads(session_file(home).read_text())["refresh_token"]
        assert saved == [new] and new != data["refresh_token"] and fake.tokens[token][0] == c
        assert anyio.run(s.bearer) == token  # no renewal while it has more than 5 minutes left
        s.set_machine("c1c1c1c1-0000-0000-0000-000000000001")
        assert json.loads(session_file(home).read_text())["machine_id"] == s.machine_id
    finally:
        s.close()
    s = session.Session(transport=fake.transport)  # free again after close()
    fake.refresh_tokens.clear()  # Supabase no longer takes the token
    try:
        with pytest.raises(supa.SignedOut, match="run python -m harpor.webapp.worker login"):
            anyio.run(s.bearer)
    finally:
        s.close()

    write_session(fake, home, c, refresh_token=None)
    with pytest.raises(SystemExit, match="this machine is signed out of the HARPOR site; run python -m"):
        session.Session(transport=fake.transport)
    for broken in ({"version": 2}, {"url": ""}, {"publishable_key": fs.SECRET}):
        write_session(fake, home, c, **broken)
        with pytest.raises(SystemExit, match="is not a HARPOR session file"):
            session.read()
    write_session(fake, home, c)
    session_file(home).chmod(0o644)
    with pytest.raises(SystemExit, match="mode 0600"):
        session.Session(transport=fake.transport)
    session_file(home).unlink()
    with pytest.raises(SystemExit, match="this machine has no HARPOR session"):
        session.Session(transport=fake.transport)


def test_status_and_logout(fake, home, capsys):
    assert session.status() == 1 and "this machine has no HARPOR session" in capsys.readouterr().out
    c = fake.add_user("c@example.org", approved=True, agents=["opus"], max_result_bytes=200_000_000)
    mid = fake.put("machines", {"user_id": c, "name": "lab-box"})["id"]
    data = write_session(fake, home, c, mid)
    fake.object("job-files", f"{c}/x/0123456789abcdef/a.glb", size=12_000_000)
    assert session.status(transport=fake.transport) == 0
    out = capsys.readouterr().out
    assert f"account: c@example.org (github)\nmachine: lab-box ({mid})\n" in out
    assert "approved: yes; agents: opus\n" in out and "storage on the site: 12 of 200 MB" in out
    assert "the worker does not run" in out and not any(s in out for s in secrets_of(fake))
    rotated = json.loads(session_file(home).read_text())["refresh_token"]
    assert rotated != data["refresh_token"]  # status renewed once and saved the new token

    p = hold_lock(home)
    try:
        assert session.status(transport=fake.transport) == 0
        assert f"the HARPOR server on this machine holds the session (pid {p.pid})" in capsys.readouterr().out
        assert session.logout(transport=fake.transport) == 1
        assert "another HARPOR process holds" in capsys.readouterr().out
    finally:
        p.kill()
        p.wait()

    n = len(fake.tokens)
    assert session.logout(transport=fake.transport) == 0
    assert "signed out on this machine; the machine and its jobs stay on the site" in capsys.readouterr().out
    left = json.loads(session_file(home).read_text())
    assert left["refresh_token"] is None and left["machine_id"] == mid and len(fake.tokens) == n  # its token went
    assert session.status(transport=fake.transport) == 1 and "signed out on this machine" in capsys.readouterr().out
    assert session.logout(transport=fake.transport) == 0 and "signed out already" in capsys.readouterr().out


def test_logout_then_login_keeps_the_machine(fake, home, tmp_path):
    c = fake.add_user("c@example.org", approved=True)
    assert login(fake, c, home, tmp_path) == 0
    mid = json.loads(session_file(home).read_text())["machine_id"]
    assert session.logout(transport=fake.transport) == 0
    assert login(fake, c, home, tmp_path) == 0
    assert json.loads(session_file(home).read_text())["machine_id"] == mid and len(fake.rows("machines")) == 1


def test_local_key_reads_one_line_now_and_never_quotes_it(home):
    conf = home / ".config" / "harpor"
    conf.mkdir(parents=True)
    conf.chmod(0o700)
    key = "sk-ant-test-" + "k" * 30
    f = conf / "anthropic.env"
    f.write_text(f"# a key\nOTHER=1\nexport ANTHROPIC_API_KEY='{key}'\n")
    f.chmod(0o600)
    assert session.local_key("opus") == key
    f.write_text(f"ANTHROPIC_API_KEY={key}\n")  # read again at each use
    assert session.local_key("opus") == key
    for text, error, match in ((f"ANTHROPIC_API_KEY={key} two\n", ValueError, "is not one word"),
                               (f"OPENAI_API_KEY={key}\n", ValueError, r"has no ANTHROPIC_API_KEY= line")):
        f.write_text(text)
        with pytest.raises(error, match=match) as e:
            session.local_key("opus")
        assert key not in str(e.value)
    f.chmod(0o644)
    with pytest.raises(SystemExit, match="mode 0600") as e:
        session.local_key("opus")
    assert key not in str(e.value)
    with pytest.raises(SystemExit, match="openai.env is missing"):
        session.local_key("astra")
    with pytest.raises(ValueError):
        session.local_key("none")


def test_local_key_reads_the_file_it_is_given_by_the_same_rules(tmp_path):
    """The terminal command's --key-file."""
    d = tmp_path / "keys"
    d.mkdir(mode=0o700)
    key = "sk-ant-test-" + "k" * 30
    f = d / "my.env"
    f.write_text(f"ANTHROPIC_API_KEY={key}\n")
    f.chmod(0o600)
    assert session.local_key("opus", f) == key and session.local_key("opus", str(f)) == key
    f.chmod(0o644)
    with pytest.raises(SystemExit, match="mode 0600") as e:
        session.local_key("opus", f)
    assert key not in str(e.value)
    f.chmod(0o600)
    d.chmod(0o755)
    with pytest.raises(SystemExit, match="mode 0700"):
        session.local_key("opus", f)
    d.chmod(0o700)
    with pytest.raises(SystemExit, match="other.env is missing"):
        session.local_key("opus", d / "other.env")


class Upload:
    """A stand-in for the server on this machine: it keeps each PUT /api/upload (headers, query, body)."""

    def __init__(self, status=200, answer=None):
        import http.server

        seen = self.seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_PUT(self):
                n = int(self.headers["Content-Length"])
                seen.append((self.path, dict(self.headers), self.rfile.read(n)))
                data = json.dumps(answer or {"id": "20261024-101010-abcdef"}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.http = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.http.server_address[1]
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def close(self):
        self.http.shutdown()
        self.http.server_close()


def webapp_json(home, port, worker=True, token="tok-" + "t" * 40):
    conf = home / ".config" / "harpor"
    conf.mkdir(parents=True, exist_ok=True)
    conf.chmod(0o700)
    fileio.write_private(conf / "webapp.json", json.dumps({"node": "box", "port": port, "token": token,
                                                           "worker": worker}).encode())
    return token


def test_add_streams_the_file_to_the_local_server_with_the_cookie(home, tmp_path, capsys):
    src = tmp_path / "my clip.mp4"
    src.write_bytes(b"v" * 300_000)
    assert session.add(src) == 1
    assert "webapp.json is missing); start it: python -m harpor.webapp --worker\n" in capsys.readouterr().out
    up = Upload()
    try:
        token = webapp_json(home, up.port)
        assert session.add(src, "cup.mp4") == 0
        out = capsys.readouterr().out
        assert out == "job 20261024-101010-abcdef made; the worker copies it to the site within 10 s\n"
        path, headers, body = up.seen[0]
        assert path == "/api/upload?name=cup.mp4" and body == src.read_bytes()
        assert headers["Content-Length"] == "300000" and headers["Cookie"] == f"{server.COOKIE}={token}"
        webapp_json(home, up.port, worker=False, token=token)
        assert session.add(src) == 0 and up.seen[1][0] == "/api/upload?name=my+clip.mp4"
        out = capsys.readouterr().out
        assert "the server runs without --worker, so this job stays on this machine" in out
        assert out.endswith("job 20261024-101010-abcdef made\n") and token not in out
    finally:
        up.close()
    refused = Upload(400, {"detail": "the file is not a video (.mp4 .mov .m4v .avi .mkv .webm) ..."})
    try:
        webapp_json(home, refused.port)
        assert session.add(src) == 1 and "the server refused the file: the file is not a video" in \
            capsys.readouterr().out
    finally:
        refused.close()
    with socket.socket() as s:  # a port where nothing listens
        s.bind(("127.0.0.1", 0))
        webapp_json(home, s.getsockname()[1])
    assert session.add(src) == 1 and "no HARPOR server answers on 127.0.0.1:" in capsys.readouterr().out
    assert session.add(tmp_path / "none.mp4") == 1


def test_the_server_starts_the_real_session(fake, home, capsys):
    """server.worker_of with the real session.py: a Session that holds the lock, or tunnel mode when it cannot."""
    c = fake.add_user("c@example.org", approved=True)
    write_session(fake, home, c)
    s = server.worker_of(())
    try:
        assert type(s) is session.Session and s.user_id == c
        assert server.worker_of(()) is None  # the lock is taken
        out = capsys.readouterr().out
        assert "the personal worker does not start: another HARPOR process holds" in out
    finally:
        s.close()
    write_session(fake, home, c, refresh_token=None)
    assert server.worker_of(()) is None and "signed out of the HARPOR site" in capsys.readouterr().out


def test_a_personal_worker_process_copies_a_job_added_with_worker_add(fake, home, tmp_path, monkeypatch):
    """The real server process with --worker in fake mode, against the fake Supabase over HTTP and a real session
    file: the worker renews the token (and saves it), binds the machine, a file added with `worker add` becomes a
    job on the site, and SIGTERM stops the worker and releases the session."""
    pytest.importorskip("av")
    from test_webapp_server import write_video
    from test_webapp_worker import FakeHTTP

    c = fake.add_user("c@example.org", approved=True)
    bridge = FakeHTTP(fake)
    first = write_session(fake, home, c, url=bridge.url)
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower() and not k.startswith("SLURM")}
    env.update(HOME=str(home), TMPDIR=str(tmp_path), PYTHONPATH=str(ROOT))
    log = tmp_path / "server.log"
    with open(log, "wb") as out:
        p = subprocess.Popen([sys.executable, "-m", "harpor.webapp", "--fake", "--worker", "--runs",
                              str(tmp_path / "runs")], cwd=ROOT, env=env, stdout=out,
                             stderr=subprocess.STDOUT)
    info = home / ".config" / "harpor" / "webapp.json"
    try:
        t0 = time.time()
        while not (fake.row("workers", user_id=c) and info.exists()):
            assert p.poll() is None and time.time() - t0 < 60, log.read_text()
            time.sleep(0.1)
        assert json.loads(info.read_text())["worker"] is True
        mid = fake.row("machines", user_id=c)["id"]
        saved = json.loads(session_file(home).read_text())
        assert saved["machine_id"] == mid and saved["refresh_token"] != first["refresh_token"]
        assert saved["refresh_token"] in fake.refresh_tokens
        video = write_video(tmp_path / "cup.mp4")
        assert session.add(video) == 0
        while not fake.row("jobs", user_id=c):
            assert p.poll() is None and time.time() - t0 < 90, log.read_text()
            time.sleep(0.1)
        row = fake.row("jobs", user_id=c)
        assert (row["user_id"], row["machine_id"], row["input_name"]) == (c, mid, "cup.mp4")
        wid = fake.row("workers", user_id=c)["id"]
        p.send_signal(signal.SIGTERM)
        assert p.wait(60) == 128 + signal.SIGTERM, log.read_text()
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        bridge.close()
    text = log.read_text()
    assert "the personal worker of lab-box runs in this server" in text and "Traceback" not in text, text
    assert not any(s in text for s in secrets_of(fake)) and fake.row("workers", id=wid)["stopped_at"] is not None
    assert session.Lock().holder() is False  # the process let the session go
