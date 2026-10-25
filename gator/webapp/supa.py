"""The personal worker's link to Supabase: its config and sign-in, an httpx client for PostgREST, RPC and
Storage, and the Store that worker.py talks through (addendum section 6.2).

A personal worker runs on a user's own GPU machine (python -m gator.webapp --worker). It signs in as that
user with PKCE and a loopback redirect, keeps the refresh token in a 0600 file and calls Supabase with the
user's access token and the publishable key, as role authenticated, under RLS and through the own_*
functions (Store). session.py builds that session from the parts here: sign_in (with pkce_pair,
authorize_url, Loopback and Auth), Tokens, read_private and write_private. No part of HARPOR uses the
project's secret key: Supa and Auth refuse one.

Secrets. The tokens and the PKCE verifier stay in Python variables. They never go to os.environ
(Runner._run_stage passes os.environ to every stage), to a command line, to a log or to an error. An Error
names the method, the path without its query, the status and Supabase's own message (PostgREST quotes a
function's exception, never a header), cut to 300 characters, with the key and the access token replaced
by [key]. An answer that is not JSON is not quoted. httpx logs every request's URL at INFO, so this module
sets the httpx and httpcore loggers to WARNING.

Supa sends `apikey: <publishable key>` and `Authorization: Bearer <the user's access token>`. A request
that never reached Supabase (no connection) is tried again, 4 tries in all, after 1, 2 and 4 s. A request
that may have reached it (a timeout, a broken answer, a 5xx or a 429) is tried again only when a second
copy does no harm (again=True, the default for GET and DELETE): a second own_job_create would make a
second job. A 401 (or Storage's "jwt expired") renews the token once and tries again.
"""

import base64
import contextlib
import hashlib
import http.server
import logging
import os
import re
import secrets
import socketserver
import stat
import tempfile
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import anyio
import httpx

log = logging.getLogger("harpor.supa")
for _name in ("httpx", "httpcore"):  # their INFO and DEBUG lines name every URL
    logging.getLogger(_name).setLevel(logging.WARNING)

AGENT = "harpor-worker"  # User-Agent
TIMEOUT = httpx.Timeout(60, connect=10)  # an upload of a large result file takes a while
AUTH_TIMEOUT = httpx.Timeout(6, connect=5)  # a refresh answers well inside the reuse window below
TRIES = 4
REFRESH_WINDOW = 8  # s: Supabase takes a used refresh token again for 10 s, so a second try stays inside that
REFRESH_MARGIN = 300  # s: Tokens.bearer gives a token with at least this much left
BUCKET = "job-files"
IMMUTABLE = "private, max-age=31536000, immutable"  # job-files names hold the content's hash
LOGIN_TIMEOUT = 300
PROVIDERS = ("github", "google")
NAME = re.compile(r"[a-z_][a-z0-9_]{0,62}")  # tables and functions
BUCKET_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")
OBJECT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_./-]{0,399}")
NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ProxyError)
OWN_ROW = ("status", "note", "prompt", "queue_pos", "stopped_at")  # what own_publish takes
SNAPSHOTS = ("frames", "selection", "parts")
JOB_COLUMNS = ("id,user_id,machine_id,created_at,updated_at,input_name,input_kind,input_bytes,status,note,prompt,"
               "queue_pos,stopped_at,deleted_at")


class Error(Exception):
    """Supabase refused a call, gave no answer, or a sign-in step failed. str() names the call and Supabase's
    message, never a key, a token or a header. status: the HTTP status (None without an answer); message:
    Supabase's own text (for PostgREST the function's exception, which it answers with 400 and code
    P0001); code: its error code."""

    def __init__(self, text, status=None, message="", code=None):
        super().__init__(text)
        self.status, self.message, self.code = status, message, code


class Offline(Error):
    """No answer from Supabase after the tries."""


class SignedOut(Error):
    """Supabase refused the refresh token. The user signs in again: python -m gator.webapp.worker login."""


def iso(t):
    """A timestamp for PostgREST: a datetime in ISO 8601, else t as it is (None or a string)."""
    return t.isoformat() if isinstance(t, datetime) else t


def now():
    return datetime.now(timezone.utc).isoformat()


def check_url(url):
    """url without a trailing slash, if it is https, or http on this machine (the Supabase CLI's local stack)."""
    u = urllib.parse.urlsplit(url if isinstance(url, str) else "")
    local = u.scheme == "http" and u.hostname in ("127.0.0.1", "localhost")
    if not (u.scheme == "https" or local) or not u.hostname or u.query or u.fragment or u.username:
        raise ValueError(f"not a Supabase project URL: {url!r} (https://<ref>.supabase.co)")
    return url.rstrip("/")


def object_name(user_id, job_id, sha256, path):
    """The job-files object of a served file: <user>/<job>/<sha256[:16]>/<path> (job_files' check)."""
    return f"{user_id}/{job_id}/{sha256[:16]}/{path}"


# ---- private files: the session file, the key files ----

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


# ---- answers ----

def error_of(method, path, r, hide=()):
    """The Error for answer r: Supabase's own message from a JSON body, cut, with every value of hide replaced."""
    try:
        body = r.json()
    except ValueError:
        body = None
    message, code = "", None
    if isinstance(body, dict):
        message = next((str(body[k]) for k in ("message", "msg", "error_description", "error") if body.get(k)), "")
        code = body.get("code") or body.get("error_code") or body.get("statusCode")
    for s in hide:
        if s:
            message = message.replace(s, "[key]")
    message = message[:300]
    text = f"{method} {path}: {r.status_code} {message or r.reason_phrase}".rstrip()
    return Error(text, r.status_code, message, code)


def expired(r):
    """True for an answer that refuses an expired access token: a 401, or Storage's 400 or 403 "jwt expired"."""
    if r.status_code == 401:
        return True
    if r.status_code not in (400, 403):
        return False
    try:
        message = str(r.json().get("message", ""))
    except (ValueError, AttributeError):
        return False
    return bool(re.search(r'jwt expired|"exp" claim', message, re.I))


def duplicate(r):
    """True for Storage's answer to an upload over an existing object (409, or 400 with statusCode 409)."""
    if r.status_code == 409:
        return True
    try:
        body = r.json()
    except ValueError:
        return False
    return r.status_code == 400 and isinstance(body, dict) and (str(body.get("statusCode")) == "409"
                                                                or body.get("error") == "Duplicate")


def retry_after(r, default):
    """The wait that a 429 or 503 asks for (Retry-After in seconds, at most 30), else default."""
    with contextlib.suppress(TypeError, ValueError):
        return min(float(r.headers.get("Retry-After")), 30)
    return default


def filter_of(v):
    """A PostgREST filter that matches v."""
    if v is None or isinstance(v, bool):
        return f"is.{str(v).lower() if v is not None else 'null'}"
    return f"eq.{v}"


def name(n, pattern=NAME):
    if not isinstance(n, str) or not pattern.fullmatch(n):
        raise ValueError(f"not a table, function or bucket name: {n!r}")
    return n


def quote(obj):
    """A Storage object name for a URL path. Refuses a name that could leave its folder."""
    if not isinstance(obj, str) or not OBJECT.fullmatch(obj) or ".." in obj or "//" in obj:
        raise ValueError(f"not a Storage object name: {obj!r}")
    return urllib.parse.quote(obj, safe="/")


def object_ok(obj):
    """True for a name that quote() takes."""
    try:
        quote(obj)
        return True
    except ValueError:
        return False


# ---- PostgREST, RPC and Storage ----

class Supa:
    """PostgREST, RPC and Storage of one project, as a user. key is the publishable key, which goes in the apikey
    header; a secret key is refused. bearer is an async callable that returns the user's access token
    (Tokens.bearer, session.Session.bearer). A 401 calls refresh() once (by default the refresh method of bearer's
    object) and tries again. transport: an httpx transport for tests (httpx.MockTransport) or the fake Supabase."""

    def __init__(self, url, key, bearer, refresh=None, transport=None, tries=TRIES, wait=1.0):
        self.url = check_url(url)
        if not isinstance(key, str) or not key:
            raise ValueError("no Supabase key")
        if key.startswith("sb_secret_"):
            raise ValueError("a user's session goes with the publishable key, never the secret key")
        if bearer is None:
            raise ValueError("a call to Supabase needs the user's session")
        self.key, self.bearer, self.tries, self.wait = key, bearer, tries, wait
        self.refresh = refresh or getattr(getattr(bearer, "__self__", None), "refresh", None)
        self.token = None  # the last access token sent, to hide in errors
        self.http = httpx.AsyncClient(headers={"apikey": key, "User-Agent": AGENT}, timeout=TIMEOUT,
                                      transport=transport)

    def __repr__(self):
        return f"Supa({self.url})"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def close(self):
        await self.http.aclose()

    async def auth(self):
        """The Authorization header: the bearer's access token."""
        self.token = await self.bearer()
        return {"Authorization": f"Bearer {self.token}"}

    async def call(self, method, path, *, params=None, json=None, content=None, headers=None, again=None, ok=None):
        """One request to <url><path>, tried again as the module docstring says. Returns the answer when its
        status is below 400 or ok(answer) is true; else raises Error (Offline without an answer, SignedOut when
        the session's refresh token was refused)."""
        again = method in ("GET", "HEAD", "DELETE") if again is None else again
        tries, renewed = 0, False
        while True:
            tries += 1
            try:
                r = await self.http.request(method, self.url + path, params=params, json=json, content=content,
                                            headers={**(headers or {}), **await self.auth()})
            except httpx.TransportError as e:
                if tries >= self.tries or not (again or isinstance(e, NOT_SENT)):
                    raise Offline(f"{method} {path}: no answer from Supabase ({type(e).__name__})") from None
                log.warning("%s %s: %s, trying again", method, path, type(e).__name__)
                await anyio.sleep(self.wait * 2 ** (tries - 1))
                continue
            if r.status_code < 400:
                return r
            if self.refresh and not renewed and expired(r):
                renewed, tries = True, tries - 1  # the renewal does not use up a try
                await self.refresh()
                continue
            if ok and ok(r):
                return r
            if again and tries < self.tries and (r.status_code >= 500 or r.status_code == 429):
                log.warning("%s %s: %s, trying again", method, path, r.status_code)
                await anyio.sleep(retry_after(r, self.wait * 2 ** (tries - 1)))
                continue
            raise error_of(method, path, r, (self.key, self.token))

    async def rpc(self, fn, again=False, **args):
        """POST /rest/v1/rpc/<fn> with args (p_...) as JSON: the function's answer, None for void."""
        r = await self.call("POST", f"/rest/v1/rpc/{name(fn)}", json=args, again=again)
        return r.json() if r.content else None

    async def select(self, table, columns="*", filters=None, order=None, limit=None, **eq):
        """Rows of table that RLS lets the user read: eq gives column=value filters; filters gives PostgREST
        filters as they are ({"or": "(a.is.null,b.eq.1)"})."""
        params = {"select": columns, **{k: filter_of(v) for k, v in eq.items()}, **(filters or {})}
        if order:
            params["order"] = order
        if limit:
            params["limit"] = str(limit)
        return (await self.call("GET", f"/rest/v1/{name(table)}", params=params)).json()

    async def upload(self, bucket, obj, data, content_type, cache=IMMUTABLE):
        """The standard upload of data (bytes) to bucket/obj, never over an existing object (x-upsert: false).
        An object that is already there counts as uploaded: job-files names hold the content's hash, and a try
        whose answer was lost finds its own upload. Tried again after a 5xx or a network error. A refusal by
        the Storage policies (the user's storage is full) raises Error with status 400 or 403."""
        await self.call("POST", f"/storage/v1/object/{name(bucket, BUCKET_NAME)}/{quote(obj)}", content=data,
                        again=True, ok=duplicate,
                        headers={"Content-Type": content_type, "Cache-Control": cache, "x-upsert": "false"})

    async def remove(self, bucket, objs):
        """Delete objects through the Storage API (a SQL delete leaves the file behind), 1,000 a call. Returns
        how many it deleted; a name that is not there, or that the policies keep, is skipped."""
        objs, n = list(objs), 0
        for o in objs:
            quote(o)  # refuses a name that could leave its folder
        for i in range(0, len(objs), 1000):
            r = await self.call("DELETE", f"/storage/v1/object/{name(bucket, BUCKET_NAME)}",
                                json={"prefixes": objs[i:i + 1000]})
            n += len(r.json()) if r.content else 0
        return n


# ---- sign-in of a personal worker ----

def pkce_pair():
    """(verifier, challenge) for a PKCE sign-in: 64 random bytes in base64url (86 characters), and
    base64url(SHA-256(verifier)) without padding."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    return verifier, base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


def authorize_url(url, provider, redirect, challenge):
    """The page that starts the sign-in with provider (github or google) at Supabase Auth. It sends the browser
    back to redirect with ?code=, which Auth.exchange turns into a session."""
    if provider not in PROVIDERS:
        raise ValueError(f"sign in with {' or '.join(PROVIDERS)}, not {provider!r}")
    q = urllib.parse.urlencode({"provider": provider, "redirect_to": redirect, "code_challenge": challenge,
                                "code_challenge_method": "s256"})
    return f"{check_url(url)}/auth/v1/authorize?{q}"


class Server(socketserver.TCPServer):
    """The loopback's listener. Not http.server.HTTPServer, which looks up the host's name at start (slow
    without DNS). A second login soon after the first finds the port free."""

    allow_reuse_address = True


class Loopback:
    """The sign-in's redirect target, http://127.0.0.1:<port>/auth/callback (addendum 7.3). It listens on
    127.0.0.1 only, answers only GET /auth/callback with a plain text page and no script, never logs a
    request line (it holds the code), and gives up after timeout seconds. port 0 takes a free port."""

    PATH = "/auth/callback"

    def __init__(self, port=8767, timeout=LOGIN_TIMEOUT):
        self.result, self.timeout = None, timeout  # result: ("code", code) or ("error", why)
        box = self

        class Handler(http.server.BaseHTTPRequestHandler):
            timeout = 10  # a connection that sends nothing does not hold up the wait

            def do_GET(self):
                u = urllib.parse.urlsplit(self.path)
                if u.path != Loopback.PATH:
                    return self.reply(404, "Not found.")
                q = urllib.parse.parse_qs(u.query)
                if box.result is not None:
                    return self.reply(409, "This sign-in is already done. You can close this tab.")
                if q.get("code"):
                    box.result = ("code", q["code"][0])
                    return self.reply(200, "Signed in. You can close this tab.")
                if q.get("error"):
                    box.result = ("error", (q.get("error_description") or q["error"])[0][:300])
                    return self.reply(400, f"The sign-in failed: {box.result[1]}")
                self.reply(400, "This address holds no sign-in code. Start the sign-in again.")

            def reply(self, status, text):
                data = (text + "\n").encode()
                self.send_response(status)
                for k, v in (("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(data))),
                             ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
                             ("Referrer-Policy", "no-referrer")):
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        try:
            self.server = Server(("127.0.0.1", port), Handler)
        except OSError as e:
            raise Error(f"cannot listen on 127.0.0.1:{port} ({e.strerror}); pass another --port") from None
        self.port = self.server.server_address[1]
        self.deadline = time.monotonic() + timeout

    @property
    def redirect(self):
        return f"http://127.0.0.1:{self.port}{self.PATH}"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.server.server_close()

    def wait(self):
        """The code that the browser brought back; Error for an error answer or after the timeout."""
        while self.result is None:
            left = self.deadline - time.monotonic()
            if left <= 0:
                raise Error(f"the browser did not come back within {self.timeout:g} s; run login again")
            self.server.timeout = min(left, 1.0)
            self.server.handle_request()
        kind, value = self.result
        if kind == "error":
            raise Error(f"the sign-in failed: {value}")
        return value


async def sign_in(url, publishable_key, provider="github", port=8767, browser=None, say=print,
                  timeout=LOGIN_TIMEOUT, transport=None):
    """Steps 3 to 7 of `worker login` (addendum 7.3): a PKCE pair, the loopback on 127.0.0.1:port, the
    authorize page (printed with say, and opened with browser(url) when given), the code it brings back, and
    the session for that code (Grant). The verifier never leaves this function; the page holds only the
    challenge. Error when the browser does not come back within timeout seconds or brings an error."""
    auth = Auth(url, publishable_key, transport)  # refuses a secret key before anything opens
    verifier, challenge = pkce_pair()
    with Loopback(port, timeout) as lb:
        page = authorize_url(url, provider, lb.redirect, challenge)
        say(f"Open this page to sign in with {provider}:\n  {page}")
        if browser:
            browser(page)
        code = await anyio.to_thread.run_sync(lb.wait)
    return await auth.exchange(code, verifier)


@dataclass
class Grant:
    """A session from Supabase Auth. repr() hides the tokens."""
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_in: float
    user_id: str | None
    email: str | None = None
    provider: str | None = None

    @classmethod
    def of(cls, data):
        try:
            user = data.get("user") or {}
            return cls(data["access_token"], data["refresh_token"], float(data.get("expires_in") or 3600),
                       user.get("id"), user.get("email"), (user.get("app_metadata") or {}).get("provider"))
        except (KeyError, TypeError, ValueError, AttributeError):
            raise Error("Supabase Auth answered without a session") from None


class Auth:
    """Supabase Auth for a personal worker's session: the PKCE code exchange, the refresh and the sign-out. Each
    call opens its own short client, so a session works from any event loop. The key is the publishable key."""

    def __init__(self, url, publishable_key, transport=None):
        self.url = check_url(url)
        if not isinstance(publishable_key, str) or not publishable_key or publishable_key.startswith("sb_secret_"):
            raise ValueError("a user's session goes with the publishable key, never the secret key")
        self.key, self.transport = publishable_key, transport

    def __repr__(self):
        return f"Auth({self.url})"

    async def exchange(self, code, verifier):
        """The session for the code that the loopback got: POST /auth/v1/token?grant_type=pkce."""
        r = await self.post("/auth/v1/token", {"grant_type": "pkce"}, {"auth_code": code, "code_verifier": verifier})
        g = Grant.of(r.json())
        if not g.user_id:
            raise Error("Supabase Auth answered without a user")
        return g

    async def refresh(self, refresh_token):
        """A new session for refresh_token, which Supabase takes no more after that. SignedOut when Supabase
        refuses it. A try whose answer was lost goes again only within REFRESH_WINDOW, where Supabase still
        takes the same token; a second use later would end the whole session."""
        r = await self.post("/auth/v1/token", {"grant_type": "refresh_token"}, {"refresh_token": refresh_token},
                            again=True)
        return Grant.of(r.json())

    async def logout(self, access_token):
        """Ends this session at Supabase (scope local: this machine only). A session that is already gone is fine."""
        await self.post("/auth/v1/logout", {"scope": "local"}, None, bearer=access_token, ok=(401, 403, 404))

    async def post(self, path, params, body, bearer=None, again=False, ok=()):
        headers = {"apikey": self.key, "User-Agent": AGENT, **({"Authorization": f"Bearer {bearer}"} if bearer else {})}
        hide, t0 = (self.key, bearer, *(body or {}).values()), time.monotonic()
        async with httpx.AsyncClient(timeout=AUTH_TIMEOUT, transport=self.transport) as c:
            for i in range(1, TRIES + 1):
                try:
                    r = await c.post(self.url + path, params=params, json=body, headers=headers)
                except httpx.TransportError as e:
                    if i == TRIES or not (isinstance(e, NOT_SENT) or again and time.monotonic() - t0 < REFRESH_WINDOW):
                        raise Offline(f"POST {path}: no answer from Supabase Auth ({type(e).__name__})") from None
                    await anyio.sleep(0.5 * i)
                    continue
                if r.status_code < 400 or r.status_code in ok:
                    return r
                if r.status_code >= 500 and i < TRIES and again and time.monotonic() - t0 < REFRESH_WINDOW:
                    await anyio.sleep(0.5 * i)
                    continue
                e = error_of("POST", path, r, hide)
                if body and "refresh_token" in body and r.status_code in (400, 401, 403):
                    raise SignedOut(f"the HARPOR site signed this machine out ({e.message or r.status_code}); "
                                    "run python -m gator.webapp.worker login", e.status, e.message, e.code)
                raise e


class Tokens:
    """A personal worker's access token, renewed before it runs out. save(refresh_token) runs after each renewal
    and before the new access token is used: Supabase takes a refresh token once, and a second use after its
    10 s window ends the whole session, so the session file must hold the new token before anything else can
    fail. The access token lives only in memory."""

    def __init__(self, auth, refresh_token, save, margin=REFRESH_MARGIN):
        self.auth, self.refresh_token, self.save, self.margin = auth, refresh_token, save, margin
        self.access, self.expires, self.grant, self.renewals = None, 0.0, None, 0
        self.lock = anyio.Lock()

    def __repr__(self):
        return f"Tokens({self.renewals} renewals)"

    def left(self):
        """Seconds the access token has left (0 without one)."""
        return max(0.0, self.expires - time.monotonic()) if self.access else 0.0

    async def bearer(self):
        """An access token with at least margin seconds left; renews it first when needed. When the renewal gets
        no answer, a token with 30 s left still serves."""
        if self.left() > self.margin:
            return self.access
        seen = self.renewals
        try:
            async with self.lock:
                if self.renewals == seen:  # no other task renewed it meanwhile
                    await self._renew()
        except Offline:
            if self.left() > 30:
                return self.access
            raise
        return self.access

    async def refresh(self):
        """Renews now (Supa calls this after a 401)."""
        async with self.lock:
            await self._renew()

    async def _renew(self):
        g = await self.auth.refresh(self.refresh_token)
        self.refresh_token = g.refresh_token  # memory first: the old token is spent now
        self.save(g.refresh_token)
        self.access, self.expires, self.grant = g.access_token, time.monotonic() + g.expires_in, g
        self.renewals += 1


# ---- the Store: what worker.py needs from Supabase ----

class Store:
    """What worker.py needs from Supabase (addendum 6.2): the personal worker of one machine, with its user's
    session (role authenticated, under RLS) and the own_* functions. Rows are dicts as PostgREST gives them. An
    own_* function's refusal raises Error with its text in .message ("worker ... was stopped", "your account is
    waiting for approval", "no such job on this machine", "the job's log on the site is full")."""

    def __init__(self, supa, user_id, machine_id=None):
        self.supa, self.user_id, self.machine_id, self.worker_id = supa, user_id, machine_id, None

    @classmethod
    def of(cls, session, **kw):
        """The store of a session: session.Session, or anything with url, publishable_key, user_id, machine_id,
        bearer() and refresh(). kw goes to Supa."""
        supa = Supa(session.url, session.publishable_key, bearer=session.bearer, refresh=session.refresh, **kw)
        return cls(supa, session.user_id, session.machine_id)

    def __repr__(self):
        return f"Store({self.supa.url}, worker {self.worker_id})"

    async def close(self):
        await self.supa.close()

    async def bind(self, machine_name):
        """own_machine_bind: a new machine on the site, or a new name for this one. Sets and returns machine_id."""
        self.machine_id = await self.supa.rpc("own_machine_bind", again=self.machine_id is not None,
                                              p_machine=self.machine_id, p_name=machine_name)
        return self.machine_id

    async def storage_used(self):
        """The bytes of the user's job-files objects."""
        return await self.supa.rpc("storage_used", again=True)

    async def start(self, public_key, ends_at, version, local_keys=()):
        """A new workers row with the KeyBox's public key, the end of the Slurm job (or None) and the agents whose
        key the machine holds. Sets and returns worker_id. Older workers of this machine stop."""
        if not self.machine_id:
            raise ValueError("this machine has no id on the site yet; bind it first")
        self.worker_id = await self.supa.rpc("own_worker_start", again=True, p_machine=self.machine_id,
                                             p_public_key=public_key, p_ends_at=iso(ends_at), p_version=version,
                                             p_local_keys=list(local_keys))
        return self.worker_id

    async def beat(self, queue_len, accepting, ends_at):
        """The heartbeat, every 10 s."""
        await self.supa.rpc("own_worker_beat", again=True, p_worker=self.worker_id, p_queue_len=queue_len,
                            p_accepting=accepting, p_ends_at=iso(ends_at))

    async def stop(self):
        await self.supa.rpc("own_worker_stop", again=True, p_worker=self.worker_id)

    async def claim(self, kinds):
        """One pending request of this machine's jobs, now taken, or None. kinds: the routes to take (near the end
        of a Slurm job), or None for all. own_claim_request does not wait: the loop sleeps 2 s when this gives
        None."""
        rows = await self.supa.rpc("own_claim_request", p_worker=self.worker_id,
                                   p_kinds=None if kinds is None else list(kinds))
        return rows[0] if rows else None

    async def take_key(self, req):
        """The sealed key of a claimed request, {"agent", "epk", "iv", "ct"}, or None. own_claim_request deleted
        the secret row already."""
        return req.pop("sealed", None)

    async def answer(self, req_id, status, response):
        """server.py's status and JSON for a request that this worker took. A request that was answered meanwhile
        (a retry after a lost answer) is left as it is."""
        try:
            await self.supa.rpc("own_answer", again=True, p_worker=self.worker_id, p_request=req_id,
                                p_status=status, p_response=response)
        except Error as e:
            if not e.message.startswith("the request is not taken by this worker"):
                raise

    async def settings(self):
        """The one settings row."""
        rows = await self.supa.select("settings", id=1)
        return rows[0] if rows else {}

    async def access(self):
        """The user's access row, or None."""
        rows = await self.supa.select("access", user_id=self.user_id)
        return rows[0] if rows else None

    async def job_row(self, job_id):
        """The row, or None; RLS hides a job that waits in "deleting"."""
        rows = await self.supa.select("jobs", JOB_COLUMNS, id=job_id)
        return rows[0] if rows else None

    async def job_rows(self):
        """The rows of this machine's jobs."""
        return await self.supa.select("jobs", JOB_COLUMNS, machine_id=self.machine_id)

    async def create_job(self, name, kind, nbytes):
        """A row for a local upload (own_job_create); its uuid."""
        return await self.supa.rpc("own_job_create", p_worker=self.worker_id, p_name=name, p_kind=kind,
                                   p_bytes=int(nbytes))

    async def publish(self, job_id, view=None, live=None, snapshots=None, row=None):
        """job_views.body, job_live.state, job_snapshots ({"frames"|"selection"|"parts": {"status", "body"}})
        and the jobs row (status, note, prompt, queue_pos and stopped_at), in one own_publish call: one
        transaction for all four parts. None leaves a part as it is."""
        if row is not None and set(row) - set(OWN_ROW):
            raise ValueError(f"own_publish's row takes {', '.join(OWN_ROW)}")
        if snapshots is not None and set(snapshots) - set(SNAPSHOTS):
            raise ValueError(f"the snapshots are {', '.join(SNAPSHOTS)}")
        if view is None and live is None and snapshots is None and row is None:
            return
        await self.supa.rpc("own_publish", again=True, p_worker=self.worker_id, p_job=job_id, p_view=view,
                            p_live=live, p_snapshots=snapshots,
                            p_row={k: iso(v) for k, v in row.items()} if row is not None else None)

    async def files(self, job_id, put, drop):
        """job_files through own_files, 500 put rows a call: put rows {path, object, bytes, sha256, role,
        content_type} replace their paths' rows; the drop paths go. Upload the objects first."""
        put = [{k: f[k] for k in ("path", "object", "bytes", "sha256", "role", "content_type")} for f in put]
        drop = list(drop)
        for i in range(0, max(len(put), 1 if drop else 0), 500):  # one call for drop alone
            await self.supa.rpc("own_files", again=True, p_worker=self.worker_id, p_job=job_id, p_put=put[i:i + 500],
                                p_drop=drop if i == 0 else [])

    async def manifest(self, job_id):
        """{path: {"object", "sha256"}} of the job's files on the site."""
        rows = await self.supa.select("job_files", "path,object,sha256", job_id=job_id)
        return {r["path"]: {"object": r["object"], "sha256": r["sha256"]} for r in rows}

    async def logs(self, job_id, rows):
        """job_logs rows {stage, run_started, pos, text}, scrubbed, through own_logs, 100 a call."""
        rows = [{k: r[k] for k in ("stage", "pos", "text")} | {"run_started": r.get("run_started")} for r in rows]
        for i in range(0, len(rows), 100):
            await self.supa.rpc("own_logs", p_worker=self.worker_id, p_job=job_id, p_rows=rows[i:i + 100])

    async def upload(self, name, data, content_type):
        """One job-files object (object_name), never over another."""
        await self.supa.upload(BUCKET, name, data, content_type)

    async def remove(self, names):
        """Delete job-files objects of the user's folder; how many went."""
        return await self.supa.remove(BUCKET, names)

    async def job_objects(self, job_id):
        """The names of a job's objects (job_objects), oldest first, up to 1,000; also of a job in "deleting"."""
        return [r["name"] for r in await self.supa.rpc("job_objects", again=True, p_job=job_id) or []]

    async def orphans(self, limit=1000):
        """The names of objects in the user's folder that nothing names (own_orphans), oldest first."""
        return [r["name"] for r in await self.supa.rpc("own_orphans", again=True, p_worker=self.worker_id,
                                                       p_limit=limit) or []]

    async def delete_row(self, job_id):
        """The jobs row goes (own_job_delete), after its folder and objects went."""
        await self.supa.rpc("own_job_delete", again=True, p_worker=self.worker_id, p_job=job_id)
