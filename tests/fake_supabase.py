"""An in-memory Supabase project for the worker tests: PostgREST (tables and RPC), Storage and the Auth
token endpoints, as an httpx transport (FakeSupabase.transport), written to the rules of
supabase/migrations.

One kind of caller besides anon, as on Supabase: a user's access token is role authenticated, with RLS
(users read their own rows), no table writes, and the functions of users and personal workers. The
publishable key alone is anon, which gets nothing. The secret key (SECRET) is refused with 401: no part of
HARPOR uses it, and tests check that the code never sends it. A function called by the wrong role answers
"permission denied", as PostgREST does. A raised exception answers 400 with {"code": "P0001", "message"}, a
table check 400 with "violates check constraint", and each call runs as one transaction (a failed call
changes nothing).

Storage keeps the objects in memory and runs the job-files policies of 20261101000400_harpor_storage.sql:
uploads through own_object_ok, deletes in the caller's own folder. Auth hands out opaque access
tokens (access_token, authorize) and turns refresh tokens once, as Supabase does (with a 10 s reuse window).
sweep() applies private.sweep, which pg_cron runs on Supabase.

It is not Supabase. It models what the worker and the case table need: PostgREST filters eq, neq, gt,
gte, lt, lte, is, in, like, not.<op> and or=(...), order and limit; no embedding, no Realtime. run_case()
runs one case of tests/supabase_cases.json against it, so the fake and the SQL answer the same way.
"""

import base64
import copy
import hashlib
import json
import os
import re
import secrets
import threading
import urllib.parse
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import anyio
import httpx
from supabase_cases import IDS, INPUT, OWNER, SHA, USERS, WORKERS, names, names_in
from supabase_cases import MACHINE as MACHINE_OF  # user -> the machine of the case's job by default
from supabase_cases import load as load_cases  # noqa: F401 (the worker tests read the table through it)

URL = "https://harpor-test.supabase.co"
SECRET = "sb_secret_" + "fakefakefakefake" * 2  # refused: a test's stand-in for the project's secret key
PUBLISHABLE = "sb_publishable_" + "fakefakefakefake"
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
NAME = re.compile(r"^(?!.*\.\.)[A-Za-z0-9_-][A-Za-z0-9_.-]{0,199}$")  # jobs.input_name: no two dots in a row
MACHINE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,59}$")
OWN_OBJECT = re.compile(r"^[0-9a-f-]{36}/[0-9a-f-]{36}/[0-9a-f]{16}/[A-Za-z0-9_-][A-Za-z0-9_./-]{0,199}$")
OWN_STATUSES = ("uploaded", "analyzing", "choose_topic", "ready", "queued", "running", "review", "done", "failed",
                "cancelled")
STATUSES = OWN_STATUSES + ("deleting",)
BUSY = ("uploaded", "analyzing", "queued", "running")  # relay: wait until it stops; the sweep keeps its log text
ACTIVE = ("analyzing", "queued", "running")  # delete_job: cancel first while the machine is online
ROUTES = ("topic", "calibration_put", "calibration_delete", "start", "overrides", "parts", "continue", "cancel",
          "delete")
AGENTS = ("astra", "opus")
JSON_DEPTH = 32  # relay() refuses a body or query nested deeper
REUSE = 10  # s: Auth takes a used refresh token again for this long
TOKEN_S = 3600
MISSING = object()


def now_utc():
    return datetime.now(timezone.utc)


def kind_of(name):
    """private.kind_of: video, rosbag or stray by the suffix, else None."""
    suffix = Path(str(name)).suffix.lower() if "." in str(name) else ""
    return {".mp4": "video", ".mov": "video", ".m4v": "video", ".avi": "video", ".mkv": "video", ".webm": "video",
            ".bag": "rosbag", ".mcap": "rosbag", ".db3": "rosbag", ".zip": "stray"}.get(suffix)


def jtext(v):
    """jsonb ->> of a value: None, the string itself, or the JSON text."""
    if v is None or isinstance(v, str):
        return v
    return json.dumps(v)


def jsize(v):
    """octet_length(v::text) of a jsonb value, near enough."""
    return len(json.dumps(v, ensure_ascii=False).encode())


def too_deep(v, levels):
    """private.too_deep: True when v nests dicts and lists more than levels deep."""
    if not isinstance(v, (dict, list)):
        return False
    return levels <= 0 or any(too_deep(x, levels - 1) for x in (v.values() if isinstance(v, dict) else v))


def try_uuid(v):
    return v.lower() if isinstance(v, str) and UUID.fullmatch(v) else None


def limit_of(n):
    """greatest(1, least(coalesce(n, 1000), 1000))."""
    return max(1, min(1000 if n is None else n, 1000))


class Fail(Exception):
    """An error as PostgREST or Storage answers it: (HTTP status, JSON body)."""

    def __init__(self, status, body):
        super().__init__(body.get("message"))
        self.status, self.body = status, body

    @property
    def message(self):
        return self.body.get("message", "")


def raise_(text):
    """A plpgsql `raise exception`: 400 with code P0001."""
    raise Fail(400, {"code": "P0001", "message": text, "details": None, "hint": None})


def check(ok, table, name):
    if not ok:
        raise Fail(400, {"code": "23514", "message": f'new row for relation "{table}" violates check constraint '
                                                     f'"{table}_{name}_check"'})


def denied(role, what, name):
    raise Fail(401 if role == "anon" else 403, {"code": "42501", "message": f"permission denied for {what} {name}"})


# ---- the tables: column -> (type, default); a callable default takes the fake ----

def _now(f):
    return f.now()


def _uuid(f):
    return str(uuid.uuid4())


SCHEMA = {
    "settings": {"pk": ("id",), "cols": {
        "id": ("int", 1), "plan_file_limit": ("int", 52428800), "worker_online_s": ("int", 30),
        "pending_days": ("int", 7), "near_end_minutes": ("int", 60), "lean_files": ("bool", True),
        "max_machines": ("int", 5), "own_log_bytes": ("int", 2097152)}},
    "access": {"pk": ("user_id",), "cols": {
        "user_id": ("uuid", None), "email": ("text", None), "approved": ("bool", False),
        "max_result_bytes": ("int", 2147483648), "max_requests_per_hour": ("int", 60), "max_own_jobs": ("int", 20),
        "max_own_per_day": ("int", 20), "agents": ("text[]", ["astra", "opus"]), "request_note": ("text", None),
        "request_at": ("ts", None), "created_at": ("ts", _now)}},
    "usage_daily": {"pk": ("user_id", "day"), "cols": {
        "user_id": ("uuid", None), "day": ("date", lambda f: f.today()), "own_jobs": ("int", 0)}},
    "machines": {"pk": ("id",), "cols": {
        "id": ("uuid", _uuid), "user_id": ("uuid", None), "name": ("text", None), "created_at": ("ts", _now)}},
    "workers": {"pk": ("id",), "cols": {
        "id": ("uuid", _uuid), "user_id": ("uuid", None), "machine_id": ("uuid", None), "public_key": ("text", None),
        "version": ("text", None), "local_keys": ("text[]", []), "started_at": ("ts", _now),
        "last_seen": ("ts", _now), "ends_at": ("ts", None), "accepting": ("bool", True), "queue_len": ("int", 0),
        "stopped_at": ("ts", None)}},
    "jobs": {"pk": ("id",), "cols": {
        "id": ("uuid", _uuid), "user_id": ("uuid", None), "machine_id": ("uuid", None), "created_at": ("ts", _now),
        "updated_at": ("ts", _now), "input_name": ("text", None), "input_kind": ("text", None),
        "input_bytes": ("int", None), "status": ("text", "uploaded"), "note": ("text", None), "prompt": ("text", None),
        "queue_pos": ("int", None), "stopped_at": ("ts", None), "deleted_at": ("ts", None)}},
    "job_views": {"pk": ("job_id",), "cols": {
        "job_id": ("uuid", None), "user_id": ("uuid", None), "rev": ("int", 0), "body": ("json", None),
        "updated_at": ("ts", _now)}},
    "job_live": {"pk": ("job_id",), "cols": {
        "job_id": ("uuid", None), "user_id": ("uuid", None), "state": ("json", None), "updated_at": ("ts", _now)}},
    "job_snapshots": {"pk": ("job_id", "path"), "cols": {
        "job_id": ("uuid", None), "path": ("text", None), "user_id": ("uuid", None), "status": ("int", None),
        "body": ("json", None), "updated_at": ("ts", _now)}},
    "job_files": {"pk": ("job_id", "path"), "cols": {
        "job_id": ("uuid", None), "path": ("text", None), "user_id": ("uuid", None), "object": ("text", None),
        "bytes": ("int", None), "sha256": ("text", None), "role": ("text", None), "content_type": ("text", None),
        "updated_at": ("ts", _now)}},
    "job_logs": {"pk": ("id",), "identity": "id", "cols": {
        "id": ("int", None), "job_id": ("uuid", None), "user_id": ("uuid", None), "stage": ("text", None),
        "run_started": ("float", None), "pos": ("int", None), "text": ("text", None), "created_at": ("ts", _now)}},
    "requests": {"pk": ("id",), "identity": "id", "cols": {
        "id": ("int", None), "user_id": ("uuid", None), "job_id": ("uuid", None), "route": ("text", None),
        "query": ("json", {}), "body": ("json", {}), "body_text": ("text", None), "key_ref": ("uuid", None),
        "state": ("text", "pending"), "sent_online": ("bool", False), "created_at": ("ts", _now),
        "taken_at": ("ts", None), "worker_id": ("uuid", None), "done_at": ("ts", None), "status": ("int", None),
        "response": ("json", None)}},
    "job_secrets": {"pk": ("ref",), "cols": {
        "ref": ("uuid", None), "job_id": ("uuid", None), "user_id": ("uuid", None), "worker_id": ("uuid", None),
        "agent": ("text", None), "epk": ("text", None), "iv": ("text", None), "ct": ("text", None),
        "created_at": ("ts", _now)}},
}
NOT_NULL = {"access": ("user_id",), "usage_daily": ("user_id",), "machines": ("user_id", "name"),
            "workers": ("user_id", "machine_id", "public_key"),
            "jobs": ("user_id", "machine_id", "input_name", "input_kind", "input_bytes"),
            "job_views": ("job_id", "user_id", "body"), "job_live": ("job_id", "user_id", "state"),
            "job_snapshots": ("job_id", "path", "user_id", "status", "body"),
            "job_files": ("job_id", "path", "user_id", "object", "bytes", "sha256", "role", "content_type"),
            "job_logs": ("job_id", "user_id", "stage", "pos", "text"),
            "requests": ("user_id", "job_id", "route"),
            "job_secrets": ("ref", "job_id", "user_id", "worker_id", "agent", "epk", "iv", "ct")}
JOB_TABLES = ("job_views", "job_live", "job_snapshots", "job_files", "job_logs", "requests", "job_secrets")
USER_TABLES = ("settings", "access", "usage_daily", "machines", "workers", "jobs", "job_views", "job_live",
               "job_snapshots", "job_files", "job_logs", "requests")
# function -> (the role that may call it, {argument: default}); REQUIRED arguments have no default
REQUIRED = object()
FUNCTIONS = {
    "relay": ("authenticated", {"p_job": REQUIRED, "p_route": REQUIRED, "p_query": {}, "p_body": {}, "p_text": None,
                                "p_sealed": None}),
    "withdraw_request": ("authenticated", {"p_id": REQUIRED}),
    "delete_job": ("authenticated", {"p_job": REQUIRED}),
    "request_access": ("authenticated", {"p_note": REQUIRED}),
    "own_machine_bind": ("authenticated", {"p_machine": REQUIRED, "p_name": REQUIRED}),
    "own_worker_start": ("authenticated", {"p_machine": REQUIRED, "p_public_key": REQUIRED, "p_ends_at": REQUIRED,
                                           "p_version": REQUIRED, "p_local_keys": []}),
    "own_worker_beat": ("authenticated", {"p_worker": REQUIRED, "p_queue_len": REQUIRED, "p_accepting": REQUIRED,
                                          "p_ends_at": None}),
    "own_worker_stop": ("authenticated", {"p_worker": REQUIRED}),
    "own_claim_request": ("authenticated", {"p_worker": REQUIRED, "p_kinds": None}),
    "own_answer": ("authenticated", {"p_worker": REQUIRED, "p_request": REQUIRED, "p_status": REQUIRED,
                                     "p_response": REQUIRED}),
    "own_job_create": ("authenticated", {"p_worker": REQUIRED, "p_name": REQUIRED, "p_kind": REQUIRED,
                                         "p_bytes": REQUIRED}),
    "own_publish": ("authenticated", {"p_worker": REQUIRED, "p_job": REQUIRED, "p_view": None, "p_live": None,
                                      "p_snapshots": None, "p_row": None}),
    "own_files": ("authenticated", {"p_worker": REQUIRED, "p_job": REQUIRED, "p_put": [], "p_drop": []}),
    "own_logs": ("authenticated", {"p_worker": REQUIRED, "p_job": REQUIRED, "p_rows": REQUIRED}),
    "own_job_delete": ("authenticated", {"p_worker": REQUIRED, "p_job": REQUIRED}),
    "storage_used": ("authenticated", {}),
    "own_object_ok": ("authenticated", {"p_name": REQUIRED}),
    "job_objects": ("authenticated", {"p_job": REQUIRED, "p_limit": 1000}),
    "own_orphans": ("authenticated", {"p_worker": REQUIRED, "p_limit": 1000}),
}
UUID_ARGS = {"p_job", "p_machine", "p_worker"}
TS_ARGS = {"p_ends_at"}
BUCKETS = {"job-files": {"limit": 52428800, "mime": ("model/gltf-binary", "image/jpeg", "image/png", "application/json",
                                                     "application/zip", "text/plain; charset=utf-8", "text/plain",
                                                     "video/mp4")}}


class Caller:
    """Who makes a call: role authenticated (uid set) or anon."""

    def __init__(self, role, uid=None):
        self.role, self.uid = role, uid

    @property
    def label(self):
        return self.uid or "anon"


class FakeSupabase:
    """One project's tables, functions, Storage and Auth in memory. transport serves them to httpx (a Supa or
    an AsyncClient). calls lists the function calls and Storage writes that succeeded, as (caller, kind, name):
    caller is the user's id or "anon", kind rpc, upload or remove. watch, when set, is called with the same
    three after each of them, while the state is as that call left it."""

    url, publishable = URL, PUBLISHABLE

    def __init__(self):
        self.lock = threading.RLock()
        self.shift = timedelta()
        self.t = {name: {} for name in SCHEMA}
        self.ident = {name: 0 for name in SCHEMA}
        self.rate = {}                  # (user, hour) -> requests relayed in that hour (private.rate)
        self.objects = {}               # (bucket, name) -> {bucket_id, name, owner, size, data, content_type,
        #                                   created_at}
        self.buckets = copy.deepcopy(BUCKETS)
        self.users = {}                 # id -> email
        self.tokens = {}                # access token -> (user, expires)
        self.refresh_tokens = {}        # refresh token -> {"user", "used": None or (time, child grant)}
        self.codes = {}                 # PKCE code -> (user, challenge)
        self.calls = []
        self.fail_next = {}             # (kind, name) -> Fail to raise once (a test's outage)
        self.watch = None
        self._insert("settings", {"id": 1})
        self.transport = httpx.MockTransport(self.handle)

    def done(self, caller, kind, name):
        self.calls.append((caller, kind, name))
        if self.watch is not None:
            self.watch(caller, kind, name)

    # ---- time and helpers ----

    def now(self):
        return now_utc() + self.shift

    def today(self):
        return self.now().date().isoformat()

    def advance(self, seconds):
        self.shift += timedelta(seconds=seconds)

    def settings(self):
        return self.t["settings"][(1,)]

    def rows(self, table, **eq):
        """Copies of a table's rows (and private.rate as "rate") whose columns equal eq, in insertion order."""
        with self.lock:
            if table == "rate":
                return [{"user_id": u, "hour": h, "n": n} for (u, h), n in self.rate.items()
                        if all({"user_id": u, "hour": h}.get(c) == v for c, v in eq.items())]
            return [dict(r) for r in self.t[table].values() if all(r.get(c) == v for c, v in eq.items())]

    def row(self, table, **eq):
        rows = self.rows(table, **eq)
        return rows[0] if rows else None

    def put(self, table, values):
        """Insert a row as the superuser (test setup): defaults, checks and keys apply, no RLS. Returns it."""
        with self.lock:
            return dict(self._insert(table, values))

    def set(self, table, match, **fields):
        """Update rows as the superuser (test setup)."""
        with self.lock:
            for k, r in list(self.t[table].items()):
                if all(r.get(c) == v for c, v in match.items()):
                    self._store(table, {**r, **{c: self._coerce(table, c, v) for c, v in fields.items()}}, k)

    def object(self, bucket, name, data=b"", size=None, owner=None, age=0):
        """A Storage object as the superuser (test setup); age in seconds."""
        with self.lock:
            self.objects[(bucket, name)] = {"bucket_id": bucket, "name": name, "owner": owner,
                                            "size": len(data) if size is None else size, "data": data,
                                            "content_type": "application/octet-stream",
                                            "created_at": self.now() - timedelta(seconds=age)}

    # ---- users, tokens and the Auth endpoints ----

    def add_user(self, email, user_id=None, **access):
        """A new account (auth.users), whose access row the sign-up trigger makes; access sets its columns."""
        with self.lock:
            uid = user_id or str(uuid.uuid4())
            self.users[uid] = email
            self._insert("access", {"user_id": uid, "email": email})
            if access:
                self.set("access", {"user_id": uid}, **access)
            return uid

    def access_token(self, user_id, seconds=TOKEN_S):
        with self.lock:
            token = "fake-access-" + secrets.token_urlsafe(24)
            self.tokens[token] = (user_id, self.now() + timedelta(seconds=seconds))
            return token

    def expire_tokens(self, user_id=None):
        """Every access token (of user_id) runs out now, so the next call gets 401."""
        with self.lock:
            for t, (u, _) in list(self.tokens.items()):
                if user_id in (None, u):
                    self.tokens[t] = (u, self.now() - timedelta(seconds=1))

    def grant(self, user_id):
        """A session as Auth answers it: access and refresh token, expires_in and the user."""
        refresh = "fake-refresh-" + secrets.token_urlsafe(16)
        self.refresh_tokens[refresh] = {"user": user_id, "used": None}
        return {"access_token": self.access_token(user_id), "token_type": "bearer", "expires_in": TOKEN_S,
                "refresh_token": refresh, "user": {"id": user_id, "email": self.users.get(user_id),
                                                   "app_metadata": {"provider": "github"}}}

    def authorize(self, user_id, challenge):
        """What the browser brings back after a sign-in at /auth/v1/authorize: a code for this challenge."""
        with self.lock:
            code = secrets.token_urlsafe(16)
            self.codes[code] = (user_id, challenge)
            return code

    def session(self, user_id, machine_id=None, machine_name="lab-box", mirror_since="2000-01-01T00:00:00"):
        """A personal worker's session (what session.Session gives Store.of)."""
        return FakeSession(self, user_id, machine_id, machine_name, mirror_since)

    def _auth(self, request):
        path, q = request.url.path, dict(urllib.parse.parse_qsl(request.url.query.decode()))
        body = json.loads(request.content or b"{}")
        bad = {"error": "invalid_grant", "error_description": "Invalid Refresh Token: Already Used"}
        with self.lock:
            if path == "/auth/v1/token" and q.get("grant_type") == "pkce":
                code = self.codes.pop(body.get("auth_code"), None)
                verifier = body.get("code_verifier") or ""
                sha = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
                if code is None or code[1] != sha:
                    return httpx.Response(400, json={"error": "invalid_grant", "error_description":
                                                     "invalid flow state, no valid flow state found"})
                return httpx.Response(200, json=self.grant(code[0]))
            if path == "/auth/v1/token" and q.get("grant_type") == "refresh_token":
                rec = self.refresh_tokens.get(body.get("refresh_token"))
                if rec is None:
                    return httpx.Response(400, json={**bad, "error_description": "Invalid Refresh Token: Not Found"})
                if rec["used"] is not None:
                    at, child = rec["used"]
                    if (self.now() - at).total_seconds() <= REUSE:
                        return httpx.Response(200, json=child)
                    return httpx.Response(400, json=bad)
                child = self.grant(rec["user"])
                rec["used"] = (self.now(), child)
                return httpx.Response(200, json=child)
            if path == "/auth/v1/logout":
                token = request.headers.get("authorization", "").removeprefix("Bearer ")
                self.tokens.pop(token, None)
                return httpx.Response(204)
        return httpx.Response(404, json={"message": "not found"})

    # ---- the HTTP surface ----

    def caller(self, request):
        """The role of a request, from its apikey and Authorization headers; Fail(401) for a key other than the
        publishable key (the secret key too) or an access token that ran out."""
        key = request.headers.get("apikey")
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if key != PUBLISHABLE or token == SECRET:
            raise Fail(401, {"message": "Invalid API key"})
        if token in ("", PUBLISHABLE):
            return Caller("anon")
        with self.lock:
            found = self.tokens.get(token)
        if found is None or found[1] <= self.now():
            raise Fail(401, {"code": "PGRST301", "message": "JWT expired"})
        return Caller("authenticated", found[0])

    def handle(self, request):
        """The transport's handler."""
        path = request.url.path
        try:
            if path.startswith("/auth/v1/"):
                return self._auth(request)
            c = self.caller(request)
            if path.startswith("/rest/v1/rpc/"):
                args = json.loads(request.content or b"{}")
                return self._json(self.rpc_call(c, path.removeprefix("/rest/v1/rpc/"), args))
            if path.startswith("/rest/v1/"):
                return self._rest(c, request, path.removeprefix("/rest/v1/"))
            if path.startswith("/storage/v1/"):
                return self._storage(c, request, path.removeprefix("/storage/v1/"))
        except Fail as e:
            return httpx.Response(e.status, json=e.body)
        return httpx.Response(404, json={"message": f"no route {path}"})

    def _json(self, value, status=200):
        if value is None:
            return httpx.Response(204 if status == 200 else status)
        return httpx.Response(status, content=json.dumps(self._out(value)).encode(),
                              headers={"Content-Type": "application/json"})

    def _out(self, v):
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, dict):
            return {k: self._out(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self._out(x) for x in v]
        return v

    # ---- RPC ----

    def rpc(self, who, name, **args):
        """A function call without HTTP: who is a user id or "anon". Raises Fail as the HTTP call would answer."""
        c = Caller("anon") if who == "anon" else Caller("authenticated", who)
        return self._out(self.rpc_call(c, name, args))

    def rpc_call(self, c, name, args):
        if name not in FUNCTIONS:
            raise Fail(404, {"code": "PGRST202", "message": f"Could not find the function public.{name}"})
        role, params = FUNCTIONS[name]
        if c.role != role:
            denied(c.role, "function", name)
        if not isinstance(args, dict) or set(args) - set(params):
            raise Fail(404, {"code": "PGRST202", "message": f"Could not find the function public.{name} with these "
                                                            "arguments"})
        kw = {}
        for p, default in params.items():
            v = args.get(p, MISSING)
            if v is MISSING:
                if default is REQUIRED:
                    raise Fail(404, {"code": "PGRST202", "message": f"Could not find the function public.{name}"})
                v = copy.deepcopy(default)
            if p in UUID_ARGS and v is not None:
                if try_uuid(v) is None:
                    raise Fail(400, {"code": "22P02", "message": f'invalid input syntax for type uuid: "{v}"'})
                v = v.lower()
            if p in TS_ARGS and isinstance(v, str):
                v = datetime.fromisoformat(v)
            kw[p] = v
        with self.lock:
            saved = self._save()
            try:
                out = getattr(self, "fn_" + name)(c, **kw)
            except BaseException:
                self._restore(saved)
                raise
            self.done(c.label, "rpc", name)
            return out

    def _save(self):
        return ({k: dict(v) for k, v in self.t.items()}, dict(self.ident), dict(self.rate), dict(self.objects))

    def _restore(self, saved):
        self.t, self.ident, self.rate, self.objects = saved

    # ---- rows: defaults, checks, keys, cascades (rows are never changed in place, so _save can copy shallowly) ----

    def _coerce(self, table, col, v):
        cols = SCHEMA[table]["cols"]
        if col not in cols:
            raise Fail(400, {"code": "PGRST204", "message": f"Could not find the '{col}' column of '{table}'"})
        kind = cols[col][0]
        if v is None:
            return None
        if kind == "uuid":
            if try_uuid(v) is None:
                raise Fail(400, {"code": "22P02", "message": f'invalid input syntax for type uuid: "{v}"'})
            return v.lower()
        if kind == "ts":
            return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))
        if kind == "date":
            return v.isoformat() if isinstance(v, date) else str(v)
        if kind == "int":
            if isinstance(v, bool) or not isinstance(v, (int, float, str)):
                raise Fail(400, {"code": "22P02", "message": f'invalid input syntax for type bigint: "{v}"'})
            return int(v)
        if kind == "float":
            return float(v)
        if kind == "bool":
            if isinstance(v, str):
                return v == "true"
            return bool(v)
        if kind == "text[]":
            return list(v)
        if kind == "text":
            return v if isinstance(v, str) else jtext(v)
        return copy.deepcopy(v)

    def _full(self, table, values):
        spec, row = SCHEMA[table], {}
        for col, (_, default) in spec["cols"].items():
            if col in values:
                row[col] = self._coerce(table, col, values[col])
            else:
                row[col] = default(self) if callable(default) else copy.deepcopy(default)
        for col in values:
            if col not in spec["cols"]:
                self._coerce(table, col, None)
        if spec.get("identity") and values.get(spec["identity"]) is None:
            self.ident[table] += 1
            row[spec["identity"]] = self.ident[table]
        return row

    def _key(self, table, row):
        return tuple(row[c] for c in SCHEMA[table]["pk"])

    def _insert(self, table, values, upsert=False):
        row = self._full(table, values)
        k = self._key(table, row)
        old = self.t[table].get(k)
        if old is not None:
            if not upsert:
                raise Fail(409, {"code": "23505", "message": f'duplicate key value violates unique constraint '
                                                             f'"{table}_pkey"'})
            row = {**old, **{c: self._coerce(table, c, v) for c, v in values.items()}}
        self._store(table, row, k)
        return row

    def _store(self, table, row, k=None):
        self._check(table, row)
        k = self._key(table, row) if k is None else k
        if table == "requests" and k in self.t[table] and row["state"] in ("done", "withdrawn", "expired"):
            # private.on_request_done: a finished request keeps a small body only, and no calibration text
            row = {**row, "body_text": None, "body": row["body"] if jsize(row["body"]) <= 1024 else {}}
        if self._key(table, row) != k:
            self.t[table].pop(k, None)
            k = self._key(table, row)
        self.t[table][k] = row

    def _check(self, table, r):
        for col in NOT_NULL.get(table, ()):
            if r.get(col) is None:
                raise Fail(400, {"code": "23502", "message": f'null value in column "{col}" of relation "{table}" '
                                                             "violates not-null constraint"})
        def fk(ok, name):
            if not ok:
                raise Fail(409, {"code": "23503", "message": f'insert or update on table "{table}" violates foreign '
                                                             f'key constraint "{table}_{name}_fkey"'})

        if table == "settings":
            check(r["id"] == 1, table, "id")
        elif table == "access":
            check(r["max_result_bytes"] >= 0, table, "max_result_bytes")
            check(0 <= r["max_requests_per_hour"] <= 1000, table, "max_requests_per_hour")
            check(0 <= r["max_own_jobs"] <= 1000, table, "max_own_jobs")
            check(0 <= r["max_own_per_day"] <= 1000, table, "max_own_per_day")
            check(set(r["agents"]) <= set(AGENTS), table, "agents")
            check(r["request_note"] is None or len(r["request_note"]) <= 500, table, "request_note")
        elif table == "machines":
            check(bool(MACHINE.match(r["name"])), table, "name")
        elif table == "workers":
            check(len(r["public_key"]) == 88, table, "public_key")
            check(set(r["local_keys"]) <= set(AGENTS), table, "local_keys")
            fk(self._machine_of(r["machine_id"], r["user_id"]), "machine_id")
        elif table == "jobs":
            check(bool(NAME.match(r["input_name"])), table, "input_name")
            check(r["input_kind"] in ("video", "rosbag", "stray"), table, "input_kind")
            check(r["input_bytes"] > 0, table, "input_bytes")
            check(r["status"] in STATUSES, table, "status")
            check(r["note"] is None or len(r["note"]) <= 300, table, "note")
            check(r["prompt"] is None or len(r["prompt"]) <= 60, table, "prompt")
            fk(self._machine_of(r["machine_id"], r["user_id"]), "machine_id")
        elif table == "job_views":
            check(jsize(r["body"]) <= 262144, table, "body")
        elif table == "job_live":
            check(jsize(r["state"]) <= 65536, table, "state")
        elif table == "job_snapshots":
            check(r["path"] in ("frames", "selection", "parts") and jsize(r["body"]) <= 1048576, table, "body")
        elif table == "job_files":
            check(len(r["path"]) <= 200 and len(r["object"]) <= 400 and r["bytes"] >= 0, table, "path")
            check(bool(re.fullmatch(r"[0-9a-f]{64}", r["sha256"])), table, "sha256")
            check(r["role"] in ("review", "result", "log") and len(r["content_type"]) <= 100, table, "role")
            check(r["object"] == f"{r['user_id']}/{r['job_id']}/{r['sha256'][:16]}/{r['path']}", table, "check")
        elif table == "job_logs":
            check(bool(re.fullmatch(r"[a-z0-9_]{1,20}", r["stage"])), table, "stage")
            check(len(r["text"]) <= 65536, table, "text")
        elif table == "requests":
            check(r["route"] in ROUTES, table, "route")
            check(isinstance(r["query"], dict) and jsize(r["query"]) <= 1024, table, "query")
            check(isinstance(r["body"], dict) and jsize(r["body"]) <= 131072, table, "body")
            check(r["body_text"] is None or len(r["body_text"].encode()) <= 65536, table, "body_text")
            check(r["state"] in ("pending", "taken", "done", "withdrawn", "expired"), table, "state")
            check(r["response"] is None or jsize(r["response"]) <= 65536, table, "response")
        elif table == "job_secrets":
            check(r["agent"] in AGENTS and len(r["epk"]) == 88 and len(r["iv"]) == 16
                  and 24 <= len(r["ct"]) <= 704, table, "sealed")
            fk((r["worker_id"],) in self.t["workers"], "worker_id")
        if table in JOB_TABLES:
            fk((r["job_id"],) in self.t["jobs"], "job_id")

    def _machine_of(self, machine, user):
        m = self.t["machines"].get((machine,))
        return m is not None and m["user_id"] == user

    def _delete(self, table, keys):
        """Delete rows by key, with the cascades of the migrations."""
        gone = []
        for k in keys:
            r = self.t[table].pop(k, None)
            if r is None:
                continue
            gone.append(r)
            if table == "jobs":
                for t in JOB_TABLES:
                    self._delete(t, [kk for kk, x in self.t[t].items() if x["job_id"] == r["id"]])
            elif table == "machines":
                self._delete("workers", [kk for kk, x in self.t["workers"].items() if x["machine_id"] == r["id"]])
                self._delete("jobs", [kk for kk, x in self.t["jobs"].items() if x["machine_id"] == r["id"]])
            elif table == "workers":
                self._delete("job_secrets", [kk for kk, x in self.t["job_secrets"].items()
                                             if x["worker_id"] == r["id"]])
        return gone

    def _update(self, table, pred, **fields):
        n = 0
        for k, r in list(self.t[table].items()):
            if pred(r):
                self._store(table, {**r, **{c: self._coerce(table, c, v) for c, v in fields.items()}}, k)
                n += 1
        return n

    def _find(self, table, pred):
        return [r for r in self.t[table].values() if pred(r)]

    # ---- private helpers of the migrations ----

    def live_worker(self, machine):
        """private.live_worker: the newest worker of a machine that is online, or None."""
        s, now = self.settings(), self.now()
        ws = [w for w in self.t["workers"].values() if w["machine_id"] == machine and w["stopped_at"] is None
              and w["last_seen"] > now - timedelta(seconds=s["worker_online_s"])]
        return max(reversed(ws), key=lambda w: w["started_at"], default=None)

    def hour(self):
        return self.now().replace(minute=0, second=0, microsecond=0)

    def bump_rate(self, user):
        k = (user, self.hour())
        self.rate[k] = self.rate.get(k, 0) + 1
        return self.rate[k]

    def access(self, user):
        return self.t["access"].get((user,))

    def size_of(self, bucket, name):
        o = self.objects.get((bucket, name))
        return o["size"] if o else None

    def withdraw_pending(self, job, detail):
        """Withdraw a job's pending requests and delete their sealed keys; how many."""
        now, n = self.now(), 0
        for r in list(self.t["requests"].values()):
            if r["job_id"] == job and r["state"] == "pending":
                self._store("requests", {**r, "state": "withdrawn", "done_at": now, "status": 409,
                                         "response": {"detail": detail}})
                if r["key_ref"]:
                    self._delete("job_secrets", [(r["key_ref"],)])
                n += 1
        return n

    def uid_of(self, c):
        if c.uid is None:
            raise_("sign in first")
        return c.uid

    def sweep(self):
        """private.sweep, which pg_cron runs every 15 minutes: requests that waited longer than pending_days
        expire, jobs that waited in "deleting" as long go, sealed keys after an hour, large answers of finished
        requests after an hour, finished requests after 7 days, rate counts after 2 days, and the log text of a
        job 14 days after it stopped. Returns how many requests expired."""
        with self.lock:
            s, now = self.settings(), self.now()
            n = self._update("requests", lambda r: r["state"] == "pending"
                             and r["created_at"] < now - timedelta(days=s["pending_days"]), state="expired",
                             done_at=now, status=410,
                             response={"detail": f"the request waited more than {s['pending_days']} days for its "
                                                 "machine; send it again"})
            self._delete("jobs", [k for k, j in self.t["jobs"].items() if j["status"] == "deleting"
                                  and j["deleted_at"] < now - timedelta(days=s["pending_days"])])
            self._delete("job_secrets", [k for k, x in self.t["job_secrets"].items()
                                         if x["created_at"] < now - timedelta(hours=1)])
            for k, r in list(self.t["requests"].items()):  # large answers go an hour after they came
                if r["state"] in ("done", "withdrawn", "expired") and r["done_at"] and r["done_at"] < now - timedelta(
                        hours=1) and r["response"] is not None and jsize(r["response"]) > 1024:
                    self._store("requests", {**r, "response": None}, k)
            self._delete("requests", [k for k, r in self.t["requests"].items()
                                      if r["state"] in ("done", "withdrawn", "expired") and r["done_at"]
                                      and r["done_at"] < now - timedelta(days=7)])
            for k in [k for k in self.rate if k[1] < now - timedelta(days=2)]:
                del self.rate[k]
            stopped = {j["id"] for j in self.t["jobs"].values() if j["status"] not in BUSY
                       and (j["stopped_at"] or j["updated_at"]) < now - timedelta(days=14)}
            self._delete("job_logs", [k for k, x in self.t["job_logs"].items() if x["job_id"] in stopped])
            return n

    # ---- the functions that users call ----

    def fn_relay(self, c, p_job, p_route, p_query, p_body, p_text, p_sealed):
        uid = self.uid_of(c)
        a = self.access(uid)
        if a is None or not a["approved"]:
            raise_("your account is waiting for approval")
        j = self.t["jobs"].get((p_job,))
        if j is None or j["user_id"] != uid or j["deleted_at"] is not None:
            raise_("no such job")
        if p_route not in ROUTES[:-1]:
            raise_("unknown request")
        p_query = {} if p_query is None else p_query
        p_body = {} if p_body is None else p_body
        if not isinstance(p_body, dict) or jsize(p_body) > 131072:
            raise_("the request body is not a JSON object of at most 128 KB")
        if not isinstance(p_query, dict) or set(p_query) - {"name", "lidar_topic"} or "lidar_topic" in p_query and (
                p_route != "calibration_put" or not isinstance(p_query["lidar_topic"], str)):
            raise_("bad query")  # part 7: lidar_topic is text, for a calibration upload only
        if too_deep(p_body, JSON_DEPTH) or too_deep(p_query, JSON_DEPTH):
            raise_(f"the request nests lists and objects more than {JSON_DEPTH} deep")
        if p_body.get("api_key") is not None:
            raise_("send the API key sealed, not in the body")
        agent = p_body.get("agent", MISSING)
        if agent == "none" or agent is False:
            raise_("the public site runs the agents astra and opus")
        if (p_text is not None) != (p_route == "calibration_put"):
            raise_("bad calibration body")
        if p_route == "calibration_put" and not isinstance(p_query.get("name"), str):
            raise_("the calibration needs a file name")
        if j["status"] in BUSY and p_route != "cancel":
            raise_(f"the job is {j['status']}; wait until it stops, or cancel it")
        if p_route == "parts" and j["status"] != "review":
            raise_("the job does not wait for its part list")
        if p_route in ("topic", "calibration_put", "calibration_delete") and j["input_kind"] != "rosbag":
            raise_("this job's input is not a rosbag")
        open_ = [r for r in self.t["requests"].values() if r["job_id"] == p_job and r["state"] in ("pending", "taken")]
        if p_route == "cancel":
            if any(r["route"] == "cancel" and r["state"] == "pending" for r in open_):
                raise_("a cancel for this job already waits")
            self.withdraw_pending(p_job, "withdrawn by your cancel")
        elif open_:
            raise_("an earlier request for this job still waits for its machine")
        if self.bump_rate(uid) > a["max_requests_per_hour"]:
            raise_(f"too many requests: at most {a['max_requests_per_hour']} an hour")
        w = self.live_worker(j["machine_id"])
        ref = None
        if p_sealed is not None:
            if p_route not in ("start", "overrides", "parts", "continue"):
                raise_("this request takes no API key")
            if not isinstance(p_sealed, dict):
                raise_("the sealed key is not an object")
            if w is None:
                raise_("worker_offline: the machine that runs this job is offline; send the request without the key")
            if not w["accepting"]:
                raise_("worker_ending: the machine stops soon; send the request without the key")
            if try_uuid(jtext(p_sealed.get("worker_id"))) != w["id"]:
                raise_("worker_restarted: the machine restarted; paste the key again")
            sa = jtext(p_sealed.get("agent")) or ""
            if sa not in AGENTS or sa not in a["agents"]:
                raise_(f"this account may use the agents {', '.join(a['agents'])}")
            ref = try_uuid(jtext(p_sealed.get("ref")))
            lens = [len(jtext(p_sealed.get(k)) or "") for k in ("epk", "iv", "ct")]
            if ref is None or lens[0] != 88 or lens[1] != 16 or not 24 <= lens[2] <= 704:
                raise_("the sealed key is malformed")
            self._insert("job_secrets", {"ref": ref, "job_id": p_job, "user_id": uid, "worker_id": w["id"], "agent": sa,
                                         "epk": jtext(p_sealed["epk"]), "iv": jtext(p_sealed["iv"]),
                                         "ct": jtext(p_sealed["ct"])})
        r = self._insert("requests", {"user_id": uid, "job_id": p_job, "route": p_route, "query": p_query,
                                      "body": p_body, "body_text": p_text, "key_ref": ref,
                                      "sent_online": w is not None and w["accepting"]})
        return {"id": r["id"], "worker_online": w is not None, "accepting": bool(w and w["accepting"])}

    def fn_withdraw_request(self, c, p_id):
        r = self.t["requests"].get((p_id,))
        if r is None or r["user_id"] != c.uid or r["state"] != "pending" or r["route"] == "delete":
            raise_("the request is no longer waiting")
        self._store("requests", {**r, "state": "withdrawn", "done_at": self.now(), "status": 409,
                                 "response": {"detail": "you withdrew this request"}})
        if r["key_ref"]:
            self._delete("job_secrets", [(r["key_ref"],)])
        return "withdrawn"

    def fn_delete_job(self, c, p_job):
        uid = self.uid_of(c)
        j = self.t["jobs"].get((p_job,))
        if j is None or j["user_id"] != uid or j["deleted_at"] is not None:
            raise_("no such job")
        w = self.live_worker(j["machine_id"])
        if j["status"] in ACTIVE and w is not None:
            raise_(f"the job is {j['status']}; cancel it first")
        self.withdraw_pending(p_job, "the job was deleted")
        for t in ("job_views", "job_live", "job_snapshots", "job_files", "job_logs"):  # the row alone waits
            self._delete(t, [k for k, x in self.t[t].items() if x["job_id"] == p_job])
        now = self.now()
        self._store("jobs", {**j, "deleted_at": now, "status": "deleting", "updated_at": now})
        self._insert("requests", {"user_id": uid, "job_id": p_job, "route": "delete", "sent_online": w is not None})
        return "deleting"

    def fn_request_access(self, c, p_note):
        if p_note is None or not 1 <= len(p_note) <= 500:
            raise_("write 1 to 500 characters")
        a = self.access(c.uid)
        if a is None or (a["request_at"] is not None and a["request_at"] >= self.now() - timedelta(days=1)):
            raise_("you can send one request a day")
        self._store("access", {**a, "request_note": p_note, "request_at": self.now()})
        return "sent"

    # ---- the functions of a personal worker ----

    def own_worker(self, c, p_worker):
        w = self.t["workers"].get((p_worker,)) if p_worker else None
        if w is None or w["user_id"] != c.uid:
            raise_("no such worker")
        if w["stopped_at"] is not None:
            raise_(f"worker {p_worker} was stopped: another worker took over")
        if not self.approved(c.uid):
            raise_("your account is waiting for approval")
        return w

    def own_job(self, w, p_job, deleting=False):
        j = self.t["jobs"].get((p_job,))
        if j is None or j["user_id"] != w["user_id"] or j["machine_id"] != w["machine_id"] \
                or (j["deleted_at"] is not None) != deleting:
            raise_("no such job on this machine")
        return j

    def approved(self, uid):
        a = self.access(uid)
        return a is not None and a["approved"]

    def fn_own_machine_bind(self, c, p_machine, p_name):
        uid = self.uid_of(c)
        if not self.approved(uid):
            raise_("your account is waiting for approval")
        if p_name is None or not MACHINE.match(p_name):
            raise_("the machine name has 1 to 60 letters, digits, _ . and -, and starts with a letter or digit")
        if p_machine is not None:
            m = self.t["machines"].get((p_machine,))
            if m is None or m["user_id"] != uid:
                raise_("no such machine")
            self._store("machines", {**m, "name": p_name})
            return p_machine
        n = len(self._find("machines", lambda m: m["user_id"] == uid))
        if n >= self.settings()["max_machines"]:
            raise_(f"you have {n} machines; ask the site owner to remove one")
        return self._insert("machines", {"user_id": uid, "name": p_name})["id"]

    def fn_own_worker_start(self, c, p_machine, p_public_key, p_ends_at, p_version, p_local_keys):
        uid = self.uid_of(c)
        if not self.approved(uid):
            raise_("your account is waiting for approval")
        m = self.t["machines"].get((p_machine,)) if p_machine else None
        if m is None or m["user_id"] != uid:
            raise_("no such machine")
        now = self.now()
        self._update("workers", lambda w: w["machine_id"] == p_machine and w["stopped_at"] is None, stopped_at=now,
                     accepting=False)
        wid = self._insert("workers", {"user_id": uid, "machine_id": p_machine, "public_key": p_public_key,
                                       "ends_at": p_ends_at, "version": p_version,
                                       "local_keys": p_local_keys or []})["id"]
        old = {w["id"] for w in self.t["workers"].values() if w["machine_id"] == p_machine and w["id"] != wid}
        self._delete("job_secrets", [k for k, s in self.t["job_secrets"].items() if s["worker_id"] in old])
        self._update("requests", lambda r: r["state"] == "taken" and r["worker_id"] in old, state="done", done_at=now,
                     status=409, response={"detail": "your machine restarted during this request; look at the job "
                                                     "and try again"})
        jobs = {j["id"] for j in self.t["jobs"].values() if j["machine_id"] == p_machine}
        self._update("requests", lambda r: r["state"] == "pending" and r["key_ref"] and r["job_id"] in jobs
                     and (r["key_ref"],) not in self.t["job_secrets"], state="done", done_at=now, status=400,
                     response={"detail": "your machine restarted after you sent the key; paste the key again"})
        mine = sorted((w for w in self.t["workers"].values() if w["machine_id"] == p_machine),
                      key=lambda w: (w["stopped_at"] is not None, -w["started_at"].timestamp(), w["id"]))
        keep = {w["id"] for w in mine[:10]}
        self._delete("workers", [(w["id"],) for w in mine if w["id"] not in keep and w["stopped_at"] is not None])
        return wid

    def fn_own_worker_beat(self, c, p_worker, p_queue_len, p_accepting, p_ends_at):
        w = self.own_worker(c, p_worker)
        self._store("workers", {**w, "last_seen": self.now(), "queue_len": max(p_queue_len or 0, 0),
                                "accepting": True if p_accepting is None else bool(p_accepting),
                                "ends_at": p_ends_at if p_ends_at is not None else w["ends_at"]})

    def fn_own_worker_stop(self, c, p_worker):
        self._update("workers", lambda w: w["id"] == p_worker and w["user_id"] == c.uid and w["stopped_at"] is None,
                     stopped_at=self.now(), accepting=False)

    def fn_own_claim_request(self, c, p_worker, p_kinds):
        """The oldest pending request of the machine's jobs whose job has no request in progress and (with kinds)
        no earlier pending request that kinds leaves out; now taken, with its sealed key."""
        w = self.own_worker(c, p_worker)
        reqs = sorted(self.t["requests"].values(), key=lambda r: r["id"])
        for r in reqs:
            j = self.t["jobs"].get((r["job_id"],))
            if r["state"] != "pending" or r["user_id"] != w["user_id"] or j is None \
                    or j["machine_id"] != w["machine_id"] or (p_kinds is not None and r["route"] not in p_kinds):
                continue
            if any(b["job_id"] == r["job_id"] and b["state"] == "taken" for b in reqs):
                continue
            if p_kinds is not None and any(o["job_id"] == r["job_id"] and o["state"] == "pending"
                                           and o["id"] < r["id"] and o["route"] not in p_kinds for o in reqs):
                continue
            r = {**r, "state": "taken", "taken_at": self.now(), "worker_id": p_worker}
            self._store("requests", r)
            s = self.t["job_secrets"].get((r["key_ref"],)) if r["key_ref"] else None
            if s is not None and s["worker_id"] == p_worker:
                self._delete("job_secrets", [(r["key_ref"],)])
            else:
                s = None
            return [{**r, "sealed": None if s is None else {k: s[k] for k in ("agent", "epk", "iv", "ct")}}]
        return []

    def fn_own_answer(self, c, p_worker, p_request, p_status, p_response):
        self.own_worker(c, p_worker)
        if p_status is None or not 100 <= p_status <= 599:
            raise_("the status is an HTTP status")
        r = self.t["requests"].get((p_request,))
        if r is None or r["worker_id"] != p_worker or r["state"] != "taken":
            raise_("the request is not taken by this worker")
        self._store("requests", {**r, "state": "done", "done_at": self.now(), "status": p_status,
                                 "response": p_response})

    def fn_own_job_create(self, c, p_worker, p_name, p_kind, p_bytes):
        w = self.own_worker(c, p_worker)
        a = self.access(w["user_id"])
        if p_name is None or not NAME.match(p_name):
            raise_("the file name has 1 to 200 letters, digits, _ . and -, does not start with a dot and has no two "
                   "dots in a row")
        if p_kind is None or p_kind != kind_of(p_name):
            raise_("the input is a video (.mp4 .mov .m4v .avi .mkv .webm), a rosbag (.bag .mcap .db3) or a Stray "
                   "Scanner export (.zip)")
        if p_bytes is None or p_bytes < 1:
            raise_("the file is empty")
        n = len(self._find("jobs", lambda j: j["user_id"] == w["user_id"] and j["deleted_at"] is None))
        if n >= a["max_own_jobs"]:
            raise_(f"you have {n} jobs from your own machines on the site; delete older ones first")
        day = (w["user_id"], self.today())
        u = self.t["usage_daily"].get(day) or self._insert("usage_daily", {"user_id": w["user_id"]})
        if u["own_jobs"] >= a["max_own_per_day"]:
            raise_(f"you reached {a['max_own_per_day']} jobs from your own machines today")
        self._store("usage_daily", {**u, "own_jobs": u["own_jobs"] + 1})
        return self._insert("jobs", {"user_id": w["user_id"], "machine_id": w["machine_id"], "input_name": p_name,
                                     "input_kind": p_kind, "input_bytes": p_bytes, "status": "uploaded"})["id"]

    def fn_own_publish(self, c, p_worker, p_job, p_view, p_live, p_snapshots, p_row):
        w = self.own_worker(c, p_worker)
        j = self.own_job(w, p_job)
        now = self.now()
        if p_row is not None:
            if not isinstance(p_row, dict) or set(p_row) - {"status", "note", "prompt", "queue_pos", "stopped_at"}:
                raise_("the row takes status, note, prompt, queue_pos and stopped_at")
            st = jtext(p_row["status"]) if p_row.get("status") is not None else j["status"]
            if st not in OWN_STATUSES:
                raise_(f"bad status {st}")
            new = {**j, "status": st, "updated_at": now}
            for k in ("note", "prompt"):
                if k in p_row:
                    new[k] = jtext(p_row[k])
            if "queue_pos" in p_row:
                new["queue_pos"] = None if p_row["queue_pos"] is None else int(p_row["queue_pos"])
            if "stopped_at" in p_row:  # least() skips a null, so a null gives now()
                t = None if p_row["stopped_at"] is None else self.timestamp(p_row["stopped_at"])
                if t is not None and t < datetime(2020, 1, 1, tzinfo=timezone.utc):
                    raise_("stopped_at is a time from 2020 on")
                new["stopped_at"] = min(t, now) if t is not None else now
            self._store("jobs", new)
        if p_view is not None:
            if not isinstance(p_view, dict):
                raise_("the view is a JSON object")
            old = self.t["job_views"].get((p_job,))
            self._insert("job_views", {"job_id": p_job, "user_id": w["user_id"], "body": p_view,
                                       "rev": old["rev"] + 1 if old else 0, "updated_at": now}, upsert=True)
        if p_live is not None:
            if not isinstance(p_live, dict):
                raise_("the live state is a JSON object")
            self._insert("job_live", {"job_id": p_job, "user_id": w["user_id"], "state": p_live, "updated_at": now},
                         upsert=True)
        if p_snapshots is not None:
            if not isinstance(p_snapshots, dict) or set(p_snapshots) - {"frames", "selection", "parts"}:
                raise_("the snapshots are frames, selection and parts")
            for k, snap in p_snapshots.items():
                self._insert("job_snapshots", {"job_id": p_job, "path": k, "user_id": w["user_id"],
                                               "status": (snap or {}).get("status"), "body": (snap or {}).get("body"),
                                               "updated_at": now}, upsert=True)

    @staticmethod
    def timestamp(v):
        """A JSON value cast to timestamptz: infinity, -infinity and BC dates as times that are not after 2020
        (Postgres takes them), anything else that is not ISO 8601 refused as Postgres refuses it."""
        text = jtext(v).strip()
        if text in ("-infinity", "infinity") or text.upper().endswith(" BC"):
            return datetime.min.replace(tzinfo=timezone.utc)  # -infinity and BC dates; infinity is refused too
        try:
            t = datetime.fromisoformat(text)
        except ValueError:
            raise Fail(400, {"code": "22007", "message":
                             f'invalid input syntax for type timestamp with time zone: "{text}"'}) from None
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)

    def fn_own_files(self, c, p_worker, p_job, p_put, p_drop):
        w = self.own_worker(c, p_worker)
        self.own_job(w, p_job)
        p_put = [] if p_put is None else p_put
        if not isinstance(p_put, list) or len(p_put) > 500:
            raise_("put is a list of at most 500 files")
        drop = set(p_drop or [])
        self._delete("job_files", [k for k, f in self.t["job_files"].items()
                                   if f["job_id"] == p_job and f["path"] in drop])
        for f in p_put:
            if f.get("bytes") is None or self.size_of("job-files", jtext(f.get("object"))) != int(f["bytes"]):
                raise_(f"job-files holds no object for {jtext(f.get('path'))} with {jtext(f.get('bytes'))} bytes")
            self._insert("job_files", {"job_id": p_job, "path": f.get("path"), "user_id": w["user_id"],
                                       "object": f.get("object"), "bytes": f.get("bytes"), "sha256": f.get("sha256"),
                                       "role": f.get("role"), "content_type": f.get("content_type"),
                                       "updated_at": self.now()}, upsert=True)
        if len(self._find("job_files", lambda f: f["job_id"] == p_job)) > 2000:
            raise_("a job keeps at most 2000 files on the site")

    def fn_own_logs(self, c, p_worker, p_job, p_rows):
        w = self.own_worker(c, p_worker)
        self.own_job(w, p_job)
        cap = self.settings()["own_log_bytes"]
        if not isinstance(p_rows, list) or not 1 <= len(p_rows) <= 100:
            raise_("rows is a list of 1 to 100 log parts")
        used = sum(len(r["text"].encode()) for r in self.t["job_logs"].values() if r["job_id"] == p_job)
        more = sum(len((jtext(e.get("text")) or "").encode()) for e in p_rows)
        if used + more > cap:
            raise_(f"the job's log on the site is full ({cap // 1048576} MB); the rest stays on your machine")
        for e in p_rows:
            self._insert("job_logs", {"job_id": p_job, "user_id": w["user_id"], "stage": jtext(e.get("stage")),
                                      "run_started": e.get("run_started"), "pos": e.get("pos"),
                                      "text": jtext(e.get("text"))})

    def fn_own_job_delete(self, c, p_worker, p_job):
        w = self.own_worker(c, p_worker)
        j = self.own_job(w, p_job, deleting=True)
        self._delete("jobs", [(j["id"],)])

    def fn_storage_used(self, c):
        return self.storage_used(c.uid)

    def storage_used(self, uid):
        return sum(o["size"] for (b, n), o in self.objects.items()
                   if b == "job-files" and uid and n.startswith(uid + "/"))

    def fn_own_object_ok(self, c, p_name):
        return self.own_object_ok(c.uid, p_name)

    def own_object_ok(self, uid, name):
        """True when uid may upload name: approved, a job of uid that is not deleted, the name's form, and room
        under max_result_bytes."""
        if not uid or not isinstance(name, str):
            return False
        parts, a = name.split("/"), self.access(uid)
        j = self.t["jobs"].get((parts[1],)) if len(parts) > 1 else None
        return bool(a and a["approved"] and j and j["user_id"] == uid and j["deleted_at"] is None and parts[0] == uid
                    and OWN_OBJECT.match(name) and not re.search(r"\.\.|//", name)
                    and self.storage_used(uid) < a["max_result_bytes"])

    def fn_job_objects(self, c, p_job, p_limit):
        uid = c.uid
        j = self.t["jobs"].get((p_job,))
        if j is None or j["user_id"] != uid:
            raise_("no such job")
        prefix = f"{uid}/{p_job}/"
        out = sorted((o for (b, n), o in self.objects.items() if b == "job-files" and n.startswith(prefix)),
                     key=lambda o: o["created_at"])
        return [{"name": o["name"], "created_at": o["created_at"]} for o in out[:limit_of(p_limit)]]

    def fn_own_orphans(self, c, p_worker, p_limit):
        w = self.own_worker(c, p_worker)
        now, named = self.now(), {f["object"] for f in self.t["job_files"].values()}
        out = []
        for (b, n), o in sorted(self.objects.items(), key=lambda kv: kv[1]["created_at"]):
            if b != "job-files" or not n.startswith(w["user_id"] + "/") or o["created_at"] >= now - timedelta(hours=1) \
                    or n in named:
                continue
            parts = n.split("/")
            j = self.t["jobs"].get((try_uuid(parts[1]) if len(parts) > 1 else None,))
            if j is None or j["machine_id"] == w["machine_id"]:
                out.append({"name": n})
        return out[:limit_of(p_limit)]

    # ---- PostgREST tables ----

    def _rest(self, c, request, table):
        """A read of a table, under its grant and RLS. Users write no table: every other method is denied."""
        if table not in SCHEMA:
            if table == "rate":
                denied(c.role, "table", table)
            raise Fail(404, {"code": "42P01", "message": f'relation "public.{table}" does not exist'})
        if c.role == "anon" or request.method != "GET" or table not in USER_TABLES:
            denied(c.role, "table", table)
        q = urllib.parse.parse_qsl(request.url.query.decode(), keep_blank_values=True)
        params = [(k, v) for k, v in q if k not in ("select", "order", "limit", "offset")]
        opts = dict((k, v) for k, v in q if k in ("select", "order", "limit"))
        with self.lock:
            columns = self._columns(table, opts.get("select", "*"))
            rows = [r for r in self.t[table].values() if self._match(table, r, params) and self._visible(c, table, r)]
            rows = self._order(table, rows, opts.get("order"))
            if opts.get("limit"):
                rows = rows[:int(opts["limit"])]
            return self._json([{k: r[k] for k in columns} for r in rows])

    def _visible(self, c, table, r):
        """The RLS policies of the migrations, for role authenticated."""
        if table == "settings":
            return True
        if table == "workers":
            return r["user_id"] == c.uid and r["last_seen"] > self.now() - timedelta(days=30)
        if table == "jobs":
            return r["user_id"] == c.uid and r["deleted_at"] is None
        return r.get("user_id") == c.uid

    def _columns(self, table, select):
        if select.strip() == "*":
            return list(SCHEMA[table]["cols"])
        cols = [s.strip() for s in select.split(",") if s.strip()]
        for col in cols:
            if col not in SCHEMA[table]["cols"]:
                raise Fail(400, {"code": "42703", "message": f"column {table}.{col} does not exist"})
        return cols

    def _value(self, table, col, text):
        if col not in SCHEMA[table]["cols"]:
            raise Fail(400, {"code": "42703", "message": f"column {table}.{col} does not exist"})
        kind = SCHEMA[table]["cols"][col][0]
        if kind == "int":
            return int(text)
        if kind == "float":
            return float(text)
        if kind == "bool":
            return text == "true"
        if kind == "uuid":
            return text.lower()
        if kind == "ts":
            return datetime.fromisoformat(text)
        return text

    def _test(self, table, col, expr, r):
        neg = expr.startswith("not.")
        expr = expr.removeprefix("not.")
        op, _, val = expr.partition(".")
        v = r.get(col)
        if op == "is":
            ok = v is None if val == "null" else v is (val == "true")
        elif op == "in":
            ok = v in [self._value(table, col, x) for x in split_list(val.strip("()"))]
        elif op == "like":
            pattern = re.escape(val).replace(r"\*", ".*").replace("%", ".*")
            ok = v is not None and re.fullmatch(pattern, str(v)) is not None
        else:
            x = self._value(table, col, val)
            ok = v is not None and {"eq": v == x, "neq": v != x, "gt": v > x, "gte": v >= x, "lt": v < x,
                                    "lte": v <= x}[op]
        return not ok if neg else ok

    def _match(self, table, r, params):
        for k, expr in params:
            if k == "or":
                items = split_list(expr.strip()[1:-1])
                if not any(self._test(table, *it.split(".", 1), r) for it in items):
                    return False
            elif not self._test(table, k, expr, r):
                return False
        return True

    def _order(self, table, rows, order):
        for part in reversed((order or "").split(",") if order else []):
            col, _, way = part.partition(".")
            desc = way.startswith("desc")
            have = [r for r in rows if r.get(col) is not None]
            none = [r for r in rows if r.get(col) is None]
            have.sort(key=lambda r: r[col], reverse=desc)
            rows = (none + have) if desc else (have + none)
        return rows

    # ---- Storage ----

    def _storage(self, c, request, path):
        method = request.method
        m = re.fullmatch(r"object/([a-z0-9_-]+)(?:/(.+))?", path)
        if not m:
            return httpx.Response(404, json={"statusCode": "404", "error": "not_found", "message": "not found"})
        bucket, name = m[1], m[2]  # httpx gives the path decoded
        if bucket not in self.buckets:
            return httpx.Response(400, json={"statusCode": "404", "error": "Bucket not found",
                                             "message": "Bucket not found"})
        if c.role == "anon":
            return httpx.Response(400, json={"statusCode": "403", "error": "Unauthorized",
                                             "message": "new row violates row-level security policy"})
        with self.lock:
            if method == "POST" and name:
                return self._upload(c, request, bucket, name)
            if method == "DELETE" and not name:
                names = json.loads(request.content or b"{}").get("prefixes") or []
                gone = []
                for n in names:
                    if (bucket, n) in self.objects and self._may_delete(c.uid, n):
                        del self.objects[(bucket, n)]
                        gone.append({"bucket_id": bucket, "name": n})
                self.done(c.label, "remove", bucket)
                return self._json(gone)
        return httpx.Response(405, json={"statusCode": "405", "error": "method", "message": "not allowed"})

    def _upload(self, c, request, bucket, name):
        data = request.content
        fail = self.fail_next.pop(("upload", name), None) or self.fail_next.pop(("upload", "*"), None)
        if fail is not None:
            raise fail
        if (bucket, name) in self.objects and request.headers.get("x-upsert", "false") != "true":
            return httpx.Response(400, json={"statusCode": "409", "error": "Duplicate",
                                             "message": "The resource already exists"})
        if len(data) > self.buckets[bucket]["limit"]:
            return httpx.Response(413, json={"statusCode": "413", "error": "Payload too large",
                                             "message": "The object exceeded the maximum allowed size"})
        mime, ctype = self.buckets[bucket]["mime"], request.headers.get("content-type")
        if mime is not None and ctype not in mime:
            return httpx.Response(400, json={"statusCode": "415", "error": "invalid_mime_type",
                                             "message": f"mime type {ctype} is not supported"})
        if not self.own_object_ok(c.uid, name):
            return httpx.Response(403, json={"statusCode": "403", "error": "Unauthorized",
                                             "message": "new row violates row-level security policy"})
        self.objects[(bucket, name)] = {"bucket_id": bucket, "name": name, "owner": c.uid, "size": len(data),
                                        "data": data, "content_type": request.headers.get("content-type"),
                                        "created_at": self.now()}
        self.done(c.label, "upload", name)
        return httpx.Response(200, json={"Key": f"{bucket}/{name}", "Id": str(uuid.uuid4())})

    def storage_insert(self, uid, bucket, name, size):
        """An insert into storage.objects under the policies, as the case table's storage_insert does it."""
        with self.lock:
            if bucket not in self.buckets or not self.own_object_ok(uid, name):
                raise Fail(403, {"statusCode": "403", "error": "Unauthorized",
                                 "message": "new row violates row-level security policy"})
            self.object(bucket, name, size=size, owner=uid)

    @staticmethod
    def _may_delete(uid, name):
        """The delete policy: an object in the caller's own folder, with or without approval."""
        return bool(uid) and name.split("/")[0] == uid


def split_list(text):
    """The items of a PostgREST list (in.(...) or or=(...)): commas outside double quotes and parentheses split,
    quotes and backslash escapes are taken off."""
    out, cur, depth, quoted, esc = [], "", 0, False, False
    for ch in text:
        if esc:
            cur, esc = cur + ch, False
        elif ch == "\\" and quoted:
            esc = True
        elif ch == '"':
            quoted = not quoted
        elif ch == "(" and not quoted:
            depth, cur = depth + 1, cur + ch
        elif ch == ")" and not quoted:
            depth, cur = depth - 1, cur + ch
        elif ch == "," and not quoted and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur] if cur or out else out


class FakeSession:
    """A personal worker's session with the fake, with what Store.of and the worker read from session.Session:
    url, publishable_key, user_id, email, machine_id, machine_name, mirror_since, bearer(), refresh() and
    set_machine()."""

    def __init__(self, fake, user_id, machine_id=None, machine_name="lab-box", mirror_since="2000-01-01T00:00:00"):
        self.fake, self.url, self.publishable_key = fake, fake.url, PUBLISHABLE
        self.user_id, self.email = user_id, fake.users.get(user_id)
        self.machine_id, self.machine_name, self.mirror_since = machine_id, machine_name, mirror_since
        self.token, self.refreshes = fake.access_token(user_id), 0

    def __repr__(self):
        return f"FakeSession({self.user_id})"

    async def bearer(self):
        return self.token

    async def refresh(self):
        self.token = self.fake.access_token(self.user_id)
        self.refreshes += 1

    def set_machine(self, machine_id):
        self.machine_id = machine_id

    def close(self):
        pass


# ---- sealed keys, as the page makes them ----

def seal(public_b64, key, aad):
    """What the page does (supa.js seal()): {"epk", "iv", "ct"} in base64, key sealed under aad to a worker's public
    key."""
    from cryptography.hazmat.primitives.asymmetric import ec

    from gator.webapp.keybox import aes, b64, peer, point, unb64

    eph = ec.generate_private_key(ec.SECP256R1())
    epk, iv = point(eph), os.urandom(12)
    shared = eph.exchange(ec.ECDH(), peer(unb64(public_b64, "public key"), "worker's public key"))
    return {"epk": b64(epk), "iv": b64(iv), "ct": b64(aes(shared, epk, aad).encrypt(iv, key.encode(), aad))}


def sealed(worker, key, job, user, agent, ref=None):
    """relay()'s p_sealed for key sealed to a workers row ({"id", "public_key"}), as supa.js seal() makes it:
    {ref, worker_id, agent, epk, iv, ct}."""
    from gator.webapp.keybox import aad

    ref = ref or str(uuid.uuid4())
    return {"ref": ref, "worker_id": worker["id"], "agent": agent,
            **seal(worker["public_key"], key, aad(job, user, agent, worker["id"], ref))}


# ---- the shared case table (its ids, names() and names_in() are supabase_cases.py's) ----

def case_base():
    """A fake with the rows that every case starts from (the table's "_format")."""
    f = FakeSupabase()
    for u in USERS:
        f.add_user(f"{u}@example.org", IDS[u])
    for u in ("c", "d"):
        f.set("access", {"user_id": IDS[u]}, approved=True)
    f.put("machines", {"id": IDS["mc"], "user_id": IDS["c"], "name": "lab-box"})
    f.put("machines", {"id": IDS["md"], "user_id": IDS["d"], "name": "d-box"})
    f.put("workers", {"id": IDS["wc"], "user_id": IDS["c"], "machine_id": IDS["mc"], "public_key": "C" * 88})
    f.put("workers", {"id": IDS["wd"], "user_id": IDS["d"], "machine_id": IDS["md"], "public_key": "D" * 88})
    return f


def case_setup(case):
    """(fake, the caller's headers, the job's user) for one case, after its state."""
    f = case_base()
    who, state = case["as"], case.get("state", {})
    job = state.get("job", {})
    machine = job.get("machine", MACHINE_OF.get(who, "mc"))
    job_user = OWNER[machine]
    access_user = who if who in USERS else "c"
    status, kind = job.get("status", "ready"), job.get("input_kind", "video")
    if state.get("settings"):
        f.set("settings", {"id": 1}, **state["settings"])
    if state.get("access"):
        f.set("access", {"user_id": IDS[access_user]}, **state["access"])
    if "own_worker" in state and state["own_worker"] is None:
        f.set("workers", {}, stopped_at=f.now(), accepting=False)
    else:
        f.set("workers", {}, last_seen=f.now())
    f.put("jobs", {"id": IDS["job"], "user_id": IDS[job_user], "machine_id": IDS[machine], "input_name": INPUT[kind],
                   "input_kind": kind, "input_bytes": 1000, "status": status,
                   "deleted_at": f.now() if status == "deleting" else None})
    if state.get("rate"):
        f.rate[(IDS[job_user], f.hour())] = state["rate"]
    for i, r in enumerate(state.get("requests", [])):
        ref = None
        if r.get("sealed") is not None:
            ref = f"5ea1ed00-0000-0000-0000-{i:012d}"
            f.put("job_secrets", {"ref": ref, "job_id": IDS["job"], "user_id": IDS[job_user],
                                  "worker_id": IDS[WORKERS[r["sealed"].get("worker")]],
                                  "agent": r["sealed"].get("agent", "opus"), "epk": "B" * 88, "iv": "C" * 16,
                                  "ct": "D" * 40})
        f.put("requests", {"user_id": IDS[job_user], "job_id": IDS["job"], "route": r["route"],
                           "state": r.get("state", "pending"), "key_ref": ref})
    for o in state.get("objects", []):
        f.object(o["bucket"], names(o["name"]), size=o["size"], age=o.get("age", 0))
    if who in USERS:
        headers = {"apikey": PUBLISHABLE, "Authorization": f"Bearer {f.access_token(IDS[who])}"}
    else:
        headers = {"apikey": PUBLISHABLE, "Authorization": f"Bearer {PUBLISHABLE}"}
    return f, headers, job_user


def case_call(f, case, job_user):
    """(method, path, json) of the case's call over HTTP, or ("storage_insert", uid, args) for an insert that
    the table makes as SQL."""
    fn, a = case["fn"], case.get("args", {})
    wid = lambda name: IDS[WORKERS[name]] if name else None  # noqa: E731
    if fn == "relay":
        sealed = a.get("sealed")
        if sealed is not None:
            sealed = {} if sealed == "valid" else sealed
            jw = f.live_worker(f.t["jobs"][(IDS["job"],)]["machine_id"])
            w = wid(sealed["worker"]) if "worker" in sealed else jw["id"] if jw else str(uuid.uuid4())
            sealed = {"ref": str(uuid.uuid4()), "worker_id": w, "agent": sealed.get("agent", "opus"), "epk": "B" * 88,
                      "iv": "C" * 16, "ct": "D" * 40}
        return "POST", "/rest/v1/rpc/relay", {"p_job": IDS["job"], "p_route": a.get("route"),
                                              "p_query": a.get("query", {}), "p_body": a.get("body", {}),
                                              "p_text": a.get("text"), "p_sealed": sealed}
    if fn == "delete_job":
        return "POST", "/rest/v1/rpc/delete_job", {"p_job": IDS["job"]}
    if fn == "job_objects":
        return "POST", "/rest/v1/rpc/job_objects", {"p_job": IDS["job"], "p_limit": a.get("limit", 1000)}
    if fn == "own_orphans":
        return "POST", "/rest/v1/rpc/own_orphans", {"p_worker": wid(a.get("worker")), "p_limit": a.get("limit", 1000)}
    if fn == "own_claim_request":
        return "POST", "/rest/v1/rpc/own_claim_request", {"p_worker": wid(a.get("worker")), "p_kinds": a.get("kinds")}
    if fn == "own_worker_start":
        m = a.get("machine")
        return "POST", "/rest/v1/rpc/own_worker_start", {"p_machine": IDS[m] if m else None, "p_public_key": "K" * 88,
                                                         "p_ends_at": None, "p_version": "test",
                                                         "p_local_keys": a.get("local_keys", [])}
    if fn == "own_publish":
        parts = {f"p_{k}": a.get(k) for k in ("view", "live", "snapshots", "row")}
        return "POST", "/rest/v1/rpc/own_publish", {"p_worker": wid(a.get("worker")), "p_job": IDS["job"], **parts}
    if fn == "own_files":
        put = []
        for row in a.get("put", []):
            sha = row.get("sha256", SHA)
            obj = names(row["object"]) if "object" in row else f"{IDS[job_user]}/{IDS['job']}/{sha[:16]}/{row['path']}"
            put.append({"path": row["path"], "object": obj, "bytes": row["bytes"], "sha256": sha,
                        "role": row.get("role", "result"), "content_type": row.get("content_type",
                                                                                   "application/octet-stream")})
        return "POST", "/rest/v1/rpc/own_files", {"p_worker": wid(a.get("worker")), "p_job": IDS["job"], "p_put": put,
                                                  "p_drop": a.get("drop", [])}
    if fn == "own_logs":
        return "POST", "/rest/v1/rpc/own_logs", {"p_worker": wid(a.get("worker")), "p_job": IDS["job"],
                                                 "p_rows": a.get("rows")}
    if fn == "own_job_create":
        return "POST", "/rest/v1/rpc/own_job_create", {"p_worker": wid(a.get("worker")), "p_name": a.get("name"),
                                                       "p_kind": a.get("kind"), "p_bytes": a.get("bytes")}
    if fn == "storage_delete":
        return "DELETE", f"/storage/v1/object/{a['bucket']}", {"prefixes": [names(a["name"])]}
    if fn == "storage_insert":
        return "storage_insert", None, a
    raise ValueError(f"unknown fn {fn!r}")


def contains(got, want):
    """jsonb @>: every key of want is in got with a value that contains want's."""
    if isinstance(want, dict):
        return isinstance(got, dict) and all(k in got and contains(got[k], v) for k, v in want.items())
    if isinstance(want, list):
        return isinstance(got, list) and all(any(contains(g, w) for g in got) for w in want)
    if isinstance(want, bool) or isinstance(got, bool):  # jsonb: true is not 1
        return type(got) is type(want) and got == want
    return got == want


def case_match(got, want):
    """tests.case_match of tests/supabase_cases.py."""
    if want == "uuid":
        return isinstance(got, str) and UUID.fullmatch(got) is not None and got == got.lower()
    if isinstance(want, list):
        return isinstance(got, list) and len(got) == len(want) and all(contains(g, w) for g, w in zip(got, want))
    if isinstance(want, dict):
        return contains(got, want)
    return got == want


def run_case(case):
    """(ok, what happened) for one case of the table, made over HTTP like the real callers."""
    f, headers, job_user = case_setup(case)
    method, path, body = case_call(f, case, job_user)
    expect = case["expect"]
    if method == "storage_insert":
        uid = IDS[case["as"]] if case["as"] in USERS else None
        try:
            f.storage_insert(uid, body["bucket"], names(body["name"]), body["size"])
            status, got = 200, None
        except Fail as e:
            status, got = e.status, e.body
    else:
        async def call():
            async with httpx.AsyncClient(transport=f.transport) as client:
                return await client.request(method, URL + path, json=body, headers=headers)
        r = anyio.run(call)
        status, got = r.status_code, r.json() if r.content else None
    if "error" in expect:
        if status < 400:
            return False, f"expected an error like {expect['error']!r}, got {status} {got!r}"
        message = (got or {}).get("message", "")
        ok = re.search(expect["error"], message, re.I) is not None
        return ok, f"{status} {message!r}"
    if status >= 400:
        return False, f"expected success, got {status} {got!r}"
    want = names_in(expect["ok"])
    if case["fn"] == "storage_delete":
        return len(got or []) == want, f"deleted {got!r}"
    if case["fn"] in ("own_publish", "own_files", "own_logs", "storage_insert"):
        return want in (None, True), "done"
    if isinstance(want, dict) and "withdrawn" in want:
        want = dict(want)
        n = want.pop("withdrawn")
        withdrawn = len(f.rows("requests", job_id=IDS["job"], state="withdrawn"))
        if withdrawn != n:
            return False, f"{withdrawn} withdrawn, not {n}"
    return case_match(got, want), repr(got)
