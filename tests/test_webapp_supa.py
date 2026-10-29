"""The personal worker's link to Supabase (harpor/webapp/supa.py): the client with the publishable key and the
user's access token, Storage, the sign-in, the token renewal and the Store. The private files are in
test_fileio.py."""

import base64
import functools
import hashlib
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import anyio
import httpx
import pytest

from harpor.webapp import supa

URL = "https://abcdefghijklmnop.supabase.co"
SECRET = "sb_secret_" + "fakefakefake" * 3
PUB = "sb_publishable_fakefakefake"
TOKEN, TOKEN2 = "access-token-one-" + "a" * 20, "access-token-two-" + "b" * 20
REFRESH, REFRESH2 = "refresh-token-one", "refresh-token-two"
USER, JOB, MACHINE = "9a7e0d2e-0000-4000-8000-000000000002", "6f1c0d2e-0000-4000-8000-000000000001", \
    "5d4c3b2a-0000-4000-8000-000000000005"
P0001 = {"code": "P0001", "message": "no such job on this machine", "details": None, "hint": None}


class Fake:
    """An httpx.MockTransport handler. Records each request and answers with the next item of answers (the
    last one repeats). An item is a Response, an exception to raise, or a function of the request."""

    def __init__(self, *answers):
        self.answers, self.seen = list(answers), []
        self.transport = httpx.MockTransport(self)

    def __call__(self, request):
        self.seen.append(request)
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        a = a(request) if callable(a) and not isinstance(a, BaseException) else a
        if isinstance(a, BaseException):
            raise a
        return a

    def body(self, i=-1):
        return json.loads(self.seen[i].content) if self.seen[i].content else None


def ok(data=None, status=200, **kw):
    if data is None:
        return httpx.Response(204 if status == 200 else status, **kw)
    return httpx.Response(status, json=data, **kw)


def params(request):
    return dict(urllib.parse.parse_qsl(request.url.query.decode()))


@pytest.fixture
def naps(monkeypatch):
    """The waits between tries, without waiting."""
    seen = []

    async def nap(s):
        seen.append(s)

    monkeypatch.setattr(supa.anyio, "sleep", nap)
    return seen


def run(fn, *args, **kw):
    return anyio.run(functools.partial(fn, *args, **kw))


async def bearer():
    """The user's access token, as session.Session.bearer gives it."""
    return TOKEN


def client(transport, **kw):
    return supa.Supa(URL, PUB, bearer, transport=transport, **kw)


# ---- Supa: headers, calls, retries, errors ----

def test_headers_and_the_postgrest_calls():
    fake = Fake(lambda r: ok([{"id": 1}]) if r.method == "GET" else ok("w1"))

    async def main():
        async with supa.Supa(URL + "/", PUB, bearer, transport=fake.transport) as s:
            assert await s.rpc("x", p_a=1, p_b=None) == "w1"
            assert await s.select("jobs", "id,status", filters={"or": "(a.is.null,b.eq.1)"}, order="id", limit=5,
                                  id=JOB, deleted_at=None, approved=True) == [{"id": 1}]
            with pytest.raises(ValueError, match="not a table"):
                await s.select("jobs?select=*")
            with pytest.raises(ValueError, match="not a table"):
                await s.rpc("x/../y")
            assert PUB not in repr(s) and TOKEN not in repr(s)

    run(main)
    r = fake.seen
    assert all(q.headers["apikey"] == PUB and q.headers["authorization"] == f"Bearer {TOKEN}"
               and q.headers["user-agent"] == "harpor-worker" for q in r)
    assert (r[0].method, r[0].url.path, fake.body(0)) == ("POST", "/rest/v1/rpc/x", {"p_a": 1, "p_b": None})
    assert r[1].url.path == "/rest/v1/jobs" and params(r[1]) == {
        "select": "id,status", "id": f"eq.{JOB}", "deleted_at": "is.null", "approved": "is.true",
        "or": "(a.is.null,b.eq.1)", "order": "id", "limit": "5"}
    assert len(r) == 2
    assert not any(hasattr(supa.Supa, n) for n in ("insert", "upsert", "patch", "delete", "download"))


def test_supa_refuses_a_secret_key_and_a_call_without_a_session():
    with pytest.raises(ValueError, match="never the secret key"):
        supa.Supa(URL, SECRET, bearer)
    with pytest.raises(ValueError, match="needs the user's session"):
        supa.Supa(URL, PUB, None)
    with pytest.raises(ValueError, match="no Supabase key"):
        supa.Supa(URL, "", bearer)
    for gone in ("read_env", "Secrets", "SharedStore", "OwnStore", "Download", "ENV"):
        assert not hasattr(supa, gone), gone


def test_rpc_answers_void_scalars_and_lists():
    fake = Fake(ok(), ok(7), ok([{"id": 3}]), httpx.Response(200, content=b"null"))

    async def main():
        async with client(fake.transport) as s:
            return [await s.rpc("storage_used") for _ in range(4)]

    assert run(main) == [None, 7, [{"id": 3}], None]


def test_retries_and_what_is_never_sent_twice(naps):
    async def main(answers, fn, *args, **kw):
        fake = Fake(*answers)
        async with client(fake.transport) as s:
            try:
                return await getattr(s, fn)(*args, **kw), len(fake.seen)
            except supa.Error as e:
                return e, len(fake.seen)

    out, n = run(main, [ok(status=503), ok(status=502), ok([1])], "select", "jobs")
    assert (out, n, naps) == ([1], 3, [1.0, 2.0])
    out, n = run(main, [ok(status=503)], "select", "jobs")
    assert isinstance(out, supa.Error) and out.status == 503 and n == 4  # 4 tries in all
    out, n = run(main, [ok(status=503)], "rpc", "own_job_create")  # it may have run: not again
    assert out.status == 503 and n == 1
    out, n = run(main, [httpx.ConnectError("refused"), ok("j1")], "rpc", "own_job_create")  # never reached it
    assert (out, n) == ("j1", 2)
    out, n = run(main, [httpx.ReadTimeout("slow"), ok("j1")], "rpc", "own_job_create")
    assert isinstance(out, supa.Offline) and n == 1 and "ReadTimeout" in str(out) and out.status is None
    out, n = run(main, [httpx.ReadTimeout("slow"), ok()], "rpc", "own_publish", again=True)
    assert (out, n) == (None, 2)
    out, n = run(main, [httpx.ConnectError("refused")], "select", "jobs")
    assert isinstance(out, supa.Offline) and n == 4 and TOKEN not in str(out)
    naps.clear()
    out, n = run(main, [ok(status=429, headers={"Retry-After": "7"}), ok(status=503, headers={"Retry-After": "900"}),
                        ok()], "rpc", "own_worker_beat", again=True)
    assert (out, n, naps) == (None, 3, [7.0, 30])


def test_errors_carry_supabases_message_and_no_secret():
    long = "x" * 400

    async def main():
        fake = Fake(ok(P0001, 400), ok({"message": f"bad key {PUB} for {TOKEN}"}, 403),
                    httpx.Response(502, text=f"<html>gateway {TOKEN}</html>"), ok({"message": long}, 400))
        async with client(fake.transport) as s:
            out = []
            for call in (s.rpc("own_publish", p_job=JOB), s.select("jobs", id=JOB), s.rpc("storage_used"),
                         s.rpc("own_logs")):
                with pytest.raises(supa.Error) as e:
                    await call
                out.append(e.value)
            return out

    a, b, c, d = run(main)
    assert (a.status, a.code, a.message) == (400, "P0001", "no such job on this machine")
    assert str(a) == "POST /rest/v1/rpc/own_publish: 400 no such job on this machine"
    assert str(b) == "GET /rest/v1/jobs: 403 bad key [key] for [key]" and JOB not in str(b)  # no query in the text
    assert str(c) == "POST /rest/v1/rpc/storage_used: 502 Bad Gateway" and c.message == ""  # HTML is not quoted
    assert len(d.message) == 300
    assert all(TOKEN not in str(e) + repr(e) and PUB not in str(e) + repr(e) for e in (a, b, c, d))


def test_timeouts():
    s = client(None)
    assert s.http.timeout.connect == 10 and s.http.timeout.read >= 30  # a large upload takes a while
    assert supa.AUTH_TIMEOUT.read + supa.AUTH_TIMEOUT.connect < 2 * supa.REFRESH_WINDOW
    assert supa.AUTH_TIMEOUT.read < supa.REFRESH_WINDOW
    with pytest.raises(ValueError, match="not a Supabase project URL"):
        supa.Supa("http://abc.supabase.co", PUB, bearer)


def test_a_bearer_and_one_renewal_after_a_401(naps):
    class Session:
        def __init__(self):
            self.token, self.renewals = TOKEN, 0

        async def bearer(self):
            return self.token

        async def refresh(self):
            self.renewals += 1
            self.token = TOKEN2

    async def main(*answers, renew=True):
        fake, sess = Fake(*answers), Session()
        s = supa.Supa(URL, PUB, bearer=sess.bearer if renew else bearer, transport=fake.transport)
        try:
            return await s.select("jobs"), fake, sess
        except supa.Error as e:
            return e, fake, sess
        finally:
            await s.close()

    out, fake, sess = run(main, ok({"message": "JWT expired"}, 401), ok([{"id": 1}]))
    assert out == [{"id": 1}] and sess.renewals == 1
    assert [q.headers["authorization"] for q in fake.seen] == [f"Bearer {TOKEN}", f"Bearer {TOKEN2}"]
    assert all(q.headers["apikey"] == PUB for q in fake.seen)
    out, fake, sess = run(main, ok({"message": "JWT expired"}, 401))
    assert out.status == 401 and sess.renewals == 1 and len(fake.seen) == 2  # once only
    assert TOKEN not in str(out) and TOKEN2 not in str(out)
    storage = ok({"statusCode": "403", "error": "Unauthorized", "message": "jwt expired"}, 400)
    out, fake, sess = run(main, storage, ok([]))
    assert out == [] and sess.renewals == 1
    out, fake, sess = run(main, ok({"message": "Invalid API key"}, 401), renew=False)
    assert out.status == 401 and len(fake.seen) == 1  # a bearer without a refresh: nothing to renew

    async def signed_out():
        async def refuse():
            raise supa.SignedOut("signed out")

        s = supa.Supa(URL, PUB, bearer=bearer, refresh=refuse, transport=Fake(ok({}, 401)).transport)
        with pytest.raises(supa.SignedOut):
            await s.rpc("own_worker_beat")

    run(signed_out)


# ---- Storage ----

def test_upload_headers_tries_and_a_duplicate_counts_as_done(naps):
    obj = supa.object_name(USER, JOB, "ab" * 32, "frames/thumbs/0001.jpg")
    dup = ok({"statusCode": "409", "error": "Duplicate", "message": "The resource already exists"}, 400)
    policy = ok({"statusCode": "403", "error": "Unauthorized",
                 "message": "new row violates row-level security policy"}, 400)

    async def main(*answers):
        fake = Fake(*answers)
        async with client(fake.transport) as s:
            try:
                await s.upload("job-files", obj, b"jpeg bytes", "image/jpeg")
                return None, fake
            except supa.Error as e:
                return e, fake

    out, fake = run(main, ok(status=503), httpx.ReadTimeout("slow"), ok({"Key": "x"}))
    q = fake.seen[0]
    assert out is None and len(fake.seen) == 3
    assert q.method == "POST"
    assert q.url.path == f"/storage/v1/object/job-files/{USER}/{JOB}/{'ab' * 8}/frames/thumbs/0001.jpg"
    assert (q.headers["content-type"], q.headers["x-upsert"], q.headers["cache-control"], q.content) == (
        "image/jpeg", "false", supa.IMMUTABLE, b"jpeg bytes")
    assert run(main, httpx.ReadTimeout("slow"), dup)[0] is None  # the lost try had put it there
    assert run(main, ok({"message": "exists"}, 409))[0] is None
    out, fake = run(main, policy)
    assert out.status == 400 and "row-level security" in out.message and len(fake.seen) == 1

    async def bad():
        s = client(Fake(ok()).transport)
        for name in ("../x", "a//b", "/abs", "a b", "a/../b", "", None):
            with pytest.raises(ValueError, match="not a Storage object name"):
                await s.upload("job-files", name, b"", "text/plain")
        with pytest.raises(ValueError, match="not a table, function or bucket"):
            await s.upload("Job Files", "a", b"", "text/plain")

    run(bad)


def test_remove_deletes_1000_names_a_call():
    names = [f"{USER}/{JOB}/{i:016x}/run/posed.glb" for i in range(2500)]
    fake = Fake(lambda r: ok([{"name": n} for n in json.loads(r.content)["prefixes"]]))

    async def main():
        async with client(fake.transport) as s:
            return await s.remove("job-files", names)

    assert run(main) == 2500
    assert [(q.method, q.url.path, len(json.loads(q.content)["prefixes"])) for q in fake.seen] == [
        ("DELETE", "/storage/v1/object/job-files", 1000)] * 2 + [("DELETE", "/storage/v1/object/job-files", 500)]
    assert json.loads(fake.seen[2].content)["prefixes"] == names[2000:]


# ---- sign-in ----

def test_pkce_pair_and_the_authorize_url():
    v, c = supa.pkce_pair()
    assert len(v) == 86 and re.fullmatch(r"[A-Za-z0-9_-]+", v) and re.fullmatch(r"[A-Za-z0-9_-]{43}", c)
    assert c == base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode()
    assert supa.pkce_pair()[0] != v
    u = urllib.parse.urlsplit(supa.authorize_url(URL + "/", "github", "http://127.0.0.1:8767/auth/callback", c))
    assert (u.scheme, u.netloc, u.path) == ("https", "abcdefghijklmnop.supabase.co", "/auth/v1/authorize")
    assert dict(urllib.parse.parse_qsl(u.query)) == {"provider": "github", "code_challenge": c,
                                                     "redirect_to": "http://127.0.0.1:8767/auth/callback",
                                                     "code_challenge_method": "s256"}
    assert "redirect_to=http%3A%2F%2F127.0.0.1%3A8767%2Fauth%2Fcallback" in u.query
    with pytest.raises(ValueError, match="github or google"):
        supa.authorize_url(URL, "gitlab", "http://127.0.0.1:8767/auth/callback", c)


def browse(url, out, delay=0.2):
    """A browser in a thread: GET url, keep (status, text, content type)."""
    def go():
        time.sleep(delay)
        try:
            r = urllib.request.urlopen(url, timeout=10)
        except urllib.error.HTTPError as e:
            r = e
        with r:
            out.append((r.status, r.read().decode(), r.headers["Content-Type"], r.headers["X-Content-Type-Options"]))
    t = threading.Thread(target=go)
    t.start()
    return t


def test_loopback_takes_the_code_and_prints_nothing(capfd):
    with supa.Loopback(port=0, timeout=20) as lb:
        assert re.fullmatch(r"http://127\.0\.0\.1:\d+/auth/callback", lb.redirect)
        seen = []
        ts = [browse(f"http://127.0.0.1:{lb.port}/favicon.ico", seen, 0.1), browse(lb.redirect, seen, 0.3),
              browse(lb.redirect + "?code=code-1234&state=x", seen, 0.6)]
        assert lb.wait() == "code-1234"
        for t in ts:
            t.join()
        late = []
        t = browse(lb.redirect + "?code=other", late, 0)
        lb.server.timeout = 5
        lb.server.handle_request()
        t.join()
    assert [s[0] for s in seen] == [404, 400, 200]
    assert seen[2][1] == "Signed in. You can close this tab.\n" and "<" not in seen[2][1]
    assert all(s[2] == "text/plain; charset=utf-8" and s[3] == "nosniff" for s in seen)
    assert late and late[0][0] == 409
    out, err = capfd.readouterr()
    assert "code-1234" not in out + err and "GET" not in out + err


def test_loopback_reports_an_error_and_gives_up_in_time():
    with supa.Loopback(port=0, timeout=20) as lb:
        seen = []
        t = browse(lb.redirect + "?error=access_denied&error_description=The+user+said+no", seen)
        with pytest.raises(supa.Error, match="the sign-in failed: The user said no"):
            lb.wait()
        t.join()
    assert seen[0][:2] == (400, "The sign-in failed: The user said no\n")
    t0 = time.monotonic()
    with supa.Loopback(port=0, timeout=0.4) as lb:
        with pytest.raises(supa.Error, match="did not come back within 0.4 s"):
            lb.wait()
    assert time.monotonic() - t0 < 3
    with supa.Loopback(port=0) as lb:
        with pytest.raises(supa.Error, match=f"cannot listen on 127.0.0.1:{lb.port}"):
            supa.Loopback(port=lb.port)


SESSION = {"access_token": TOKEN, "refresh_token": REFRESH2, "expires_in": 3600, "expires_at": 1,
           "token_type": "bearer",
           "user": {"id": USER, "email": "a@example.org", "app_metadata": {"provider": "github"}}}


def test_auth_exchange_refresh_and_logout(naps):
    async def main(fn, *args, answers):
        fake = Fake(*answers)
        try:
            return await getattr(supa.Auth(URL, PUB, transport=fake.transport), fn)(*args), fake
        except supa.Error as e:
            return e, fake

    g, fake = run(main, "exchange", "code-1", "verifier-1", answers=[ok(SESSION)])
    q = fake.seen[0]
    assert (q.url.path, params(q), fake.body(0)) == ("/auth/v1/token", {"grant_type": "pkce"},
                                                     {"auth_code": "code-1", "code_verifier": "verifier-1"})
    assert q.headers["apikey"] == PUB and "authorization" not in q.headers
    assert (g.access_token, g.refresh_token, g.expires_in, g.user_id, g.email, g.provider) == (
        TOKEN, REFRESH2, 3600.0, USER, "a@example.org", "github")
    assert TOKEN not in repr(g) and REFRESH2 not in repr(g)
    e, fake = run(main, "exchange", "code-1", "verifier-1",
                  answers=[ok({"code": 400, "error_code": "flow_state_not_found", "msg": "invalid flow state"}, 400)])
    assert str(e) == "POST /auth/v1/token: 400 invalid flow state" and not isinstance(e, supa.SignedOut)
    e, fake = run(main, "exchange", "code-1", "verifier-1", answers=[httpx.ReadTimeout("slow"), ok(SESSION)])
    assert isinstance(e, supa.Offline) and len(fake.seen) == 1  # a code works once
    g, fake = run(main, "exchange", "code-1", "verifier-1", answers=[httpx.ConnectError("no"), ok(SESSION)])
    assert g.user_id == USER and len(fake.seen) == 2
    e, fake = run(main, "exchange", "c", "v", answers=[ok({**SESSION, "user": None})])
    assert "without a user" in str(e)

    g, fake = run(main, "refresh", REFRESH, answers=[ok(SESSION)])
    assert params(fake.seen[0]) == {"grant_type": "refresh_token"} and fake.body(0) == {"refresh_token": REFRESH}
    assert g.refresh_token == REFRESH2
    used = ok({"code": 400, "error_code": "refresh_token_already_used",
               "msg": "Invalid Refresh Token: Already Used"}, 400)
    e, fake = run(main, "refresh", REFRESH, answers=[used])
    assert isinstance(e, supa.SignedOut) and "Already Used" in str(e) and "worker login" in str(e)
    assert REFRESH not in str(e) and e.status == 400
    g, fake = run(main, "refresh", REFRESH, answers=[httpx.ReadTimeout("slow"), ok(SESSION)])
    assert g.refresh_token == REFRESH2 and len(fake.seen) == 2  # inside the reuse window
    g, fake = run(main, "refresh", REFRESH, answers=[ok(status=503), ok(SESSION)])
    assert g.refresh_token == REFRESH2 and len(fake.seen) == 2

    out, fake = run(main, "logout", TOKEN, answers=[ok()])
    assert out is None and fake.seen[0].url.path == "/auth/v1/logout" and params(fake.seen[0]) == {"scope": "local"}
    assert fake.seen[0].headers["authorization"] == f"Bearer {TOKEN}"
    assert run(main, "logout", TOKEN, answers=[ok({"msg": "session not found"}, 403)])[0] is None
    e, fake = run(main, "logout", TOKEN, answers=[ok({"msg": "down"}, 500)])
    assert e.status == 500 and TOKEN not in str(e)
    with pytest.raises(ValueError, match="never the secret key"):
        supa.Auth(URL, SECRET)


def test_sign_in_runs_pkce_through_the_loopback():
    printed, opened = [], []

    def browser(page, answer="?code=code-77"):
        opened.append(page)  # the provider and Supabase send the browser back to the loopback
        browse(dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(page).query))["redirect_to"] + answer, [], 0.2)

    def token(request):
        body = json.loads(request.content)
        page = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(opened[-1]).query))
        made = base64.urlsafe_b64encode(hashlib.sha256(body["code_verifier"].encode()).digest()).rstrip(b"=").decode()
        ok_ = body["auth_code"] == "code-77" and made == page["code_challenge"] and params(request) == {
            "grant_type": "pkce"}
        return ok(SESSION) if ok_ else ok({"msg": "wrong code or verifier"}, 400)

    fake = Fake(token)
    g = run(supa.sign_in, URL, PUB, "google", 0, browser, printed.append, 10, fake.transport)
    assert g.user_id == USER and g.refresh_token == REFRESH2 and len(fake.seen) == 1
    page = urllib.parse.urlsplit(opened[0])
    q = dict(urllib.parse.parse_qsl(page.query))
    assert q["provider"] == "google" and re.fullmatch(r"http://127\.0\.0\.1:\d+/auth/callback", q["redirect_to"])
    assert printed == [f"Open this page to sign in with google:\n  {opened[0]}"]
    assert json.loads(fake.seen[0].content)["code_verifier"] not in printed[0]

    fake = Fake(token)
    with pytest.raises(supa.Error, match="the sign-in failed: access_denied"):
        run(supa.sign_in, URL, PUB, "github", 0, lambda p: browser(p, "?error=access_denied"), printed.append, 10,
            fake.transport)
    assert fake.seen == []  # no code, no token call
    with pytest.raises(ValueError, match="never the secret key"):
        run(supa.sign_in, URL, SECRET, port=0)


def test_a_refresh_is_not_tried_again_after_the_reuse_window(naps, monkeypatch):
    monkeypatch.setattr(supa, "REFRESH_WINDOW", 0)
    fake = Fake(httpx.ReadTimeout("slow"), ok(SESSION))

    async def main():
        await supa.Auth(URL, PUB, transport=fake.transport).refresh(REFRESH)

    with pytest.raises(supa.Offline):
        run(main)
    assert len(fake.seen) == 1


def tokens(*answers, save=None, delay=0.0):
    seen, saved = [], []

    async def handle(request):
        seen.append(json.loads(request.content)["refresh_token"])
        await anyio.sleep(delay)
        a = answers[min(len(seen) - 1, len(answers) - 1)]
        if isinstance(a, BaseException):
            raise a
        return a

    t = supa.Tokens(supa.Auth(URL, PUB, transport=httpx.MockTransport(handle)), REFRESH,
                    save or (lambda rt: saved.append((rt, t.access))))
    return t, seen, saved


def grant(n):
    return ok({**SESSION, "access_token": f"access-{n}" + "x" * 20, "refresh_token": f"refresh-{n}"})


def test_tokens_renew_before_expiry_and_save_first(naps):
    t, seen, saved = tokens(grant(1), grant(2), grant(3))

    async def main():
        a = await t.bearer()
        assert await t.bearer() == a  # an hour left: no renewal
        t.expires = time.monotonic() + 200  # under 5 minutes left
        b = await t.bearer()
        await t.refresh()  # Supa's call after a 401: always
        return a, b

    a, b = run(main)
    assert (a, b) == ("access-1" + "x" * 20, "access-2" + "x" * 20) and t.access == "access-3" + "x" * 20
    assert seen == [REFRESH, "refresh-1", "refresh-2"] and t.refresh_token == "refresh-3"
    # each new refresh token was saved before its access token was in use
    assert saved == [("refresh-1", None), ("refresh-2", a), ("refresh-3", b)]
    assert "access-" not in repr(t) and "refresh-" not in repr(t)


def test_tokens_renew_once_for_many_tasks():
    t, seen, saved = tokens(grant(1), grant(2), delay=0.05)

    async def main():
        out = []

        async def one():
            out.append(await t.bearer())

        async with anyio.create_task_group() as tg:
            for _ in range(5):
                tg.start_soon(one)
        return out

    assert run(main) == ["access-1" + "x" * 20] * 5 and seen == [REFRESH]


def test_tokens_when_supabase_refuses_is_away_or_the_file_cannot_be_written(naps):
    used = ok({"error_code": "refresh_token_already_used", "msg": "Invalid Refresh Token: Already Used"}, 400)
    t, seen, saved = tokens(used)
    with pytest.raises(supa.SignedOut):
        run(t.bearer)
    assert t.refresh_token == REFRESH and t.access is None and saved == []

    t, seen, saved = tokens(grant(1), httpx.ConnectError("no"))

    async def away(left):
        await t.bearer()
        t.expires = time.monotonic() + left
        return await t.bearer()

    assert run(away, 200) == "access-1" + "x" * 20  # under the margin but still good: it serves
    t, seen, saved = tokens(grant(1), httpx.ConnectError("no"))
    with pytest.raises(supa.Offline):
        run(away, 10)

    def full(rt):
        raise OSError("disk full")

    t, seen, _ = tokens(grant(1), grant(2), save=full)
    with pytest.raises(OSError):
        run(t.bearer)
    assert t.refresh_token == "refresh-1" and t.access is None  # the spent token is not used again
    t.save = lambda rt: None
    assert run(t.bearer) == "access-2" + "x" * 20 and seen == [REFRESH, "refresh-1"]


def test_supa_with_tokens_renews_after_a_401():
    t, seen, saved = tokens(grant(1), grant(2))
    fake = Fake(ok({"message": "JWT expired"}, 401), ok([]))

    async def main():
        async with supa.Supa(URL, PUB, bearer=t.bearer, transport=fake.transport) as s:
            assert s.refresh == t.refresh
            return await s.select("jobs")

    assert run(main) == [] and seen == [REFRESH, "refresh-1"]
    assert [q.headers["authorization"] for q in fake.seen] == ["Bearer access-1" + "x" * 20,
                                                               "Bearer access-2" + "x" * 20]


# ---- the Store ----

def store_fake(answers=None):
    """Answers by (method, path); 204 for the rest."""
    answers = answers or {}

    def handle(request):
        a = answers.get((request.method, request.url.path), ok())
        return a(request) if callable(a) else a

    return Fake(handle)


def calls(fake):
    return [(q.method, q.url.path.removeprefix("/rest/v1"), params(q), fake.body(i)) for i, q in enumerate(fake.seen)]


def test_store_interface():
    have = {n for n in vars(supa.Store) if callable(getattr(supa.Store, n)) and not n.startswith("_")}
    assert have == {"of", "close", "bind", "storage_used", "start", "beat", "stop", "claim", "take_key", "answer",
                    "settings", "access", "job_row", "job_rows", "create_job", "publish", "files", "manifest", "logs",
                    "upload", "remove", "job_objects", "orphans", "delete_row"}
    assert supa.Store.__bases__ == (object,)


class Session:
    url, publishable_key, user_id, machine_id = URL, PUB, USER, None

    async def bearer(self):
        return TOKEN

    async def refresh(self):
        pass


def test_store_calls():
    sealed = {"agent": "opus", "epk": "e", "iv": "i", "ct": "c"}
    stopped = ok({"code": "P0001", "message": "worker w-2 was stopped: another worker took over"}, 400)
    answered = {"n": 0}

    def own_answer(request):
        answered["n"] += 1
        return ok({"code": "P0001", "message": "the request is not taken by this worker"}, 400) \
            if answered["n"] == 2 else stopped if answered["n"] == 3 else ok()

    names = [{"name": f"{USER}/{JOB}/{'0' * 16}/a.jpg", "created_at": "2026-10-25T00:00:00+00:00"}]
    fake = store_fake({("POST", "/rest/v1/rpc/own_machine_bind"): ok(MACHINE),
                       ("POST", "/rest/v1/rpc/own_worker_start"): ok("w-2"),
                       ("POST", "/rest/v1/rpc/own_claim_request"): ok([{"id": 9, "key_ref": "r", "sealed": sealed}]),
                       ("POST", "/rest/v1/rpc/own_answer"): own_answer,
                       ("POST", "/rest/v1/rpc/own_job_create"): ok(JOB),
                       ("POST", "/rest/v1/rpc/storage_used"): ok(1234),
                       ("POST", "/rest/v1/rpc/job_objects"): ok(names),
                       ("POST", "/rest/v1/rpc/own_orphans"): ok([{"name": "o1"}, {"name": "o2"}]),
                       ("GET", "/rest/v1/settings"): ok([{"id": 1, "lean_files": True}]),
                       ("GET", "/rest/v1/access"): ok([]),
                       ("GET", "/rest/v1/jobs"): ok([{"id": JOB}]),
                       ("GET", "/rest/v1/job_files"): ok([{"path": "run/posed.glb", "object": "o", "sha256": "s"}]),
                       ("DELETE", "/storage/v1/object/job-files"): ok([{"name": "o1"}]),
                       ("POST", "/rest/v1/rpc/own_worker_beat"): stopped})
    put = [{"path": f"frames/thumbs/{i:04d}.jpg", "object": f"o{i}", "bytes": 1, "sha256": "s", "role": "review",
            "content_type": "image/jpeg", "extra": 1} for i in range(1200)]

    async def main():
        st = supa.Store.of(Session(), transport=fake.transport)
        assert (st.user_id, st.machine_id, st.supa.key) == (USER, None, PUB)
        with pytest.raises(ValueError, match="bind it first"):
            await st.start("pk", None, "v1")
        assert await st.bind("lab-box") == MACHINE == st.machine_id
        assert await st.start("pk", datetime(2026, 10, 25, 5, 35, tzinfo=timezone.utc), "v1", ("opus",)) == "w-2"
        req = await st.claim(None)
        assert await st.take_key(req) == sealed and "sealed" not in req and await st.take_key(req) is None
        await st.answer(9, 200, {"ok": 1})
        await st.answer(9, 200, {"ok": 1})  # a retry after a lost answer: already done
        with pytest.raises(supa.Error, match="was stopped") as e:
            await st.answer(9, 200, {})
        assert e.value.message.startswith("worker w-2 was stopped")
        assert (await st.settings())["lean_files"] is True and await st.access() is None
        assert await st.job_row(JOB) == {"id": JOB} and await st.job_rows() == [{"id": JOB}]
        assert await st.create_job("clip.mp4", "video", 123) == JOB
        await st.publish(JOB, view={"v": 1}, snapshots={"frames": {"status": 200, "body": {}}},
                         row={"status": "review", "stopped_at": datetime(2026, 1, 2, tzinfo=timezone.utc)})
        await st.publish(JOB)  # nothing to write
        with pytest.raises(ValueError, match="row takes"):
            await st.publish(JOB, row={"local_id": "20261024-120000-abcdef"})
        with pytest.raises(ValueError, match="snapshots"):
            await st.publish(JOB, snapshots={"thumbs": {}})
        await st.files(JOB, put, ["old.jpg"])
        await st.files(JOB, [], ["gone.jpg"])
        await st.files(JOB, [], [])
        assert await st.manifest(JOB) == {"run/posed.glb": {"object": "o", "sha256": "s"}}
        await st.logs(JOB, [{"stage": "frames", "pos": i, "text": "x", "extra": 1} for i in range(250)])
        assert await st.job_objects(JOB) == [names[0]["name"]]
        assert await st.orphans() == ["o1", "o2"] and await st.orphans(5) == ["o1", "o2"]
        assert await st.remove(["o1", "o2"]) == 1
        await st.delete_row(JOB)
        assert await st.storage_used() == 1234
        await st.stop()
        with pytest.raises(supa.Error, match="was stopped"):
            await st.beat(0, True, None)
        assert TOKEN not in repr(st)
        await st.close()

    run(main)
    c = calls(fake)
    assert all(q.headers["apikey"] == PUB and q.headers["authorization"] == f"Bearer {TOKEN}" for q in fake.seen)
    assert c[0] == ("POST", "/rpc/own_machine_bind", {}, {"p_machine": None, "p_name": "lab-box"})
    assert c[1] == ("POST", "/rpc/own_worker_start", {}, {"p_machine": MACHINE, "p_public_key": "pk",
                                                          "p_ends_at": "2026-10-25T05:35:00+00:00", "p_version": "v1",
                                                          "p_local_keys": ["opus"]})
    assert c[2] == ("POST", "/rpc/own_claim_request", {}, {"p_worker": "w-2", "p_kinds": None})
    assert c[3] == ("POST", "/rpc/own_answer", {}, {"p_worker": "w-2", "p_request": 9, "p_status": 200,
                                                    "p_response": {"ok": 1}})
    assert [x[1] for x in c[4:6]] == ["/rpc/own_answer"] * 2
    assert c[6] == ("GET", "/settings", {"select": "*", "id": "eq.1"}, None)
    assert c[7] == ("GET", "/access", {"select": "*", "user_id": f"eq.{USER}"}, None)
    cols = supa.JOB_COLUMNS
    assert c[8] == ("GET", "/jobs", {"select": cols, "id": f"eq.{JOB}"}, None)
    assert c[9] == ("GET", "/jobs", {"select": cols, "machine_id": f"eq.{MACHINE}"}, None)
    assert c[10] == ("POST", "/rpc/own_job_create", {}, {"p_worker": "w-2", "p_name": "clip.mp4", "p_kind": "video",
                                                         "p_bytes": 123})
    assert c[11] == ("POST", "/rpc/own_publish", {}, {
        "p_worker": "w-2", "p_job": JOB, "p_view": {"v": 1}, "p_live": None,
        "p_snapshots": {"frames": {"status": 200, "body": {}}},
        "p_row": {"status": "review", "stopped_at": "2026-01-02T00:00:00+00:00"}})
    files = c[12:16]
    assert [len(x[3]["p_put"]) for x in files] == [500, 500, 200, 0]
    assert [x[3]["p_drop"] for x in files] == [["old.jpg"], [], [], ["gone.jpg"]]
    assert files[0][3]["p_put"][0] == {k: v for k, v in put[0].items() if k != "extra"}
    assert c[16] == ("GET", "/job_files", {"select": "path,object,sha256", "job_id": f"eq.{JOB}"}, None)
    logs = c[17:20]
    assert [len(x[3]["p_rows"]) for x in logs] == [100, 100, 50] and logs[0][1] == "/rpc/own_logs"
    assert logs[0][3]["p_rows"][0] == {"stage": "frames", "pos": 0, "text": "x", "run_started": None}
    assert c[20] == ("POST", "/rpc/job_objects", {}, {"p_job": JOB})
    assert c[21] == ("POST", "/rpc/own_orphans", {}, {"p_worker": "w-2", "p_limit": 1000})
    assert c[22][3] == {"p_worker": "w-2", "p_limit": 5}
    assert c[23] == ("DELETE", "/storage/v1/object/job-files", {}, {"prefixes": ["o1", "o2"]})
    assert c[24] == ("POST", "/rpc/own_job_delete", {}, {"p_worker": "w-2", "p_job": JOB})
    assert [x[1] for x in c[25:]] == ["/rpc/storage_used", "/rpc/own_worker_stop", "/rpc/own_worker_beat"]


def test_store_renames_a_bound_machine_and_tries_again_safely(naps):
    fake = store_fake({("POST", "/rest/v1/rpc/own_machine_bind"): ok(MACHINE)})
    sess = Session()
    sess.machine_id = MACHINE

    async def main():
        st = supa.Store.of(sess, transport=fake.transport)
        return await st.bind("new-name")

    assert run(main) == MACHINE and calls(fake)[0][3] == {"p_machine": MACHINE, "p_name": "new-name"}
    flaky = Fake(httpx.ReadTimeout("slow"), ok(MACHINE))

    async def new():
        return await supa.Store.of(Session(), transport=flaky.transport).bind("lab-box")

    with pytest.raises(supa.Offline):  # a second try could make a second machine
        run(new)
    assert len(flaky.seen) == 1


def test_no_secret_in_logs_or_errors(caplog, naps):
    caplog.set_level(logging.DEBUG)
    assert logging.getLogger("httpx").level == logging.WARNING == logging.getLogger("httpcore").level

    async def renew():
        pass

    async def main():
        errors = []
        echo = f"echo {PUB} {TOKEN}"  # what the client holds
        fake = Fake(ok(status=503), httpx.ConnectError("refused"), httpx.ReadTimeout("slow"),
                    ok({"message": echo}, 401), ok({"message": echo}, 400))
        async with supa.Supa(URL, PUB, bearer=bearer, refresh=renew, transport=fake.transport) as s:
            for _ in range(3):
                try:
                    await s.select("jobs", id=JOB)
                except supa.Error as e:
                    errors.append(e)
        return errors

    errors = run(main)
    assert errors and "trying again" in caplog.text
    text = caplog.text + " ".join(str(e) + repr(e) for e in errors)
    assert TOKEN not in text and PUB not in text
