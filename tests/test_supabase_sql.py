"""The Supabase migrations (supabase/migrations) on a throwaway local Postgres.

tests/pgcheck/run.py makes the cluster under pytest's tmp folder, on 127.0.0.1 at a free port with no
Unix socket, and removes it at the end. tests/pgcheck/supabase_stub.sql stands in for Supabase's auth,
storage and Realtime. It is not Supabase: here no PostgREST, Storage server or Realtime runs, and the
superuser plays the role that owns the functions. The tests that need Postgres skip when Postgres 15
or newer (pg_ctl, initdb, psql) is not installed; HARPOR_PG_BIN names its folder.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPABASE = ROOT / "supabase"
_spec = importlib.util.spec_from_file_location("pgcheck_run", ROOT / "tests" / "pgcheck" / "run.py")
pgcheck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pgcheck)
cases_mod = pgcheck.supabase_cases
CASES = cases_mod.load()


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    bindir = pgcheck.find_bin()
    if bindir is None:
        pytest.skip("no Postgres 15 or newer (pg_ctl, initdb, psql); set HARPOR_PG_BIN to its folder")
    c = pgcheck.Cluster(tmp_path_factory.mktemp("pg"), bindir).start()
    yield c
    c.stop()


@pytest.fixture(scope="module")
def cases_db(cluster):
    cluster.project("cases")
    pgcheck.cases_setup(cluster, "cases")
    return "cases"


@pytest.mark.parametrize("old_grants", [True, False], ids=["old-project", "new-project"])
def test_checks(cluster, old_grants):
    """Every check file passes, on a project with Supabase's old default grants and on a new project
    without them, and every check in each file ran."""
    db = cluster.project("old_project" if old_grants else "new_project", old_grants=old_grants)
    try:
        results = pgcheck.run_checks(cluster, db)
    except pgcheck.PsqlError as e:
        pytest.fail(str(e)[-6000:])
    for name, text, expected in results:
        assert expected >= 30, name
        assert pgcheck.passed(text) == expected, (name, text[-3000:])
        assert re.search(r"^ALL .*CHECKS PASSED$", text, re.M), name


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_shared_case(cluster, cases_db, case):
    """One case of tests/supabase_cases.json, which tests/fake_supabase.py must answer the same way."""
    ok, text = pgcheck.run_case(cluster, cases_db, case)
    assert ok, f"{cases_mod.case_sql(case)}\n{text}"


def test_a_restart_waits_for_a_relay_that_seals_a_key(cluster):
    """own_worker_start takes the user's lock that relay() holds. A restart of the machine while relay() seals a key
    to the old worker waits for it, then drops that key and answers the request with 400, so that no request waits
    with a key that no worker can open."""
    import subprocess
    import time

    db = cluster.project("restart_race")
    as_c = ("reset role; set request.jwt.claims to '{\"sub\": \"cccccccc-cccc-cccc-cccc-cccccccccccc\", "
            "\"role\": \"authenticated\"}'; set role authenticated;\n")
    cluster.psql(db, """
      insert into tests.ids (name, id) values ('C', 'cccccccc-cccc-cccc-cccc-cccccccccccc');
      insert into auth.users (id, email) values (tests.id('C'), 'c@example.org');
      update public.access set approved = true;
    """ + as_c + """
      insert into tests.ids (name, id) values ('mc', public.own_machine_bind(null, 'lab-box'));
      insert into tests.ids (name, id)
        values ('wc', public.own_worker_start(tests.id('mc'), repeat('C', 88), null, 't'));
      insert into tests.ids (name, id) values ('jc', public.own_job_create(tests.id('wc'), 'mug.mp4', 'video', 10));
      select public.own_publish(tests.id('wc'), tests.id('jc'), p_row => '{"status": "ready"}');
    """)
    relay = subprocess.Popen(cluster._cmd(db) + ["-f", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=pgcheck._env())
    relay.stdin.write(as_c + """
      begin;
      insert into tests.ids (name, n) values ('r', (public.relay(tests.id('jc'), 'continue', '{}', '{}', null,
        tests.sealed(tests.id('wc'), 'opus', gen_random_uuid()::text)) ->> 'id')::bigint);
      select pg_sleep(1.5);
      commit;
    """)
    relay.stdin.close()
    t0 = time.monotonic()
    while cluster.value(db, "select count(*) from pg_locks where locktype = 'advisory' and granted") == "0":
        assert relay.poll() is None and time.monotonic() - t0 < 30, relay.stdout.read()
        time.sleep(0.05)
    cluster.psql(db, as_c + "select public.own_worker_start(tests.id('mc'), repeat('D', 88), null, 't');")
    out = relay.stdout.read()
    assert relay.wait(timeout=60) == 0, out
    assert cluster.value(db, "select state || ' ' || status || ' ' || (response ->> 'detail') from public.requests "
                             "where id = tests.n('r')") == \
        "done 400 your machine restarted after you sent the key; paste the key again"
    assert cluster.value(db, "select count(*) from public.job_secrets") == "0"


def _call(src, start):
    """The text of the call whose "(" is at or after start, up to its matching ")"."""
    i = src.index("(", start)
    depth = 0
    for j in range(i, len(src)):
        depth += {"(": 1, ")": -1}.get(src[j], 0)
        if depth == 0:
            return src[i:j + 1]
    raise ValueError(f"no end of the call at {start}")


def test_the_page_the_worker_and_the_fake_name_what_the_migrations_have(cluster):
    """The tables, columns, functions and arguments that supa.js and account.js (the page), the worker (supa.py,
    worker.py, session.py) and fake_supabase.py name are the ones the migrations make, read from the catalog. So a
    renamed column or function fails here, and not first on the live site."""
    import fake_supabase as fs

    from harpor.webapp import supa

    db = cluster.project("names")
    cols = {}
    for tc in cluster.value(db, "select string_agg(table_name || '.' || column_name, ' ') "
                                "from information_schema.columns where table_schema = 'public'").split():
        t, c = tc.split(".")
        cols.setdefault(t, set()).add(c)
    fns, optional = {}, {}
    for sig in cluster.value(db, "select string_agg(p.proname || '(' || pg_get_function_arguments(p.oid) || ')', ';') "
                                 "from pg_proc p join pg_namespace n on n.oid = p.pronamespace where n.nspname = "
                                 "'public' and has_function_privilege('authenticated', p.oid, 'execute')").split(";"):
        name, args = re.fullmatch(r"(\w+)\((.*)\)", sig).groups()
        args = [a for a in args.split(", ") if a]
        fns[name] = {a.split()[0] for a in args}
        optional[name] = {a.split()[0] for a in args if " DEFAULT " in a}

    assert {t: set(s["cols"]) for t, s in fs.SCHEMA.items()} == cols
    assert {f: set(args) for f, (_, args) in fs.FUNCTIONS.items()} == fns
    assert {f: {a for a, d in args.items() if d is not fs.REQUIRED} for f, (_, args) in fs.FUNCTIONS.items()} \
        == optional
    assert all(role == "authenticated" for role, _ in fs.FUNCTIONS.values())

    def check_rpcs(src, expect):
        seen = set()
        for m in re.finditer(r'\brpc\("(\w+)"', src):
            seen.add(m[1])
            assert m[1] in fns, m[1]
            args = set(re.findall(r"\b(p_\w+)\s*[:=]", _call(src, m.start())))
            assert args <= fns[m[1]], (m[1], args - fns[m[1]])
        assert seen == expect

    def check_columns(table, names, where):
        bad = {c for c in names if c != "*"} - cols[table]
        assert not bad, (where, table, bad)

    js = "".join((ROOT / "harpor" / "webapp" / "static" / f).read_text() for f in ("supa.js", "account.js"))
    check_rpcs(js, {"relay", "withdraw_request", "delete_job", "request_access", "storage_used", "job_objects"})
    consts = dict(re.findall(r'const (\w+_COLS) = "([^"]+)"', js))
    reads = 0
    for m in re.finditer(r'\bsb\.from\("(\w+)"\)', js):
        chain = js[m.end():].split("sb.from(", 1)[0].split(";", 1)[0]
        sel = re.match(r'\.select\(("([^"]*)"|(\w+))\)', chain)
        if sel:
            reads += 1
            check_columns(m[1], (sel[2] if sel[2] is not None else consts[sel[3]]).split(","), "the page")
        check_columns(m[1], re.findall(r'\.(?:eq|neq|in|gt|gte|lt|lte|is|order)\("(\w+)"', chain), "the page")
    assert reads >= 15
    for m in re.finditer(r'table: "(\w+)", filter: `(\w+)=', js):
        check_columns(m[1], [m[2]], "the page's Realtime")

    py = "".join((ROOT / "harpor" / "webapp" / f).read_text() for f in ("supa.py", "worker.py", "session.py"))
    check_rpcs(py, {"own_machine_bind", "own_worker_start", "own_worker_beat", "own_worker_stop", "own_claim_request",
                    "own_answer", "own_job_create", "own_publish", "own_files", "own_logs", "own_job_delete",
                    "own_orphans", "job_objects", "storage_used"})
    check_columns("jobs", supa.JOB_COLUMNS.split(","), "supa.JOB_COLUMNS")
    for m in re.finditer(r'\.select\("(\w+)"', py):
        call = _call(py, m.start() + len(".select"))
        named = re.match(r'\("\w+",\s*"([^"]+)"', call)
        check_columns(m[1], named[1].split(",") if named else [], "the worker")
        check_columns(m[1], set(re.findall(r"\b(\w+)=", call)) - {"filters", "order", "limit"}, "the worker")
        check_columns(m[1], re.findall(r'"(\w+)":\s*f?"(?:in|gt|eq)\.', call), "the worker")


def test_seed_makes_the_local_accounts(cluster):
    """seed.sql applies after the migrations and makes three accounts: two approved ones, to see that users
    stay apart, and one that waits for approval. It holds no password."""
    db = cluster.project("seed")
    cluster.psql(db, file=SUPABASE / "seed.sql")
    rows = cluster.value(db, "select string_agg(email || ':' || approved, ',' order by email) from public.access")
    assert rows == "a@example.org:true,b@example.org:false,c@example.org:true"
    assert cluster.value(db, "select count(*) from auth.identities where provider = 'email'") == "3"
    assert cluster.value(db, "select count(*) from auth.users where encrypted_password <> ''") == "0"


def test_no_postgres_means_a_skip(monkeypatch, tmp_path):
    """Without pg_ctl, initdb and psql of Postgres 15 or newer, find_bin gives None and the tests above
    skip; an old Postgres does not count."""
    monkeypatch.setattr(pgcheck, "BIN_DIRS", ())
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("HARPOR_PG_BIN", raising=False)
    assert pgcheck.find_bin() is None
    for tool in pgcheck.TOOLS:
        (tmp_path / tool).write_text("#!/bin/sh\necho 'pg_ctl (PostgreSQL) 14.9'\n")
        (tmp_path / tool).chmod(0o755)
    monkeypatch.setenv("HARPOR_PG_BIN", str(tmp_path))
    assert pgcheck.find_bin() is None


def test_migration_files():
    """Seven migrations in order; every function has a fixed search_path; nothing secret is in supabase/."""
    names = sorted(p.name for p in (SUPABASE / "migrations").glob("*.sql"))
    assert [n.split("_", 1)[1] for n in names] == [
        "harpor_tables.sql", "harpor_user_functions.sql", "harpor_worker_functions.sql", "harpor_storage.sql",
        "harpor_realtime.sql", "harpor_cron.sql", "harpor_relay_lidar_topic.sql"]
    assert all(n.startswith("20261101000") for n in names)
    assert [p.name for p in pgcheck.migrations()] == [n for n in names if not n.endswith("_cron.sql")]  # no pg_cron
    for p in (SUPABASE / "migrations").glob("*.sql"):
        text = p.read_text()
        heads = re.findall(r"create (?:or replace )?function .*?\bas \$\$", text, re.S)
        assert all("set search_path = ''" in h for h in heads), p.name
        text.encode("ascii")
    for p in SUPABASE.rglob("*"):
        if p.is_file() and p.suffix in (".sql", ".toml", ".json", ".md"):
            text = p.read_text()
            assert "sb_secret_" not in text and "sb_publishable_" not in text, p
            assert not re.search(r"eyJ[A-Za-z0-9_-]{10,}\.", text), p        # no JWT
            assert not re.search(r"sk-[A-Za-z0-9_-]{16,}", text), p         # no provider key


def test_job_files_takes_the_types_the_workers_send():
    """allowed_mime_types of job-files: every content type of relay.SITE_TYPES (what the workers upload), and the
    same list in the fake. No XML, HTML or SVG: a browser would run their scripts from the project's domain."""
    from harpor.webapp import relay

    import fake_supabase as fs

    text = (SUPABASE / "migrations" / "20261101000400_harpor_storage.sql").read_text()
    row = re.search(r"\('job-files', 'job-files', false, 52428800,\s+array\[(.*?)\]\)", text, re.S)
    types = set(re.findall(r"'([^']+)'", row[1]))
    assert set(relay.SITE_TYPES.values()) <= types and types - set(relay.SITE_TYPES.values()) == {"text/plain"}
    assert not [t for t in types if re.search(r"xml|html|svg|javascript", t)]
    assert set(fs.BUCKETS["job-files"]["mime"]) == types



def relay_function(name):
    """public.relay as a migration makes it, from its create line to its end $$."""
    text = (SUPABASE / "migrations" / name).read_text()
    return re.search(r"create (?:or replace )?function public\.relay\(.*?\nend \$\$;", text, re.S)[0]


def test_the_lidar_topic_migration_changes_only_the_query_check():
    """Part 7 restates part 2's relay() whole: the same function apart from its create line and the query check,
    which also takes lidar_topic, as text, for a calibration upload only. Its grants stay those of part 2."""
    old, new = relay_function("20261101000200_harpor_user_functions.sql"), relay_function(
        "20261101000700_harpor_relay_lidar_topic.sql")
    query = re.compile(r"\n(?:  --[^\n]*\n)*  if jsonb_typeof\(p_query\) <> 'object'.*?end if;", re.S)
    norm = lambda f: re.sub(r"\s+", " ", query.sub("", f.replace("create or replace function", "create function")))
    assert norm(old) == norm(new)
    assert "(p_query - 'name') <> '{}'::jsonb" in query.search(old)[0]
    assert "(p_query - '{name,lidar_topic}'::text[]) <> '{}'::jsonb" in query.search(new)[0]
    assert "jsonb_typeof(p_query -> 'lidar_topic') <> 'string'" in query.search(new)[0]
    text = (SUPABASE / "migrations" / "20261101000700_harpor_relay_lidar_topic.sql").read_text()
    assert "grant execute on function public.relay(uuid, text, jsonb, jsonb, text, jsonb) to authenticated;" in text

def test_cron_migration():
    """The pg_cron file is required: it runs the sweep every 15 minutes, since no worker sweeps, and keeps
    cron's run log short."""
    text = (SUPABASE / "migrations" / "20261101000600_harpor_cron.sql").read_text()
    assert "Required" in text
    assert "create extension if not exists pg_cron" in text
    assert "cron.schedule('harpor-sweep', '*/15 * * * *', $$select private.sweep()$$)" in text
    assert re.search(r"cron\.schedule\('harpor-cron-cleanup', '0 4 \* \* \*',\s+"
                     r"\$\$delete from cron\.job_run_details", text)


def test_local_config():
    """config.toml is for the local stack: email sign-in for tests, no OAuth secrets, the loopback page of
    `worker login` in the redirect list, and the seed file in place."""
    tomllib = pytest.importorskip("tomllib")                          # Python 3.11 and newer
    cfg = tomllib.loads((SUPABASE / "config.toml").read_text())
    assert cfg["auth"]["enable_anonymous_sign_ins"] is False
    assert cfg["auth"]["email"]["enable_signup"] is True
    assert cfg["auth"]["enable_refresh_token_rotation"] is True
    assert "http://127.0.0.1:8767/auth/callback" in cfg["auth"]["additional_redirect_urls"]
    for provider in ("github", "google"):
        assert cfg["auth"]["external"][provider] == {"enabled": False}
    assert [(SUPABASE / p).resolve() for p in cfg["db"]["seed"]["sql_paths"]] == [(SUPABASE / "seed.sql").resolve()]
    assert "config push" in (SUPABASE / "config.toml").read_text()


def test_case_table():
    """Every case has a unique id and compiles to SQL; a key the generator does not know is refused, so a
    typo in the table fails here instead of passing unseen."""
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    assert len(CASES) >= 55
    fns = {c["fn"] for c in CASES}
    assert fns == set(cases_mod.ARGS)
    for c in CASES:
        sql = cases_mod.case_sql(c)
        assert sql.startswith(f"-- case {c['id']}\nbegin;") and sql.rstrip().endswith("rollback;")
        assert "tests.check(" in sql or "tests.expect_error(" in sql
    good = {"id": "x", "fn": "relay", "as": "c", "state": {"job": {"status": "ready"}}, "args": {"route": "cancel"},
            "expect": {"ok": {}}}
    cases_mod.case_sql(good)
    for bad in ({**good, "expext": {}}, {**good, "state": {"jbo": {}}}, {**good, "args": {"rout": "cancel"}},
                {**good, "state": {"job": {"compute": "own"}}}, {**good, "state": {"worker": None}},
                {**good, "state": {"rate": {"request": 1}}}, {**good, "state": {"job": {"machine": "ms"}}},
                {**good, "state": {"objects": [{"bucket": "job-files", "name": "x", "size": 1, "owner": "c"}]}},
                {**good, "fn": "claim_request"}, {**good, "as": "a"}, {**good, "as": "worker"}, {**good, "as": "e"}):
        with pytest.raises(ValueError):
            cases_mod.case_sql(bad)


def test_case_table_format_is_documented():
    """The table says what each field means, for tests/fake_supabase.py as much as for the SQL."""
    doc = json.loads((ROOT / "tests" / "supabase_cases.json").read_text())
    text = " ".join(doc["_format"])
    for word in ("own_worker", "sealed", "objects", "rate", "storage_insert", "storage_delete", "withdrawn", "<job>"):
        assert word in text, word
