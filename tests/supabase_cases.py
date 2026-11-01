"""The shared case table, tests/supabase_cases.json, as SQL for the throwaway Postgres of tests/pgcheck.

tests/fake_supabase.py reads the same table, so the SQL and the fake give the same answers. The
table's "_format" says what each field means. base_sql() makes the users, machines and workers that
every case starts from, in a database that has the migrations and tests/pgcheck/helpers.sql.
case_sql(case) makes one transaction: it sets up the case's rows as the superuser, makes the call as
the case's role, checks the answer and rolls back. <c>, <d> and <job> in names and in expected answers
become ids. A case with a key this module does not know is an error, so a typo in the table cannot pass
unseen.
"""

import json
import re
from pathlib import Path

PATH = Path(__file__).with_name("supabase_cases.json")

IDS = {
    "b": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "c": "cccccccc-cccc-cccc-cccc-cccccccccccc",
    "d": "dddddddd-dddd-dddd-dddd-dddddddddddd",
    "mc": "c1c1c1c1-0000-0000-0000-000000000001", "md": "d1d1d1d1-0000-0000-0000-000000000001",
    "wc": "c2c2c2c2-0000-0000-0000-000000000001", "wd": "d2d2d2d2-0000-0000-0000-000000000001",
    "job": "e0e0e0e0-0000-0000-0000-000000000001",
}
USERS = ("b", "c", "d")
OWNER = {"mc": "c", "md": "d"}                 # machine -> its user
MACHINE = {"c": "mc", "d": "md"}               # user -> its machine
WORKERS = {"wc": "wc", "wd": "wd"}             # worker names in the table -> IDS
SHA = "0123456789abcdef" + "0" * 48           # the sha256 of a put row that names none
INPUT = {"video": "case.mp4", "rosbag": "case.bag", "stray": "case.zip"}

CASE_KEYS = {"id", "fn", "as", "args", "state", "expect"}
STATE_KEYS = {"settings", "access", "own_worker", "job", "requests", "rate", "objects"}
JOB_KEYS = {"status", "input_kind", "machine"}
REQUEST_KEYS = {"route", "state", "sealed"}
SEALED_KEYS = {"worker", "agent"}
OBJECT_KEYS = {"bucket", "name", "size", "age"}
ARGS = {
    "relay": {"route", "query", "body", "text", "sealed"},
    "delete_job": set(),
    "own_claim_request": {"worker", "kinds"},
    "own_worker_start": {"machine", "local_keys"},
    "own_publish": {"worker", "view", "live", "snapshots", "row"},
    "own_files": {"worker", "put", "drop"},
    "own_logs": {"worker", "rows"},
    "own_job_create": {"worker", "name", "kind", "bytes"},
    "job_objects": {"limit"},
    "own_orphans": {"worker", "limit"},
    "storage_insert": {"bucket", "name", "size"},
    "storage_delete": {"bucket", "name"},
}
STATEMENTS = {"own_publish", "own_files", "own_logs", "storage_insert"}   # calls that answer nothing
COUNTS = {"storage_delete"}                                                # the answer is a row count
COLUMN = re.compile(r"^[a-z_][a-z0-9_]*$")

# Compares an answer with a case's "ok" value (see "_format" in the table).
MATCH_SQL = r"""
create function tests.case_match(got jsonb, want jsonb) returns boolean language plpgsql immutable as $$
declare
  i int;
begin
  if want = '"uuid"'::jsonb then
    return jsonb_typeof(got) = 'string'
       and got #>> '{}' ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';
  elsif jsonb_typeof(want) = 'array' then
    if jsonb_typeof(got) is distinct from 'array' or jsonb_array_length(got) <> jsonb_array_length(want) then
      return false;
    end if;
    for i in 0 .. jsonb_array_length(want) - 1 loop
      if not (got -> i) @> (want -> i) then return false; end if;
    end loop;
    return true;
  elsif jsonb_typeof(want) = 'object' then
    return jsonb_typeof(got) = 'object' and got @> want;
  end if;
  return got = want;
end $$;
grant execute on function tests.case_match(jsonb, jsonb) to public;
"""


def load(path=PATH):
    return json.loads(Path(path).read_text())["cases"]


def lit(v):
    """A SQL literal: a quoted string, a number, a boolean or null."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def jlit(v):
    return "null::jsonb" if v is None else lit(json.dumps(v)) + "::jsonb"


def alit(v):
    """A text[] literal."""
    if v is None:
        return "null::text[]"
    return "array[" + ", ".join(lit(x) for x in v) + "]::text[]"


def uid(name):
    return lit(IDS[name]) + "::uuid"


def names(text):
    """<b> <c> <d> and <job> in a name become ids."""
    for u in USERS:
        text = text.replace(f"<{u}>", IDS[u])
    return text.replace("<job>", IDS["job"])


def names_in(v):
    """names() in every string of an expected answer."""
    if isinstance(v, str):
        return names(v)
    if isinstance(v, list):
        return [names_in(x) for x in v]
    if isinstance(v, dict):
        return {k: names_in(x) for k, x in v.items()}
    return v


def worker_id(name):
    if name is None:
        return "null::uuid"
    if name not in WORKERS:
        raise ValueError(f"unknown worker {name!r}")
    return uid(WORKERS[name])


def known(where, got, allowed):
    extra = {k for k in got if not k.startswith("_")} - allowed
    if extra:
        raise ValueError(f"{where}: unknown keys {sorted(extra)}")


def base_sql():
    """The rows every case starts from, and tests.case_match."""
    out = ["insert into auth.users (id, email) values "
           + ", ".join(f"({uid(u)}, '{u}@example.org')" for u in USERS) + ";",
           f"update public.access set approved = true where user_id in ({uid('c')}, {uid('d')});",
           f"insert into public.machines (id, user_id, name) values ({uid('mc')}, {uid('c')}, 'lab-box'), "
           f"({uid('md')}, {uid('d')}, 'd-box');",
           "insert into public.workers (id, user_id, machine_id, public_key) values "
           f"({uid('wc')}, {uid('c')}, {uid('mc')}, repeat('C', 88)), "
           f"({uid('wd')}, {uid('d')}, {uid('md')}, repeat('D', 88));",
           "insert into tests.ids (name, id) values "
           + ", ".join(f"({lit(k)}, {uid(k)})" for k in IDS) + ";",
           MATCH_SQL]
    return "\n".join(out) + "\n"


def _set_columns(table, values, where=""):
    out = []
    for k, v in values.items():
        if not COLUMN.match(k):
            raise ValueError(f"bad column name {k!r}")
        out.append(f"update public.{table} set {k} = (jsonb_populate_record(null::public.{table}, "
                   f"{lit(json.dumps({k: v}))}::jsonb)).{k}{where};")
    return out


def _call(fn, args, job_user):
    """The SQL of the call: a statement for the calls in STATEMENTS, a data-modifying statement whose
    rows are counted for COUNTS, else an expression that gives the answer as jsonb."""
    a = dict(args)
    if fn == "relay":
        sealed = a.get("sealed")
        if sealed is None:
            sealed_sql = "null::jsonb"
        else:
            if sealed == "valid":
                sealed = {}
            known("sealed", sealed, SEALED_KEYS)
            w = worker_id(sealed["worker"]) if "worker" in sealed else \
                "coalesce(tests.id('jobworker'), gen_random_uuid())"
            sealed_sql = f"tests.sealed({w}, {lit(sealed.get('agent', 'opus'))}, gen_random_uuid()::text)"
        return (f"public.relay({uid('job')}, {lit(a.get('route'))}, {jlit(a.get('query', {}))}, "
                f"{jlit(a.get('body', {}))}, {lit(a.get('text'))}, {sealed_sql})")
    if fn == "delete_job":
        return f"to_jsonb(public.delete_job({uid('job')}))"
    if fn == "own_claim_request":
        return (f"(select coalesce(jsonb_agg(v), '[]'::jsonb) from public.own_claim_request("
                f"{worker_id(a.get('worker'))}, {alit(a.get('kinds'))}) v)")
    if fn == "own_worker_start":
        m = a.get("machine")
        return (f"to_jsonb(public.own_worker_start({'null::uuid' if m is None else uid(m)}, repeat('K', 88), null, "
                f"'test', {alit(a.get('local_keys', []))}))")
    if fn == "own_publish":
        return (f"select public.own_publish({worker_id(a.get('worker'))}, {uid('job')}, {jlit(a.get('view'))}, "
                f"{jlit(a.get('live'))}, {jlit(a.get('snapshots'))}, {jlit(a.get('row'))})")
    if fn == "own_files":
        put = []
        for row in a.get("put", []):
            sha = row.get("sha256", SHA)
            obj = names(row["object"]) if "object" in row else \
                f"{IDS[job_user]}/{IDS['job']}/{sha[:16]}/{row['path']}"
            put.append({"path": row["path"], "object": obj, "bytes": row["bytes"], "sha256": sha,
                        "role": row.get("role", "result"),
                        "content_type": row.get("content_type", "application/octet-stream")})
        return (f"select public.own_files({worker_id(a.get('worker'))}, {uid('job')}, {jlit(put)}, "
                f"{alit(a.get('drop', []))})")
    if fn == "own_logs":
        return f"select public.own_logs({worker_id(a.get('worker'))}, {uid('job')}, {jlit(a.get('rows'))})"
    if fn == "own_job_create":
        return (f"to_jsonb(public.own_job_create({worker_id(a.get('worker'))}, {lit(a.get('name'))}, "
                f"{lit(a.get('kind'))}, {lit(a.get('bytes'))}))")
    if fn == "job_objects":
        return (f"(select coalesce(jsonb_agg(jsonb_build_object('name', o.name, 'created_at', o.created_at) order by o.n), "
                f"'[]'::jsonb) from public.job_objects({uid('job')}, {lit(a.get('limit', 1000))}) "
                "with ordinality o (name, created_at, n))")
    if fn == "own_orphans":
        return (f"(select coalesce(jsonb_agg(jsonb_build_object('name', o.name) order by o.n), '[]'::jsonb) "
                f"from public.own_orphans({worker_id(a.get('worker'))}, {lit(a.get('limit', 1000))}) "
                "with ordinality o (name, n))")
    if fn == "storage_insert":
        return (f"insert into storage.objects (bucket_id, name, metadata) values ({lit(a['bucket'])}, "
                f"{lit(names(a['name']))}, jsonb_build_object('size', {lit(a['size'])}))")
    if fn == "storage_delete":
        return (f"delete from storage.objects where bucket_id = {lit(a['bucket'])} and name = {lit(names(a['name']))} "
                "returning 1")
    raise ValueError(f"unknown fn {fn!r}")


def case_sql(case):
    """One case as one transaction that ends with a rollback."""
    known(f"case {case.get('id')}", case, CASE_KEYS)
    cid, fn, who = case["id"], case["fn"], case["as"]
    if fn not in ARGS:
        raise ValueError(f"{cid}: unknown fn {fn!r}")
    if who not in (*USERS, "anon"):
        raise ValueError(f"{cid}: unknown caller {who!r}")
    known(f"{cid}: args", case.get("args", {}), ARGS[fn])
    state = case.get("state", {})
    known(f"{cid}: state", state, STATE_KEYS)
    job = state.get("job", {})
    known(f"{cid}: job", job, JOB_KEYS)
    machine = job.get("machine", MACHINE.get(who, "mc"))
    if machine not in OWNER:
        raise ValueError(f"{cid}: unknown machine {machine!r}")
    job_user = OWNER[machine]
    access_user = who if who in USERS else "c"
    status = job.get("status", "ready")
    kind = job.get("input_kind", "video")
    rate = state.get("rate")
    if rate is not None and (isinstance(rate, bool) or not isinstance(rate, int)):
        raise ValueError(f"{cid}: rate is this hour's request count, a number")

    out = [f"-- case {cid}", "begin;"]
    out += _set_columns("settings", state.get("settings", {}))
    out += _set_columns("access", state.get("access", {}), f" where user_id = {uid(access_user)}")
    if "own_worker" in state:
        if state["own_worker"] is not None:
            raise ValueError(f"{cid}: own_worker is null or absent")
        out.append("update public.workers set stopped_at = now(), accepting = false;")
    else:
        out.append("update public.workers set last_seen = now();")
    out.append("insert into public.jobs (id, user_id, machine_id, input_name, input_kind, input_bytes, status, deleted_at) "
               f"values ({uid('job')}, {uid(job_user)}, {uid(machine)}, {lit(INPUT[kind])}, {lit(kind)}, 1000, "
               f"{lit(status)}, {'now()' if status == 'deleting' else 'null'});")
    out.append("insert into tests.ids (name, id) select 'jobworker', (private.live_worker(j.machine_id)).id "
               f"from public.jobs j where j.id = {uid('job')};")
    if rate is not None:
        out.append(f"insert into private.rate (user_id, hour, n) values ({uid(job_user)}, date_trunc('hour', now()), "
                   f"{lit(rate)});")
    for i, r in enumerate(state.get("requests", [])):
        known(f"{cid}: request", r, REQUEST_KEYS)
        ref = "null"
        if r.get("sealed") is not None:
            known(f"{cid}: sealed", r["sealed"], SEALED_KEYS)
            ref = lit(f"5ea1ed00-0000-0000-0000-{i:012d}") + "::uuid"
            out.append("insert into public.job_secrets (ref, job_id, user_id, worker_id, agent, epk, iv, ct) values "
                       f"({ref}, {uid('job')}, {uid(job_user)}, {worker_id(r['sealed'].get('worker'))}, "
                       f"{lit(r['sealed'].get('agent', 'opus'))}, repeat('B', 88), repeat('C', 16), repeat('D', 40));")
        out.append(f"insert into public.requests (user_id, job_id, route, state, key_ref) values ({uid(job_user)}, "
                   f"{uid('job')}, {lit(r['route'])}, {lit(r.get('state', 'pending'))}, {ref});")
    for o in state.get("objects", []):
        known(f"{cid}: object", o, OBJECT_KEYS)
        out.append(f"insert into storage.objects (bucket_id, name, metadata, created_at) values ({lit(o['bucket'])}, "
                   f"{lit(names(o['name']))}, jsonb_build_object('size', {lit(o['size'])}), "
                   f"now() - make_interval(secs => {lit(o.get('age', 0))}));")

    if who in USERS:
        out.append("set local role authenticated;")
        out.append(f"set local request.jwt.claims to {lit(json.dumps({'sub': IDS[who], 'role': 'authenticated'}))};")
    else:
        out.append("set local role anon;")

    call = _call(fn, case.get("args", {}), job_user)
    is_statement = fn in STATEMENTS
    if fn in COUNTS:
        # A data-modifying WITH must stand at the top level, so the count goes through tests.ids.
        call = f"with done as ({call}) insert into tests.ids (name, n) select 'counted', count(*) from done"
        is_statement = True
    expect = case["expect"]
    label = lit(f"case {cid}")
    if "error" in expect:
        stmt = call if is_statement else f"select {call}"
        if "$c$" in stmt:
            raise ValueError(f"{cid}: the call holds $c$")
        out.append(f"select tests.expect_error($c${stmt}$c$, {lit(expect['error'])});")
    elif "ok" in expect:
        want = names_in(expect["ok"])
        if fn in COUNTS:
            out.append(f"{call};")
            out.append(f"select tests.check(tests.n('counted') = {lit(want)}, {label});")
        elif is_statement:
            if want not in (None, True):
                raise ValueError(f"{cid}: {fn} answers nothing to compare with {want!r}")
            out.append(f"{call};")
            out.append(f"select tests.check(true, {label});")
        else:
            extra = []
            if isinstance(want, dict) and "withdrawn" in want:
                want = dict(want)
                n = want.pop("withdrawn")
                extra.append(f"select tests.check((select count(*) from public.requests where job_id = {uid('job')} "
                             f"and state = 'withdrawn') = {lit(n)}, {lit(f'case {cid}: withdrawn')});")
            out.append(f"select tests.check(tests.case_match({call}, {jlit(want)}), {label});")
            out += extra
    else:
        raise ValueError(f"{cid}: expect needs error or ok")
    out.append("rollback;")
    return "\n".join(out) + "\n"
