-- Helpers for the checks and the generated cases (tests/supabase_cases.py). Run as the superuser
-- after the migrations. Each check prints one NOTICE that starts with "ok"; a failed check raises,
-- and psql stops there (ON_ERROR_STOP).
create schema tests;
grant usage on schema tests to public;
-- Named ids and numbers that later statements read back, under any role.
create table tests.ids (name text primary key, id uuid, n bigint);
grant all on tests.ids to public;
-- Values for the checks, such as SHA-256 strings.
create table tests.sha (name text primary key, v text);
grant all on tests.sha to public;
create function tests.id(p text) returns uuid language sql stable as $$ select id from tests.ids where name = p $$;
create function tests.n(p text) returns bigint language sql stable as $$ select n from tests.ids where name = p $$;
create function tests.expect_error(stmt text, pattern text) returns void language plpgsql as $$
begin
  begin
    execute stmt;
  exception when others then
    if sqlerrm !~* pattern then
      raise exception 'expected an error like "%" from: %; got: %', pattern, stmt, sqlerrm;
    end if;
    raise notice 'ok (refused): % -> %', left(regexp_replace(stmt, '\s+', ' ', 'g'), 60), sqlerrm;
    return;
  end;
  raise exception 'expected an error like "%" from: %', pattern, stmt;
end $$;
create function tests.check(ok boolean, what text) returns void language plpgsql as $$
begin
  if not coalesce(ok, false) then raise exception 'FAILED: %', what; end if;
  raise notice 'ok: %', what;
end $$;
-- A sealed key as the page would send it (lengths only; the stand-in never opens it).
create function tests.sealed(p_worker uuid, p_agent text, p_ref text, p_ct int default 40) returns jsonb
language sql immutable as $$
  select jsonb_build_object('ref', p_ref, 'worker_id', p_worker, 'agent', p_agent, 'epk', repeat('B', 88),
                            'iv', repeat('C', 16), 'ct', repeat('D', p_ct))
$$;
-- A job-files object name: <user>/<job>/<sha256[:16]>/<path>.
create function tests.obj(p_user uuid, p_job uuid, p_sha text, p_path text) returns text language sql immutable as $$
  select p_user::text || '/' || p_job::text || '/' || left(p_sha, 16) || '/' || p_path
$$;
grant execute on all functions in schema tests to public;
