-- HARPOR public site, part 2: helpers, triggers, the sweep and the functions that users call.
-- Every function that users call is security definer with an empty search_path and schema-qualified
-- names. Postgres grants EXECUTE on new functions to PUBLIC, so each function revokes it and grants it
-- back only to authenticated (at the end of this file). The default privileges below take back what an old
-- project grants the API roles on functions made later. They cannot take back PUBLIC's EXECUTE, so a later
-- function must still revoke that itself.
alter default privileges in schema public revoke execute on functions from public, anon, authenticated, service_role;

-- ---- helpers (schema private is not exposed by the Data API) ----

create function private.try_uuid(t text) returns uuid language plpgsql immutable set search_path = '' as $$
begin
  return t::uuid;
exception when others then
  return null;
end $$;

-- The input kind of a file name, by its suffix (jobs.kind in Python): video, rosbag, stray or null.
create function private.kind_of(p_name text) returns text language sql immutable set search_path = '' as $$
  select case lower(substring(p_name from '(\.[^.]*)$'))
    when '.mp4' then 'video' when '.mov' then 'video' when '.m4v' then 'video' when '.avi' then 'video'
    when '.mkv' then 'video' when '.webm' then 'video'
    when '.bag' then 'rosbag' when '.mcap' then 'rosbag' when '.db3' then 'rosbag'
    when '.zip' then 'stray' end
$$;

-- True when j nests objects and arrays more than p_levels deep. relay() refuses such a request: jsonb
-- takes thousands of levels, but the worker's JSON parser stops much sooner and could not read it.
create function private.too_deep(j jsonb, p_levels int) returns boolean language sql immutable set search_path = '' as $$
  select case
    when jsonb_typeof(j) not in ('object', 'array') then false
    when p_levels <= 0 then true
    when jsonb_typeof(j) = 'object' then exists (select 1 from jsonb_each(j) e where private.too_deep(e.value, p_levels - 1))
    else exists (select 1 from jsonb_array_elements(j) e where private.too_deep(e.value, p_levels - 1))
  end
$$;

-- The newest worker of a machine that is online, or a row of nulls.
create function private.live_worker(p_machine uuid) returns public.workers language sql stable
set search_path = '' as $$
  select w.* from public.workers w, public.settings s
   where s.id = 1 and w.machine_id = p_machine and w.stopped_at is null
     and w.last_seen > now() - make_interval(secs => s.worker_online_s)
   order by w.started_at desc limit 1
$$;

-- This hour's relayed requests of a user, after adding one.
create function private.bump_rate(p_user uuid) returns int language sql set search_path = '' as $$
  insert into private.rate as r (user_id, hour, n) values (p_user, date_trunc('hour', now()), 1)
  on conflict (user_id, hour) do update set n = r.n + 1
  returning n
$$;

-- What no worker cleans up, run by pg_cron every 15 minutes (part 6): a request that waited longer than
-- pending_days expires, and a job that waited in "deleting" as long goes (its machine did not come back;
-- the page deleted its files when the user deleted it). Sealed keys that no worker took go after an hour,
-- large answers of finished requests after an hour (the page has read them by then), finished requests
-- after 7 days and old rate counts after 2 days. The log text of a job goes 14 days after the job stopped
-- (worker.REVIEW_DAYS). Storage objects are never deleted here: a SQL delete leaves the file behind.
-- Returns how many requests expired.
create function private.sweep() returns int language plpgsql set search_path = '' as $$
declare
  s public.settings;
  n int;
begin
  select * into s from public.settings where id = 1;
  update public.requests set state = 'expired', done_at = now(), status = 410,
         response = jsonb_build_object('detail',
           format('the request waited more than %s days for its machine; send it again', s.pending_days))
   where state = 'pending' and created_at < now() - make_interval(days => s.pending_days);
  get diagnostics n = row_count;
  delete from public.jobs where status = 'deleting' and deleted_at < now() - make_interval(days => s.pending_days);
  delete from public.job_secrets where created_at < now() - interval '1 hour';
  update public.requests set response = null
   where state in ('done', 'withdrawn', 'expired') and done_at < now() - interval '1 hour'
     and octet_length(response::text) > 1024;
  delete from public.requests where state in ('done', 'withdrawn', 'expired') and done_at < now() - interval '7 days';
  delete from private.rate where hour < now() - interval '2 days';
  delete from public.job_logs l using public.jobs j
   where j.id = l.job_id and j.status not in ('uploaded', 'analyzing', 'queued', 'running')
     and coalesce(j.stopped_at, j.updated_at) < now() - interval '14 days';
  return n;
end $$;

-- ---- triggers ----

create function private.on_signup() returns trigger language plpgsql security definer set search_path = '' as $$
begin
  insert into public.access (user_id, email) values (new.id, new.email) on conflict (user_id) do nothing;
  return new;
end $$;
create trigger harpor_on_signup after insert on auth.users
  for each row execute function private.on_signup();

-- A request that is done, withdrawn or expired no longer needs what it carried to the worker: the
-- calibration text goes, and a body over 1 KB becomes {} (a small one stays, for the page's line about
-- the request).
create function private.on_request_done() returns trigger language plpgsql set search_path = '' as $$
begin
  if new.state in ('done', 'withdrawn', 'expired') then
    new.body_text := null;
    if octet_length(new.body::text) > 1024 then new.body := '{}'::jsonb; end if;
  end if;
  return new;
end $$;
create trigger harpor_on_request_done before update of state on public.requests
  for each row execute function private.on_request_done();

-- ---- functions that users call ----

-- One relayed request. p_sealed is {ref, worker_id, agent, epk, iv, ct} from the page. The checks here
-- are coarse and fast; the personal worker of the job's machine runs server.py's own checks against
-- job.json. While that worker is offline, the request waits as pending (no key can be sealed then: the
-- page sends key_later instead). Public jobs run astra or opus, never "none".
-- Returns {id, worker_online, accepting}.
create function public.relay(p_job uuid, p_route text, p_query jsonb default '{}'::jsonb,
                             p_body jsonb default '{}'::jsonb, p_text text default null,
                             p_sealed jsonb default null) returns jsonb
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
  a public.access;
  j public.jobs;
  w public.workers;
  v_ref uuid;
  rid bigint;
begin
  if uid is null then raise exception 'sign in first'; end if;
  perform pg_advisory_xact_lock(hashtextextended('harpor:user:' || uid::text, 0));
  select * into a from public.access where user_id = uid;
  if not found or not a.approved then raise exception 'your account is waiting for approval'; end if;
  select * into j from public.jobs where id = p_job and user_id = uid and deleted_at is null for update;
  if not found then raise exception 'no such job'; end if;
  if p_route is null or p_route not in ('topic', 'calibration_put', 'calibration_delete', 'start', 'overrides',
                                        'parts', 'continue', 'cancel') then
    raise exception 'unknown request';
  end if;
  p_query := coalesce(p_query, '{}'::jsonb);
  p_body := coalesce(p_body, '{}'::jsonb);
  if jsonb_typeof(p_body) <> 'object' or octet_length(p_body::text) > 131072 then
    raise exception 'the request body is not a JSON object of at most 128 KB';
  end if;
  if jsonb_typeof(p_query) <> 'object' or (p_query - 'name') <> '{}'::jsonb then raise exception 'bad query'; end if;
  if private.too_deep(p_body, 32) or private.too_deep(p_query, 32) then
    raise exception 'the request nests lists and objects more than 32 deep';
  end if;
  if p_body ? 'api_key' and jsonb_typeof(p_body -> 'api_key') <> 'null' then
    raise exception 'send the API key sealed, not in the body';
  end if;
  -- "none" or false names no agent (steps.agent). The worker checks the agent that server.py will
  -- run against access.agents as well, since a body without "agent" takes the job's agent.
  if p_body -> 'agent' in ('"none"'::jsonb, 'false'::jsonb) then
    raise exception 'the public site runs the agents astra and opus';
  end if;
  if (p_text is not null) <> (p_route = 'calibration_put') then raise exception 'bad calibration body'; end if;
  if p_route = 'calibration_put' and jsonb_typeof(p_query -> 'name') is distinct from 'string' then
    raise exception 'the calibration needs a file name';
  end if;
  -- coarse state rules (the row can be behind job.json; the worker decides from job.json)
  if j.status in ('uploaded', 'analyzing', 'queued', 'running') and p_route <> 'cancel' then
    raise exception 'the job is %; wait until it stops, or cancel it', j.status;
  end if;
  if p_route = 'parts' and j.status <> 'review' then raise exception 'the job does not wait for its part list'; end if;
  if p_route in ('topic', 'calibration_put', 'calibration_delete') and j.input_kind <> 'rosbag' then
    raise exception 'this job''s input is not a rosbag';
  end if;
  -- one open request per job; a cancel takes back the job's pending requests instead of waiting
  if p_route = 'cancel' then
    if exists (select 1 from public.requests where job_id = p_job and route = 'cancel' and state = 'pending') then
      raise exception 'a cancel for this job already waits';
    end if;
    with gone as (
      update public.requests set state = 'withdrawn', done_at = now(), status = 409,
             response = '{"detail": "withdrawn by your cancel"}'::jsonb
       where job_id = p_job and state = 'pending' returning key_ref)
    delete from public.job_secrets where ref in (select key_ref from gone where key_ref is not null);
  elsif exists (select 1 from public.requests where job_id = p_job and state in ('pending', 'taken')) then
    raise exception 'an earlier request for this job still waits for its machine';
  end if;
  -- the hourly request limit, counted where a job delete does not reach (an exception undoes the count)
  if private.bump_rate(uid) > a.max_requests_per_hour then
    raise exception 'too many requests: at most % an hour', a.max_requests_per_hour;
  end if;
  w := private.live_worker(j.machine_id);
  if p_sealed is not null then
    if p_route not in ('start', 'overrides', 'parts', 'continue') then raise exception 'this request takes no API key'; end if;
    if jsonb_typeof(p_sealed) <> 'object' then raise exception 'the sealed key is not an object'; end if;
    if w.id is null then
      raise exception 'worker_offline: the machine that runs this job is offline; send the request without the key';
    end if;
    if not w.accepting then
      raise exception 'worker_ending: the machine stops soon; send the request without the key';
    end if;
    -- the key goes only to the newest live worker of the job's machine
    if private.try_uuid(p_sealed ->> 'worker_id') is distinct from w.id then
      raise exception 'worker_restarted: the machine restarted; paste the key again';
    end if;
    if coalesce(p_sealed ->> 'agent', '') not in ('astra', 'opus') or not ((p_sealed ->> 'agent') = any (a.agents)) then
      raise exception 'this account may use the agents %', array_to_string(a.agents, ', ');
    end if;
    v_ref := private.try_uuid(p_sealed ->> 'ref');
    if v_ref is null or coalesce(char_length(p_sealed ->> 'epk'), 0) <> 88
       or coalesce(char_length(p_sealed ->> 'iv'), 0) <> 16
       or coalesce(char_length(p_sealed ->> 'ct'), 0) not between 24 and 704 then
      raise exception 'the sealed key is malformed';
    end if;
    insert into public.job_secrets (ref, job_id, user_id, worker_id, agent, epk, iv, ct)
    values (v_ref, p_job, uid, w.id, p_sealed ->> 'agent', p_sealed ->> 'epk', p_sealed ->> 'iv', p_sealed ->> 'ct');
  end if;
  insert into public.requests (user_id, job_id, route, query, body, body_text, key_ref, sent_online)
  values (uid, p_job, p_route, p_query, p_body, p_text, v_ref, w.id is not null and w.accepting)
  returning id into rid;
  return jsonb_build_object('id', rid, 'worker_online', w.id is not null, 'accepting', coalesce(w.accepting, false));
end $$;

-- Take back a request that no worker has taken yet, with its sealed key.
create function public.withdraw_request(p_id bigint) returns text
language plpgsql security definer set search_path = '' as $$
declare
  r public.requests;
begin
  update public.requests set state = 'withdrawn', done_at = now(), status = 409,
         response = '{"detail": "you withdrew this request"}'::jsonb
   where id = p_id and user_id = auth.uid() and state = 'pending' and route <> 'delete'
  returning * into r;
  if not found then raise exception 'the request is no longer waiting'; end if;
  if r.key_ref is not null then delete from public.job_secrets where ref = r.key_ref; end if;
  return 'withdrawn';
end $$;

-- Delete a job. It is hidden at once and waits in "deleting" for its machine, whose worker removes the
-- job folder there and then the row (own_job_delete). The page deletes the job's Storage objects itself
-- (job_objects), so a machine that never comes back leaves no files, and the sweep removes the row after
-- pending_days. The job's view, live state, snapshots, file list and log text go at once, so the row holds
-- little while it waits. A job that analyzes, waits in the queue or runs on a machine that is online must be
-- cancelled first; any other job may go, since the worker cancels a job that still runs before it deletes
-- the folder. On a machine that is offline a job may go in any state, since that machine may never come
-- back. A user whose approval was taken back may still delete.
create function public.delete_job(p_job uuid) returns text
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
  j public.jobs;
  w public.workers;
begin
  if uid is null then raise exception 'sign in first'; end if;
  perform pg_advisory_xact_lock(hashtextextended('harpor:user:' || uid::text, 0));
  select * into j from public.jobs where id = p_job and user_id = uid and deleted_at is null for update;
  if not found then raise exception 'no such job'; end if;
  w := private.live_worker(j.machine_id);
  if j.status in ('analyzing', 'queued', 'running') and w.id is not null then
    raise exception 'the job is %; cancel it first', j.status;
  end if;
  with gone as (
    update public.requests set state = 'withdrawn', done_at = now(), status = 409,
           response = '{"detail": "the job was deleted"}'::jsonb
     where job_id = p_job and state = 'pending' returning key_ref)
  delete from public.job_secrets where ref in (select key_ref from gone where key_ref is not null);
  delete from public.job_views where job_id = p_job;
  delete from public.job_live where job_id = p_job;
  delete from public.job_snapshots where job_id = p_job;
  delete from public.job_files where job_id = p_job;
  delete from public.job_logs where job_id = p_job;
  update public.jobs set deleted_at = now(), status = 'deleting', updated_at = now() where id = p_job;
  insert into public.requests (user_id, job_id, route, sent_online) values (uid, p_job, 'delete', w.id is not null);
  return 'deleting';
end $$;

-- The "Request access" box of an account that waits for approval: once a day at most.
create function public.request_access(p_note text) returns text
language plpgsql security definer set search_path = '' as $$
begin
  if p_note is null or char_length(p_note) not between 1 and 500 then raise exception 'write 1 to 500 characters'; end if;
  update public.access set request_note = p_note, request_at = now()
   where user_id = auth.uid() and (request_at is null or request_at < now() - interval '1 day');
  if not found then raise exception 'you can send one request a day'; end if;
  return 'sent';
end $$;

-- ---- who may call what ----
revoke execute on function private.try_uuid(text), private.kind_of(text), private.too_deep(jsonb, int),
  private.live_worker(uuid), private.bump_rate(uuid), private.sweep(), private.on_signup(),
  private.on_request_done() from public, anon, authenticated, service_role;
revoke execute on function public.relay(uuid, text, jsonb, jsonb, text, jsonb), public.withdraw_request(bigint),
  public.delete_job(uuid), public.request_access(text) from public, anon, authenticated, service_role;
grant execute on function public.relay(uuid, text, jsonb, jsonb, text, jsonb), public.withdraw_request(bigint),
  public.delete_job(uuid), public.request_access(text) to authenticated;
