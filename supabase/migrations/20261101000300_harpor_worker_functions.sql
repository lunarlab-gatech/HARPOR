-- HARPOR public site, part 3: the functions that a personal worker calls. A personal worker runs
-- HARPOR's server on a user's own GPU machine and signs in as that user (role authenticated). Every
-- function but own_machine_bind and own_worker_start takes the worker id that own_worker_start returned,
-- and private.own_worker checks that it is a live worker of the caller and that the caller is still
-- approved. Each function writes only rows of the caller's jobs on that worker's machine. The caller's
-- browser holds a session of the same user and could call these functions too; that reaches only the
-- same user's rows, within the limits below, so it needs no extra guard.

-- The row of a live personal worker of the caller, else an error.
create function private.own_worker(p_worker uuid) returns public.workers
language plpgsql stable set search_path = '' as $$
declare
  w public.workers;
begin
  select * into w from public.workers where id = p_worker and user_id = auth.uid();
  if not found then raise exception 'no such worker'; end if;
  if w.stopped_at is not null then raise exception 'worker % was stopped: another worker took over', p_worker; end if;
  if not exists (select 1 from public.access where user_id = auth.uid() and approved) then
    raise exception 'your account is waiting for approval';
  end if;
  return w;
end $$;

-- A job of the worker's machine, locked, else an error. p_deleting picks the jobs that wait in "deleting"
-- instead of the live ones.
create function private.own_job(w public.workers, p_job uuid, p_deleting boolean default false) returns public.jobs
language plpgsql set search_path = '' as $$
declare
  j public.jobs;
begin
  select * into j from public.jobs
   where id = p_job and user_id = w.user_id and machine_id = w.machine_id
     and (case when p_deleting then deleted_at is not null else deleted_at is null end)
   for update;
  if not found then raise exception 'no such job on this machine'; end if;
  return j;
end $$;

-- `worker login`: a new machine (p_machine null) or a new name for one of the caller's machines.
-- Returns the machine id, which the session file keeps.
create function public.own_machine_bind(p_machine uuid, p_name text) returns uuid
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
  mid uuid;
  n int;
begin
  if uid is null then raise exception 'sign in first'; end if;
  perform pg_advisory_xact_lock(hashtextextended('harpor:user:' || uid::text, 0));
  if not exists (select 1 from public.access where user_id = uid and approved) then
    raise exception 'your account is waiting for approval';
  end if;
  if p_name is null or p_name !~ '^[A-Za-z0-9][A-Za-z0-9_.-]{0,59}$' then
    raise exception 'the machine name has 1 to 60 letters, digits, _ . and -, and starts with a letter or digit';
  end if;
  if p_machine is not null then
    update public.machines set name = p_name where id = p_machine and user_id = uid returning id into mid;
    if mid is null then raise exception 'no such machine'; end if;
    return mid;
  end if;
  select count(*) into n from public.machines where user_id = uid;
  if n >= (select max_machines from public.settings where id = 1) then
    raise exception 'you have % machines; ask the site owner to remove one', n;
  end if;
  insert into public.machines (user_id, name) values (uid, p_name) returning id into mid;
  return mid;
end $$;

-- A personal worker starts on a machine: its row and a clean slate for that machine only. Older
-- workers of the machine are stopped, keys sealed to them go, requests they took and never answered
-- get 409, and pending requests of the machine's jobs whose key is gone get 400. p_local_keys names
-- the agents whose key the machine holds in its own env (the page then sends no key for them).
-- It takes the user's lock that relay() holds, so a key sealed to the old worker during a start is
-- either dropped here (its request gets 400) or refused by relay() as worker_restarted.
create function public.own_worker_start(p_machine uuid, p_public_key text, p_ends_at timestamptz, p_version text,
                                        p_local_keys text[] default '{}') returns uuid
language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
  wid uuid;
begin
  if uid is null then raise exception 'sign in first'; end if;
  perform pg_advisory_xact_lock(hashtextextended('harpor:user:' || uid::text, 0));
  if not exists (select 1 from public.access where user_id = uid and approved) then
    raise exception 'your account is waiting for approval';
  end if;
  perform 1 from public.machines where id = p_machine and user_id = uid for update;   -- one start at a time
  if not found then raise exception 'no such machine'; end if;
  update public.workers set stopped_at = now(), accepting = false
   where machine_id = p_machine and stopped_at is null;
  insert into public.workers (user_id, machine_id, public_key, ends_at, version, local_keys)
  values (uid, p_machine, p_public_key, p_ends_at, p_version, coalesce(p_local_keys, '{}'))
  returning id into wid;
  delete from public.job_secrets s using public.workers w
   where w.id = s.worker_id and w.machine_id = p_machine and w.id <> wid;
  update public.requests r set state = 'done', done_at = now(), status = 409,
         response = '{"detail": "your machine restarted during this request; look at the job and try again"}'::jsonb
    from public.workers w
   where r.state = 'taken' and w.id = r.worker_id and w.machine_id = p_machine and w.id <> wid;
  update public.requests r set state = 'done', done_at = now(), status = 400,
         response = '{"detail": "your machine restarted after you sent the key; paste the key again"}'::jsonb
    from public.jobs j
   where r.state = 'pending' and r.key_ref is not null and j.id = r.job_id
     and j.machine_id = p_machine and not exists (select 1 from public.job_secrets s where s.ref = r.key_ref);
  -- Keep the machine's newest 10 rows, so that a loop of starts cannot fill the table. The rows that go
  -- are stopped, and the requests they took were answered when they stopped.
  delete from public.workers w
   where w.machine_id = p_machine and w.stopped_at is not null
     and w.id not in (select id from public.workers where machine_id = p_machine
                       order by stopped_at is null desc, started_at desc, id limit 10);
  return wid;
end $$;

-- The heartbeat, every 10 s: last_seen, the Runner's queue length, accepting (false in the last
-- near_end_minutes of a Slurm job) and the job's end. p_ends_at is null on a machine outside Slurm.
create function public.own_worker_beat(p_worker uuid, p_queue_len int, p_accepting boolean,
                                       p_ends_at timestamptz default null) returns void
language plpgsql security definer set search_path = '' as $$
begin
  perform private.own_worker(p_worker);
  update public.workers set last_seen = now(), queue_len = greatest(coalesce(p_queue_len, 0), 0),
         accepting = coalesce(p_accepting, true), ends_at = coalesce(p_ends_at, ends_at)
   where id = p_worker;
end $$;

-- The worker stops (server shutdown). Works also after the account lost its approval.
create function public.own_worker_stop(p_worker uuid) returns void
language sql security definer set search_path = '' as $$
  update public.workers set stopped_at = now(), accepting = false
   where id = p_worker and user_id = auth.uid() and stopped_at is null;
$$;

-- The oldest pending request of the machine's jobs whose job has no request in progress. It does not
-- wait: user sessions have a short statement timeout, and each waiting call would hold a database
-- connection, so the worker calls it every 2 s. p_kinds null takes every route. Near the end of a
-- Slurm job the worker passes the routes that start no build; a job's later request is then not taken
-- before an earlier one that waits for the next start. Returns the request row as JSON with
-- "sealed": {agent, epk, iv, ct} of its key, or null; the key's row is deleted in the same call.
create function public.own_claim_request(p_worker uuid, p_kinds text[] default null) returns setof jsonb
language plpgsql security definer set search_path = '' as $$
declare
  w public.workers := private.own_worker(p_worker);
  r public.requests;
  s public.job_secrets;
begin
  update public.requests q set state = 'taken', taken_at = clock_timestamp(), worker_id = p_worker
   where q.id = (
     select c.id from public.requests c join public.jobs j on j.id = c.job_id
      where c.state = 'pending' and c.user_id = w.user_id and j.machine_id = w.machine_id
        and (p_kinds is null or c.route = any (p_kinds))
        and not exists (select 1 from public.requests b where b.job_id = c.job_id and b.state = 'taken')
        and (p_kinds is null or not exists (
              select 1 from public.requests o
               where o.job_id = c.job_id and o.state = 'pending' and o.id < c.id and not (o.route = any (p_kinds))))
      order by c.id limit 1 for update of c skip locked)
  returning q.* into r;
  if r.id is null then
    return;
  end if;
  if r.key_ref is not null then
    delete from public.job_secrets where ref = r.key_ref and worker_id = p_worker returning * into s;
  end if;
  return next to_jsonb(r) || jsonb_build_object('sealed', case when s.ref is null then null
    else jsonb_build_object('agent', s.agent, 'epk', s.epk, 'iv', s.iv, 'ct', s.ct) end);
end $$;

-- The answer to a request that this worker took: server.py's status and JSON, after own_publish.
create function public.own_answer(p_worker uuid, p_request bigint, p_status int, p_response jsonb) returns void
language plpgsql security definer set search_path = '' as $$
begin
  perform private.own_worker(p_worker);
  if p_status is null or p_status not between 100 and 599 then raise exception 'the status is an HTTP status'; end if;
  update public.requests set state = 'done', done_at = now(), status = p_status, response = p_response
   where id = p_request and worker_id = p_worker and state = 'taken';
  if not found then raise exception 'the request is not taken by this worker'; end if;
end $$;

-- A job that the machine made from a local upload (its own UI, or `worker add PATH`). The input
-- stays on the machine; only its name, kind and size come here. Status "uploaded"; own_publish
-- follows job.json from then on. A job that waits in "deleting" does not count toward max_own_jobs:
-- the user no longer sees it, and delete_job left only its row.
create function public.own_job_create(p_worker uuid, p_name text, p_kind text, p_bytes bigint) returns uuid
language plpgsql security definer set search_path = '' as $$
declare
  w public.workers := private.own_worker(p_worker);
  a public.access;
  n int;
  jid uuid;
begin
  perform pg_advisory_xact_lock(hashtextextended('harpor:user:' || w.user_id::text, 0));
  select * into a from public.access where user_id = w.user_id;
  if p_name is null or p_name !~ '^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,199}$' or p_name ~ '\.\.' then
    raise exception 'the file name has 1 to 200 letters, digits, _ . and -, does not start with a dot and has no two dots in a row';
  end if;
  if p_kind is null or p_kind is distinct from private.kind_of(p_name) then
    raise exception 'the input is a video (.mp4 .mov .m4v .avi .mkv .webm), a rosbag (.bag .mcap .db3) or a Stray Scanner export (.zip)';
  end if;
  if p_bytes is null or p_bytes < 1 then raise exception 'the file is empty'; end if;
  select count(*) into n from public.jobs where user_id = w.user_id and deleted_at is null;
  if n >= a.max_own_jobs then
    raise exception 'you have % jobs from your own machines on the site; delete older ones first', n;
  end if;
  insert into public.usage_daily (user_id) values (w.user_id) on conflict (user_id, day) do nothing;
  if (select own_jobs from public.usage_daily where user_id = w.user_id and day = current_date) >= a.max_own_per_day then
    raise exception 'you reached % jobs from your own machines today', a.max_own_per_day;
  end if;
  update public.usage_daily set own_jobs = own_jobs + 1 where user_id = w.user_id and day = current_date;
  insert into public.jobs (user_id, machine_id, input_name, input_kind, input_bytes, status)
  values (w.user_id, w.machine_id, p_name, p_kind, p_bytes, 'uploaded')
  returning id into jid;
  return jid;
end $$;

-- One publish of a job, in one transaction, so the page sees the view, the live state, the snapshots
-- and the row change together: p_view (job_views.body), p_live (job_live.state), p_snapshots
-- ({"frames"|"selection"|"parts": {"status": 200, "body": ...}}) and p_row ({status, note, prompt,
-- queue_pos, stopped_at}). A null argument leaves that part as it is. The worker uploads the files
-- and calls own_files first, and answers the request only after this call. stopped_at is never later
-- than now, so a worker cannot put off the sweep's retention of log text, which counts from it, and it
-- is a finite time from 2020 on.
create function public.own_publish(p_worker uuid, p_job uuid, p_view jsonb default null, p_live jsonb default null,
                                   p_snapshots jsonb default null, p_row jsonb default null) returns void
language plpgsql security definer set search_path = '' as $$
declare
  w public.workers := private.own_worker(p_worker);
  j public.jobs := private.own_job(w, p_job);
  k text;
  st text;
  t timestamptz;
begin
  if p_row is not null then
    if jsonb_typeof(p_row) <> 'object'
       or (p_row - array['status', 'note', 'prompt', 'queue_pos', 'stopped_at']) <> '{}'::jsonb then
      raise exception 'the row takes status, note, prompt, queue_pos and stopped_at';
    end if;
    st := coalesce(p_row ->> 'status', j.status);
    if st not in ('uploaded', 'analyzing', 'choose_topic', 'ready', 'queued', 'running', 'review', 'done', 'failed',
                  'cancelled') then
      raise exception 'bad status %', st;
    end if;
    t := (p_row ->> 'stopped_at')::timestamptz;
    if t is not null and (not isfinite(t) or t < '2020-01-01 00:00:00+00') then
      raise exception 'stopped_at is a time from 2020 on';
    end if;
    update public.jobs set status = st,
           note = case when p_row ? 'note' then p_row ->> 'note' else note end,
           prompt = case when p_row ? 'prompt' then p_row ->> 'prompt' else prompt end,
           queue_pos = case when p_row ? 'queue_pos' then (p_row ->> 'queue_pos')::int else queue_pos end,
           stopped_at = case when p_row ? 'stopped_at' then least(t, now()) else stopped_at end,
           updated_at = now()
     where id = p_job;
  end if;
  if p_view is not null then
    if jsonb_typeof(p_view) <> 'object' then raise exception 'the view is a JSON object'; end if;
    insert into public.job_views (job_id, user_id, body) values (p_job, w.user_id, p_view)
    on conflict (job_id) do update set body = excluded.body, rev = public.job_views.rev + 1, updated_at = now();
  end if;
  if p_live is not null then
    if jsonb_typeof(p_live) <> 'object' then raise exception 'the live state is a JSON object'; end if;
    insert into public.job_live (job_id, user_id, state) values (p_job, w.user_id, p_live)
    on conflict (job_id) do update set state = excluded.state, updated_at = now();
  end if;
  if p_snapshots is not null then
    if jsonb_typeof(p_snapshots) <> 'object' or (p_snapshots - array['frames', 'selection', 'parts']) <> '{}'::jsonb then
      raise exception 'the snapshots are frames, selection and parts';
    end if;
    for k in select jsonb_object_keys(p_snapshots) loop
      insert into public.job_snapshots (job_id, path, user_id, status, body)
      values (p_job, k, w.user_id, (p_snapshots -> k ->> 'status')::int, p_snapshots -> k -> 'body')
      on conflict (job_id, path) do update set status = excluded.status, body = excluded.body, updated_at = now();
    end loop;
  end if;
end $$;

-- The file list after the worker uploaded the objects: p_put rows {path, object, bytes, sha256, role,
-- content_type} replace the rows of their paths, and p_drop paths go. Each object must exist in
-- job-files with that size, and its name must be <user>/<job>/<sha256[:16]>/<path> (a check on
-- job_files). The old object of a changed path is the worker's to delete, 10 minutes later.
create function public.own_files(p_worker uuid, p_job uuid, p_put jsonb default '[]'::jsonb,
                                 p_drop text[] default '{}') returns void
language plpgsql security definer set search_path = '' as $$
declare
  w public.workers := private.own_worker(p_worker);
  j public.jobs := private.own_job(w, p_job);
  f jsonb;
begin
  p_put := coalesce(p_put, '[]'::jsonb);
  if jsonb_typeof(p_put) <> 'array' or jsonb_array_length(p_put) > 500 then
    raise exception 'put is a list of at most 500 files';
  end if;
  delete from public.job_files where job_id = p_job and path = any (coalesce(p_drop, '{}'));
  for f in select * from jsonb_array_elements(p_put) loop
    if not exists (select 1 from storage.objects o where o.bucket_id = 'job-files' and o.name = f ->> 'object'
                    and (o.metadata ->> 'size')::bigint = (f ->> 'bytes')::bigint) then
      raise exception 'job-files holds no object for % with % bytes', f ->> 'path', f ->> 'bytes';
    end if;
    insert into public.job_files (job_id, path, user_id, object, bytes, sha256, role, content_type)
    values (p_job, f ->> 'path', w.user_id, f ->> 'object', (f ->> 'bytes')::bigint, f ->> 'sha256', f ->> 'role',
            f ->> 'content_type')
    on conflict (job_id, path) do update set object = excluded.object, bytes = excluded.bytes, sha256 = excluded.sha256,
           role = excluded.role, content_type = excluded.content_type, updated_at = now();
  end loop;
  if (select count(*) from public.job_files where job_id = p_job) > 2000 then
    raise exception 'a job keeps at most 2000 files on the site';
  end if;
end $$;

-- Log text of running stages, scrubbed by the worker: p_rows [{stage, run_started, pos, text}], at
-- most 100 a call. A job's log text on the site is capped at settings.own_log_bytes.
create function public.own_logs(p_worker uuid, p_job uuid, p_rows jsonb) returns void
language plpgsql security definer set search_path = '' as $$
declare
  w public.workers := private.own_worker(p_worker);
  j public.jobs := private.own_job(w, p_job);
  cap bigint := (select own_log_bytes from public.settings where id = 1);
  used bigint;
  more bigint;
begin
  if jsonb_typeof(p_rows) is distinct from 'array' or jsonb_array_length(p_rows) not between 1 and 100 then
    raise exception 'rows is a list of 1 to 100 log parts';
  end if;
  select coalesce(sum(octet_length(text)), 0) into used from public.job_logs where job_id = p_job;
  select coalesce(sum(octet_length(e ->> 'text')), 0) into more from jsonb_array_elements(p_rows) e;
  if used + more > cap then
    raise exception 'the job''s log on the site is full (% MB); the rest stays on your machine', cap / 1048576;
  end if;
  insert into public.job_logs (job_id, user_id, stage, run_started, pos, text)
  select p_job, w.user_id, e ->> 'stage', (e ->> 'run_started')::double precision, (e ->> 'pos')::bigint, e ->> 'text'
    from jsonb_array_elements(p_rows) e;
end $$;

-- A delete request is done: the worker removed the job folder on its machine and the job's Storage
-- objects (job_objects). The row goes.
create function public.own_job_delete(p_worker uuid, p_job uuid) returns void
language plpgsql security definer set search_path = '' as $$
declare
  w public.workers := private.own_worker(p_worker);
  j public.jobs := private.own_job(w, p_job, true);
begin
  delete from public.jobs where id = j.id;
end $$;

-- ---- who may call what ----
revoke execute on function private.own_worker(uuid), private.own_job(public.workers, uuid, boolean)
  from public, anon, authenticated, service_role;
revoke execute on function public.own_machine_bind(uuid, text),
  public.own_worker_start(uuid, text, timestamptz, text, text[]),
  public.own_worker_beat(uuid, int, boolean, timestamptz), public.own_worker_stop(uuid),
  public.own_claim_request(uuid, text[]), public.own_answer(uuid, bigint, int, jsonb),
  public.own_job_create(uuid, text, text, bigint), public.own_publish(uuid, uuid, jsonb, jsonb, jsonb, jsonb),
  public.own_files(uuid, uuid, jsonb, text[]), public.own_logs(uuid, uuid, jsonb), public.own_job_delete(uuid, uuid)
  from public, anon, authenticated, service_role;
grant execute on function public.own_machine_bind(uuid, text),
  public.own_worker_start(uuid, text, timestamptz, text, text[]),
  public.own_worker_beat(uuid, int, boolean, timestamptz), public.own_worker_stop(uuid),
  public.own_claim_request(uuid, text[]), public.own_answer(uuid, bigint, int, jsonb),
  public.own_job_create(uuid, text, text, bigint), public.own_publish(uuid, uuid, jsonb, jsonb, jsonb, jsonb),
  public.own_files(uuid, uuid, jsonb, text[]), public.own_logs(uuid, uuid, jsonb), public.own_job_delete(uuid, uuid)
  to authenticated;
