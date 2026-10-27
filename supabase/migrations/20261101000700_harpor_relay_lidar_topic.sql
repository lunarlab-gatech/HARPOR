-- HARPOR public site, part 7: relay() takes the LiDAR topic of a calibration upload. The page sends the topic
-- that was chosen when the file was uploaded (query lidar_topic), and the worker passes it to server.py, which
-- checks it against the bag and keeps it in job.json. Before this, a calibration sent from the site named no
-- topic, so Start could not refuse a LiDAR topic that the calibration was not made for.
-- The function is part 2's, restated whole (create or replace keeps its grants); only the query check differs.
-- A project set up with `supabase db push` gets this file from `supabase db push` again, which sends only new
-- files. A project set up by pasting the files gets it by pasting this file into the SQL editor (there db push
-- would start again from part 1 and fail). Add it before the site that sends lidar_topic: until then, relay()
-- refuses that query as a bad query.

-- One relayed request. p_sealed is {ref, worker_id, agent, epk, iv, ct} from the page. The checks here
-- are coarse and fast; the personal worker of the job's machine runs server.py's own checks against
-- job.json. While that worker is offline, the request waits as pending (no key can be sealed then: the
-- page sends key_later instead). Public jobs run astra or opus, never "none".
-- Returns {id, worker_online, accepting}.
create or replace function public.relay(p_job uuid, p_route text, p_query jsonb default '{}'::jsonb,
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
  -- the query holds the calibration's file name and the LiDAR topic it was made for (server.py's ?name= and
  -- ?lidar_topic=); lidar_topic is text, and only a calibration upload has it
  if jsonb_typeof(p_query) <> 'object' or (p_query - '{name,lidar_topic}'::text[]) <> '{}'::jsonb
     or (p_query ? 'lidar_topic' and (p_route is distinct from 'calibration_put'
                                       or jsonb_typeof(p_query -> 'lidar_topic') <> 'string')) then
    raise exception 'bad query';
  end if;
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

revoke execute on function public.relay(uuid, text, jsonb, jsonb, text, jsonb)
  from public, anon, authenticated, service_role;
grant execute on function public.relay(uuid, text, jsonb, jsonb, text, jsonb) to authenticated;
