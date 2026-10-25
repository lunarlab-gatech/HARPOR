-- Checks of the HARPOR migrations on the local stand-in (supabase_stub.sql): grants and RLS from the catalog,
-- sign-up and approval, request_access, relay() and delete_job() on jobs of a machine, sealed keys, the hourly
-- limit, finished requests, the sweep, the owner's list of orphan folders and Realtime. Run as the superuser,
-- after helpers.sql. 20_own_checks.sql and 30_isolation_checks.sql follow in the same database.
\set ON_ERROR_STOP on
\set QUIET on
\set as_a 'reset role; set request.jwt.claims to ''{"sub": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "role": "authenticated"}''; set role authenticated;'
\set as_b 'reset role; set request.jwt.claims to ''{"sub": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "role": "authenticated"}''; set role authenticated;'
\set as_nobody 'reset role; set request.jwt.claims to ''{"role": "authenticated"}''; set role authenticated;'
\set as_anon 'reset role; set request.jwt.claims to ''''; set role anon;'
\set as_service 'reset role; set request.jwt.claims to ''''; set role service_role;'
\set as_admin 'reset role; set request.jwt.claims to '''';'

-- ---------------------------------------------------------------- tables, grants and RLS, from the catalog
select tests.check((select array_agg(c.relname::text order by c.relname) from pg_class c
                     where c.relnamespace = 'public'::regnamespace and c.relkind = 'r')
                   = '{access,job_files,job_live,job_logs,job_secrets,job_snapshots,job_views,jobs,machines,requests,settings,usage_daily,workers}',
                   'public holds the thirteen HARPOR tables');
select tests.check((select array_agg(a.attname::text order by a.attname) from pg_attribute a
                     where a.attrelid = 'public.jobs'::regclass and a.attnum > 0 and not a.attisdropped)
                   = '{created_at,deleted_at,id,input_bytes,input_kind,input_name,machine_id,note,prompt,queue_pos,status,stopped_at,updated_at,user_id}',
                   'jobs holds only columns that its user may read, since the grant covers the whole table');
select tests.check((select string_agg(c.relname || '(' || (select string_agg(a.attname::text, ',' order by a.attname) from pg_attribute a
                                                          where a.attrelid = c.oid and a.attnum > 0 and not a.attisdropped) || ')',
                                      ' ' order by c.relname)
                      from pg_class c where c.oid in ('public.access'::regclass, 'public.settings'::regclass, 'public.usage_daily'::regclass,
                                                      'public.workers'::regclass, 'private.rate'::regclass))
                   = 'access(agents,approved,created_at,email,max_own_jobs,max_own_per_day,max_requests_per_hour,max_result_bytes,'
                     'request_at,request_note,user_id) rate(hour,n,user_id) settings(id,lean_files,max_machines,near_end_minutes,'
                     'own_log_bytes,pending_days,plan_file_limit,worker_online_s) usage_daily(day,own_jobs,user_id) workers(accepting,'
                     'ends_at,id,last_seen,local_keys,machine_id,public_key,queue_len,started_at,stopped_at,user_id,version)',
                   'access, settings, usage_daily, workers and private.rate hold only what personal workers use');
select tests.check((select bool_and(not has_table_privilege('anon', c.oid, 'select, insert, update, delete, truncate, references, trigger'))
                      from pg_class c where c.relnamespace in ('public'::regnamespace, 'private'::regnamespace) and c.relkind = 'r'),
                   'anon has no privilege on any HARPOR table (also with the old default grants)');
select tests.check((select bool_and(not has_function_privilege('anon', p.oid, 'execute'))
                      from pg_proc p where p.pronamespace in ('public'::regnamespace, 'private'::regnamespace)),
                   'anon may run no HARPOR function');
select tests.check((select bool_and(not has_table_privilege('authenticated', c.oid, 'insert, update, delete, truncate, references, trigger'))
                      from pg_class c where c.relnamespace = 'public'::regnamespace and c.relkind = 'r'),
                   'authenticated cannot write any public table');
select tests.check(not has_table_privilege('authenticated', 'public.job_secrets', 'select')
                   and not has_table_privilege('authenticated', 'private.rate', 'select, insert, update, delete'),
                   'authenticated cannot read job_secrets or private.rate');
select tests.check((select bool_and(has_table_privilege('authenticated', c.oid, 'select'))
                      from pg_class c where c.relnamespace = 'public'::regnamespace and c.relkind = 'r' and c.relname <> 'job_secrets'),
                   'authenticated may read every other public table under RLS, jobs as a whole (select=*)');
select tests.check((select bool_and(not has_function_privilege('authenticated', p.oid, 'execute'))
                      from pg_proc p where p.pronamespace = 'private'::regnamespace),
                   'authenticated may run no private function');
select tests.check((select count(*) = 19 and count(distinct p.proname) = 19
                           and bool_and(has_function_privilege('authenticated', p.oid, 'execute'))
                           and array_agg(p.proname::text) <@ array['relay', 'withdraw_request', 'delete_job', 'request_access',
                                 'storage_used', 'own_object_ok', 'job_objects', 'own_orphans', 'own_machine_bind', 'own_worker_start',
                                 'own_worker_beat', 'own_worker_stop', 'own_claim_request', 'own_answer', 'own_job_create',
                                 'own_publish', 'own_files', 'own_logs', 'own_job_delete']
                      from pg_proc p where p.pronamespace = 'public'::regnamespace),
                   'public holds exactly the 19 functions of the interface, and authenticated may run each');
select tests.check((select bool_and(not has_table_privilege('service_role', c.oid, 'select, insert, update, delete, truncate, references, trigger'))
                      from pg_class c where c.relnamespace in ('public'::regnamespace, 'private'::regnamespace) and c.relkind = 'r'),
                   'service_role has no privilege on any HARPOR table (also with the old default grants)');
select tests.check((select bool_and(not has_function_privilege('service_role', p.oid, 'execute'))
                      from pg_proc p where p.pronamespace in ('public'::regnamespace, 'private'::regnamespace)),
                   'service_role may run no HARPOR function: no part of HARPOR uses the secret key');
select tests.check((select count(*) = 6 and bool_and(not has_sequence_privilege(r::name, c.oid, 'usage, select, update'))
                      from pg_class c, unnest(array['anon', 'authenticated', 'service_role']) r
                     where c.relnamespace = 'public'::regnamespace and c.relkind = 'S'),
                   'no API role may use the two sequences');
select tests.check(not has_schema_privilege('anon', 'private', 'usage') and not has_schema_privilege('authenticated', 'private', 'usage')
                   and not has_schema_privilege('service_role', 'private', 'usage'),
                   'no API role may use schema private');
select tests.check((select bool_and(p.prosecdef and array_to_string(p.proconfig, ',') like '%search_path=%')
                      from pg_proc p where p.pronamespace = 'public'::regnamespace),
                   'every public function is security definer with a fixed search_path');
select tests.check((select bool_and(array_to_string(p.proconfig, ',') like '%search_path=%')
                      from pg_proc p where p.pronamespace = 'private'::regnamespace),
                   'every private function has a fixed search_path');
select tests.check(not exists (select 1 from pg_proc p, aclexplode(coalesce(p.proacl, acldefault('f', p.proowner))) x
                                where p.pronamespace in ('public'::regnamespace, 'private'::regnamespace) and x.grantee = 0),
                   'no HARPOR function keeps the EXECUTE that Postgres gives PUBLIC');
select tests.check((select bool_and(c.relrowsecurity) from pg_class c
                     where c.relnamespace = 'public'::regnamespace and c.relkind = 'r'),
                   'RLS is on for every public table');
select tests.check((select bool_and(roles = '{authenticated}'::name[]) from pg_policies where schemaname in ('public', 'storage')),
                   'every policy is for authenticated only');
select tests.check((select count(*) from pg_policies where schemaname = 'public' and tablename = 'job_secrets') = 0,
                   'job_secrets has no policy at all');
select tests.check((select count(*) from pg_policies where schemaname = 'public' and cmd <> 'SELECT') = 0,
                   'no policy lets users write a public table');
select tests.check((select array_agg(cmd::text order by cmd) = '{DELETE,INSERT,SELECT}'
                           and bool_and(coalesce(qual, with_check) like '%job-files%')
                      from pg_policies where schemaname = 'storage' and tablename = 'objects'),
                   'Storage has a read, an upload and a delete policy, each for job-files, and no update policy');
select tests.check((select array_agg(id order by id) from storage.buckets) = '{job-files}',
                   'job-files is the only bucket: no input file comes to the site');
create function public.made_later() returns int language sql as $$ select 1 $$;
select tests.check(not exists (select 1 from aclexplode(coalesce((select proacl from pg_proc where oid = 'public.made_later()'::regprocedure),
                                                                   acldefault('f', 10))) x
                                where x.grantee in ('anon'::regrole, 'authenticated'::regrole, 'service_role'::regrole)),
                   'a function made later in public gets no grant to an API role, also on an old project');
drop function public.made_later();

-- ---------------------------------------------------------------- sign-up, approval and request_access
insert into tests.ids (name, id) values ('A', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'), ('B', 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb');
insert into auth.users (id, email) values (tests.id('A'), 'a@example.org'), (tests.id('B'), 'b@example.org');
select tests.check((select count(*) = 2 and bool_and(not approved and agents = '{astra,opus}' and request_at is null)
                      from public.access), 'sign-up made two access rows: not approved, agents astra and opus');
select tests.expect_error($$update public.access set agents = '{astra,opus,none}' where email = 'a@example.org'$$,
                          'check constraint');
update public.access set approved = true where email = 'a@example.org';
:as_b
select tests.check((select count(*) from public.access) = 1 and (select not approved from public.access),
                   'B sees only its own access row');
select tests.expect_error($$update public.access set approved = true$$, 'permission denied');
select tests.expect_error($$select public.own_machine_bind(null, 'b-box')$$, 'waiting for approval');
select tests.expect_error($$select public.relay(gen_random_uuid(), 'cancel')$$, 'waiting for approval');
select tests.expect_error($$select public.request_access('')$$, 'write 1 to 500 characters');
select tests.expect_error($$select public.request_access(repeat('x', 501))$$, 'write 1 to 500 characters');
select tests.check(public.request_access('I work in the lab') = 'sent', 'B asks for access');
select tests.expect_error($$select public.request_access('again')$$, 'you can send one request a day');
:as_admin
select tests.check((select not approved and request_note = 'I work in the lab' and request_at > now() - interval '1 minute'
                      from public.access where email = 'b@example.org'),
                   'B''s note waits for the owner, and B could not approve itself');
update public.access set request_at = now() - interval '25 hours' where email = 'b@example.org';
:as_b
select tests.check(public.request_access('still waiting') = 'sent', 'a day later B may ask again');

-- ---------------------------------------------------------------- anon, the secret key's role, and a session without a user
:as_anon
select tests.expect_error($$select id from public.jobs$$, 'permission denied');
select tests.expect_error($$select * from public.settings$$, 'permission denied');
select tests.expect_error($$select public.relay(gen_random_uuid(), 'cancel')$$, 'permission denied');
select tests.expect_error($$select public.request_access('hello')$$, 'permission denied');
select tests.expect_error($$select public.storage_used()$$, 'permission denied');
:as_service
select tests.expect_error($$select id from public.jobs$$, 'permission denied');
select tests.expect_error($$select * from public.job_secrets$$, 'permission denied');
select tests.expect_error($$insert into public.jobs (user_id, machine_id, input_name, input_kind, input_bytes)
                            values (gen_random_uuid(), gen_random_uuid(), 'x.mp4', 'video', 1)$$, 'permission denied');
select tests.expect_error($$select public.relay(gen_random_uuid(), 'cancel')$$, 'permission denied');
select tests.expect_error($$select public.own_worker_start(gen_random_uuid(), repeat('S', 88), null, 'x')$$, 'permission denied');
select tests.expect_error($$select private.sweep()$$, 'permission denied');
:as_nobody
select tests.expect_error($$select public.relay(gen_random_uuid(), 'cancel')$$, 'sign in first');
select tests.expect_error($$select public.delete_job(gen_random_uuid())$$, 'sign in first');
select tests.expect_error($$select public.own_machine_bind(null, 'box')$$, 'sign in first');

-- ---------------------------------------------------------------- A's machine makes jobs
:as_a
insert into tests.ids (name, id) values ('m1', public.own_machine_bind(null, 'a-lab'));
insert into tests.ids (name, id) values ('w1', public.own_worker_start(tests.id('m1'), repeat('A', 88), null, 'test', '{}'));
insert into tests.ids (name, id) values ('j1', public.own_job_create(tests.id('w1'), 'cup.mp4', 'video', 1000));
insert into tests.ids (name, id) values ('j2', public.own_job_create(tests.id('w1'), 'walk.bag', 'rosbag', 2000));
insert into tests.ids (name, id) values ('j3', public.own_job_create(tests.id('w1'), 'mug.mp4', 'video', 3000));
select tests.check((select count(*) = 3 and bool_and(status = 'uploaded' and machine_id = tests.id('m1') and deleted_at is null)
                      from public.jobs), 'A''s machine made three jobs: uploaded, on that machine');
select tests.check((select count(*) from (select * from public.jobs) j) = 3, 'A reads its jobs with select *');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}')$$,
                          'the job is uploaded; wait until it stops, or cancel it');
select tests.expect_error($$update public.jobs set status = 'done'$$, 'permission denied');
select tests.expect_error($$insert into public.requests (user_id, job_id, route) values (auth.uid(), tests.id('j1'), 'start')$$,
                          'permission denied');
select public.own_publish(tests.id('w1'), tests.id('j1'), p_row => '{"status": "ready"}');
select public.own_publish(tests.id('w1'), tests.id('j2'), p_row => '{"status": "choose_topic"}');
select public.own_publish(tests.id('w1'), tests.id('j3'), p_row => '{"status": "ready"}');

-- ---------------------------------------------------------------- what relay refuses
:as_a
select tests.expect_error($$select public.relay(gen_random_uuid(), 'cancel')$$, 'no such job');
select tests.expect_error($$select public.relay(tests.id('j1'), 'delete')$$, 'unknown request');
select tests.expect_error($$select public.relay(tests.id('j1'), null)$$, 'unknown request');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '[1]')$$, 'not a JSON object of at most 128 KB');
select tests.expect_error($$select public.relay(tests.id('j1'), 'overrides', '{}', jsonb_build_object('pad', repeat('x', 131072)))$$,
                          'not a JSON object of at most 128 KB');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{"other": 1}')$$, 'bad query');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '[1]')$$, 'bad query');
select tests.expect_error($$select public.relay(tests.id('j1'), 'overrides', '{}',
  jsonb_build_object('instance', (repeat('[', 32) || repeat(']', 32))::jsonb))$$, 'nests lists and objects more than 32 deep');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"api_key": "sk-abc"}')$$,
                          'send the API key sealed, not in the body');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup", "agent": "none"}')$$,
                          'the public site runs the agents astra and opus');
select tests.expect_error($$select public.relay(tests.id('j1'), 'continue', '{}', '{"agent": false}')$$, 'astra and opus');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{}', 'a: 1')$$, 'bad calibration body');
select tests.expect_error($$select public.relay(tests.id('j2'), 'calibration_put', '{"name": "cal.yaml"}')$$, 'bad calibration body');
select tests.expect_error($$select public.relay(tests.id('j2'), 'calibration_put', '{}', '{}', 'a: 1')$$,
                          'the calibration needs a file name');
select tests.expect_error($$select public.relay(tests.id('j1'), 'topic', '{}', '{"topic": "/cam"}')$$,
                          'this job''s input is not a rosbag');
select tests.expect_error($$select public.relay(tests.id('j1'), 'parts', '{}', '{"parts": []}')$$,
                          'the job does not wait for its part list');
select tests.expect_error($$select public.relay(tests.id('j1'), 'cancel', '{}', '{}', null,
  tests.sealed(tests.id('w1'), 'opus', gen_random_uuid()::text))$$, 'this request takes no API key');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}', null, '"sk-abc"')$$,
                          'the sealed key is not an object');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}', null,
  tests.sealed(gen_random_uuid(), 'opus', gen_random_uuid()::text))$$, 'worker_restarted: the machine restarted; paste the key again');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}', null,
  tests.sealed(tests.id('w1'), 'none', gen_random_uuid()::text))$$, 'this account may use the agents astra, opus');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}', null,
  tests.sealed(tests.id('w1'), 'opus', gen_random_uuid()::text, 900))$$, 'the sealed key is malformed');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}', null,
  tests.sealed(tests.id('w1'), 'opus', 'not-a-uuid'))$$, 'the sealed key is malformed');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}', null,
  tests.sealed(tests.id('w1'), 'opus', gen_random_uuid()::text) || '{"iv": "short"}')$$, 'the sealed key is malformed');
:as_admin
update public.access set agents = '{opus}' where email = 'a@example.org';
:as_a
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}', null,
  tests.sealed(tests.id('w1'), 'astra', gen_random_uuid()::text))$$, 'this account may use the agents opus');
:as_admin
update public.access set agents = '{astra,opus}' where email = 'a@example.org';
select tests.check((select count(*) from public.requests) = 0 and (select count(*) from public.job_secrets) = 0
                   and (select count(*) from private.rate) = 0,
                   'no refused call left a request, a key or a count toward the hourly limit');

-- ---------------------------------------------------------------- one open request per job; a key goes to the machine's live worker
:as_a
insert into tests.ids (name, id) values ('ref1', gen_random_uuid());
create temp table rv1 as select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup", "api_key": null}',
  null, tests.sealed(tests.id('w1'), 'opus', tests.id('ref1')::text)) as v;
insert into tests.ids (name, n) select 'r1', (v ->> 'id')::bigint from rv1;
select tests.check((select v ? 'id' and v - 'id' = '{"worker_online": true, "accepting": true}'::jsonb from rv1),
                   'relay answers {id, worker_online, accepting}');
:as_admin
select tests.check((select state = 'pending' and sent_online and key_ref = tests.id('ref1') and user_id = tests.id('A')
                      from public.requests where id = tests.n('r1'))
                   and (select worker_id = tests.id('w1') and agent = 'opus' and job_id = tests.id('j1') and user_id = tests.id('A')
                          from public.job_secrets where ref = tests.id('ref1')),
                   'a sealed Start waits as pending, with its key sealed to the machine''s live worker');
:as_a
select tests.expect_error($$select public.relay(tests.id('j1'), 'continue')$$,
                          'an earlier request for this job still waits for its machine');
insert into tests.ids (name, n) values ('r2', (public.relay(tests.id('j1'), 'cancel') ->> 'id')::bigint);
:as_admin
select tests.check((select state = 'withdrawn' and status = 409 and response ->> 'detail' = 'withdrawn by your cancel'
                      from public.requests where id = tests.n('r1'))
                   and (select count(*) from public.job_secrets where ref = tests.id('ref1')) = 0,
                   'a cancel takes back the pending Start and its key instead of waiting behind it');
:as_a
select tests.expect_error($$select public.relay(tests.id('j1'), 'cancel')$$, 'a cancel for this job already waits');
select tests.check(public.withdraw_request(tests.n('r2')) = 'withdrawn', 'A withdraws its cancel');
select tests.expect_error($$select public.withdraw_request(tests.n('r2'))$$, 'the request is no longer waiting');
insert into tests.ids (name, id) values ('ref3', gen_random_uuid());
insert into tests.ids (name, n) values ('r3', (public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}',
  null, tests.sealed(tests.id('w1'), 'opus', tests.id('ref3')::text)) ->> 'id')::bigint);
select tests.check((select (v ->> 'id')::bigint = tests.n('r3') and v -> 'sealed' ->> 'agent' = 'opus'
                      from public.own_claim_request(tests.id('w1')) v), 'the machine''s worker takes the Start with its key');
select tests.expect_error($$select public.relay(tests.id('j1'), 'continue')$$, 'still waits for its machine');
insert into tests.ids (name, n) values ('r4', (public.relay(tests.id('j1'), 'cancel') ->> 'id')::bigint);
select tests.check((select count(*) from public.own_claim_request(tests.id('w1'))) = 0,
                   'a cancel may be sent while the Start is in progress, and waits until the Start is answered');
select public.own_answer(tests.id('w1'), tests.n('r3'), 200, '{"status": "queued"}');
select tests.check((select (v ->> 'id')::bigint from public.own_claim_request(tests.id('w1')) v) = tests.n('r4'),
                   'then the worker takes the cancel');
select public.own_answer(tests.id('w1'), tests.n('r4'), 200, '{"status": "cancelled"}');

-- ---------------------------------------------------------------- near the end of a Slurm job, and while the machine is offline
:as_a
select public.own_worker_beat(tests.id('w1'), 0, false, now() + interval '30 minutes');
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}', null,
  tests.sealed(tests.id('w1'), 'opus', gen_random_uuid()::text))$$, 'worker_ending: the machine stops soon; send the request without the key');
create temp table rv5 as select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup", "key_later": true}') as v;
select tests.check((select v - 'id' = '{"worker_online": true, "accepting": false}'::jsonb from rv5)
                   and (select state = 'pending' and not sent_online and key_ref is null from public.requests
                         where id = (select (v ->> 'id')::bigint from rv5)),
                   'near the end, a Start with key_later waits as pending, sent while the machine took no builds');
select public.withdraw_request((select (v ->> 'id')::bigint from rv5));
select public.own_worker_beat(tests.id('w1'), 0, true, null);
:as_admin
update public.workers set last_seen = now() - interval '31 seconds' where id = tests.id('w1');
:as_a
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}', null,
  tests.sealed(tests.id('w1'), 'opus', gen_random_uuid()::text))$$,
  'worker_offline: the machine that runs this job is offline; send the request without the key');
create temp table rv6 as select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup", "key_later": true}') as v;
select tests.check((select v - 'id' = '{"worker_online": false, "accepting": false}'::jsonb from rv6)
                   and (select state = 'pending' and not sent_online from public.requests
                         where id = (select (v ->> 'id')::bigint from rv6)),
                   'a worker not seen for 30 s is offline: a Start with key_later waits as pending for its machine');
select public.withdraw_request((select (v ->> 'id')::bigint from rv6));

-- ---------------------------------------------------------------- a key never goes to an old worker or to another machine
:as_a
insert into tests.ids (name, id) values ('w2', public.own_worker_start(tests.id('m1'), repeat('E', 88), null, 'test', '{}'));
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}', null,
  tests.sealed(tests.id('w1'), 'opus', gen_random_uuid()::text))$$, 'worker_restarted');
insert into tests.ids (name, id) values ('m2', public.own_machine_bind(null, 'a-laptop'));
insert into tests.ids (name, id) values ('w3', public.own_worker_start(tests.id('m2'), repeat('F', 88), null, 'test', '{}'));
select tests.expect_error($$select public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}', null,
  tests.sealed(tests.id('w3'), 'opus', gen_random_uuid()::text))$$, 'worker_restarted');
select tests.check(true, 'a key for a job of a-lab cannot be sealed to the live worker of a-laptop, A''s other machine');
insert into tests.ids (name, id) values ('ref7', gen_random_uuid());
insert into tests.ids (name, n) values ('r7', (public.relay(tests.id('j1'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}', null,
  tests.sealed(tests.id('w2'), 'opus', tests.id('ref7')::text)) ->> 'id')::bigint);
:as_admin
select tests.check((select worker_id from public.job_secrets where ref = tests.id('ref7')) = tests.id('w2'),
                   'the key goes to the new worker of a-lab');
:as_a
select public.withdraw_request(tests.n('r7'));
:as_admin
select tests.check((select count(*) from public.job_secrets where ref = tests.id('ref7')) = 0, 'a withdrawn request takes its key along');

-- ---------------------------------------------------------------- the hourly request count survives a job delete
:as_admin
update public.access set max_requests_per_hour =
  (select n from private.rate where user_id = tests.id('A') and hour = date_trunc('hour', now())) + 1
  where email = 'a@example.org';
:as_a
insert into tests.ids (name, n) values ('r8', (public.relay(tests.id('j1'), 'continue', '{}', '{"from": "sam3d"}') ->> 'id')::bigint);
select public.withdraw_request(tests.n('r8'));
select tests.expect_error($$select public.relay(tests.id('j1'), 'continue')$$, 'too many requests: at most \d+ an hour');
select tests.check(public.delete_job(tests.id('j1')) = 'deleting', 'A deletes a stopped job while its machine is online');
select tests.expect_error($$select public.relay(tests.id('j3'), 'continue')$$, 'too many requests: at most');
select tests.check(true, 'deleting a job did not reset the hourly request count');
:as_admin
update public.access set max_requests_per_hour = 60 where email = 'a@example.org';

-- ---------------------------------------------------------------- delete_job
:as_admin
select tests.check((select count(*) = 1 and bool_and(state = 'pending' and sent_online and user_id = tests.id('A'))
                      from public.requests where job_id = tests.id('j1') and route = 'delete')
                   and (select status = 'deleting' and deleted_at is not null from public.jobs where id = tests.id('j1')),
                   'the delete waits for the machine as a request, sent while the machine was online');
:as_a
select tests.check((select count(*) from public.jobs where id = tests.id('j1')) = 0, 'A no longer sees the deleted job');
select tests.expect_error($$select public.relay(tests.id('j1'), 'cancel')$$, 'no such job');
select tests.expect_error($$select public.delete_job(tests.id('j1'))$$, 'no such job');
select tests.expect_error($$select public.withdraw_request((select id from public.requests where job_id = tests.id('j1') and route = 'delete'))$$,
                          'the request is no longer waiting');
select tests.check(true, 'a delete cannot be withdrawn');
select tests.expect_error($$select public.delete_job(gen_random_uuid())$$, 'no such job');
insert into tests.ids (name, id) values ('ref9', gen_random_uuid());
insert into tests.ids (name, n) values ('r9', (public.relay(tests.id('j3'), 'start', '{}', '{"mode": "auto", "prompt": "mug"}', null,
  tests.sealed(tests.id('w2'), 'opus', tests.id('ref9')::text)) ->> 'id')::bigint);
select tests.check(public.delete_job(tests.id('j3')) = 'deleting', 'a stopped job is deleted while a request for it waits');
:as_admin
select tests.check((select state = 'withdrawn' and status = 409 and response ->> 'detail' = 'the job was deleted'
                      from public.requests where id = tests.n('r9'))
                   and (select count(*) from public.job_secrets where ref = tests.id('ref9')) = 0,
                   'the delete withdrew the pending Start and dropped its key');
:as_a
select public.own_publish(tests.id('w2'), tests.id('j2'), p_row => '{"status": "analyzing"}');
select tests.expect_error($$select public.delete_job(tests.id('j2'))$$, 'the job is analyzing; cancel it first');
select public.own_publish(tests.id('w2'), tests.id('j2'), p_row => '{"status": "queued"}');
select tests.expect_error($$select public.delete_job(tests.id('j2'))$$, 'the job is queued; cancel it first');
select public.own_publish(tests.id('w2'), tests.id('j2'), p_row => '{"status": "running"}');
select tests.expect_error($$select public.delete_job(tests.id('j2'))$$, 'the job is running; cancel it first');
insert into tests.ids (name, n) values ('r10', (public.relay(tests.id('j2'), 'cancel') ->> 'id')::bigint);
:as_admin
update public.workers set last_seen = now() - interval '31 seconds' where id = tests.id('w2');
:as_a
select tests.check(public.delete_job(tests.id('j2')) = 'deleting', 'a running job whose machine is offline can be deleted');
:as_admin
select tests.check((select state from public.requests where id = tests.n('r10')) = 'withdrawn'
                   and (select state = 'pending' and not sent_online from public.requests
                         where job_id = tests.id('j2') and route = 'delete'),
                   'its pending cancel was withdrawn, and the delete waits for the machine, sent while it was offline');

-- ---------------------------------------------------------------- finished requests keep little
:as_admin
insert into public.jobs (id, user_id, machine_id, input_name, input_kind, input_bytes, status) values
  ('66666666-6666-6666-6666-666666666666', tests.id('A'), tests.id('m1'), 'r.bag', 'rosbag', 10, 'ready');
insert into public.requests (user_id, job_id, route, query, body, body_text) values
  (tests.id('A'), '66666666-6666-6666-6666-666666666666', 'calibration_put', '{"name": "c.yaml"}',
   jsonb_build_object('pad', repeat('x', 2000)), repeat('y', 5000));
insert into public.requests (user_id, job_id, route, body) values
  (tests.id('A'), '66666666-6666-6666-6666-666666666666', 'continue', '{"from": "agent"}');
insert into tests.ids (name, n)
  select case route when 'calibration_put' then 'rbig' else 'rsmall' end, id from public.requests
   where job_id = '66666666-6666-6666-6666-666666666666';
update public.requests set state = 'done', done_at = now(), status = 200, response = '{"status": "queued"}'
 where job_id = '66666666-6666-6666-6666-666666666666';
select tests.check((select body = '{}'::jsonb and body_text is null from public.requests where id = tests.n('rbig'))
                   and (select body = '{"from": "agent"}'::jsonb from public.requests where id = tests.n('rsmall')),
                   'a finished request loses its calibration text and a body over 1 KB; a small body stays');
update public.requests set state = 'pending', done_at = null, body = jsonb_build_object('pad', repeat('x', 2000))
 where id = tests.n('rbig');
select tests.check((select octet_length(body::text) > 1024 from public.requests where id = tests.n('rbig')),
                   'a request that waits keeps its whole body');
delete from public.jobs where id = '66666666-6666-6666-6666-666666666666';

-- ---------------------------------------------------------------- the sweep, which pg_cron runs as the superuser
:as_a
select tests.expect_error($$select private.sweep()$$, 'permission denied');
:as_admin
insert into tests.ids (name, id) values
  ('s15', '5a5a5a5a-0000-0000-0000-000000000015'), ('s13', '5a5a5a5a-0000-0000-0000-000000000013'),
  ('srun', '5a5a5a5a-0000-0000-0000-000000000030'), ('snull', '5a5a5a5a-0000-0000-0000-000000000099'),
  ('sdel8', '5a5a5a5a-0000-0000-0000-000000000008'), ('sdel6', '5a5a5a5a-0000-0000-0000-000000000006');
insert into public.jobs (id, user_id, machine_id, input_name, input_kind, input_bytes, status, stopped_at, updated_at, deleted_at) values
  (tests.id('s15'), tests.id('A'), tests.id('m1'), 'a.mp4', 'video', 1, 'done', now() - interval '15 days', now(), null),
  (tests.id('s13'), tests.id('A'), tests.id('m1'), 'b.mp4', 'video', 1, 'failed', now() - interval '13 days', now(), null),
  (tests.id('srun'), tests.id('A'), tests.id('m1'), 'c.mp4', 'video', 1, 'running', now() - interval '30 days',
   now() - interval '30 days', null),
  (tests.id('snull'), tests.id('A'), tests.id('m1'), 'd.mp4', 'video', 1, 'cancelled', null, now() - interval '15 days', null),
  (tests.id('sdel8'), tests.id('A'), tests.id('m1'), 'e.mp4', 'video', 1, 'deleting', null, now() - interval '8 days',
   now() - interval '8 days'),
  (tests.id('sdel6'), tests.id('A'), tests.id('m1'), 'f.mp4', 'video', 1, 'deleting', null, now() - interval '6 days',
   now() - interval '6 days');
insert into public.job_logs (job_id, user_id, stage, pos, text)
  select id, user_id, 'frames', 1, 'x' from public.jobs where id in (tests.id('s15'), tests.id('s13'), tests.id('srun'), tests.id('snull'));
with q as (insert into public.requests (user_id, job_id, route, created_at) values
             (tests.id('A'), tests.id('s13'), 'continue', now() - interval '8 days'),
             (tests.id('A'), tests.id('srun'), 'cancel', now() - interval '6 days'),
             (tests.id('A'), tests.id('sdel8'), 'delete', now() - interval '8 days')
           returning id, route)
insert into tests.ids (name, n) select 'q_' || route, id from q;
with q as (insert into public.requests (user_id, job_id, route, body) values
             (tests.id('A'), tests.id('s15'), 'start', '{"n": 1}'), (tests.id('A'), tests.id('s15'), 'continue', '{"n": 2}'),
             (tests.id('A'), tests.id('s15'), 'overrides', '{"n": 3}'), (tests.id('A'), tests.id('s15'), 'parts', '{"n": 4}'),
             (tests.id('A'), tests.id('s15'), 'topic', '{"n": 5}')
           returning id, body ->> 'n' as n)
insert into tests.ids (name, n) select 'f' || n, id from q;
update public.requests set state = 'done', status = 200,
       done_at = case route when 'start' then now() - interval '8 days' when 'continue' then now() - interval '6 days'
                 when 'overrides' then now() - interval '2 hours' else now() - interval '30 minutes' end,
       response = case when route in ('overrides', 'parts') then jsonb_build_object('pad', repeat('z', 2000))
                  else '{"status": "queued"}'::jsonb end
 where job_id = tests.id('s15') and route <> 'topic';
update public.requests set state = 'withdrawn', status = 409, done_at = now() - interval '8 days'
 where job_id = tests.id('s15') and route = 'topic';
insert into public.job_secrets (ref, job_id, user_id, worker_id, agent, epk, iv, ct, created_at) values
  ('5a5a5a5a-0000-0000-0000-0000000000a1', tests.id('srun'), tests.id('A'), tests.id('w2'), 'opus', repeat('B', 88),
   repeat('C', 16), repeat('D', 40), now() - interval '2 hours'),
  ('5a5a5a5a-0000-0000-0000-0000000000a2', tests.id('srun'), tests.id('A'), tests.id('w2'), 'opus', repeat('B', 88),
   repeat('C', 16), repeat('D', 40), now() - interval '30 minutes');
insert into private.rate (user_id, hour, n) values (tests.id('A'), date_trunc('hour', now()) - interval '3 days', 5);
insert into storage.objects (bucket_id, name, metadata) values
  ('job-files', tests.obj(tests.id('A'), tests.id('s15'), repeat('a', 64), 'run/x.glb'), '{"size": 3}'),
  ('job-files', tests.obj(tests.id('A'), tests.id('srun'), repeat('a', 64), 'run/x.glb'), '{"size": 3}'),
  ('job-files', tests.obj(tests.id('A'), tests.id('sdel6'), repeat('a', 64), 'run/x.glb'), '{"size": 3}'),
  ('job-files', tests.obj(tests.id('A'), tests.id('sdel8'), repeat('a', 64), 'run/x.glb'), '{"size": 4}');
insert into tests.ids (name, n) values ('objects', (select count(*) from storage.objects));
select tests.check(private.sweep() = 2, 'the sweep answers how many requests expired: the two that waited 8 days');
select tests.check((select state = 'expired' and status = 410
                           and response ->> 'detail' = 'the request waited more than 7 days for its machine; send it again'
                      from public.requests where id = tests.n('q_continue')),
                   'a request that waited 8 days for its machine expired with 410');
select tests.check((select state from public.requests where id = tests.n('q_cancel')) = 'pending', 'one that waited 6 days still waits');
select tests.check((select count(*) from public.jobs where id = tests.id('sdel8')) = 0
                   and (select count(*) from public.requests where id = tests.n('q_delete')) = 0
                   and (select count(*) from public.jobs where id = tests.id('sdel6')) = 1,
                   'a job that waited 8 days in deleting goes with its delete request; one that waited 6 days stays');
select tests.check((select count(*) from public.job_secrets where ref = '5a5a5a5a-0000-0000-0000-0000000000a1') = 0
                   and (select count(*) from public.job_secrets where ref = '5a5a5a5a-0000-0000-0000-0000000000a2') = 1,
                   'a sealed key that no worker took goes after an hour; a newer one stays');
select tests.check((select count(*) from public.requests where id in (tests.n('f1'), tests.n('f5'))) = 0
                   and (select response = '{"status": "queued"}'::jsonb from public.requests where id = tests.n('f2')),
                   'finished requests go after 7 days, done or withdrawn; one finished 6 days ago stays with its answer');
select tests.check((select response is null from public.requests where id = tests.n('f3'))
                   and (select octet_length(response::text) > 1024 from public.requests where id = tests.n('f4')),
                   'an answer over 1 KB goes an hour after it came; one that came 30 minutes ago stays');
select tests.check((select count(*) from private.rate where hour < now() - interval '2 days') = 0
                   and (select count(*) from private.rate where user_id = tests.id('A') and hour = date_trunc('hour', now())) = 1,
                   'hourly counts go after 2 days; this hour''s count stays');
select tests.check((select array_agg(job_id order by job_id) from public.job_logs where job_id::text like '5a5a5a5a-%')
                   = array[tests.id('s13'), tests.id('srun')],
                   'log text goes 14 days after its job stopped, or last changed; it stays at 13 days and while the job runs');
select tests.check((select count(*) from storage.objects) = tests.n('objects'),
                   'the sweep deletes no Storage object (a SQL delete would leave the file behind)');
select tests.check(private.sweep() = 0, 'a second sweep finds nothing more to expire');

-- ---------------------------------------------------------------- the owner's list of folders that no worker can clean up
:as_admin
insert into auth.users (id, email) values ('eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee', 'e@example.org');
insert into storage.objects (bucket_id, name, metadata) values
  ('job-files', 'eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee/e0e0e0e0-0000-0000-0000-000000000001/0123456789abcdef/run/x.glb', '{"size": 5}');
delete from auth.users where email = 'e@example.org';
select tests.check((select array_agg(folder || ' ' || objects || ' ' || bytes || ' ' || account_gone order by folder)
                      from private.orphan_folders())
                   = array['aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/5a5a5a5a-0000-0000-0000-000000000008/ 1 4 false',
                           'eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee/e0e0e0e0-0000-0000-0000-000000000001/ 1 5 true'],
                   'orphan_folders lists the folders of a job row and of an account that went, never one of a job that is there');
:as_a
select tests.expect_error($$select * from private.orphan_folders()$$, 'permission denied');
:as_admin
delete from storage.objects where name like '%/5a5a5a5a-%' or name like 'eeeeeeee-%';
delete from public.jobs where id::text like '5a5a5a5a-%';

-- ---------------------------------------------------------------- Realtime
:as_admin
select tests.check((select array_agg(tablename::text order by tablename) from pg_publication_tables
                     where pubname = 'supabase_realtime') = '{job_live,job_logs,requests}',
                   'Realtime publishes job_live, job_logs and requests only');
\echo ALL CHECKS PASSED
