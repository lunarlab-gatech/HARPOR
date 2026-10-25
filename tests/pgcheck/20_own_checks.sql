-- Checks of the personal workers on users' own GPU machines, on the local stand-in, after 10_checks.sql in the
-- same database: machines, workers and the own_* functions, Storage uploads and deletes, job_objects and
-- own_orphans. C and D are approved users with machines of their own.
\set ON_ERROR_STOP on
\set QUIET on
\set as_c 'reset role; set request.jwt.claims to ''{"sub": "cccccccc-cccc-cccc-cccc-cccccccccccc", "role": "authenticated"}''; set role authenticated;'
\set as_d 'reset role; set request.jwt.claims to ''{"sub": "dddddddd-dddd-dddd-dddd-dddddddddddd", "role": "authenticated"}''; set role authenticated;'
\set as_admin 'reset role; set request.jwt.claims to '''';'

:as_admin
insert into tests.ids (name, id) values ('C', 'cccccccc-cccc-cccc-cccc-cccccccccccc'), ('D', 'dddddddd-dddd-dddd-dddd-dddddddddddd');
insert into auth.users (id, email) values (tests.id('C'), 'c@example.org'), (tests.id('D'), 'd@example.org');
select tests.check((select agents = '{astra,opus}' and not approved and max_own_jobs = 20 and max_own_per_day = 20
                      from public.access where email = 'c@example.org'),
                   'a new account: not approved, agents astra and opus, 20 jobs at once and 20 a day');
select tests.check(to_regprocedure('public.own_object_ok(text)') is not null
                   and to_regprocedure('public.own_object_ok(text, boolean)') is null
                   and to_regprocedure('public.job_objects(uuid, integer)') is not null
                   and to_regprocedure('public.own_orphans(uuid, integer)') is not null,
                   'own_object_ok takes the object name only; job_objects and own_orphans take a limit');

-- ---------------------------------------------------------------- C binds a machine and starts a worker
:as_c
select tests.expect_error($$select public.own_machine_bind(null, 'lab-box')$$, 'waiting for approval');
:as_admin
update public.access set approved = true where email in ('c@example.org', 'd@example.org');
:as_c
select tests.expect_error($$select public.own_machine_bind(gen_random_uuid(), 'box')$$, 'no such machine');
select tests.expect_error($$select public.own_machine_bind(null, 'bad name!')$$, 'machine name');
insert into tests.ids (name, id) values ('mc', public.own_machine_bind(null, 'lab-box'));
select tests.check(public.own_machine_bind(tests.id('mc'), 'lab-box-2') = tests.id('mc'), 'login again renames the machine');
select tests.check((select count(*) = 1 and bool_and(name = 'lab-box-2') from public.machines), 'C reads its machine''s new name');
select tests.expect_error($$select public.own_worker_start(tests.id('mc'), repeat('C', 88), null, 'test', '{opus,none}')$$,
                          'check constraint');
select tests.expect_error($$select public.own_worker_start(tests.id('mc'), 'short', null, 'test', '{}')$$, 'check constraint');
insert into tests.ids (name, id) values ('wc', public.own_worker_start(tests.id('mc'), repeat('C', 88), null, 'test', '{opus}'));
select tests.check((select user_id = auth.uid() and machine_id = tests.id('mc') and local_keys = '{opus}' and accepting
                           and stopped_at is null from public.workers where id = tests.id('wc')),
                   'C''s worker row names its user, its machine and the agents whose key the machine holds');
select tests.expect_error($$insert into public.workers (user_id, machine_id, public_key) values (auth.uid(), tests.id('mc'), repeat('Z', 88))$$,
                          'permission denied');
select tests.expect_error($$insert into public.machines (user_id, name) values (auth.uid(), 'x')$$, 'permission denied');
select public.own_worker_beat(tests.id('wc'), 3, false, now() + interval '20 minutes');
select tests.check((select queue_len = 3 and not accepting and ends_at > now() from public.workers where id = tests.id('wc')),
                   'the heartbeat writes the queue length, accepting and the end of the Slurm job');
select public.own_worker_beat(tests.id('wc'), -1, null);
select tests.check((select queue_len = 0 and accepting and ends_at > now() from public.workers where id = tests.id('wc')),
                   'a heartbeat without an end keeps the known end; a bad queue length counts as 0, no accepting as true');

-- ---------------------------------------------------------------- a job, published by C's worker
insert into tests.ids (name, id) values ('jc', public.own_job_create(tests.id('wc'), 'mug.mp4', 'video', 30000000000));
select tests.check((select status = 'uploaded' and machine_id = tests.id('mc') and input_bytes = 30000000000 from public.jobs
                     where id = tests.id('jc')), 'a 30 GB input on the machine makes a job; the input never comes here');
select tests.check((select own_jobs from public.usage_daily where user_id = auth.uid() and day = current_date) = 1,
                   'the job counts in usage_daily.own_jobs');
select tests.expect_error($$select public.own_job_create(tests.id('wc'), 'mug.bag', 'video', 1)$$, 'the input is a video');
select tests.expect_error($$select public.own_job_create(tests.id('wc'), 'mug..x.mp4', 'video', 1)$$, 'no two dots in a row');
select tests.expect_error($$select public.own_job_create(tests.id('wc'), '.hidden.mp4', 'video', 1)$$, 'file name');
select tests.expect_error($$select public.own_job_create(tests.id('wc'), 'empty.mp4', 'video', 0)$$, 'the file is empty');
select public.own_publish(tests.id('wc'), tests.id('jc'), '{"status": "analyzing"}', '{"status": "analyzing"}',
                          '{"frames": {"status": 404, "body": {"detail": "no frames yet"}}}', '{"status": "analyzing"}');
select tests.check((select status from public.jobs where id = tests.id('jc')) = 'analyzing'
                   and (select count(*) from public.job_views where job_id = tests.id('jc')) = 1
                   and (select count(*) from public.job_live where job_id = tests.id('jc')) = 1
                   and (select status from public.job_snapshots where job_id = tests.id('jc') and path = 'frames') = 404,
                   'own_publish writes the row, view, live state and snapshot');
select tests.expect_error($$select public.own_publish(tests.id('wc'), tests.id('jc'), p_row => '{"machine_id": null}')$$, 'the row takes');
select tests.expect_error($$select public.own_publish(tests.id('wc'), tests.id('jc'), p_row => '{"status": "deleting"}')$$, 'bad status');
select tests.expect_error($$select public.own_publish(tests.id('wc'), tests.id('jc'), p_row => '{"status": "uploading"}')$$, 'bad status');
select tests.expect_error($$select public.own_publish(tests.id('wc'), tests.id('jc'), p_snapshots => '{"other": {}}')$$, 'snapshots are');

-- ---------------------------------------------------------------- D cannot reach C's machine, worker or job
:as_d
insert into tests.ids (name, id) values ('md', public.own_machine_bind(null, 'd-box'));
insert into tests.ids (name, id) values ('wd', public.own_worker_start(tests.id('md'), repeat('D', 88), null, 'test', '{}'));
select tests.expect_error($$select public.own_worker_start(tests.id('mc'), repeat('D', 88), null, 'x', '{}')$$, 'no such machine');
select tests.expect_error($$select public.own_machine_bind(tests.id('mc'), 'mine')$$, 'no such machine');
select tests.expect_error($$select public.own_publish(tests.id('wc'), tests.id('jc'), '{}')$$, 'no such worker');
select tests.expect_error($$select public.own_publish(tests.id('wd'), tests.id('jc'), '{}')$$, 'no such job on this machine');
select tests.expect_error($$select public.own_job_create(tests.id('wc'), 'x.mp4', 'video', 1)$$, 'no such worker');
select tests.check((select count(*) from public.machines) = 1 and (select count(*) from public.workers) = 1
                   and (select count(*) from public.jobs) = 0, 'D sees only its own machine and worker, and no job of C');
select tests.expect_error($$insert into storage.objects (bucket_id, name, metadata)
  values ('job-files', tests.obj(tests.id('C'), tests.id('jc'), repeat('a', 64), 'run/posed.glb'), '{"size": 10}')$$,
  'row-level security');
select tests.expect_error($$insert into storage.objects (bucket_id, name, metadata)
  values ('job-files', tests.obj(tests.id('D'), tests.id('jc'), repeat('a', 64), 'run/posed.glb'), '{"size": 10}')$$,
  'row-level security');

-- ---------------------------------------------------------------- C's second machine cannot touch the first one's job
:as_c
insert into tests.ids (name, id) values ('mc2', public.own_machine_bind(null, 'laptop'));
insert into tests.ids (name, id) values ('wc2', public.own_worker_start(tests.id('mc2'), repeat('L', 88), null, 'test', '{}'));
select tests.expect_error($$select public.own_publish(tests.id('wc2'), tests.id('jc'), '{}')$$, 'no such job on this machine');
select tests.check((select count(*) from public.workers where stopped_at is null) = 2,
                   'starting a worker on the laptop did not stop the one on lab-box');
insert into tests.ids (name, id) values ('jl', public.own_job_create(tests.id('wc2'), 'desk.mp4', 'video', 10));
select tests.check((select machine_id from public.jobs where id = tests.id('jl')) = tests.id('mc2'), 'the laptop''s job is on the laptop');

-- ---------------------------------------------------------------- Storage uploads: own folder, a job of the caller, under the quota
:as_admin
insert into tests.sha values ('sha1', '0123456789abcdef' || repeat('1', 48)), ('sha2', 'fedcba9876543210' || repeat('2', 48)),
                             ('sha3', '00112233aabbccdd' || repeat('3', 48));
:as_c
select tests.check(public.own_object_ok(tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha1'), 'run/posed.glb'))
                   and not public.own_object_ok(tests.obj(tests.id('D'), tests.id('jc'), (select v from tests.sha where name = 'sha1'), 'run/posed.glb'))
                   and not public.own_object_ok(tests.obj(auth.uid(), gen_random_uuid(), repeat('a', 64), 'run/posed.glb'))
                   and not public.own_object_ok(auth.uid()::text || '/' || tests.id('jc')::text || '/0123456789abcdef/../x.glb')
                   and not public.own_object_ok(auth.uid()::text || '/' || tests.id('jc')::text || '/0123456789abcdef//x.glb')
                   and not public.own_object_ok(auth.uid()::text || '/' || tests.id('jc')::text || '/short/run/x.glb'),
                   'own_object_ok: yes for <C>/<C''s job>/<16 hex>/<path>; no for another folder, an unknown job, "..", "//" or a short hash');
insert into storage.objects (bucket_id, name, metadata)
  values ('job-files', tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha1'), 'run/posed.glb'),
          '{"size": 10}');
select tests.check(true, 'C''s worker uploads into its job''s folder');
select tests.expect_error($$insert into storage.objects (bucket_id, name, metadata)
  values ('job-files', tests.obj(auth.uid(), gen_random_uuid(), repeat('a', 64), 'run/posed.glb'), '{"size": 10}')$$,
  'row-level security');
select tests.expect_error($$insert into storage.objects (bucket_id, name, metadata)
  values ('job-files', auth.uid()::text || '/' || tests.id('jc')::text || '/0123456789abcdef/../x.glb', '{"size": 10}')$$,
  'row-level security');
select tests.expect_error($$insert into storage.objects (bucket_id, name, metadata)
  values ('job-files', auth.uid()::text || '/' || tests.id('jc')::text || '/short/run/x.glb', '{"size": 10}')$$,
  'row-level security');
update storage.objects set name = name || '.x' where bucket_id = 'job-files';
select tests.check((select count(*) from storage.objects where name like '%.x') = 0,
                   'an update of an object changes nothing: there is no update policy, so nobody renames or overwrites one');
select tests.check(public.storage_used() = 10, 'storage_used counts C''s objects');
select public.own_files(tests.id('wc'), tests.id('jc'), jsonb_build_array(jsonb_build_object(
  'path', 'run/posed.glb', 'object', tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha1'), 'run/posed.glb'),
  'bytes', 10, 'sha256', (select v from tests.sha where name = 'sha1'), 'role', 'result', 'content_type', 'model/gltf-binary')));
select tests.check((select count(*) from public.job_files where job_id = tests.id('jc')) = 1, 'own_files lists the uploaded object');
select tests.expect_error($$select public.own_files(tests.id('wc'), tests.id('jc'), jsonb_build_array(jsonb_build_object(
  'path', 'run/posed.glb', 'object', tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha1'), 'run/posed.glb'),
  'bytes', 11, 'sha256', (select v from tests.sha where name = 'sha1'), 'role', 'result', 'content_type', 'model/gltf-binary')))$$,
  'holds no object');
select tests.expect_error($$select public.own_files(tests.id('wc'), tests.id('jc'), jsonb_build_array(jsonb_build_object(
  'path', 'refined.glb', 'object', tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha1'), 'run/posed.glb'),
  'bytes', 10, 'sha256', (select v from tests.sha where name = 'sha1'), 'role', 'result', 'content_type', 'model/gltf-binary')))$$,
  'check constraint');
:as_admin
update public.access set max_result_bytes = 10 where email = 'c@example.org';
:as_c
select tests.expect_error($$insert into storage.objects (bucket_id, name, metadata)
  values ('job-files', tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha2'), 'run/x.glb'), '{"size": 5}')$$,
  'row-level security');
select tests.check(true, 'no upload once C''s objects reach max_result_bytes');
:as_admin
update public.access set max_result_bytes = 2147483648 where email = 'c@example.org';

-- ---------------------------------------------------------------- job_objects and own_orphans
:as_c
insert into storage.objects (bucket_id, name, metadata) values
  ('job-files', tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha2'), 'run/old.glb'), '{"size": 5}'),
  ('job-files', tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha3'), 'run/new.glb'), '{"size": 5}'),
  ('job-files', tests.obj(auth.uid(), tests.id('jl'), (select v from tests.sha where name = 'sha2'), 'run/desk.glb'), '{"size": 5}');
:as_d
insert into tests.ids (name, id) values ('jd', public.own_job_create(tests.id('wd'), 'd.mp4', 'video', 10));
insert into storage.objects (bucket_id, name, metadata)
  values ('job-files', tests.obj(auth.uid(), tests.id('jd'), repeat('d', 64), 'run/d.glb'), '{"size": 5}');
:as_admin
insert into storage.objects (bucket_id, name, metadata, created_at) values
  ('job-files', tests.obj(tests.id('C'), '99999999-9999-9999-9999-999999999999', repeat('9', 64), 'run/x.glb'), '{"size": 2}',
   now() - interval '3 hours');
update storage.objects set created_at = now() - interval '2 hours'
 where name like '%/run/posed.glb' or name like '%/run/old.glb' or name like '%/run/desk.glb' or name like '%/run/d.glb';
:as_c
select tests.check((select array_agg(name order by name) from public.job_objects(tests.id('jc')))
                   = array[tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha3'), 'run/new.glb'),
                           tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha1'), 'run/posed.glb'),
                           tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha2'), 'run/old.glb')]
                   and (select name from public.job_objects(tests.id('jc')) offset 2) like '%/run/new.glb',
                   'job_objects lists every object of the job, named in job_files or not, oldest first');
select tests.check((select count(*) from public.job_objects(tests.id('jc'), 2)) = 2
                   and (select count(*) from public.job_objects(tests.id('jc'), 0)) = 1,
                   'job_objects gives at most p_limit names, 1 to 1000');
select tests.expect_error($$select * from public.job_objects(tests.id('jd'))$$, 'no such job');
select tests.expect_error($$select * from public.job_objects(gen_random_uuid())$$, 'no such job');
select tests.check((select array_agg(name order by name) from public.own_orphans(tests.id('wc')))
                   = (select array_agg(x order by x) from unnest(array[
                       tests.obj(auth.uid(), '99999999-9999-9999-9999-999999999999', repeat('9', 64), 'run/x.glb'),
                       tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha2'), 'run/old.glb')]) x)
                   and (select name from public.own_orphans(tests.id('wc')) limit 1) like '%/99999999-%',
                   'own_orphans of lab-box: the old object no file names and the folder of a gone row, oldest first; '
                   'not the named object, not the new one, not the laptop''s job, nothing of D');
select tests.check((select array_agg(name order by name) from public.own_orphans(tests.id('wc2')))
                   = (select array_agg(x order by x) from unnest(array[
                       tests.obj(auth.uid(), '99999999-9999-9999-9999-999999999999', repeat('9', 64), 'run/x.glb'),
                       tests.obj(auth.uid(), tests.id('jl'), (select v from tests.sha where name = 'sha2'), 'run/desk.glb')]) x),
                   'own_orphans of the laptop: its own job''s old object and the folder of a gone row, not lab-box''s');
select tests.check((select count(*) from public.own_orphans(tests.id('wc'), 1)) = 1, 'own_orphans gives at most p_limit names');
:as_d
select tests.check((select array_agg(name) from public.own_orphans(tests.id('wd')))
                   = array[tests.obj(auth.uid(), tests.id('jd'), repeat('d', 64), 'run/d.glb')],
                   'D''s orphans are D''s own');
select tests.expect_error($$select * from public.own_orphans(tests.id('wc'))$$, 'no such worker');

-- ---------------------------------------------------------------- Storage deletes: the caller's own folder only
:as_d
delete from storage.objects where bucket_id = 'job-files' and name like tests.id('C')::text || '/%';
delete from storage.objects;   -- no WHERE, so only the delete policy picks the rows (not the read policy too)
:as_admin
select tests.check((select count(*) from storage.objects where name like tests.id('C')::text || '/%') = 5
                   and (select count(*) from storage.objects where name like tests.id('D')::text || '/%') = 0,
                   'D deletes its own object and none of C''s, also with a delete that names no rows');
:as_c
delete from storage.objects where bucket_id = 'job-files'
  and name in (select name from public.own_orphans(tests.id('wc')) union select name from public.own_orphans(tests.id('wc2')));
:as_admin
select tests.check((select count(*) from storage.objects where name like tests.id('C')::text || '/%') = 2
                   and (select count(*) from storage.objects where name like '%/99999999-%') = 0,
                   'C deletes its orphans, also in the folder of a gone row');
update public.access set approved = false where email = 'c@example.org';
:as_c
delete from storage.objects where bucket_id = 'job-files' and name like '%/run/new.glb';
:as_admin
select tests.check((select count(*) from storage.objects where name like tests.id('C')::text || '/%') = 1,
                   'C deletes its own objects also while it waits for approval');
update public.access set approved = true where email = 'c@example.org';

-- ---------------------------------------------------------------- logs on the site are capped per job
:as_admin
update public.settings set own_log_bytes = 20;
:as_c
select public.own_logs(tests.id('wc'), tests.id('jc'), '[{"stage": "frames", "run_started": 1, "pos": 10, "text": "0123456789"}]');
select tests.expect_error($$select public.own_logs(tests.id('wc'), tests.id('jc'),
  '[{"stage": "frames", "run_started": 1, "pos": 21, "text": "01234567890"}]')$$, 'log on the site is full');
:as_admin
update public.settings set own_log_bytes = 2097152;

-- ---------------------------------------------------------------- relay to C's machine: never "none", keys only to its worker
:as_c
select public.own_publish(tests.id('wc'), tests.id('jc'), p_row => '{"status": "ready", "prompt": "mug"}');
select tests.expect_error($$select public.relay(tests.id('jc'), 'start', '{}', '{"mode": "auto", "prompt": "mug", "agent": "none"}')$$,
                          'astra and opus');
select tests.expect_error($$select public.relay(tests.id('jc'), 'start', '{}', '{"mode": "auto", "prompt": "mug"}', null,
  tests.sealed(tests.id('wc2'), 'opus', gen_random_uuid()::text))$$, 'worker_restarted');
select tests.check(true, 'a key for a job of lab-box cannot be sealed to the laptop''s worker');
select tests.expect_error($$select public.relay(tests.id('jc'), 'start', '{}', '{"mode": "auto", "prompt": "mug"}', null,
  tests.sealed(tests.id('wd'), 'opus', gen_random_uuid()::text))$$, 'worker_restarted');
select tests.check(true, 'nor to D''s worker');
insert into tests.ids (name, id) values ('refc1', gen_random_uuid());
insert into tests.ids (name, n) values ('rc1', (public.relay(tests.id('jc'), 'start', '{}', '{"mode": "auto", "prompt": "mug"}', null,
  tests.sealed(tests.id('wc'), 'opus', tests.id('refc1')::text)) ->> 'id')::bigint);
:as_admin
select tests.check((select worker_id from public.job_secrets where ref = tests.id('refc1')) = tests.id('wc'),
                   'the key is sealed to C''s worker on lab-box');
:as_d
select tests.check((select count(*) from public.own_claim_request(tests.id('wd'))) = 0, 'D''s worker does not take C''s request');
:as_c
select tests.check((select count(*) from public.own_claim_request(tests.id('wc2'))) = 0,
                   'C''s laptop does not take a request of a lab-box job');
create temp table oc as select own_claim_request as v from public.own_claim_request(tests.id('wc'));
select tests.check((select (v ->> 'id')::bigint = tests.n('rc1') and v ->> 'route' = 'start' and v ->> 'state' = 'taken'
                           and v -> 'sealed' ->> 'agent' = 'opus' and char_length(v -> 'sealed' ->> 'epk') = 88
                           and not (v -> 'sealed' ? 'ref') from oc),
                   'C''s worker takes the Start with its sealed key');
:as_admin
select tests.check((select count(*) from public.job_secrets where ref = tests.id('refc1')) = 0, 'and the sealed key''s row is gone');
select tests.check((select worker_id = tests.id('wc') and taken_at is not null from public.requests where id = tests.n('rc1')),
                   'the request names the worker that took it');
:as_c
select public.own_publish(tests.id('wc'), tests.id('jc'), p_row => '{"status": "queued", "queue_pos": 1}');
select tests.expect_error($$select public.own_answer(tests.id('wc'), tests.n('rc1'), 999, '{}')$$, 'the status is an HTTP status');
select public.own_answer(tests.id('wc'), tests.n('rc1'), 200, '{"status": "queued", "key": "Anthropic accepted the key"}');
select tests.check((select state = 'done' and status = 200 from public.requests where id = tests.n('rc1')), 'C reads the answer');
select tests.expect_error($$select public.own_answer(tests.id('wc'), tests.n('rc1'), 200, '{}')$$, 'not taken by this worker');
:as_d
select tests.expect_error($$select public.own_answer(tests.id('wd'), tests.n('rc1'), 200, '{}')$$, 'not taken by this worker');

-- ---------------------------------------------------------------- near the end of a Slurm job, the worker takes no build
:as_admin
insert into public.requests (user_id, job_id, route) values (tests.id('C'), tests.id('jc'), 'start'), (tests.id('C'), tests.id('jc'), 'cancel'),
  (tests.id('C'), tests.id('jl'), 'cancel');
:as_c
select tests.check((select count(*) from public.own_claim_request(tests.id('wc'), '{topic,calibration_put,calibration_delete,cancel,delete}')) = 0,
                   'near the end, lab-box leaves a Start, and the cancel sent after it');
select tests.check((select v ->> 'route' = 'cancel' and (v ->> 'job_id')::uuid = tests.id('jl')
                      from public.own_claim_request(tests.id('wc2'), '{topic,calibration_put,calibration_delete,cancel,delete}') v),
                   'the laptop takes the cancel of its own job');
select tests.check((select v ->> 'route' from public.own_claim_request(tests.id('wc')) v) = 'start', 'the next start takes the Start first');
:as_admin
update public.requests set state = 'withdrawn', done_at = now() where job_id in (tests.id('jc'), tests.id('jl')) and state in ('pending', 'taken');

-- ---------------------------------------------------------------- restarts of lab-box leave the laptop alone
:as_c
select public.own_publish(tests.id('wc'), tests.id('jc'), p_row => '{"status": "review", "queue_pos": null}');
insert into tests.ids (name, n) values ('rc2', (public.relay(tests.id('jc'), 'cancel') ->> 'id')::bigint);
select tests.check((select (v ->> 'id')::bigint from public.own_claim_request(tests.id('wc')) v) = tests.n('rc2'),
                   'C''s worker takes the cancel');
insert into tests.ids (name, id) values ('wc_b', public.own_worker_start(tests.id('mc'), repeat('M', 88), null, 'test', '{}'));
select tests.check((select state = 'done' and status = 409 and response ->> 'detail' like '%restarted during this request%'
                      from public.requests where id = tests.n('rc2'))
                   and (select stopped_at is not null and not accepting from public.workers where id = tests.id('wc'))
                   and (select stopped_at is null from public.workers where id = tests.id('wc2')),
                   'a restart on lab-box answers 409 to the request its old worker took, stops that worker, not the laptop''s');
insert into tests.ids (name, id) values ('refc3', gen_random_uuid());
insert into tests.ids (name, n) values ('rc3', (public.relay(tests.id('jc'), 'continue', '{}', '{}', null,
  tests.sealed(tests.id('wc_b'), 'opus', tests.id('refc3')::text)) ->> 'id')::bigint);
insert into tests.ids (name, id) values ('wc_c', public.own_worker_start(tests.id('mc'), repeat('N', 88), null, 'test', '{}'));
select tests.check((select state = 'done' and status = 400 and response ->> 'detail' like '%paste the key again%'
                      from public.requests where id = tests.n('rc3'))
                   and (select stopped_at is null from public.workers where id = tests.id('wc2')),
                   'another restart answers 400 to a pending request whose key was sealed to the old worker');
:as_admin
select tests.check((select count(*) from public.job_secrets where ref = tests.id('refc3')) = 0, 'and drops the key sealed to it');
:as_c
select tests.expect_error($$select public.own_claim_request(tests.id('wc'))$$, 'was stopped: another worker took over');
select tests.expect_error($$select public.own_worker_beat(tests.id('wc_b'), 0, true)$$, 'was stopped');
select tests.expect_error($$select public.own_publish(tests.id('wc_b'), tests.id('jc'), p_row => '{"status": "done"}')$$, 'was stopped');

-- ---------------------------------------------------------------- the machine is offline
:as_c
select public.own_worker_stop(tests.id('wc_c'));
select tests.expect_error($$select public.relay(tests.id('jc'), 'continue', '{}', '{}', null,
  tests.sealed(tests.id('wc_c'), 'opus', gen_random_uuid()::text))$$, 'worker_offline: the machine that runs this job is offline');
insert into tests.ids (name, n) values ('rc4', (public.relay(tests.id('jc'), 'continue', '{}', '{"key_later": true}') ->> 'id')::bigint);
select tests.check((select state = 'pending' and not sent_online from public.requests where id = tests.n('rc4')),
                   'with C''s machine offline, a continue waits as pending');

-- ---------------------------------------------------------------- deletes
:as_admin
select tests.check((select count(*) from public.job_views where job_id = tests.id('jc')) = 1
                   and (select count(*) from public.job_live where job_id = tests.id('jc')) = 1
                   and (select count(*) from public.job_snapshots where job_id = tests.id('jc')) = 1
                   and (select count(*) from public.job_files where job_id = tests.id('jc')) = 1
                   and (select count(*) from public.job_logs where job_id = tests.id('jc')) = 1,
                   'before its delete, C''s job has a view, a live state, a snapshot, a file and log text');
update public.jobs set status = 'running' where id = tests.id('jc');
:as_c
select tests.check(public.delete_job(tests.id('jc')) = 'deleting', 'a running job whose machine is offline can be deleted');
:as_admin
select tests.check((select count(*) from public.job_views where job_id = tests.id('jc')) = 0
                   and (select count(*) from public.job_live where job_id = tests.id('jc')) = 0
                   and (select count(*) from public.job_snapshots where job_id = tests.id('jc')) = 0
                   and (select count(*) from public.job_files where job_id = tests.id('jc')) = 0
                   and (select count(*) from public.job_logs where job_id = tests.id('jc')) = 0
                   and (select status = 'deleting' from public.jobs where id = tests.id('jc')),
                   'delete_job drops the view, live state, snapshots, file list and log text at once; the row alone waits');
:as_c
select tests.check((select count(*) from public.requests where job_id = tests.id('jc') and route = 'delete' and state = 'pending') = 1
                   and (select state from public.requests where id = tests.n('rc4')) = 'withdrawn',
                   'the delete waits for the machine, and the pending continue was withdrawn');
select tests.check((select array_agg(name) from public.job_objects(tests.id('jc')))
                   = array[tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha1'), 'run/posed.glb')],
                   'job_objects still lists the objects of a job in deleting, for the page to delete');
select tests.check(not public.own_object_ok(tests.obj(auth.uid(), tests.id('jc'), (select v from tests.sha where name = 'sha2'), 'run/x.glb')),
                   'no upload into a job in deleting');
insert into tests.ids (name, id) values ('wc_d', public.own_worker_start(tests.id('mc'), repeat('O', 88), null, 'test', '{}'));
select tests.check((select v ->> 'route' from public.own_claim_request(tests.id('wc_d')) v) = 'delete',
                   'the machine comes back and takes the delete');
select tests.expect_error($$select public.own_publish(tests.id('wc_d'), tests.id('jc'), p_row => '{"status": "done"}')$$,
                          'no such job on this machine');
select tests.expect_error($$select public.own_job_delete(tests.id('wc2'), tests.id('jc'))$$, 'no such job on this machine');
delete from storage.objects where bucket_id = 'job-files' and name in (select name from public.job_objects(tests.id('jc')));
select tests.check(public.storage_used() = 0, 'the worker deletes the objects that job_objects lists');
select public.own_job_delete(tests.id('wc_d'), tests.id('jc'));
:as_admin
select tests.check((select count(*) from public.jobs where id = tests.id('jc')) = 0
                   and (select count(*) from public.job_files where job_id = tests.id('jc')) = 0
                   and (select count(*) from public.job_logs where job_id = tests.id('jc')) = 0
                   and (select count(*) from public.requests where job_id = tests.id('jc')) = 0,
                   'own_job_delete removes the row, and its files, logs and requests go with it');
:as_c
select tests.expect_error($$select public.own_job_delete(tests.id('wc_d'), tests.id('jc'))$$, 'no such job on this machine');
select tests.expect_error($$select * from public.job_objects(tests.id('jc'))$$, 'no such job');
select tests.expect_error($$select public.own_job_delete(tests.id('wc_d'), tests.id('jl'))$$, 'no such job on this machine');
select tests.check(true, 'own_job_delete takes only a job that waits in deleting');
insert into tests.ids (name, id) values ('jc2', public.own_job_create(tests.id('wc_d'), 'b.mp4', 'video', 5));
select tests.check(public.delete_job(tests.id('jc2')) = 'deleting',
                   'a job that still says uploaded is marked for delete while its machine is online (its worker cancels it if it runs)');
:as_admin
update public.jobs set deleted_at = now() - interval '8 days' where id = tests.id('jc2');
update public.requests set created_at = now() - interval '8 days' where job_id = tests.id('jc2');
select private.sweep();
select tests.check((select count(*) from public.jobs where id = tests.id('jc2')) = 0,
                   'the sweep removes a job that waited in deleting longer than pending_days');

-- ---------------------------------------------------------------- limits and approval
:as_admin
update public.access set max_own_jobs = (select count(*) from public.jobs where user_id = tests.id('C')) where email = 'c@example.org';
:as_c
select tests.expect_error($$select public.own_job_create(tests.id('wc_d'), 'c.mp4', 'video', 5)$$, 'delete older ones first');
select tests.check(public.delete_job(tests.id('jl')) = 'deleting', 'C deletes the laptop''s job');
insert into tests.ids (name, id) values ('jc_after', public.own_job_create(tests.id('wc_d'), 'c.mp4', 'video', 5));
select tests.check((select count(*) from public.jobs where id = tests.id('jc_after')) = 1,
                   'a job that waits in deleting no longer counts toward max_own_jobs');
:as_admin
update public.access set max_own_jobs = 50, max_own_per_day = 2 where email = 'c@example.org';
:as_c
select tests.expect_error($$select public.own_job_create(tests.id('wc_d'), 'c.mp4', 'video', 5)$$, 'own machines today');
:as_admin
update public.access set max_own_per_day = 20 where email = 'c@example.org';
update public.settings set max_machines = 2;
:as_c
select tests.expect_error($$select public.own_machine_bind(null, 'third')$$, 'you have 2 machines');
:as_admin
update public.settings set max_machines = 5;
:as_c
insert into tests.ids (name, id) values ('jc3', public.own_job_create(tests.id('wc_d'), 'c.mp4', 'video', 5));
:as_admin
update public.access set approved = false where email = 'c@example.org';
:as_c
select tests.expect_error($$select public.own_publish(tests.id('wc_d'), tests.id('jc3'), p_row => '{"status": "done"}')$$,
                          'waiting for approval');
select tests.expect_error($$select public.own_claim_request(tests.id('wc_d'))$$, 'waiting for approval');
select tests.expect_error($$select * from public.own_orphans(tests.id('wc_d'))$$, 'waiting for approval');
select tests.expect_error($$insert into storage.objects (bucket_id, name, metadata)
  values ('job-files', tests.obj(auth.uid(), tests.id('jc3'), repeat('a', 64), 'run/posed.glb'), '{"size": 10}')$$,
  'row-level security');
select tests.check((select count(*) from public.job_objects(tests.id('jc3'))) = 0, 'job_objects needs no approval');
select public.own_worker_stop(tests.id('wc_d'));
select tests.check((select stopped_at is not null from public.workers where id = tests.id('wc_d')),
                   'without approval the worker may still stop, and nothing else');
:as_admin
update public.access set approved = true where email = 'c@example.org';

-- ---------------------------------------------------------------- keys between tables
:as_admin
select tests.expect_error($$insert into public.jobs (user_id, machine_id, input_name, input_kind, input_bytes)
  values ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', tests.id('mc'), 'x.mp4', 'video', 1)$$, 'foreign key');
select tests.expect_error($$insert into public.workers (user_id, machine_id, public_key)
  values ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', tests.id('mc'), repeat('X', 88))$$, 'foreign key');
select tests.check(true, 'a job or a worker of one user never sits on another user''s machine');
select tests.expect_error($$insert into public.jobs (user_id, input_name, input_kind, input_bytes)
  values (tests.id('C'), 'x.mp4', 'video', 1)$$, 'null value in column "machine_id"');
select tests.expect_error($$insert into public.workers (user_id, public_key) values (tests.id('C'), repeat('X', 88))$$,
                          'null value in column "machine_id"');
select tests.check(true, 'every job and every worker belongs to a machine');
select tests.expect_error($$insert into public.jobs (user_id, machine_id, input_name, input_kind, input_bytes, status)
  values (tests.id('C'), tests.id('mc'), 'x.mp4', 'video', 1, 'waiting_worker')$$, 'check constraint');
\echo ALL OWN CHECKS PASSED
