-- Checks that keep users and their machines apart, after 20_own_checks.sql in the same database. A and B are
-- approved, and each has a machine with a personal worker. B's worker calls Supabase with B's session, so
-- everything it may do, B may do. Run as B, nothing below may reach A's machine, worker, jobs, requests, sealed
-- keys or objects. The end of the file checks the rows that tie jobs and workers to machines.
\set ON_ERROR_STOP on
\set QUIET on
\set as_a 'reset role; set request.jwt.claims to ''{"sub": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "role": "authenticated"}''; set role authenticated;'
\set as_b 'reset role; set request.jwt.claims to ''{"sub": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "role": "authenticated"}''; set role authenticated;'
\set as_admin 'reset role; set request.jwt.claims to '''';'

:as_admin
insert into tests.sha values ('shaA', 'aaaaaaaaaaaaaaaa' || repeat('a', 48)), ('shaB', 'bbbbbbbbbbbbbbbb' || repeat('b', 48));
delete from private.rate where user_id in (tests.id('A'), tests.id('B'));
delete from public.usage_daily where user_id in (tests.id('A'), tests.id('B'));
update public.access set approved = true, max_requests_per_hour = 60, agents = '{astra,opus}' where user_id = tests.id('A');
update public.access set approved = true where user_id = tests.id('B');

-- ---------------------------------------------------------------- A: a machine, a worker, a job with files, a log and a sealed key
:as_a
insert into tests.ids (name, id) values ('ma', public.own_machine_bind(null, 'a-box'));
insert into tests.ids (name, id) values ('wa', public.own_worker_start(tests.id('ma'), repeat('W', 88), null, 'test', '{}'));
insert into tests.ids (name, id) values ('jao', public.own_job_create(tests.id('wa'), 'cup.mp4', 'video', 1000));
select tests.check((select own_jobs from public.usage_daily where user_id = auth.uid() and day = current_date) = 1,
                   'A''s job counts in A''s usage_daily.own_jobs');
insert into storage.objects (bucket_id, name, metadata)
  values ('job-files', tests.obj(auth.uid(), tests.id('jao'), (select v from tests.sha where name = 'shaA'), 'run/posed.glb'),
          '{"size": 10}');
select public.own_files(tests.id('wa'), tests.id('jao'), jsonb_build_array(jsonb_build_object(
  'path', 'run/posed.glb', 'object', tests.obj(auth.uid(), tests.id('jao'), (select v from tests.sha where name = 'shaA'), 'run/posed.glb'),
  'bytes', 10, 'sha256', (select v from tests.sha where name = 'shaA'), 'role', 'result', 'content_type', 'model/gltf-binary')));
select public.own_logs(tests.id('wa'), tests.id('jao'), '[{"stage": "frames", "run_started": 1, "pos": 5, "text": "a log"}]');
select public.own_publish(tests.id('wa'), tests.id('jao'), '{"status": "ready"}', '{"status": "ready"}',
                          '{"frames": {"status": 200, "body": {"times": []}}}', '{"status": "ready", "prompt": "cup"}');
insert into tests.ids (name, id) values ('refao', gen_random_uuid());
insert into tests.ids (name, n) values ('rao', (public.relay(tests.id('jao'), 'start', '{}', '{"mode": "auto", "prompt": "cup"}',
  null, tests.sealed(tests.id('wa'), 'opus', tests.id('refao')::text)) ->> 'id')::bigint);

-- ---------------------------------------------------------------- B: a machine, a worker and a job
:as_b
insert into tests.ids (name, id) values ('mb', public.own_machine_bind(null, 'b-box'));
insert into tests.ids (name, id) values ('wb', public.own_worker_start(tests.id('mb'), repeat('X', 88), null, 'test', '{opus}'));
insert into tests.ids (name, id) values ('jbo', public.own_job_create(tests.id('wb'), 'box.mp4', 'video', 1000));
select public.own_publish(tests.id('wb'), tests.id('jbo'), p_row => '{"status": "review"}');

-- ---------------------------------------------------------------- B's worker takes no work of A
:as_b
select tests.check((select count(*) from public.own_claim_request(tests.id('wb'))) = 0, 'B''s worker does not take A''s request');
select tests.check((select count(*) from public.own_claim_request(tests.id('wb'), '{start,continue,cancel,delete}')) = 0,
                   'not with route kinds either');
select tests.expect_error($$select public.own_claim_request(tests.id('wa'))$$, 'no such worker');
:as_admin
select tests.check((select state from public.requests where id = tests.n('rao')) = 'pending'
                   and (select count(*) from public.job_secrets where ref = tests.id('refao')) = 1,
                   'A''s request still waits, with its sealed key');

-- ---------------------------------------------------------------- B reads nothing of A
:as_admin
select tests.check((select count(*) from public.job_views where job_id = tests.id('jao')) = 1
                   and (select count(*) from public.job_live where job_id = tests.id('jao')) = 1
                   and (select count(*) from public.job_snapshots where job_id = tests.id('jao')) = 1
                   and (select count(*) from public.job_files where job_id = tests.id('jao')) = 1
                   and (select count(*) from public.job_logs where job_id = tests.id('jao')) = 1
                   and (select count(*) from storage.objects where name like 'aaaaaaaa-%' and bucket_id = 'job-files') >= 1,
                   'A has a view, live state, snapshot, file list, log and an object');
:as_b
select tests.check((select count(*) from public.jobs) = 1 and (select id from public.jobs) = tests.id('jbo'),
                   'B sees only its own job');
select tests.check((select count(*) from public.job_views) + (select count(*) from public.job_live)
                   + (select count(*) from public.job_snapshots) + (select count(*) from public.job_files)
                   + (select count(*) from public.job_logs) = 0, 'B reads none of A''s views, states, snapshots, files or logs');
select tests.check((select count(*) from public.requests where user_id = tests.id('A')) = 0
                   and (select count(*) from public.usage_daily where user_id = tests.id('A')) = 0
                   and (select count(*) from public.access where user_id = tests.id('A')) = 0,
                   'B reads none of A''s requests, usage or access');
select tests.check((select count(*) from public.machines) = 1 and (select id from public.machines) = tests.id('mb'),
                   'B reads only its own machine');
select tests.check((select count(*) from public.workers) = 1 and (select id from public.workers) = tests.id('wb'),
                   'B reads only its own worker, never A''s public key');
select tests.check((select count(*) from storage.objects) = 0, 'B sees none of A''s objects');
select tests.check(public.storage_used() = 0, 'storage_used counts none of A''s bytes for B');
select tests.check(not public.own_object_ok(tests.obj(tests.id('A'), tests.id('jao'), (select v from tests.sha where name = 'shaA'),
                                                      'run/x.glb'))
                   and not public.own_object_ok(tests.obj(auth.uid(), tests.id('jao'), (select v from tests.sha where name = 'shaA'),
                                                          'run/x.glb')),
                   'own_object_ok says no to A''s folder, and to A''s job in B''s folder');
select tests.expect_error($$select * from public.job_objects(tests.id('jao'))$$, 'no such job');
select tests.expect_error($$select * from public.own_orphans(tests.id('wa'))$$, 'no such worker');
select tests.check((select count(*) from public.own_orphans(tests.id('wb'))) = 0, 'B''s orphans hold nothing of A');
select tests.expect_error($$select * from public.job_secrets$$, 'permission denied');
select tests.expect_error($$select * from private.rate$$, 'permission denied');

-- ---------------------------------------------------------------- B writes nothing of A
select tests.expect_error($$select public.own_publish(tests.id('wb'), tests.id('jao'), p_row => '{"status": "done"}')$$,
                          'no such job on this machine');
select tests.expect_error($$select public.own_publish(tests.id('wa'), tests.id('jao'), p_row => '{"status": "done"}')$$,
                          'no such worker');
select tests.expect_error($$select public.own_files(tests.id('wb'), tests.id('jao'), p_drop => '{run/posed.glb}')$$,
                          'no such job on this machine');
select tests.expect_error($$select public.own_logs(tests.id('wb'), tests.id('jao'), '[{"stage": "frames", "pos": 9, "text": "x"}]')$$,
                          'no such job on this machine');
select tests.expect_error($$select public.own_job_delete(tests.id('wb'), tests.id('jao'))$$, 'no such job on this machine');
select tests.expect_error($$select public.own_job_create(tests.id('wa'), 'x.mp4', 'video', 1)$$, 'no such worker');
select tests.expect_error($$select public.own_answer(tests.id('wa'), tests.n('rao'), 200, '{}')$$, 'no such worker');
select tests.expect_error($$select public.own_worker_beat(tests.id('wa'), 0, true)$$, 'no such worker');
select public.own_worker_stop(tests.id('wa'));
select tests.expect_error($$select public.own_machine_bind(tests.id('ma'), 'mine')$$, 'no such machine');
select tests.expect_error($$select public.own_worker_start(tests.id('ma'), repeat('X', 88), null, 'x', '{}')$$, 'no such machine');
select tests.expect_error($$select public.own_worker_start(null, repeat('X', 88), null, 'x', '{}')$$, 'no such machine');
select tests.expect_error($$select public.relay(tests.id('jao'), 'cancel')$$, 'no such job');
select tests.expect_error($$select public.delete_job(tests.id('jao'))$$, 'no such job');
select tests.expect_error($$select public.withdraw_request(tests.n('rao'))$$, 'no longer waiting');
select tests.expect_error($$select public.own_files(tests.id('wb'), tests.id('jbo'), jsonb_build_array(jsonb_build_object(
  'path', 'run/posed.glb', 'object', tests.obj(tests.id('A'), tests.id('jao'), (select v from tests.sha where name = 'shaA'), 'run/posed.glb'),
  'bytes', 10, 'sha256', (select v from tests.sha where name = 'shaA'), 'role', 'result', 'content_type', 'model/gltf-binary')))$$,
  'check constraint');
select tests.check(true, 'B''s job cannot list A''s object as its own file');
select tests.expect_error($$insert into storage.objects (bucket_id, name, metadata)
  values ('job-files', tests.obj(tests.id('A'), tests.id('jao'), repeat('c', 64), 'run/x.glb'), '{"size": 10}')$$,
  'row-level security');
select tests.expect_error($$insert into storage.objects (bucket_id, name, metadata)
  values ('job-files', tests.obj(auth.uid(), tests.id('jao'), repeat('c', 64), 'run/x.glb'), '{"size": 10}')$$,
  'row-level security');
delete from storage.objects where name like 'aaaaaaaa-%';
delete from storage.objects where name in (select name from public.job_objects(tests.id('jbo')))
                                 or name like '%/' || tests.id('jao')::text || '/%';
delete from storage.objects;   -- no WHERE, so only the delete policy picks the rows
update storage.objects set name = 'x' where name like 'aaaaaaaa-%';
select tests.expect_error($$update public.workers set user_id = auth.uid() where id = tests.id('wa')$$, 'permission denied');
select tests.expect_error($$insert into public.job_secrets (ref, job_id, user_id, worker_id, agent, epk, iv, ct)
  values (gen_random_uuid(), tests.id('jbo'), auth.uid(), tests.id('wa'), 'opus', repeat('B', 88), repeat('C', 16), repeat('D', 40))$$,
  'permission denied');
select tests.expect_error($$update public.requests set state = 'taken', worker_id = tests.id('wb') where id = tests.n('rao')$$,
                          'permission denied');
select tests.expect_error($$delete from public.job_files$$, 'permission denied');
select tests.expect_error($$insert into public.machines (user_id, name) values (tests.id('A'), 'x')$$, 'permission denied');
:as_admin
select tests.check((select stopped_at is null from public.workers where id = tests.id('wa'))
                   and (select name from public.machines where id = tests.id('ma')) = 'a-box'
                   and (select status from public.jobs where id = tests.id('jao')) = 'ready'
                   and (select deleted_at is null from public.jobs where id = tests.id('jao'))
                   and (select count(*) from public.job_files where job_id = tests.id('jao')) = 1
                   and (select count(*) from public.job_logs where job_id = tests.id('jao')) = 1
                   and (select count(*) from storage.objects where bucket_id = 'job-files'
                          and name = tests.obj(tests.id('A'), tests.id('jao'), (select v from tests.sha where name = 'shaA'), 'run/posed.glb')) = 1
                   and (select count(*) from public.job_secrets where ref = tests.id('refao')) = 1
                   and (select state from public.requests where id = tests.n('rao')) = 'pending',
                   'after all that, A''s worker, machine, job, files, log, object, key and request are as they were');

-- ---------------------------------------------------------------- keys go only to their own machine's worker
:as_b
select tests.expect_error($$select public.relay(tests.id('jbo'), 'continue', '{}', '{}', null,
  tests.sealed(tests.id('wa'), 'opus', gen_random_uuid()::text))$$, 'worker_restarted');
select tests.check(true, 'a key for B''s job cannot be sealed to A''s worker');
insert into tests.ids (name, id) values ('refbo', gen_random_uuid());
insert into tests.ids (name, n) values ('rbo', (public.relay(tests.id('jbo'), 'continue', '{}', '{}', null,
  tests.sealed(tests.id('wb'), 'opus', tests.id('refbo')::text)) ->> 'id')::bigint);
:as_admin
select tests.check((select worker_id from public.job_secrets where ref = tests.id('refbo')) = tests.id('wb')
                   and (select worker_id from public.job_secrets where ref = tests.id('refao')) = tests.id('wa'),
                   'B''s key is sealed to B''s worker, and A''s to A''s');

-- ---------------------------------------------------------------- each request goes to its own worker
:as_a
select tests.check((select count(*) = 1 and bool_and((v ->> 'id')::bigint = tests.n('rao') and v -> 'sealed' ->> 'agent' = 'opus')
                      from public.own_claim_request(tests.id('wa')) v),
                   'A''s worker takes A''s request with its key');
select tests.check((select count(*) from public.own_claim_request(tests.id('wa'))) = 0, 'and never B''s');
:as_b
select tests.check((select count(*) = 1 and bool_and((v ->> 'id')::bigint = tests.n('rbo')
                           and char_length(v -> 'sealed' ->> 'ct') = 40)
                      from public.own_claim_request(tests.id('wb')) v),
                   'B''s worker takes B''s request with its key');
select tests.expect_error($$select public.own_answer(tests.id('wb'), tests.n('rao'), 200, '{}')$$, 'not taken by this worker');
select tests.check((select count(*) = 0 and clock_timestamp() - now() < interval '0.5 seconds'
                      from public.own_claim_request(tests.id('wb'))),
                   'own_claim_request returns at once when nothing waits');
:as_a
select tests.expect_error($$select public.own_answer(tests.id('wa'), tests.n('rbo'), 200, '{}')$$, 'not taken by this worker');
select public.own_answer(tests.id('wa'), tests.n('rao'), 200, '{"status": "queued"}');
select public.own_publish(tests.id('wa'), tests.id('jao'), p_row => '{"status": "review"}');
:as_b
select public.own_answer(tests.id('wb'), tests.n('rbo'), 200, '{"status": "queued"}');
:as_admin
select tests.check((select status from public.requests where id = tests.n('rao')) = 200
                   and (select status from public.requests where id = tests.n('rbo')) = 200,
                   'each answer reached its own request');

-- ---------------------------------------------------------------- what own_publish, own_files and own_logs refuse
:as_b
select public.own_publish(tests.id('wb'), tests.id('jbo'), p_row => '{"status": "done", "stopped_at": "2999-01-01T00:00:00Z"}');
select tests.check((select stopped_at <= now() from public.jobs where id = tests.id('jbo')),
                   'stopped_at is never later than now, so the retention of log text cannot be put off');
select tests.expect_error($$select public.own_publish(tests.id('wb'), tests.id('jbo'), p_row => '{"status": "uploading"}')$$, 'bad status');
select tests.expect_error($$select public.own_publish(tests.id('wb'), tests.id('jbo'), p_row => '{"status": "expired"}')$$, 'bad status');
select tests.expect_error($$select public.own_publish(tests.id('wb'), tests.id('jbo'), p_view => '[1]')$$, 'JSON object');
select tests.expect_error($$select public.own_publish(tests.id('wb'), tests.id('jbo'), p_live => '"x"')$$, 'JSON object');
select tests.expect_error($$select public.own_files(tests.id('wb'), tests.id('jbo'),
  (select jsonb_agg(jsonb_build_object('path', 'f' || i)) from generate_series(1, 501) i))$$, 'at most 500');
select tests.expect_error($$select public.own_logs(tests.id('wb'), tests.id('jbo'), '[]')$$, '1 to 100');
select tests.expect_error($$select public.own_logs(tests.id('wb'), tests.id('jbo'),
  (select jsonb_agg(jsonb_build_object('stage', 'frames', 'pos', i, 'text', 'x')) from generate_series(1, 101) i))$$, '1 to 100');
select tests.expect_error($$select public.own_logs(tests.id('wb'), tests.id('jbo'), '[{"stage": "Bad Stage", "pos": 1, "text": "x"}]')$$,
                          'check constraint');
select tests.expect_error($$select public.own_answer(tests.id('wb'), tests.n('rbo'), 999, '{}')$$, 'HTTP status');

-- ---------------------------------------------------------------- approval taken back
:as_admin
update public.access set approved = false where user_id = tests.id('B');
:as_b
select tests.expect_error($$select public.own_machine_bind(null, 'b-box-2')$$, 'waiting for approval');
select tests.expect_error($$select public.own_worker_start(tests.id('mb'), repeat('X', 88), null, 'x', '{}')$$, 'waiting for approval');
select tests.expect_error($$select public.own_worker_beat(tests.id('wb'), 0, true)$$, 'waiting for approval');
select tests.expect_error($$select public.own_files(tests.id('wb'), tests.id('jbo'))$$, 'waiting for approval');
select tests.expect_error($$select public.own_logs(tests.id('wb'), tests.id('jbo'), '[{"stage": "frames", "pos": 1, "text": "x"}]')$$,
                          'waiting for approval');
select tests.expect_error($$select public.own_job_create(tests.id('wb'), 'x.mp4', 'video', 1)$$, 'waiting for approval');
select tests.expect_error($$select public.relay(tests.id('jbo'), 'continue')$$, 'waiting for approval');
select tests.check(public.delete_job(tests.id('jbo')) = 'deleting', 'B may still delete its own job');
:as_admin
update public.access set approved = true where user_id = tests.id('B');
update public.jobs set deleted_at = now() - interval '30 days' where id = tests.id('jbo');
update public.requests set created_at = now() - interval '30 days' where job_id = tests.id('jbo') and state = 'pending';
select private.sweep();
select tests.check((select count(*) from public.jobs where id = tests.id('jbo')) = 0
                   and (select count(*) from public.jobs where id = tests.id('jao')) = 1,
                   'the sweep removes B''s job that waited 30 days in deleting, and leaves A''s job alone');

-- ---------------------------------------------------------------- rows that tie jobs and workers to machines
:as_admin
select tests.expect_error($$insert into public.job_files (job_id, path, user_id, object, bytes, sha256, role, content_type)
  values (tests.id('jao'), 'run/other.glb', tests.id('A'),
          tests.obj(tests.id('A'), tests.id('jao'), (select v from tests.sha where name = 'shaA'), 'run/posed.glb'), 1,
          (select v from tests.sha where name = 'shaA'), 'result', 'model/gltf-binary')$$, 'check constraint');
select tests.check(true, 'a file row names the object of its own path, even when the superuser writes it');
select tests.expect_error($$update public.jobs set machine_id = tests.id('mb') where id = tests.id('jao')$$, 'foreign key');
select tests.expect_error($$update public.workers set machine_id = tests.id('mb') where id = tests.id('wa')$$, 'foreign key');
select tests.check(true, 'A''s job and worker cannot move to B''s machine');
insert into public.workers (id, user_id, machine_id, public_key, started_at, last_seen, stopped_at)
  values ('d1d1d1d1-0000-0000-0000-000000000031', tests.id('B'), tests.id('mb'), repeat('O', 88), now() - interval '31 days',
          now() - interval '31 days', now() - interval '31 days'),
         ('d2d2d2d2-0000-0000-0000-000000000040', tests.id('B'), tests.id('mb'), repeat('O', 88), now() - interval '40 days',
          now(), now());
:as_b
select tests.check((select count(*) from public.workers where id = 'd1d1d1d1-0000-0000-0000-000000000031') = 0
                   and (select count(*) from public.workers where id = 'd2d2d2d2-0000-0000-0000-000000000040') = 1,
                   'a user reads its workers seen in the last 30 days, also one that started 40 days ago');
:as_a
select tests.check((select count(*) from public.workers where user_id = tests.id('B')) = 0, 'and nobody else reads them');
:as_admin
delete from public.workers where id in ('d1d1d1d1-0000-0000-0000-000000000031', 'd2d2d2d2-0000-0000-0000-000000000040');
update storage.objects set created_at = now() - interval '2 hours' where name like '%/' || tests.id('jao')::text || '/%';
delete from public.machines where id = tests.id('ma');
select tests.check((select count(*) from public.jobs where id = tests.id('jao')) = 0
                   and (select count(*) from public.workers where id = tests.id('wa')) = 0
                   and (select count(*) from public.job_files where job_id = tests.id('jao')) = 0,
                   'removing a machine removes its jobs and workers');
select tests.check((select count(*) from private.orphan_folders()
                     where folder = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/' || tests.id('jao')::text || '/' and not account_gone) = 1,
                   'the removed job''s objects stay, and the owner''s list shows their folder');
:as_a
select tests.check((select array_agg(name) from public.own_orphans(tests.id('w3')))
                   = array[tests.obj(auth.uid(), tests.id('jao'), (select v from tests.sha where name = 'shaA'), 'run/posed.glb')],
                   'a worker on A''s other machine finds those objects as orphans, for A to delete');
delete from storage.objects where name in (select name from public.own_orphans(tests.id('w3')));
:as_admin
select tests.check((select count(*) from private.orphan_folders()) = 0, 'then no folder is left without a job');

-- ---------------------------------------------------------------- a loop of worker starts does not fill the table
:as_b
do $$
begin
  for i in 1 .. 12 loop
    insert into tests.ids (name, id) values ('wb_last', public.own_worker_start(tests.id('mb'), repeat('Y', 88), null, 'loop', '{}'))
    on conflict (name) do update set id = excluded.id;
  end loop;
end $$;
:as_admin
select tests.check((select count(*) from public.workers where machine_id = tests.id('mb')) = 10
                   and (select count(*) from public.workers where machine_id = tests.id('mb') and stopped_at is null) = 1
                   and (select stopped_at is null from public.workers where id = tests.id('wb_last')),
                   'a machine keeps its newest 10 worker rows, and only the newest worker runs');
\echo ALL ISOLATION CHECKS PASSED
