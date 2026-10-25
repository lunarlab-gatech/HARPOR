-- HARPOR public site, part 5: Realtime. The page follows its own rows (RLS applies to each event).
-- job_live: the running job's state (about 3 KB); job_logs: new log text; requests: answers.
-- Not job_views (6 KB, read on refresh), not job_snapshots (up to 1 MB), not jobs (the list polls
-- every 5 s, as today), and never job_secrets, access or workers.
alter publication supabase_realtime add table public.job_live, public.job_logs, public.requests;
