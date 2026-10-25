-- HARPOR public site, part 6: pg_cron runs the sweep (private.sweep in part 2). Required: no worker
-- sweeps, so without it requests never expire and old log text stays.
-- Supabase: this line turns the extension on, or turn on "Cron" under Integrations in the dashboard
-- and leave the line out. The local stand-in (pgcheck) skips this file: Homebrew Postgres has no pg_cron.
create extension if not exists pg_cron with schema pg_catalog;

select cron.schedule('harpor-sweep', '*/15 * * * *', $$select private.sweep()$$);
-- cron.job_run_details grows with every run and is never cleaned up by itself (Supabase docs).
select cron.schedule('harpor-cron-cleanup', '0 4 * * *',
  $$delete from cron.job_run_details where end_time < now() - interval '7 days'$$);
