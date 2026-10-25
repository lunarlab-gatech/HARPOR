-- Supabase's default privileges on public before the 2026 change: every new table, function and
-- sequence is granted to the API roles. Projects made before 2026-05-30 kept them until 2026-10-30.
-- The migrations must revoke what they do not want, so the checks run once with this file (an old
-- project) and once without it (a new project, where only the migrations' own grants exist).
alter default privileges in schema public grant all on tables to anon, authenticated, service_role;
alter default privileges in schema public grant all on functions to anon, authenticated, service_role;
alter default privileges in schema public grant all on sequences to anon, authenticated, service_role;
