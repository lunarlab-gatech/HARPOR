-- HARPOR public site, part 1: tables, RLS and grants.
--
-- Every job runs on a user's own GPU machine. Who writes what:
--   * Users (role authenticated) read their own rows. They write only through the functions in
--     20261101000200_harpor_user_functions.sql (relay, withdraw_request, delete_job, request_access).
--   * A personal worker runs HARPOR's server on a user's own GPU machine and signs in as that user (role
--     authenticated). It writes only through the own_* functions in 20261101000300_harpor_worker_functions.sql,
--     and only rows of that user's jobs on its machine. RLS cannot tell the user's browser from the user's
--     personal worker, since both hold a session of the same user, so the functions check the job and the
--     machine, never the caller's kind.
--   * anon gets nothing, and neither does service_role: no part of HARPOR uses the project's secret key.
-- Every table revokes all grants first and then grants what users need. So a project whose old default
-- privileges grant every table to the API roles ends with exactly these grants, as a new project does.

create schema if not exists private;
revoke all on schema private from public;

-- Relayed requests per user and hour, a count that deleting a job does not lower. relay() reads and adds it.
create table private.rate (
  user_id uuid not null references auth.users (id) on delete cascade,
  hour timestamptz not null,
  n int not null default 0,
  primary key (user_id, hour)
);

-- Site-wide settings. One row. Users read it; the owner edits it in the SQL editor.
create table public.settings (
  id int primary key default 1 check (id = 1),
  plan_file_limit bigint not null default 52428800,        -- the largest file a worker copies (job-files): 50 MB on Free
  worker_online_s int not null default 30,                 -- a worker seen this recently is online
  pending_days int not null default 7,                     -- a request waits this long for its machine
  near_end_minutes int not null default 60,                -- a worker in a Slurm job takes no builds this close to its end
  lean_files boolean not null default true,                -- 150 thumbnails, JPEG review images, no preview over 30 MB
  max_machines int not null default 5,                     -- personal machines per user
  own_log_bytes bigint not null default 2097152            -- job_logs text per job (2 MiB)
);
insert into public.settings default values;

-- Who may do what, and the per-user limits. A row is made at sign-up (private.on_signup).
-- Users read their own row and never write it. The owner sets approved: the account may use the site,
-- and its machines may copy jobs and files to it.
create table public.access (
  user_id uuid primary key references auth.users (id) on delete cascade,
  email text,
  approved boolean not null default false,
  max_result_bytes bigint not null default 2147483648 check (max_result_bytes >= 0),  -- all the user's job-files objects
  max_requests_per_hour int not null default 60 check (max_requests_per_hour between 0 and 1000),
  max_own_jobs int not null default 20 check (max_own_jobs between 0 and 1000),        -- jobs on the site at once
  max_own_per_day int not null default 20 check (max_own_per_day between 0 and 1000),  -- new jobs a day
  agents text[] not null default '{astra,opus}' check (agents <@ array['astra', 'opus']),
  request_note text check (char_length(request_note) <= 500),
  request_at timestamptz,
  created_at timestamptz not null default now()
);

-- Daily counts that deleting a job does not lower.
create table public.usage_daily (
  user_id uuid not null references auth.users (id) on delete cascade,
  day date not null default current_date,
  own_jobs int not null default 0,             -- new jobs (own_job_create)
  primary key (user_id, day)
);

-- A user's own GPU machines. `python -m gator.webapp.worker login` makes the row (own_machine_bind)
-- and keeps its id in the machine's session file. A personal worker serves only the jobs of its
-- machine.
create table public.machines (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  name text not null check (name ~ '^[A-Za-z0-9][A-Za-z0-9_.-]{0,59}$'),
  created_at timestamptz not null default now(),
  unique (id, user_id)                         -- for the (machine_id, user_id) keys of workers and jobs
);

-- One row per worker start on a machine (own_worker_start). public_key is the worker's ECDH P-256 key
-- for sealing API keys (base64 of the 65-byte point: 88 characters). The page seals a job's key only to
-- the newest live row of the job's machine.
create table public.workers (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  machine_id uuid not null,
  public_key text not null check (char_length(public_key) = 88),
  version text check (char_length(version) <= 40),
  local_keys text[] not null default '{}' check (local_keys <@ array['astra', 'opus']),  -- agents whose key the machine holds
  started_at timestamptz not null default now(),
  last_seen timestamptz not null default now(),
  ends_at timestamptz,                         -- the end of the Slurm job, when the machine runs in one
  accepting boolean not null default true,     -- false in the last near_end_minutes: builds wait for the next start
  queue_len int not null default 0,
  stopped_at timestamptz,
  foreign key (machine_id, user_id) references public.machines (id, user_id) on delete cascade
);
create index workers_user_idx on public.workers (user_id, started_at desc);
create index workers_machine_idx on public.workers (machine_id, started_at desc);

-- Jobs. Each runs on the user's machine_id, and its input never leaves that machine (own_job_create).
-- status follows job.json (uploaded, analyzing, choose_topic, ready, queued, running, review, done, failed,
-- cancelled), plus deleting.
create table public.jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  machine_id uuid not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  -- relay.site_name: jobs.safe_name's output without two dots in a row, so the object name never holds ".."
  input_name text not null check (input_name ~ '^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,199}$' and input_name !~ '\.\.'),
  input_kind text not null check (input_kind in ('video', 'rosbag', 'stray')),
  input_bytes bigint not null check (input_bytes > 0),
  status text not null default 'uploaded' check (status in (
    'uploaded', 'analyzing', 'choose_topic', 'ready', 'queued', 'running', 'review', 'done', 'failed', 'cancelled',
    'deleting')),
  note text check (char_length(note) <= 300),            -- a short reason or progress line, scrubbed
  prompt text check (char_length(prompt) <= 60),         -- job.json "prompt", for the jobs list
  queue_pos int,                                         -- place in the machine's GPU queue, 1 = next
  stopped_at timestamptz,                                -- when the job last stopped, for retention
  deleted_at timestamptz,
  foreign key (machine_id, user_id) references public.machines (id, user_id) on delete cascade
);
create index jobs_user_idx on public.jobs (user_id, created_at desc);
create index jobs_machine_idx on public.jobs (machine_id);

-- GET /api/jobs/{id} as the worker last made it, rewritten for the public (about 6 KB).
create table public.job_views (
  job_id uuid primary key references public.jobs (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  rev bigint not null default 0,
  body jsonb not null check (octet_length(body::text) <= 262144),
  updated_at timestamptz not null default now()
);

-- The SSE "state" event of tunnel mode (about 3 KB). Kept apart from job_views so that each
-- Realtime event stays small.
create table public.job_live (
  job_id uuid primary key references public.jobs (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  state jsonb not null check (octet_length(state::text) <= 65536),
  updated_at timestamptz not null default now()
);

-- The other GET answers: frames (up to about 400 KB for a long bag), selection and parts. Not in the
-- Realtime publication.
create table public.job_snapshots (
  job_id uuid not null references public.jobs (id) on delete cascade,
  path text not null check (path in ('frames', 'selection', 'parts')),
  user_id uuid not null references auth.users (id) on delete cascade,
  status int not null,                                   -- 200, or the route's own 404
  body jsonb not null check (octet_length(body::text) <= 1048576),
  updated_at timestamptz not null default now(),
  primary key (job_id, path)
);

-- Which served file is which Storage object. The page signs URLs from this list.
create table public.job_files (
  job_id uuid not null references public.jobs (id) on delete cascade,
  path text not null check (char_length(path) <= 200),   -- the ALLOW path, e.g. run/posed.glb
  user_id uuid not null references auth.users (id) on delete cascade,
  object text not null check (char_length(object) <= 400),  -- <user>/<job>/<sha256[:16]>/<path> in job-files
  bytes bigint not null check (bytes >= 0),
  sha256 text not null check (sha256 ~ '^[0-9a-f]{64}$'),
  role text not null check (role in ('review', 'result', 'log')),
  content_type text not null check (char_length(content_type) <= 100),
  updated_at timestamptz not null default now(),
  primary key (job_id, path),
  check (object = user_id::text || '/' || job_id::text || '/' || left(sha256, 16) || '/' || path)
);
create index job_files_user_idx on public.job_files (user_id);
create index job_files_object_idx on public.job_files (object);

-- Log text of running stages, scrubbed by the worker. At most 256 KB per stage run (the worker's cap).
create table public.job_logs (
  id bigint generated always as identity primary key,
  job_id uuid not null references public.jobs (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  stage text not null check (stage ~ '^[a-z0-9_]{1,20}$'),
  run_started double precision,                         -- the stage run's "started" in job.json (epoch s)
  pos bigint not null,                                   -- byte offset after this text: the SSE id <stage>:<pos>
  text text not null check (char_length(text) <= 65536),
  created_at timestamptz not null default now()
);
create index job_logs_job_idx on public.job_logs (job_id, id);

-- The relayed requests. Users add rows only through relay() and delete_job(). The personal worker of the
-- job's machine claims them (own_claim_request), sends each to the unchanged FastAPI app and writes the
-- answer (own_answer). A finished request keeps only a small body (private.on_request_done), and the sweep
-- drops a large response after an hour and the row after 7 days, so that requests cannot fill the database.
create table public.requests (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users (id) on delete cascade,
  job_id uuid not null references public.jobs (id) on delete cascade,
  route text not null check (route in ('topic', 'calibration_put', 'calibration_delete', 'start', 'overrides',
                                       'parts', 'continue', 'cancel', 'delete')),
  query jsonb not null default '{}'::jsonb check (jsonb_typeof(query) = 'object' and octet_length(query::text) <= 1024),
  body jsonb not null default '{}'::jsonb check (jsonb_typeof(body) = 'object' and octet_length(body::text) <= 131072),
  body_text text check (octet_length(body_text) <= 65536),   -- the calibration file's text
  key_ref uuid,                                              -- the job_secrets row with the sealed key
  state text not null default 'pending' check (state in ('pending', 'taken', 'done', 'withdrawn', 'expired')),
  sent_online boolean not null default false,                -- the job's worker took builds at send time
  created_at timestamptz not null default now(),
  taken_at timestamptz,
  worker_id uuid,
  done_at timestamptz,
  status int,                                                -- the app's HTTP status
  response jsonb check (response is null or octet_length(response::text) <= 65536)
);
create index requests_pending_idx on public.requests (id) where state = 'pending';
create index requests_open_idx on public.requests (job_id, id) where state in ('pending', 'taken');
create index requests_user_idx on public.requests (user_id, created_at desc);

-- Sealed API keys (ECDH P-256, HKDF-SHA256, AES-256-GCM to the worker's public key). Written only by
-- relay(). Nobody reads the table: own_claim_request hands the sealed parts to the worker that claimed the
-- request and deletes the row.
create table public.job_secrets (
  ref uuid primary key,                                      -- made by the browser, part of the AAD
  job_id uuid not null references public.jobs (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  worker_id uuid not null references public.workers (id) on delete cascade,
  agent text not null check (agent in ('astra', 'opus')),
  epk text not null check (char_length(epk) = 88),
  iv text not null check (char_length(iv) = 16),
  ct text not null check (char_length(ct) between 24 and 704),
  created_at timestamptz not null default now()
);

-- ---- RLS ----
alter table public.settings enable row level security;
alter table public.access enable row level security;
alter table public.usage_daily enable row level security;
alter table public.machines enable row level security;
alter table public.workers enable row level security;
alter table public.jobs enable row level security;
alter table public.job_views enable row level security;
alter table public.job_live enable row level security;
alter table public.job_snapshots enable row level security;
alter table public.job_files enable row level security;
alter table public.job_logs enable row level security;
alter table public.requests enable row level security;
alter table public.job_secrets enable row level security;    -- no policy at all

create policy "read settings" on public.settings for select to authenticated using (true);
create policy "read own access" on public.access for select to authenticated
  using (user_id = (select auth.uid()));
create policy "read own usage" on public.usage_daily for select to authenticated
  using (user_id = (select auth.uid()));
create policy "read own machines" on public.machines for select to authenticated
  using (user_id = (select auth.uid()));
-- The user's own workers seen in the last 30 days (a worker that has run for longer stays visible).
create policy "read own recent workers" on public.workers for select to authenticated
  using (user_id = (select auth.uid()) and last_seen > now() - interval '30 days');
create policy "read own jobs" on public.jobs for select to authenticated
  using (user_id = (select auth.uid()) and deleted_at is null);
create policy "read own views" on public.job_views for select to authenticated
  using (user_id = (select auth.uid()));
create policy "read own live state" on public.job_live for select to authenticated
  using (user_id = (select auth.uid()));
create policy "read own snapshots" on public.job_snapshots for select to authenticated
  using (user_id = (select auth.uid()));
create policy "read own files" on public.job_files for select to authenticated
  using (user_id = (select auth.uid()));
create policy "read own logs" on public.job_logs for select to authenticated
  using (user_id = (select auth.uid()));
create policy "read own requests" on public.requests for select to authenticated
  using (user_id = (select auth.uid()));

-- ---- grants: revoke everything, then grant what users read ----
revoke all on public.settings, public.access, public.usage_daily, public.machines, public.workers, public.jobs,
  public.job_views, public.job_live, public.job_snapshots, public.job_files, public.job_logs, public.requests,
  public.job_secrets from public, anon, authenticated, service_role;
revoke all on private.rate from public, anon, authenticated, service_role;
revoke all on sequence public.job_logs_id_seq, public.requests_id_seq from public, anon, authenticated, service_role;

grant select on public.settings, public.access, public.usage_daily, public.machines, public.workers, public.jobs,
  public.job_views, public.job_live, public.job_snapshots, public.job_files, public.job_logs, public.requests
  to authenticated;
