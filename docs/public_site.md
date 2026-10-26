# The public HARPOR site

The public site is a static page that talks only to Supabase. Users sign in with Google or GitHub. Every job
runs on a user's own GPU machine. A personal worker runs inside the HARPOR server on that machine
(`python -m gator.webapp --worker`). It signs in as the user and serves only that user's jobs on that machine.
Inputs stay on the machine. Jobs, views, logs and result files are copied to Supabase, so the site shows them.
The site's owner can run a personal worker on any machine of theirs, like any other user.

The worker is a thread inside the server process. It shares the server's one job runner, sends each request
from the site to the unchanged server routes, and makes only outbound HTTPS calls. A server started without
`--worker` runs no worker and serves only the local UI, which the tunnel mode (`start_ice.sh`, `open_mac.sh`
and the login token) opens.

In both modes the runner writes `.runner.json` into the runs folder, and a second server on the same runs
folder refuses to start while the first one is known to run (a HARPOR server process of yours on the same
node, or a Slurm job that squeue reports as running). Stop the first server, or, if it is gone, delete
`.runner.json` in the runs folder. A file that cannot be checked (no Slurm job in it, or no answer from
squeue) counts as stale, and the server starts.

Where the code is:

| Part | Files |
|---|---|
| Database | `supabase/migrations/` (six files), `supabase/seed.sql` and `supabase/config.toml` (local stack only) |
| Worker | `gator/webapp/worker.py`, `relay.py`, `supa.py`, `keybox.py` and `session.py` (the worker's sign-in and commands) |
| Page | `gator/webapp/static/supa.js`, the transport seam in `static/app.js`, `static/help.html` and `gator/webapp/site.py` |
| Demos | `gator/webapp/demos.py` and `static/demos.html` (see Demos below) |
| Tests | `tests/test_supabase_sql.py` with `tests/pgcheck/`, `test_webapp_supa.py`, `test_webapp_keybox.py`, `test_webapp_relay.py`, `test_webapp_worker.py`, `test_webapp_session.py`, `test_webapp_contract.py`, `test_webapp_supa_js.py`, `test_site.py`, `test_help_page.py`, `test_webapp_demos.py` and `fake_supabase.py` |

Help for users is in one file, `static/help.html` (build_plan.md note 10), and `site.py` builds the public
site's copy. The build puts the public text of `site.py`'s `HELP_PUBLIC` in place of each block that the file
marks `<!-- local:NAME -->`: the upload, the SSH tunnel, the `?token=` link and the agent "None" go, and the
sections on the account, setting up a machine and the machines' status come in. On the public jobs list, the
machines panel lists each machine with its state, and its "Set up a machine" link opens the setup steps on the
help page. The banner under the header shows only the machines that are offline or stopping.

## What the worker does

On start, the worker makes a new key pair for sealed API keys (in memory only) and a workers row. The
machine's older workers stop. Then the worker checks the jobs rows of its machine against the job folders on
the machine. A row whose folder is gone becomes failed with a note. A folder whose row is gone is no longer
copied, and the folder stays. Next the worker deletes orphan objects in the user's folder in Storage. Jobs that
the last stop cut off go on from the first stage that is not done, without a key, at most twice per job. Then
every job is published once, and the loops start:

| Loop | Every | What |
|---|---|---|
| requests | 2 s while idle | up to 4 requests at once, one per job; the job is published before the answer |
| local jobs | 5 s | a new local upload gets a row on the site |
| sync | 1 s | logs, files, view, live state, snapshots and row of the jobs that change |
| heartbeat | 10 s | the workers row; inside a Slurm job, the job's end is read again every 10 min |
| housekeeping | 60 s, hourly | the settings row and old keys; every hour also retention and orphan objects |

Near the end of a Slurm job (the last `settings.near_end_minutes`, 60 by default) the worker takes no new
builds. They wait, in order, for the next start, and the page sends no key for them. When the server stops
(Ctrl-C, scancel, or the time limit's SIGTERM), the worker stops first. Active jobs keep their status with the
note "<machine> stopped. This job goes on when HARPOR runs there again.", and the workers row stops. Then the
runner cancels the stages.

The worker drops the key of a job that waited in review for more than 24 hours. The job asks for it again at
Build.

## Cleanup with no secret key

No part of HARPOR uses the project's secret key. The cleanup is split this way:

| What | Who | How |
|---|---|---|
| requests that waited 7 days (they expire), jobs that waited in "deleting" as long, sealed keys after 1 h, large answers after 1 h, finished requests after 7 days, rate counts after 2 days | pg_cron, every 15 min | `private.sweep()` |
| the log text of a job (`job_logs`), 14 days after the job stopped | pg_cron | `private.sweep()` |
| review files (images and LiDAR points) and log files 14 days after a job stopped, result files after 90 days | the worker of the job's machine, every hour | it publishes the job again without them. For a job whose folder is gone from the machine, it counts from the row's `stopped_at` and drops them from the file list. Their objects go 10 minutes later |
| objects older than 1 h that no file row names, in the folders of the machine's jobs and in folders whose job row is gone | each worker, at its start and every hour, in its user's folder | `own_orphans` |
| the view, live state, snapshots, file list and log text of a deleted job | `delete_job`, at once | the row alone waits in "deleting" for the machine |
| the files of a deleted job | the page, at once; the machine's worker, when it takes the delete request | `job_objects`, then the Storage remove call with the user's session |

What no automatic cleanup reaches:

1. A deleted account's folder `<user id>/` in job-files. When you delete a user in Auth, the rows go by
   cascade, but no session can delete the files any more. List them with
   `select * from private.orphan_folders();` (`account_gone` is true), and delete the folder in the Storage
   dashboard.
2. The files of jobs whose machine never runs HARPOR again. File retention does not run for them. They stay
   until the user deletes the job on the page (the page deletes the files), or until you delete the account
   and then its folder (case 1). Their log text still goes after 14 days.
3. The folders of jobs that went with a machine (`delete from public.machines ...`) when no machine of that
   user runs a worker again. They show in `orphan_folders()` with `account_gone` false. Ask the user to delete
   the machine's jobs on the page before you remove the machine.
4. Objects whose delete on the page broke off, of a job whose machine never comes back. They show in
   `orphan_folders()` once the sweep has removed the row.

## The owner's setup

Nothing below is done by the code. Do every account step and every git step yourself.

1. **The Supabase project.** Pick a region near the users. Start on Free. Note the project ref, the URL and
   the publishable key. HARPOR needs no secret key, so do not copy it anywhere.
2. **Auth settings.** Turn off email, phone and anonymous sign-in. Set the Site URL to the site's directory
   URL (`https://<site>/` or `https://<user>.github.io/<repo>/`). The redirect list holds that URL exactly,
   `http://localhost:8766/` for local tests and `http://127.0.0.1:8767/auth/callback` for `worker login`.
   Use no wildcard on a shared host.
3. **Google.** An OAuth client of type "Web application". JavaScript origin: the site's origin, without the
   `/<repo>/` path. Redirect URI: `https://<ref>.supabase.co/auth/v1/callback`. Scopes: openid, email and
   profile only. Move the consent screen to "In production", and paste the client id and secret into
   Supabase's Google page.
4. **GitHub.** An OAuth App (not a GitHub App). Homepage URL: the site URL. Callback URL:
   `https://<ref>.supabase.co/auth/v1/callback`. Leave device flow off, and paste the id and secret into the
   GitHub provider page. The loopback login of a personal worker needs nothing more at Google or GitHub.
5. **Migrations.** Turn on Cron under Integrations first. It is required: only pg_cron runs the sweep. On the
   Mac: `brew install supabase/tap/supabase`, then `supabase login`, `supabase link --project-ref <ref>` and
   `supabase db push` from the repo. Without the CLI, paste the six files of `supabase/migrations/` in order
   into the SQL editor. Then run the Security Advisor. Never run `supabase config push`:
   `supabase/config.toml` is for the local stack.
6. **Storage.** Check that the bucket `job-files` exists with a 50 MB file limit. On Free the whole project has
   1 GB, so give each account a smaller Storage cap when you approve it (step 9). On Pro, set the project's
   global file limit first, then run the two update lines at the top of
   `supabase/migrations/20261101000400_harpor_storage.sql`.
7. **The site config.** `python -m gator.webapp.site init --url https://<ref>.supabase.co --key
   sb_publishable_...` writes `gator/webapp/site.json`. Add `--contact TEXT` to name a way to reach you on the
   privacy page. The values are public by design, and the build refuses a secret key. Commit `site.json`:
   personal machines get it with the repo, and `python -m gator.webapp.worker login` reads it (or pass
   `--site-config PATH`). Build the site with
   `python -m gator.webapp.site build --demos ~/lunar_lab_research/harpor_demos`, which writes `site/` with the
   demos (see Demos below).
8. **The host.** The same `site/` folder works on Cloudflare Pages (upload `site/` in the dashboard, or
   `npx wrangler pages deploy site`) and on GitHub Pages. `.gitignore` leaves `site/` out of the main branch,
   so on GitHub Pages either push the contents of `site/` to a `gh-pages` branch and publish that branch, or
   add an Actions workflow that runs `python -m gator.webapp.site build` and uploads `site/` with
   `actions/upload-pages-artifact`. Cloudflare also sets the security headers of `site/_headers`. Put the
   final URL into steps 2 to 4.
9. **Approval.** Each account needs your approval, because its machines copy results into your Storage. Sign
   in once on the site, then approve accounts in the SQL editor:

   ```sql
   update public.access set approved = true where email = '<email>';
   ```

   - On Free, add a Storage cap of about 200 MB and fewer jobs: `, max_result_bytes = 209715200,
     max_own_jobs = 10`. The database has 500 MB on Free. Each job may keep up to 2 MiB of log text and about
     1 to 3 MB of views and snapshots there.
   - To remove a machine, ask the user to delete its jobs on the page first. Then run
     `delete from public.machines where id = '<id>';`. Its jobs and workers go by cascade.
   - To stop an account, set `approved = false`. Its machines stop copying at once. Its files stay, and the
     account can still download them, so they still cost Storage and egress. The page then shows only the
     waiting panel, so the user cannot delete the jobs there. To remove the files, run
     `delete from public.jobs where user_id = '<id>';` and then delete the folder `<user id>/` in job-files.
   - To delete an account, delete the user in Auth. Then delete the folder `<user id>/` in job-files.
     `select * from private.orphan_folders();` lists what is left.

   Then run one real job end to end on a personal machine, with a second test account.
10. **Before anyone else signs in.** Run gitleaks or trufflehog over the whole git history, turn on GitHub
    secret scanning and push protection, check the privacy page, and decide on the plan. Free pauses after
    about 7 days of low activity (sign-in stops too) and has 5 GB of egress a month. On Pro, turn the spend
    cap on.

**A risk that stays.** A personal worker is the user's own code. It can write anything within its user's caps
into that user's rows and folder, and the copies use your Storage and egress. The Storage cap
(`access.max_result_bytes`) is a soft limit. Storage checks it once, when an upload or a signed upload URL
starts, and counts only the files that are there already. So uploads that run at the same time can go past it,
and so can signed upload URLs, which work for 2 hours, also after you take the approval back. On Free, one
account could fill the project's 1 GB this way. Approve only people you trust, and watch the use per account:

```sql
select split_part(name, '/', 1) as user_id, sum((metadata ->> 'size')::bigint) as bytes
  from storage.objects where bucket_id = 'job-files' group by 1 order by 2 desc;
```

The page also trusts the public key in the machine's workers row. Someone who can write the database (you, or
anyone with the project's secret key or database password) could put in another key and read the keys sealed
after that. `--local-keys` avoids this, since the key then never leaves the machine.

## Live checks

Run these on the CLI's local stack first where possible, then on the cloud project.

| Check | How | If it fails |
|---|---|---|
| PKCE from Python, and the loopback entry in the redirect list | one real `worker login` | match gotrue's parameter names in the supabase-js 2.118.0 bundle |
| Refresh token rotation | a personal worker that runs over 2 h, with one restart | keep the lock; the file must hold the new token before the first call with the new access token |
| CORS and 206 for Range on signed URLs from the site's origin | open a job with a preview video and seek it; open a job with a 360 video and see it play and loop in Results (Safari refuses media without 206); open a GLB in Results; or copy a signed URL from the page and run `curl -s -o /dev/null -D - -H "Origin: https://<site>" -H "Range: bytes=0-1023" "<URL>"` (want 206 and Access-Control-Allow-Origin) | Range: the preview cannot seek and the 360 video may not play, and supa.js would need to fetch them as blobs (not built); CORS: model-viewer cannot load GLBs, so ask Supabase support |
| `storage.objects.metadata->>'size'` | SQL editor: `select metadata from storage.objects limit 1` | change the key name in the functions that read it |
| The statement timeout of authenticated | SQL editor: `select rolname, rolconfig from pg_roles where rolname in ('authenticated', 'authenticator')` | only a very large `own_publish` could come near it |
| pg_cron runs the sweep | after the first quarter hour: `select status, return_message from cron.job_run_details order by start_time desc limit 5` | turn on Cron, and push the cron migration again |
| Realtime with RLS on requests, job_live and job_logs | a second account sees none of the first one's events | fix the policy or the filter |
| Realtime events for rows written inside security definer functions (`own_publish`, `own_answer`) | the page follows a running job | the page falls back to its 2 s poll, which RelayEventSource already has |
| model-viewer and Draco under the CSP | open real GLBs on each host that you try | add only what the browser console names, to both policies |
| A Storage upload with a user's session under the insert policy | the first file of a real job | check that Storage runs the policy as authenticated with the JWT claims |
| A signed upload URL after the approval is taken back | with a test account, make a signed upload URL (`createSignedUploadUrl`) for its job folder, set `approved = false`, then upload to the URL | if the upload goes through, the soft cap of "A risk that stays" holds as written; if Storage refuses it, the risk is smaller |
| A Storage delete with a user's session under the own-folder policy | delete a job whose machine is offline, and watch Storage use in the dashboard drop | the read-own select policy may be needed too; it is there |
| The content types of `job-files` | the first log file and URDF (both `text/plain; charset=utf-8`) and GLB of a real job | change `allowed_mime_types` in the storage migration to what Storage takes, and keep it narrow: no XML, HTML or SVG |
| A 1 MB snapshot through `own_publish` | a long bag on a personal machine | send the frames snapshot in an `own_publish` call of its own, or lower the lean thumbnail count |
| The site under `/<repo>/` | sign in and open a job on GitHub Pages | a path that starts with `/` slipped in |
| The demos' videos and GLBs on the host | signed out, open `demos.html` in Safari, open each card and see its video play and loop; open 3D view and watch the network list: the GLB loads only then; `curl -s -o /dev/null -D - -H "Range: bytes=0-1023" https://<site>/demos/<name>/turntable.mp4` (want 206 and `video/mp4`) | Range: Safari does not play the video; host the videos elsewhere and add that host to `media-src` |
| The demos page under the CSP | open `demos.html` with the console open | add only what the console names, to both policies |

## Run a personal worker on your own GPU machine

The owner must approve your account before your machines can show jobs on the site.

**What the machine needs.** The full HARPOR setup: the repo checkout with `gator/webapp/site.json` from the
site, the envs (`envs/web` for the server, and the stage envs), the containers and the model checkouts. The
worker adds one package to the server's Python:

```bash
<envs/web>/bin/pip install cryptography==50.0.2
```

The machine also needs outbound HTTPS to the Supabase project, and a browser for the sign-in (or a port
forward, below). It needs no open port: the worker only calls out.

**Steps.**

1. Sign in and add the machine: `python -m gator.webapp.worker login` (GitHub by default;
   `--provider google`, and `--name lab-box` for the machine's name on the site). It listens on
   127.0.0.1:8767 for the sign-in's return, so on a machine without a browser, forward the port first
   (`ssh -L 8767:127.0.0.1:8767 <machine>`), run `login --no-browser` there, and open the printed URL on your
   laptop. A code that someone else catches is useless without the verifier, which never leaves the login.
   Before the owner approves you, the login signs in, and the server adds the machine at its first start
   after the approval.
2. Start the server with the worker: `python -m gator.webapp --worker`. It listens on 127.0.0.1 unless
   `--host` says otherwise, so the local UI stays on the machine. With `--local-keys opus` (or `astra`, or
   `astra,opus`) the worker uses the key in `~/.config/gator/anthropic.env` (or `openai.env`) when a job needs
   it, and the page sends no key. The server refuses to start when such a file is not 0600 in a 0700 folder.
3. Open the site. The machine's jobs show there, with "on lab-box". New inputs go in through the local UI, or
   with `python -m gator.webapp.worker add PATH`, which streams the file to the server on this machine.
   Jobs made before your first login stay local. The server prints the local UI's URL with its login token. On
   a machine without a browser, forward the port from your laptop (`ssh -L PORT:127.0.0.1:PORT <machine>`)
   and open that URL there.

**Inside a Slurm job.** The worker reads the job's end and takes no new builds in its last hour. Start the server
as a step of the job, from the login node, so that it shares the job's GPU and knows the job:

```bash
nohup setsid srun --jobid=JOBID --overlap --ntasks=1 bash -lc \
  'cd <repo> && exec <envs/web>/bin/python -m gator.webapp --worker --host 0.0.0.0' > worker.log 2>&1 < /dev/null &
```

- `--host 0.0.0.0` lets the login node reach the server. The login token still guards it, as in the tunnel
  mode. `start_ice.sh` and `open_mac.sh` then find this server in `webapp.json` and reuse it, so the local UI
  opens on your laptop.
- Do not run `start_ice.sh` while a worker server that listens on 127.0.0.1 runs in the job. It cannot reach
  that server, so it deletes `webapp.json` and tries to start a second server, which the first one blocks.
- Run `worker login --name NAME` on the login node before you start the server. The home folder is shared, so
  the compute node finds the session file. Forward port 8767 to the login node for the sign-in, as in step 1.
- Run `worker add PATH` on the compute node, for example with `srun --jobid=JOBID --overlap`, since it sends
  the file to 127.0.0.1.

`start_ice.sh` itself starts only the tunnel mode.

`python -m gator.webapp.worker status` prints the account and the machine. While the server holds the session,
it says so; otherwise it also prints your approval, your agents and the Storage you use on the site.
`python -m gator.webapp.worker logout` signs this machine out and takes the refresh token out of the session
file. The machine and its jobs stay on the site, and the file keeps the machine's id, so the next `login` on
this machine uses the same machine on the site. Delete the file to start over as a new machine.

A job that you delete on the page loses its files, logs and views on the site at once. The worker also deletes
the job's folder on the machine when it takes the delete request. While the machine is offline, that request waits up to 7 days.
After that the folder stays on the machine, and you can delete it in the machine's own HARPOR page.

**Files on the machine.**

| File | Mode | What it holds |
|---|---|---|
| `~/.config/gator/` | 0700 | made with 0700 when missing; refused when others may open it |
| `~/.config/gator/harpor_session.json` | 0600 | the site's URL and publishable key, your user, email and provider, the machine's id and name, the time from which local jobs are copied, and the refresh token (none after `logout`) |
| `~/.config/gator/.harpor_session.lock` | 0600 | the lock of the one process that may use the session, with its pid |
| `~/.config/gator/openai.env`, `anthropic.env` | 0600 | optional; read only with `--local-keys`, at the moment a job needs the key |
| `~/.config/gator/webapp.json` | 0600 | as in the tunnel mode, plus `"worker"`: true while the worker runs in the server, else false |
| `gator/webapp/site.json` (in the repo) | | the project's URL and publishable key, and the optional contact: public values |

No machine holds a secret key of the project. Do not copy the session file to another machine. Refresh tokens
work once, so two machines with one file end the session for both. Run `login` on each machine.

**What the worker writes.** It writes only through the `own_*` functions, and only into your own folder in
`job-files`. Once your files reach your Storage cap (`access.max_result_bytes`), new files stay on the machine,
with a note in the job's warnings, and so does a file over the bucket's file limit. A job that you delete in
the machine's own HARPOR page stays on the site as failed, and its files there follow the same retention. The site keeps at most 2 MiB of log text per job
(`settings.own_log_bytes`); the full logs stay on the machine. When the owner takes your approval back, the
worker pauses the copy and keeps the local server running. When your session is signed out, the server keeps
running in tunnel mode and says to run `login` again.

**What the site owner can see.** Your input file stays on your machine. The job's state, logs, review images,
LiDAR points and result files are copied to the site's Supabase project so that you can see them there, and the site's
owner runs that project and can read these copies. Your API key goes only to your machine, sealed in your
browser, or it stays in your machine's own key file.

## Agents

The public site runs GPT-6-Astra (`astra`) and Claude Opus 5.5 (`opus`), each with the user's own key: an
OpenAI key for astra and an Anthropic key for opus. The agent "none" stays in the tunnel mode. The owner can
narrow `access.agents` to `{opus}` or `{astra}` for one user.

GPT-6-Astra runs the Codex CLI in a container that can reach the network, and Claude Opus 5.5 runs its shell
with no network access. The help page's agent section (`help.html#agent`) says what this means for users.

## Demos

The Demos link in the header opens `demos.html`: a gallery of cards for the finished jobs that you picked. A card
opens its demo, with its 360 video, a 3D view of `aligned.glb`, its facts and its files. It works signed out. The
demos are files of the site, not rows or Storage objects, so nothing in Supabase changes.

- **The folder.** `WEBAPP_DEMOS` names it, and `--demos DIR` overrides it for the server, the site build and
  `demos add`. There is no default. Keep it outside the repo, for example `~/lunar_lab_research/harpor_demos`.
  Each demo is a folder `<name>/` (a-z, 0-9 and -) with `demo.json` and the files it lists: `turntable.mp4`,
  `poster.jpg`, `aligned.glb` and `results.zip`. The module docstring of `demos.py` gives the format.
- **Make a demo.** On the machine that holds the job (ICE for real jobs; `envs/web` has PyAV and Pillow):
  `python -m gator.webapp.demos add <job folder> --out <dir>`. It takes the job's 360 video, so start the job with
  "Make a 360 video" on, or give `--video`. Add `--title`, `--order` or `--fact LABEL=VALUE` (`LABEL=` removes a
  fact). A run folder without `job.json` needs `--video` and `--glb`. Copy the demo folder to the Mac's demos folder
  with scp, then run `python -m gator.webapp.demos check`.
- **Show it.** `python -m gator.webapp.site build --demos DIR` copies the demos into `site/`, and its second line
  says how many, with the placeholders. A folder that is not there stops the build, and so does a build without a
  folder over an earlier build that had demos or placeholders: its deploy would drop the live gallery. Add `--no-demos` when that is what you want; it
  publishes "No demos yet.". A GitHub Actions build cannot see a folder outside the repo, so build on the Mac
  (Cloudflare). The local server shows the same page at `/demos.html` with `--demos DIR`; it reads the folder when
  it starts, and a folder that is not there stops it. The page loads no video until a demo opens; then its video
  starts by itself unless the system asks for reduced motion (`static/demos.js`).
- **Placeholders.** `placeholders.json` in the demos folder adds cards that say "Coming soon" and open nothing, after
  the demos' cards: `{"version": 1, "placeholders": ["Drawer cabinet", "Laptop"]}`. The module docstring of
  `demos.py` gives the rules. The server, the build and `check` read it with the guard below. A placeholder with the
  title of a demo shows no card, so the demo takes its place. Edit the file as demos come in, or delete it.
- **The guard.** `add`, `check`, the server and the build refuse a demo that looks private (the words of
  `demos.PRIVATE`) in `demo.json`, a file name, the text in `results.zip` or the names in a GLB. `add` also checks
  the job's input name, prompt and camera topic and the text of `selection.json` and `lidar.json`, and refuses a
  rosbag job unless you add `--public-bag`, since the rosbags so far are private and a bag's name can look
  harmless. The guard reads names only, so look at each demo yourself before you publish it.
- **Clean files.** `add` packs `results.zip` again: in its text files every key shape becomes `[key]` and every
  path of a machine (`demos.MACHINE_PATH`) becomes its file name. A binary file that holds either stops `add`. The
  checks also refuse a file over the limit below, a type other than MP4, GLB, ZIP and JPEG, and a GLB that needs
  Draco or KTX2 (the demos page loads no decoder).

## Limits

| Limit | Default | Where |
|---|---|---|
| largest file on the site | 50 MB on Free, 500 MB on Pro | `settings.plan_file_limit` and the bucket |
| Storage per user | 2 GiB (about 200 MB on Free) | `access.max_result_bytes` |
| jobs on the site | 20 at once, 20 a day per user | `access.max_own_jobs`, `max_own_per_day` |
| log text of a job on the site | 2 MiB | `settings.own_log_bytes` |
| machines | 5 per user | `settings.max_machines` |
| requests | 60 an hour per user | `access.max_requests_per_hour` |
| waiting | a request 7 days | `settings.pending_days` |
| a request | a JSON body of 128 KB, nested at most 32 deep; once done, the body keeps 1 KB, an answer over 1 KB goes after an hour and the row after 7 days | `relay()`, `private.sweep()` |
| kept on the site | log text 14 days after a job stops; review images, LiDAR points and log files 14 days, results 90 days, while the machine runs | the sweep, the worker |
| a demo's file | 25 MiB (Cloudflare Pages' limit per file) | `demos.MAX_FILE` |
