-- HARPOR public site, part 4: the Storage bucket, its policies and the functions about objects.
--
-- job-files: what the page shows and downloads: review images, logs and result files. Object names are
--            <user_id>/<job_id>/<sha256[:16]>/<path>. A personal worker writes the files of its user's jobs
--            (that user's session), while the user is approved and the user's objects hold less than
--            access.max_result_bytes. The files use the owner's Storage. Users read and delete their own
--            folder; nobody overwrites an object (no update policy).
--
-- file_size_limit: the Free plan caps every file at 50 MB. On Pro, set the project's global limit
-- first (Storage settings), then raise the bucket and settings.plan_file_limit together:
--   update storage.buckets set file_size_limit = 524288000 where id = 'job-files';
--   update public.settings set plan_file_limit = 524288000;

-- allowed_mime_types: the types of relay.SITE_TYPES, which the worker sends (a test checks the list), and
-- text/plain without its charset in case Storage compares the type without parameters. A URDF goes as
-- text/plain, so the bucket takes no XML. A personal worker therefore cannot keep HTML, XML or programs
-- that a browser would run from the project's domain.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types) values
  ('job-files', 'job-files', false, 52428800,
   array['model/gltf-binary', 'image/jpeg', 'image/png', 'application/json', 'application/zip',
         'text/plain; charset=utf-8', 'text/plain', 'video/mp4'])
on conflict (id) do nothing;

-- Users read (and so may sign URLs for) only their own folder.
create policy "harpor job-files: read own" on storage.objects for select to authenticated
  using (bucket_id = 'job-files' and (storage.foldername(name))[1] = (select auth.uid())::text);

-- A policy on storage.objects cannot read storage.objects itself (Postgres refuses the recursion), so
-- the sum of the caller's objects is a security definer function. Both functions answer only for the
-- caller's own folder, so the page may call them too (storage_used shows "storage used").

-- The bytes of the caller's objects in job-files.
create function public.storage_used() returns bigint
language sql stable security definer set search_path = '' as $$
  select coalesce(sum((o.metadata ->> 'size')::bigint), 0) from storage.objects o
   where o.bucket_id = 'job-files' and o.name like auth.uid()::text || '/%'
$$;

-- True when the caller may upload p_name: the caller is approved, the name has the form
-- <user>/<job>/<16 hex>/<path> without ".." or "//", the job is the caller's and not deleted, and the
-- caller's objects hold less than max_result_bytes. Storage asks this once, when an upload or a signed
-- upload URL starts, and counts only the objects that are there already. So uploads that run at the same
-- time can go past the cap, and so can signed upload URLs, which work for 2 hours, also after the owner
-- took the approval back. The cap is a soft limit (docs/public_site.md, "A risk that stays").
create function public.own_object_ok(p_name text) returns boolean
language sql stable security definer set search_path = '' as $$
  select exists (
    select 1 from public.jobs j join public.access a on a.user_id = j.user_id
     where j.user_id = auth.uid() and a.approved and j.deleted_at is null
       and split_part(p_name, '/', 1) = auth.uid()::text and j.id::text = split_part(p_name, '/', 2)
       and p_name ~ '^[0-9a-f-]{36}/[0-9a-f-]{36}/[0-9a-f]{16}/[A-Za-z0-9_-][A-Za-z0-9_./-]{0,199}$'
       and p_name !~ '\.\.|//'
       and public.storage_used() < a.max_result_bytes)
$$;

create policy "harpor job-files: own machine uploads" on storage.objects for insert to authenticated
  with check (bucket_id = 'job-files' and public.own_object_ok(name));
-- Users delete their own objects, also without approval: the page deletes the files of a job that it
-- deletes, and a personal worker the files of its jobs and the orphans in its user's folder.
create policy "harpor job-files: delete own" on storage.objects for delete to authenticated
  using (bucket_id = 'job-files' and (storage.foldername(name))[1] = (select auth.uid())::text);

-- The objects of one of the caller's jobs, also of one that waits in "deleting", oldest first, at most
-- p_limit (1 to 1000) a call. The page calls it after delete_job and the worker for a delete request; each
-- deletes the objects through the Storage API (a SQL delete leaves the file behind) and calls again until
-- fewer than p_limit come back.
create function public.job_objects(p_job uuid, p_limit int default 1000)
returns table (name text, created_at timestamptz)
language plpgsql stable security definer set search_path = '' as $$
begin
  if not exists (select 1 from public.jobs j where j.id = p_job and j.user_id = auth.uid()) then
    raise exception 'no such job';
  end if;
  return query
    select o.name, o.created_at from storage.objects o
     where o.bucket_id = 'job-files' and o.name like auth.uid()::text || '/' || p_job::text || '/%'
     order by o.created_at
     limit greatest(1, least(coalesce(p_limit, 1000), 1000));
end $$;

-- Objects in the caller's folder that nothing names, older than an hour: in the folder of a job of this
-- worker's machine without a job_files row (a file replaced while the worker was down, or an upload
-- whose own_files never came), or in a folder whose job row is gone (deleted while the machine was away,
-- or with its machine). The worker deletes them through the Storage API at its start and every hour. At
-- most p_limit (1 to 1000), oldest first.
create function public.own_orphans(p_worker uuid, p_limit int default 1000) returns table (name text)
language plpgsql stable security definer set search_path = '' as $$
declare
  w public.workers := private.own_worker(p_worker);
begin
  return query
    select o.name from storage.objects o
     where o.bucket_id = 'job-files' and o.name like w.user_id::text || '/%'
       and o.created_at < now() - interval '1 hour'
       and not exists (select 1 from public.job_files f where f.object = o.name)
       and not exists (select 1 from public.jobs j where j.id = private.try_uuid(split_part(o.name, '/', 2))
                        and j.machine_id <> w.machine_id)
     order by o.created_at
     limit greatest(1, least(coalesce(p_limit, 1000), 1000));
end $$;

-- For the owner, in the SQL editor: the job folders in job-files whose job row is gone, with their size,
-- and whether the account is gone too. A personal worker of the same user deletes such a folder when it
-- runs; the folders of a deleted account stay until the owner deletes them in the Storage dashboard.
create function private.orphan_folders() returns table (folder text, objects bigint, bytes bigint, account_gone boolean)
language sql stable set search_path = '' as $$
  select x.folder, count(*), coalesce(sum(x.size), 0)::bigint,
         not exists (select 1 from public.access a where a.user_id = private.try_uuid(split_part(x.folder, '/', 1)))
    from (select split_part(o.name, '/', 1) || '/' || split_part(o.name, '/', 2) || '/' as folder,
                 (o.metadata ->> 'size')::bigint as size, split_part(o.name, '/', 2) as job
            from storage.objects o where o.bucket_id = 'job-files') x
   where not exists (select 1 from public.jobs j where j.id = private.try_uuid(x.job))
   group by x.folder
   order by x.folder
$$;

revoke execute on function public.storage_used(), public.own_object_ok(text), public.job_objects(uuid, int),
  public.own_orphans(uuid, int), private.orphan_folders() from public, anon, authenticated, service_role;
grant execute on function public.storage_used(), public.own_object_ok(text), public.job_objects(uuid, int),
  public.own_orphans(uuid, int) to authenticated;
