-- Test accounts for the Supabase CLI's local stack only. `supabase db reset` runs this file after the
-- migrations ([db.seed] in config.toml). Never run it on the cloud project.
--
-- Three accounts with a confirmed email and no usable password. A local test sets a password through
-- the local stack's admin API (PUT /auth/v1/admin/users/<id>) before it signs in, so the repo holds no
-- password. The access rows come from the sign-up trigger (private.on_signup).
--   a@example.org  approved
--   b@example.org  waits for approval
--   c@example.org  approved too, to see that two users stay apart
-- The token columns are empty strings, not null: Supabase Auth cannot read a null there.
insert into auth.users (id, instance_id, aud, role, email, encrypted_password, email_confirmed_at, raw_app_meta_data,
                        raw_user_meta_data, confirmation_token, recovery_token, email_change_token_new, email_change,
                        created_at, updated_at)
select u.id::uuid, '00000000-0000-0000-0000-000000000000'::uuid, 'authenticated', 'authenticated', u.email, '', now(),
       '{"provider": "email", "providers": ["email"]}'::jsonb, '{}'::jsonb, '', '', '', '', now(), now()
  from (values ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'a@example.org'),
               ('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', 'b@example.org'),
               ('cccccccc-cccc-cccc-cccc-cccccccccccc', 'c@example.org')) as u (id, email);

insert into auth.identities (provider_id, user_id, identity_data, provider, last_sign_in_at, created_at, updated_at)
select id::text, id, jsonb_build_object('sub', id::text, 'email', email, 'email_verified', true), 'email', now(), now(), now()
  from auth.users where email in ('a@example.org', 'b@example.org', 'c@example.org');

update public.access set approved = true where email in ('a@example.org', 'c@example.org');
