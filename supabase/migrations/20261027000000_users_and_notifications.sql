-- Users and notifications for the Gold Spread Monitor.
--
-- Supabase Auth owns accounts (auth.users: email + password, Google). This schema
-- adds what the app needs around them. Every table has row-level security: a signed-
-- in user can read and change only their own rows. The nightly job uses the SECRET
-- key, which bypasses RLS, to write notifications and read push devices.

-- ---------------------------------------------------------------- sign-up allow-list
-- Registration is invite-only: an account can be created only for an email listed
-- here. Opening sign-up later = add emails, or drop the trigger below.
create table if not exists public.allowed_emails (
  email      text primary key check (email = lower(email)),
  note       text,
  created_at timestamptz not null default now()
);
alter table public.allowed_emails enable row level security;   -- no policies: server only

create or replace function public.enforce_signup_allowlist()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  if not exists (select 1 from public.allowed_emails a where a.email = lower(new.email)) then
    raise exception 'Sign-up is by invitation only.' using errcode = '42501';
  end if;
  return new;
end $$;

drop trigger if exists enforce_signup_allowlist on auth.users;
create trigger enforce_signup_allowlist
  before insert on auth.users
  for each row execute function public.enforce_signup_allowlist();

-- ---------------------------------------------------------------- profiles
create table if not exists public.profiles (
  id           uuid primary key references auth.users (id) on delete cascade,
  email        text not null,
  display_name text,
  role         text not null default 'user' check (role in ('user', 'admin')),
  created_at   timestamptz not null default now()
);
alter table public.profiles enable row level security;
create policy "own profile: read"   on public.profiles for select using (auth.uid() = id);
create policy "own profile: update" on public.profiles for update using (auth.uid() = id)
  with check (auth.uid() = id and role = (select role from public.profiles where id = auth.uid()));

-- ---------------------------------------------------------------- notification preferences
create table if not exists public.notification_prefs (
  user_id      uuid primary key references auth.users (id) on delete cascade,
  trade_alerts boolean not null default true,   -- pair becomes ALERT / alert clears
  digest       boolean not null default true,   -- evening summary of unusual points
  circulars    boolean not null default true,   -- new MCX circulars
  pairs        text[],                          -- null = every pair
  updated_at   timestamptz not null default now()
);
alter table public.notification_prefs enable row level security;
create policy "own prefs: read"   on public.notification_prefs for select using (auth.uid() = user_id);
create policy "own prefs: update" on public.notification_prefs for update using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- New account -> profile + default preferences. The first allowed email is the admin.
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id, email, display_name, role)
  values (new.id, lower(new.email),
          coalesce(new.raw_user_meta_data ->> 'full_name', split_part(new.email, '@', 1)),
          case when not exists (select 1 from public.profiles) then 'admin' else 'user' end);
  insert into public.notification_prefs (user_id) values (new.id);
  return new;
end $$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ---------------------------------------------------------------- web-push devices
create table if not exists public.push_subscriptions (
  id          bigint generated always as identity primary key,
  user_id     uuid not null references auth.users (id) on delete cascade,
  endpoint    text not null unique,
  p256dh      text not null,
  auth        text not null,
  user_agent  text,
  created_at  timestamptz not null default now(),
  last_ok_at  timestamptz
);
create index if not exists push_subscriptions_user on public.push_subscriptions (user_id);
alter table public.push_subscriptions enable row level security;
create policy "own devices: read"   on public.push_subscriptions for select using (auth.uid() = user_id);
create policy "own devices: add"    on public.push_subscriptions for insert with check (auth.uid() = user_id);
create policy "own devices: remove" on public.push_subscriptions for delete using (auth.uid() = user_id);

-- ---------------------------------------------------------------- in-app notification centre
create table if not exists public.notifications (
  id         bigint generated always as identity primary key,
  user_id    uuid not null references auth.users (id) on delete cascade,
  kind       text not null check (kind in ('new_alert', 'alert_cleared', 'digest', 'circulars',
                                           'no_data', 'data_changed_same_session', 'test')),
  title      text not null,
  body       text not null,
  url        text,                              -- where tapping it should open
  run_id     text,
  created_at timestamptz not null default now(),
  read_at    timestamptz
);
create index if not exists notifications_user_time on public.notifications (user_id, created_at desc);
alter table public.notifications enable row level security;
create policy "own notifications: read" on public.notifications for select using (auth.uid() = user_id);
-- Users may only mark their own notifications read; content stays as the job wrote it.
create policy "own notifications: mark read" on public.notifications for update
  using (auth.uid() = user_id) with check (auth.uid() = user_id);
revoke update on public.notifications from authenticated;
grant update (read_at) on public.notifications to authenticated;
