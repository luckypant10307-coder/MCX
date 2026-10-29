-- Accounts created BEFORE the users-and-notifications migration have no profile or
-- notification settings, because the on-signup trigger did not exist yet. Give them
-- both. Safe to run any number of times. The earliest account becomes admin only if
-- no admin exists yet.
insert into public.profiles (id, email, display_name, role)
select u.id, lower(u.email),
       coalesce(u.raw_user_meta_data ->> 'full_name', split_part(u.email, '@', 1)),
       case when not exists (select 1 from public.profiles where role = 'admin')
                 and u.created_at = (select min(created_at) from auth.users)
            then 'admin' else 'user' end
from auth.users u
where not exists (select 1 from public.profiles p where p.id = u.id);

insert into public.notification_prefs (user_id)
select u.id from auth.users u
where not exists (select 1 from public.notification_prefs n where n.user_id = u.id);
