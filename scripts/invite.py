"""
Manage who may register. Sign-up (email or Google) is refused for any email not listed.

    python scripts/invite.py you@example.com            # allow one email
    python scripts/invite.py --list                     # who is allowed
    python scripts/invite.py --remove someone@example.com

Uses SUPABASE_URL + SUPABASE_SECRET_KEY from .env. The first account that registers
becomes the admin.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcxrv import supa  # noqa: E402
from mcxrv.stage1_intake.config import load_dotenv  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Manage the sign-up allow-list.")
    p.add_argument("email", nargs="?")
    p.add_argument("--note", default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--remove", default=None)
    a = p.parse_args(argv)
    load_dotenv()
    if not supa.configured():
        print("SUPABASE_URL and SUPABASE_SECRET_KEY must be set in .env")
        return 2
    try:
        if a.list:
            for r in supa.select("allowed_emails", {"select": "email,note,created_at", "order": "created_at"}):
                print("{:<40} {}".format(r["email"], r.get("note") or ""))
            return 0
        if a.remove:
            supa.delete("allowed_emails", {"email": a.remove.strip().lower()})
            print("removed", a.remove.strip().lower())
            return 0
        if not a.email:
            p.print_help()
            return 2
        email = a.email.strip().lower()
        supa.insert("allowed_emails", [{"email": email, "note": a.note}], upsert_on="email")
        print("allowed:", email, "- they can now register with email + password or Google")
        return 0
    except Exception as exc:
        msg = str(exc)
        if "404" in msg or "allowed_emails" in msg:
            print("The allowed_emails table does not exist yet. Run the SQL in "
                  "supabase/migrations/ first (Supabase dashboard > SQL Editor > paste > Run).")
        else:
            print("Failed:", msg)
        return 1


if __name__ == "__main__":
    sys.exit(main())
