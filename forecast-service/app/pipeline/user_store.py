"""User profile store.

Persists the small, user-editable part of a profile (display name + photo) and
sources the read-only organization details from the headcount directory — the
data captured when people are loaded into the tool (their "onboarding" record):
business area, sub business area, channel/role, site, location and manager.
"""
from __future__ import annotations

from typing import Optional

from app.pipeline.postgres import db_conn, has_dsn


def ensure_user_schema() -> None:
    if not has_dsn():
        return
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_key TEXT PRIMARY KEY,
                display_name TEXT,
                photo_url TEXT,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )


def _persisted_profile(user_key: str) -> dict:
    if not has_dsn() or not user_key:
        return {}
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT display_name, photo_url FROM user_profiles WHERE user_key = %s",
            (user_key,),
        )
        row = cur.fetchone()
    if not row:
        return {}
    return {"display_name": row[0], "photo_url": row[1]}


def lookup_org_profile(user_key: str, email: str = "", name: str = "") -> dict:
    """Find this user's record in the headcount directory and return the org
    fields captured at onboarding. Matches on BRID first (the fallback identity
    is typically a BRID), then on email / name inside the payload."""
    if not has_dsn():
        return {}
    key = (user_key or "").strip()
    email_l = (email or "").strip().lower()
    name_l = (name or "").strip().lower()
    if not (key or email_l or name_l):
        return {}
    with db_conn() as conn:
        cur = conn.cursor()
        # Single query that scores candidate rows by how well they match the
        # identity, preferring an exact BRID hit.
        cur.execute(
            """
            SELECT business_area, sub_business_area, channel, site, location, payload
            FROM headcount_entries
            WHERE (%(key)s <> '' AND lower(brid) = lower(%(key)s))
               OR (%(email)s <> '' AND lower(payload->>'email') = %(email)s)
               OR (%(email)s <> '' AND lower(payload->>'Email') = %(email)s)
               OR (%(name)s  <> '' AND lower(payload->>'Name')  = %(name)s)
            ORDER BY
                CASE WHEN %(key)s <> '' AND lower(brid) = lower(%(key)s) THEN 0 ELSE 1 END
            LIMIT 1
            """,
            {"key": key, "email": email_l, "name": name_l},
        )
        row = cur.fetchone()
    if not row:
        return {}
    business_area, sub_business_area, channel, site, location, payload = row
    payload = payload or {}

    def _p(*names):
        for n in names:
            v = payload.get(n)
            if v not in (None, ""):
                return str(v)
        return ""

    return {
        "business_area": business_area or _p("Business Area", "business_area"),
        "sub_business_area": sub_business_area or _p("Sub Business Area", "sub_business_area"),
        "role": _p("Role", "Designation", "Title", "Job Title") or (channel or ""),
        "manager": _p("Team Manager", "Manager", "Reports To", "reports_to", "Line Manager"),
        "location": location or _p("Location", "Site", "Country"),
        "site": site or _p("Site"),
        "name": _p("Name", "Full Name", "Employee Name"),
        "brid": _p("BRID", "brid"),
    }


def get_user_profile(user_key: str, email: str = "", name: str = "") -> dict:
    """Merge identity + persisted edits (name/photo) + onboarding org fields."""
    ensure_user_schema()
    persisted = _persisted_profile(user_key)
    org = lookup_org_profile(user_key, email=email, name=name)
    display_name = persisted.get("display_name") or org.get("name") or name or email or user_key
    return {
        "key": user_key,
        "email": email,
        "name": display_name,
        "photo_url": persisted.get("photo_url") or "",
        "business_area": org.get("business_area", ""),
        "sub_business_area": org.get("sub_business_area", ""),
        "role": org.get("role", ""),
        "manager": org.get("manager", ""),
        "location": org.get("location", ""),
        "site": org.get("site", ""),
        "brid": org.get("brid", ""),
        "org_matched": bool(org),
    }


def get_user_display_map(keys: list) -> dict:
    """Resolve a batch of user identifiers (BRID / email / name keys, as stamped
    on plan owner / created_by / activity actor) to a display name and photo.

    Prefers the user's own persisted profile edit (user_profiles, so it reflects
    Profile updates), then the headcount directory name, then the raw key. Used
    so owners/actors render as the person's name instead of an opaque BRID.
    """
    out: dict = {}
    uniq: list[str] = []
    seen = set()
    for k in keys or []:
        s = str(k or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            uniq.append(s)
    if not uniq or not has_dsn():
        return {k: {"name": k, "photo": ""} for k in uniq}
    ensure_user_schema()
    persisted: dict = {}
    org_names: dict = {}
    with db_conn() as conn:
        cur = conn.cursor()
        # Persisted profile edits keyed by user_key (case-insensitive).
        cur.execute(
            "SELECT user_key, display_name, photo_url FROM user_profiles "
            "WHERE lower(user_key) = ANY(%s)",
            ([k.lower() for k in uniq],),
        )
        for uk, dn, ph in cur.fetchall():
            persisted[str(uk).lower()] = {"name": (dn or "").strip(), "photo": (ph or "").strip()}
        # Directory names keyed by BRID (case-insensitive).
        try:
            cur.execute(
                "SELECT lower(brid), payload FROM headcount_entries "
                "WHERE lower(brid) = ANY(%s)",
                ([k.lower() for k in uniq],),
            )
            for brid_l, payload in cur.fetchall():
                payload = payload or {}
                nm = ""
                for n in ("Name", "Full Name", "Employee Name", "name"):
                    if payload.get(n) not in (None, ""):
                        nm = str(payload[n]).strip()
                        break
                if nm:
                    org_names[str(brid_l)] = nm
        except Exception:
            pass
    for k in uniq:
        kl = k.lower()
        p = persisted.get(kl, {})
        name = p.get("name") or org_names.get(kl) or k
        out[k] = {"name": name, "photo": p.get("photo", "")}
    return out


def update_user_profile(
    user_key: str,
    *,
    display_name: Optional[str] = None,
    photo_url: Optional[str] = None,
) -> dict:
    """Upsert the editable profile fields. Only provided fields are changed."""
    ensure_user_schema()
    if not has_dsn() or not user_key:
        return {}
    sets = []
    params: list = []
    if display_name is not None:
        sets.append("display_name = %s")
        params.append(display_name.strip() or None)
    if photo_url is not None:
        sets.append("photo_url = %s")
        params.append(photo_url or None)
    if not sets:
        return _persisted_profile(user_key)
    sets.append("updated_at = NOW()")
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            INSERT INTO user_profiles (user_key, display_name, photo_url)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_key) DO UPDATE SET {", ".join(sets)}
            """,
            [user_key, (display_name or None), (photo_url or None), *params],
        )
    return _persisted_profile(user_key)
