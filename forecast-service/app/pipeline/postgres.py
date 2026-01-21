from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg


def _dsn() -> Optional[str]:
    return os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL")


def has_dsn() -> bool:
    return bool(_dsn())


@contextmanager
def db_conn() -> Iterator[psycopg.Connection]:
    dsn = _dsn()
    if not dsn:
        raise RuntimeError("POSTGRES_DSN/DATABASE_URL not configured.")
    conn = psycopg.connect(dsn)
    try:
        conn.autocommit = True
        yield conn
    finally:
        conn.close()


def ensure_budget_schema() -> None:
    if not has_dsn():
        return
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS budget_entries (
                id BIGSERIAL PRIMARY KEY,
                business_area TEXT,
                sub_business_area TEXT,
                channel TEXT NOT NULL,
                site TEXT,
                week DATE NOT NULL,
                budget_headcount NUMERIC,
                budget_aht_sec NUMERIC,
                budget_sut_sec NUMERIC,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (business_area, sub_business_area, channel, site, week)
            )
            """
        )


def ensure_ops_schema() -> None:
    if not has_dsn():
        return
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS timeseries_entries (
                id BIGSERIAL PRIMARY KEY,
                kind TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                scope_key_norm TEXT NOT NULL,
                date DATE,
                interval TEXT,
                volume NUMERIC,
                aht_sec NUMERIC,
                sut_sec NUMERIC,
                items NUMERIC,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timeseries_kind_scope ON timeseries_entries (kind, scope_key_norm)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timeseries_kind_date ON timeseries_entries (kind, date)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS roster_entries (
                id BIGSERIAL PRIMARY KEY,
                start_date DATE,
                end_date DATE,
                fte NUMERIC,
                program TEXT,
                status TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hiring_entries (
                id BIGSERIAL PRIMARY KEY,
                start_week DATE,
                fte NUMERIC,
                program TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )


def ensure_roster_schema() -> None:
    if not has_dsn():
        return
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS roster_wide_entries (
                id BIGSERIAL PRIMARY KEY,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS roster_long_entries (
                id BIGSERIAL PRIMARY KEY,
                brid TEXT,
                date DATE,
                entry TEXT,
                is_leave BOOLEAN,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_roster_long_brid_date ON roster_long_entries (brid, date)"
        )


def ensure_newhire_schema() -> None:
    if not has_dsn():
        return
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS new_hire_entries (
                id BIGSERIAL PRIMARY KEY,
                business_area TEXT,
                class_reference TEXT UNIQUE,
                source_system_id TEXT,
                emp_type TEXT,
                status TEXT,
                class_type TEXT,
                class_level TEXT,
                grads_needed NUMERIC,
                billable_hc NUMERIC,
                training_weeks NUMERIC,
                nesting_weeks NUMERIC,
                induction_start DATE,
                training_start DATE,
                training_end DATE,
                nesting_start DATE,
                nesting_end DATE,
                production_start DATE,
                created_by TEXT,
                created_ts TIMESTAMPTZ,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )


def ensure_shrinkage_schema() -> None:
    if not has_dsn():
        return
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shrinkage_weekly_entries (
                id BIGSERIAL PRIMARY KEY,
                week DATE,
                program TEXT,
                ooo_hours NUMERIC,
                ino_hours NUMERIC,
                base_hours NUMERIC,
                ooo_pct NUMERIC,
                ino_pct NUMERIC,
                overall_pct NUMERIC,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shrinkage_raw_entries (
                id BIGSERIAL PRIMARY KEY,
                kind TEXT,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attrition_weekly_entries (
                id BIGSERIAL PRIMARY KEY,
                week DATE,
                program TEXT,
                leavers_fte NUMERIC,
                avg_active_fte NUMERIC,
                attrition_pct NUMERIC,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attrition_raw_entries (
                id BIGSERIAL PRIMARY KEY,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )


def ensure_planning_schema() -> None:
    if not has_dsn():
        return
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS planning_plans (
                id BIGSERIAL PRIMARY KEY,
                plan_key TEXT NOT NULL,
                org TEXT,
                business_entity TEXT,
                business_area TEXT,
                sub_business_area TEXT,
                channel TEXT,
                location TEXT,
                site TEXT,
                plan_name TEXT,
                plan_type TEXT,
                start_week DATE,
                end_week DATE,
                ft_weekly_hours NUMERIC,
                pt_weekly_hours NUMERIC,
                tags JSONB,
                is_current BOOLEAN DEFAULT FALSE,
                status TEXT DEFAULT 'draft',
                hierarchy_json JSONB,
                owner TEXT,
                created_by TEXT,
                updated_by TEXT,
                is_deleted BOOLEAN DEFAULT FALSE,
                deleted_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_planning_plans_key ON planning_plans (plan_key)"
        )
        # Backfill columns if schema existed before these fields were added.
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS org TEXT")
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS business_entity TEXT")
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS ft_weekly_hours NUMERIC")
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS pt_weekly_hours NUMERIC")
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS tags JSONB")
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS is_current BOOLEAN DEFAULT FALSE")
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'draft'")
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS hierarchy_json JSONB")
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS owner TEXT")
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS created_by TEXT")
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS updated_by TEXT")
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE")
        conn.execute("ALTER TABLE planning_plans ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS planning_plan_tables (
                id BIGSERIAL PRIMARY KEY,
                plan_id BIGINT NOT NULL REFERENCES planning_plans(id) ON DELETE CASCADE,
                table_name TEXT NOT NULL,
                payload JSONB NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (plan_id, table_name)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_planning_tables_plan ON planning_plan_tables (plan_id)"
        )
