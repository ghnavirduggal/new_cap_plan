from __future__ import annotations

import calendar
import datetime as dt
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .utils import df_from_payload, df_to_records
from .volume_summary import normalize_volume_df


def _month_name_to_num(value: Any) -> Optional[int]:
    if value is None:
        return None
    name = str(value).strip().lower()[:3]
    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    return month_map.get(name)


def _normalize_interval_label(val: Any) -> str:
    if val is None:
        return ""
    text = str(val).strip()
    if not text or text.lower() == "nan":
        return ""
    match = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        return f"{hour:02d}:{minute:02d}"
    dt_val = pd.to_datetime(text, errors="coerce")
    if pd.notna(dt_val):
        return dt_val.strftime("%H:%M")
    return text


def _normalize_interval_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=object)
    if series.empty:
        return series.astype(object)
    unique_vals = pd.unique(series.dropna())
    mapping = {v: _normalize_interval_label(v) for v in unique_vals}
    out = series.map(mapping)
    return out.fillna("")


def _interval_sort_key(val: Any) -> tuple[int, Any]:
    normalized = _normalize_interval_label(val)
    dt_val = pd.to_datetime(normalized, format="%H:%M", errors="coerce")
    if pd.notna(dt_val):
        return (0, dt_val.time())
    return (1, normalized)


def _normalize_weights(values: list[float]) -> tuple[list[float], float]:
    if not values:
        return [], 0.0
    arr = np.array(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        return [], 0.0
    return (arr / total).tolist(), total


def _round_to_step_with_total(values: list[float], step: float, total: float) -> list[float]:
    if not values:
        return []
    if step <= 0:
        return values
    scale = 1.0 / step
    raw = np.array(values, dtype=float) * scale
    base = np.floor(raw + 1e-9)
    base = np.clip(base, 0, None)
    target_units = int(round(float(total) * scale))
    diff = target_units - int(base.sum())
    if diff > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(frac)
        n = len(order)
        for k in range(diff):
            base[order[-1 - (k % n)]] += 1
    elif diff < 0:
        frac = raw - np.floor(raw)
        order = [int(i) for i in np.argsort(frac) if base[int(i)] > 0]
        remaining = -diff
        if order:
            k = 0
            while remaining > 0 and order:
                idx = order[k % len(order)]
                if base[idx] > 0:
                    base[idx] -= 1
                    remaining -= 1
                    if base[idx] == 0:
                        order.remove(idx)
                        k = 0
                        continue
                k += 1
    return (base / scale).tolist()


def _di_normalize_original_data(
    df: pd.DataFrame,
    group_value: Optional[str],
    group_level: str = "forecast_group",
) -> pd.DataFrame:
    d = normalize_volume_df(df)
    if d.empty:
        return d
    group_level_norm = str(group_level or "forecast_group").strip().lower()
    group_col = "forecast_group"
    if group_level_norm == "business_area" and "business_area" in d.columns:
        group_col = "business_area"
    if group_value and group_col in d.columns:
        d = d[d[group_col] == group_value]
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if "volume" in d.columns:
        d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    d = d.dropna(subset=["date", "volume"])
    return d


def _di_prepare_original_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    d = d.dropna(subset=["date", "volume"])
    if d.empty:
        return d
    d["month_period"] = d["date"].dt.to_period("M")
    d["Month_Year"] = d["date"].dt.strftime("%b-%y")
    d["Month_Year_Long"] = d["date"].dt.strftime("%b %Y")
    d["Day"] = d["date"].dt.day
    d["Weekday"] = d["date"].dt.day_name()
    d["Year"] = d["date"].dt.year
    d["Month"] = d["date"].dt.month_name()
    totals = d.groupby("month_period")["volume"].transform("sum")
    d["Volume%"] = np.where(totals > 0, d["volume"] / totals * 100, np.nan)
    d["Volume%"] = d["Volume%"].round(1)
    d["daily_volume"] = d["volume"]
    return d


def _di_match_months(df_daily: pd.DataFrame, target_month: pd.Timestamp) -> list[pd.Timestamp]:
    if df_daily.empty:
        return []
    month_starts = sorted(pd.unique(df_daily["month_start"].dropna()))
    target_weekday = target_month.weekday()
    matching = [m for m in month_starts if pd.Timestamp(m).weekday() == target_weekday]
    matching = [pd.Timestamp(m) for m in matching if pd.Timestamp(m) != target_month]
    if not matching:
        matching = [pd.Timestamp(m) for m in month_starts]
    return matching[-3:]


def _di_compute_distribution(df_daily: pd.DataFrame, target_month: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    days_in_month = calendar.monthrange(target_month.year, target_month.month)[1]
    matching_months = _di_match_months(df_daily, target_month)
    msg = "No matching months found."
    if matching_months:
        msg = "Using months: " + ", ".join(m.strftime("%b %Y") for m in matching_months)

    if matching_months:
        subset = df_daily.loc[
            df_daily["month_start"].isin(matching_months),
            ["month_start", "day", "volume_pct"],
        ]
        if not subset.empty:
            pivot = subset.pivot_table(
                index="day",
                columns="month_start",
                values="volume_pct",
                aggfunc="first",
            )
            day_series = pivot.mean(axis=1, skipna=True)
            day_vals = (
                day_series.reindex(range(1, days_in_month + 1), fill_value=0.0)
                .astype(float)
                .tolist()
            )
        else:
            day_vals = [0.0] * days_in_month
    else:
        day_vals = [0.0] * days_in_month

    total = sum(day_vals)
    if total > 0:
        day_vals = [v / total * 100 for v in day_vals]

    dates = [target_month + pd.Timedelta(days=i) for i in range(days_in_month)]
    dist = pd.DataFrame(
        {
            "Date": [d.date() for d in dates],
            "Weekday": [d.day_name() for d in dates],
            "Distribution_Pct": [round(v, 1) for v in day_vals],
        }
    )
    return dist, msg


def _di_normalize_distribution(dist: pd.DataFrame) -> pd.DataFrame:
    if dist.empty or "Distribution_Pct" not in dist.columns:
        return dist
    vals = pd.to_numeric(dist["Distribution_Pct"], errors="coerce").fillna(0.0)
    total = float(vals.sum())
    if total > 0:
        dist["Distribution_Pct"] = (vals / total * 100)
    return dist


def _di_interval_ratios(interval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if interval_df.empty:
        return pd.DataFrame(), pd.DataFrame(), "Interval data is empty."
    date_col = _pick_col(interval_df, ("date", "ds", "timestamp"))
    ivl_col = _pick_col(
        interval_df,
        ("interval", "time", "interval_start", "start_time", "timeslot", "time_slot"),
    )
    vol_col = _pick_col(interval_df, ("volume", "calls", "items", "count"))
    if not date_col or not ivl_col or not vol_col:
        return pd.DataFrame(), pd.DataFrame(), "Interval data must have date, interval, and volume columns."
    d = interval_df[[date_col, ivl_col, vol_col]].copy()
    d["date"] = pd.to_datetime(d[date_col], errors="coerce")
    d["interval"] = _normalize_interval_series(d[ivl_col])
    d["volume"] = pd.to_numeric(d[vol_col], errors="coerce")
    d = d.dropna(subset=["date", "interval", "volume"])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), "Interval data empty after cleaning."
    max_date = d["date"].max()
    if pd.notna(max_date):
        cutoff = max_date - pd.DateOffset(months=3)
        d = d[d["date"] >= cutoff]
    daily_totals = d.groupby("date")["volume"].sum()
    d = d.merge(daily_totals.rename("day_total"), on="date", how="left")
    d = d[d["day_total"] > 0]
    d["ratio"] = d["volume"] / d["day_total"]
    d["weekday"] = d["date"].dt.day_name()
    interval_ratio = d.groupby(["interval", "weekday"])["ratio"].mean().reset_index()
    overall_ratio = d.groupby("interval")["ratio"].mean().reset_index()
    if not interval_ratio.empty:
        interval_ratio["__sort"] = interval_ratio["interval"].apply(_interval_sort_key)
        interval_ratio = interval_ratio.sort_values(["__sort", "interval", "weekday"]).drop(columns="__sort")
    if not overall_ratio.empty:
        overall_ratio["__sort"] = overall_ratio["interval"].apply(_interval_sort_key)
        overall_ratio = overall_ratio.sort_values(["__sort", "interval"]).drop(columns="__sort")
    return interval_ratio, overall_ratio, "Interval ratios computed."


def _di_sort_month_str(values: list[str]) -> list[str]:
    def _key(val: str):
        return pd.to_datetime(val, format="%b-%y", errors="coerce")
    ordered = sorted(values, key=lambda v: (pd.isna(_key(v)), _key(v)))
    return [v for v in ordered if isinstance(v, str)]


def _di_matching_months_table(d: pd.DataFrame, target_month: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    if d is None or d.empty:
        return pd.DataFrame(), []
    first_days = (
        d.groupby("month_period")["date"]
        .min()
        .reset_index()
        .dropna(subset=["date"])
    )
    if first_days.empty:
        return pd.DataFrame(), []
    first_days["Month_Year"] = first_days["date"].dt.strftime("%b-%y")
    first_days["Year"] = first_days["date"].dt.year
    first_days["Month"] = first_days["date"].dt.month_name()
    first_days["FirstDayName"] = first_days["date"].dt.day_name()
    target_weekday = target_month.weekday()
    matching = first_days[first_days["date"].dt.weekday == target_weekday]
    match_vals = matching["Month_Year"].dropna().astype(str).tolist()
    return first_days, _di_sort_month_str(match_vals)


def _di_sequential_mapping_table(d: pd.DataFrame, month_year_list: list[str], value_col: str) -> pd.DataFrame:
    if d is None or d.empty or not month_year_list:
        return pd.DataFrame()
    rows = {}
    all_cols: set[str] = set()
    for month_year in month_year_list:
        try:
            month_date = pd.to_datetime(month_year, format="%b-%y")
        except Exception:
            continue
        month_data = d[d["Month_Year"] == month_year]
        if month_data.empty:
            continue
        days_in_month = calendar.monthrange(month_date.year, month_date.month)[1]
        row_vals = {}
        for day in range(1, days_in_month + 1):
            actual_date = pd.Timestamp(month_date.year, month_date.month, day)
            col = f"{actual_date.strftime('%a')}_{day}"
            val = month_data.loc[month_data["Day"] == day, value_col]
            row_vals[col] = round(float(val.sum()), 2) if not val.empty else np.nan
            all_cols.add(col)
        rows[month_year] = row_vals
    if not rows:
        return pd.DataFrame()

    def _sort_key(col: str) -> tuple[int, str]:
        match = re.search(r"_(\d+)$", str(col))
        return (int(match.group(1)) if match else 999, str(col))

    ordered_cols = sorted(all_cols, key=_sort_key)
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=ordered_cols)
    df.index.name = "MonthYear"
    df = df.reset_index().rename(columns={"MonthYear": "Month_Year"})
    avg = df[ordered_cols].mean(axis=0, skipna=True)
    avg_row = pd.DataFrame([["Average"] + avg.tolist()], columns=["Month_Year"] + ordered_cols)
    editable_row = pd.DataFrame([["Average_Editable"] + avg.tolist()], columns=["Month_Year"] + ordered_cols)
    return pd.concat([df, avg_row, editable_row], ignore_index=True)


def _di_single_month_table(
    d: pd.DataFrame,
    month_year_list: list[str],
    value_col: str,
    holidays_df: pd.DataFrame,
) -> pd.DataFrame:
    if d is None or d.empty or not month_year_list:
        return pd.DataFrame()
    max_days = 0
    parsed = []
    for month_year in month_year_list:
        try:
            month_date = pd.to_datetime(month_year, format="%b-%y")
        except Exception:
            continue
        parsed.append((month_year, month_date))
        max_days = max(max_days, calendar.monthrange(month_date.year, month_date.month)[1])
    if max_days == 0:
        return pd.DataFrame()

    holidays = pd.DataFrame()
    if isinstance(holidays_df, pd.DataFrame) and not holidays_df.empty and "holiday_date" in holidays_df.columns:
        holidays = holidays_df.copy()
        holidays["holiday_date"] = pd.to_datetime(holidays["holiday_date"], errors="coerce").dt.normalize()

    day_cols = [str(i) for i in range(1, max_days + 1)]
    rows = []
    for month_year, month_date in parsed:
        month_data = d[d["Month_Year"] == month_year]
        if month_data.empty:
            continue
        row = {"Month_Year": month_year}
        values_for_avg = []
        for day in range(1, max_days + 1):
            if day > calendar.monthrange(month_date.year, month_date.month)[1]:
                row[str(day)] = np.nan
                continue
            val = month_data.loc[month_data["Day"] == day, value_col]
            cell_val = float(val.sum()) if not val.empty else np.nan
            row[str(day)] = cell_val
            day_date = pd.Timestamp(month_date.year, month_date.month, day).normalize()
            is_holiday = False
            if not holidays.empty:
                is_holiday = day_date in holidays["holiday_date"].values
            if pd.notna(cell_val) and not is_holiday and cell_val > 0:
                values_for_avg.append(cell_val)
        row["Avg"] = float(np.mean(values_for_avg)) if values_for_avg else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out[["Month_Year"] + day_cols + ["Avg"]]
    avg_row = {"Month_Year": "Avg"}
    for col in day_cols:
        avg_row[col] = float(out[col].mean(skipna=True))
    avg_row["Avg"] = float(out["Avg"].mean(skipna=True))
    out = pd.concat([out, pd.DataFrame([avg_row])], ignore_index=True)
    return out


def _di_get_holidays_in_month(holidays_df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    if holidays_df is None or holidays_df.empty or "holiday_date" not in holidays_df.columns:
        return pd.DataFrame()
    h = holidays_df.copy()
    h["holiday_date"] = pd.to_datetime(h["holiday_date"], errors="coerce").dt.normalize()
    start = pd.Timestamp(year, month, 1).normalize()
    end = pd.Timestamp(year, month, calendar.monthrange(year, month)[1]).normalize()
    mask = (h["holiday_date"] >= start) & (h["holiday_date"] <= end)
    return h[mask]


def _di_get_impact_days_for_month(year: int, month: int, holidays_df: pd.DataFrame) -> list[tuple[int, str, bool]]:
    impact_days = []
    if month == 12:
        for day in range(24, 32):
            impact_days.append((day, "Christmas Period", day in [25, 26]))
        return impact_days
    if month == 1:
        for day in range(1, 3):
            impact_days.append((day, "New Year Period", day == 1))
        return impact_days
    if holidays_df is None or holidays_df.empty or "holiday_date" not in holidays_df.columns:
        return impact_days
    month_holidays = _di_get_holidays_in_month(holidays_df, year, month)
    if month_holidays.empty or "holiday_name" not in month_holidays.columns:
        return impact_days
    for _, row in month_holidays.iterrows():
        name = str(row.get("holiday_name", "")).strip()
        dt_val = pd.to_datetime(row.get("holiday_date"), errors="coerce")
        if pd.isna(dt_val):
            continue
        day = int(dt_val.day)
        if "good friday" in name.lower():
            days_after = 0
        elif any(k in name.lower() for k in ["easter monday", "early may", "spring bank", "summer bank"]):
            days_after = 4
        else:
            days_after = 2
        for offset in range(days_after + 1):
            impact_day = day + offset
            if impact_day <= calendar.monthrange(year, month)[1]:
                impact_days.append((impact_day, name or "Holiday", offset == 0))
    return impact_days


def _di_weekday_averages_excluding_impact_days(
    d: pd.DataFrame,
    year: int,
    month: int,
    holidays_df: pd.DataFrame,
    value_col: str,
) -> dict[str, float]:
    month_year = pd.Timestamp(year, month, 1).strftime("%b-%y")
    month_data = d[d["Month_Year"] == month_year].copy()
    if month_data.empty:
        return {}
    impact_days = _di_get_impact_days_for_month(year, month, holidays_df)
    impact_nums = {day for day, _name, _is_event in impact_days}
    weekday_vals: dict[str, list[float]] = {}
    for _, row in month_data.iterrows():
        day_num = int(row["Day"])
        if day_num in impact_nums:
            continue
        wd = row["date"].day_name()
        weekday_vals.setdefault(wd, []).append(float(row[value_col]) if pd.notna(row[value_col]) else 0.0)
    return {k: float(np.mean(v)) for k, v in weekday_vals.items() if v}


def _di_impact_analysis_table(
    d: pd.DataFrame,
    year: int,
    month: int,
    holidays_df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    month_year = pd.Timestamp(year, month, 1).strftime("%b-%y")
    if month_year not in d["Month_Year"].values:
        return pd.DataFrame()
    weekday_avg = _di_weekday_averages_excluding_impact_days(d, year, month, holidays_df, value_col)
    if not weekday_avg:
        return pd.DataFrame()
    impact_days = _di_get_impact_days_for_month(year, month, holidays_df)
    if not impact_days:
        return pd.DataFrame()

    month_data = d[d["Month_Year"] == month_year].copy()
    rows = []
    for day_num, holiday_name, is_event in impact_days:
        day_data = month_data[month_data["Day"] == day_num]
        if day_data.empty:
            continue
        actual_date = day_data.iloc[0]["date"]
        weekday_name = actual_date.strftime("%A")
        weekday_abbr = actual_date.strftime("%a")
        actual_value = float(day_data[value_col].sum())
        avg_value = float(weekday_avg.get(weekday_name, 0.0))
        vol_pct = (actual_value / avg_value * 100) if avg_value > 0 else 0.0
        change_pct = vol_pct - 100
        rows.append(
            {
                "Weekday": weekday_name,
                "Avg_Value": round(avg_value, 2),
                "Date": f"{day_num}-{actual_date.strftime('%b')}",
                "Weekday_Abbr": weekday_abbr,
                "Actual_Value": round(actual_value, 2),
                "Vol_Pct": round(vol_pct, 2),
                "Change_Pct": round(change_pct, 2),
                "Holiday_Name": holiday_name,
                "Day_Num": day_num,
                "Is_Event": bool(is_event),
            }
        )
    return pd.DataFrame(rows)


def _di_weekly_vertical_tables(
    d: pd.DataFrame,
    month_years: list[str],
    value_col: str,
) -> list[tuple[str, pd.DataFrame]]:
    if d is None or d.empty or not month_years:
        return []
    out = []
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for month_year in month_years:
        month_data = d[d["Month_Year"] == month_year].copy()
        if month_data.empty:
            continue
        try:
            month_date = pd.to_datetime(month_year, format="%b-%y")
        except Exception:
            continue
        days_in_month = calendar.monthrange(month_date.year, month_date.month)[1]
        first_day = pd.Timestamp(month_date.year, month_date.month, 1)
        first_weekday_num = first_day.weekday()
        weeks = []
        current_week = [np.nan] * first_weekday_num
        for day in range(1, days_in_month + 1):
            day_data = month_data[month_data["Day"] == day]
            value = float(day_data[value_col].sum()) if not day_data.empty else 0.0
            current_week.append(value)
            if len(current_week) == 7 or day == days_in_month:
                while len(current_week) < 7:
                    current_week.append(np.nan)
                weeks.append(current_week)
                current_week = []
        df = pd.DataFrame(weeks, columns=weekday_order)
        df.insert(0, "MonthYear", month_year)
        out.append((month_year, df))
    return out


def _di_forecast_rows(
    base_values: list[float],
    impact_table: pd.DataFrame,
    forecast_date: pd.Timestamp,
) -> tuple[pd.DataFrame, list[float]]:
    days_in_month = calendar.monthrange(forecast_date.year, forecast_date.month)[1]
    base_vals = (base_values or [])[:days_in_month]
    if len(base_vals) < days_in_month:
        base_vals = base_vals + [0.0] * (days_in_month - len(base_vals))
    adjusted = base_vals[:]
    if isinstance(impact_table, pd.DataFrame) and not impact_table.empty:
        for _, row in impact_table.iterrows():
            day_num = int(row.get("Day_Num") or 0)
            if day_num < 1 or day_num > days_in_month:
                continue
            change_pct = float(row.get("Change_Pct") or 0.0)
            base_value = float(base_vals[day_num - 1])
            adjusted_value = base_value * (1 + change_pct / 100.0)
            adjusted_value = adjusted_value * 1.1
            adjusted[day_num - 1] = adjusted_value
    total = sum(adjusted)
    if total > 0:
        normalized = [v / total * 100 for v in adjusted]
    else:
        normalized = adjusted
    cols = [str(i) for i in range(1, days_in_month + 1)]

    def _row(label: str, values: list[float]) -> dict:
        return {"Row": label, **{c: round(values[int(c) - 1], 1) for c in cols}, "Sum": round(sum(values), 1)}

    df = pd.DataFrame(
        [
            _row("Base", base_vals),
            _row("Adjusted", adjusted),
            _row("Final", normalized),
        ]
    )
    return df, normalized


def _di_forecast_vertical_table(values: list[float], forecast_date: pd.Timestamp) -> pd.DataFrame:
    if not values:
        return pd.DataFrame()
    days_in_month = len(values)
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    first_day = pd.Timestamp(forecast_date.year, forecast_date.month, 1)
    first_weekday = first_day.weekday()
    table_rows = []
    current_day = 1
    while current_day <= days_in_month:
        row = {}
        for weekday_idx, weekday in enumerate(weekday_order):
            day_num = (weekday_idx - first_weekday) + (len(table_rows) * 7) + 1
            if 1 <= day_num <= days_in_month:
                row[weekday] = round(values[day_num - 1], 1)
                current_day = day_num + 1
            else:
                row[weekday] = np.nan
        table_rows.append(row)
        if current_day > days_in_month:
            break
    return pd.DataFrame(table_rows)


def _di_weekday_pattern_chart_data(d: pd.DataFrame, month_years: list[str], value_col: str) -> Optional[dict]:
    if d is None or d.empty or not month_years:
        return None
    chart_data = []
    earliest_weekday_num = None
    max_days = 0
    for month_year in month_years:
        try:
            month_date = pd.to_datetime(month_year, format="%b-%y")
        except Exception:
            continue
        month_data = d[d["Month_Year"] == month_year].copy()
        if month_data.empty:
            continue
        weekday_num = pd.Timestamp(month_date.year, month_date.month, 1).weekday()
        days_in_month = calendar.monthrange(month_date.year, month_date.month)[1]
        if earliest_weekday_num is None or weekday_num < earliest_weekday_num:
            earliest_weekday_num = weekday_num
        max_days = max(max_days, days_in_month)
        chart_data.append({"month": month_year, "data": month_data, "weekday_num": weekday_num, "days": days_in_month})
    if not chart_data or earliest_weekday_num is None:
        return None
    total_positions = (max_days + (7 - 1)) + (chart_data[0]["weekday_num"] - earliest_weekday_num)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_abbr = {"Monday": "Mon", "Tuesday": "Tue", "Wednesday": "Wed", "Thursday": "Thu", "Friday": "Fri", "Saturday": "Sat", "Sunday": "Sun"}
    x_labels = []
    for pos in range(total_positions):
        weekday_idx = (earliest_weekday_num + pos) % 7
        week_num = (pos // 7) + 1
        x_labels.append(f"{weekday_abbr[weekday_order[weekday_idx]]}_{week_num}")

    series = []
    all_values_by_pos = {i: [] for i in range(total_positions)}
    for item in chart_data:
        month_date = pd.to_datetime(item["month"], format="%b-%y", errors="coerce")
        if pd.isna(month_date):
            continue
        offset = item["weekday_num"] - earliest_weekday_num
        y_values = [None] * total_positions
        for day in range(1, item["days"] + 1):
            pos = offset + day - 1
            if pos >= total_positions:
                continue
            day_val = item["data"].loc[item["data"]["Day"] == day, value_col]
            val = float(day_val.sum()) if not day_val.empty else 0.0
            y_values[pos] = val
            if val > 0:
                all_values_by_pos[pos].append(val)
        series.append(
            {"name": item["month"], "points": [{"x": x_labels[i], "y": y_values[i]} for i in range(total_positions)]}
        )
    avg_values = [np.mean(all_values_by_pos[i]) if all_values_by_pos[i] else None for i in range(total_positions)]
    series.append(
        {"name": "Average", "points": [{"x": x_labels[i], "y": avg_values[i]} for i in range(total_positions)]}
    )
    return {"x": x_labels, "series": series}


def _di_interval_section_tables(interval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    if interval_df is None or interval_df.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    date_col = _pick_col(interval_df, ("date", "ds", "timestamp"))
    ivl_col = _pick_col(
        interval_df,
        ("interval", "time", "interval_start", "start_time", "timeslot", "time_slot"),
    )
    vol_col = _pick_col(interval_df, ("volume", "calls", "items", "count"))
    if not date_col or not ivl_col or not vol_col:
        return pd.DataFrame(), pd.DataFrame(), None
    d = interval_df[[date_col, ivl_col, vol_col]].copy()
    d["date"] = pd.to_datetime(d[date_col], errors="coerce")
    d["interval"] = _normalize_interval_series(d[ivl_col])
    d["volume"] = pd.to_numeric(d[vol_col], errors="coerce")
    d = d.dropna(subset=["date", "interval", "volume"])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    max_date = d["date"].max()
    if pd.notna(max_date):
        cutoff = max_date - pd.DateOffset(months=3)
        d = d[d["date"] >= cutoff]
    d["weekday"] = d["date"].dt.day_name()
    d["day_total"] = d.groupby("date")["volume"].transform("sum")
    d = d[d["day_total"] > 0]
    d["volume_ratio"] = d["volume"] / d["day_total"] * 100
    section1 = d.pivot_table(index="interval", columns=["date", "weekday"], values="volume_ratio", aggfunc="sum", fill_value=0)
    section1_cols = []
    for col in section1.columns:
        try:
            date_val, wd = col
            label = f"{pd.to_datetime(date_val).strftime('%d %b %Y')} ({wd})"
        except Exception:
            label = str(col)
        section1_cols.append(label)
    section1.columns = section1_cols
    section1 = section1.reset_index()
    if not section1.empty and "interval" in section1.columns:
        section1["__sort"] = section1["interval"].apply(_interval_sort_key)
        section1 = section1.sort_values(["__sort", "interval"]).drop(columns="__sort")

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    section2_data = {"interval": section1["interval"]}
    for wd in weekdays:
        cols = [c for c in section1.columns if isinstance(c, str) and f"({wd})" in c]
        if cols:
            section2_data[wd] = section1[cols].mean(axis=1)
    section2 = pd.DataFrame(section2_data)
    chart_data = None
    if not section2.empty:
        chart_data = {
            "x": section2["interval"].tolist(),
            "series": [
                {
                    "name": wd,
                    "points": [{"x": x, "y": float(y) if pd.notna(y) else None} for x, y in zip(section2["interval"], section2[wd])],
                }
                for wd in [c for c in section2.columns if c != "interval"]
            ],
        }
    return section1, section2, chart_data


def _interval_summary_text(interval_tbl: pd.DataFrame) -> str:
    if interval_tbl is None or interval_tbl.empty:
        return ""
    total = pd.to_numeric(interval_tbl.get("Interval_Forecast"), errors="coerce").sum()
    if not np.isfinite(total):
        return ""
    return f"Total Interval Forecast: {total:,.0f}"


def _json_safe_value(val: Any):
    if isinstance(val, pd.Period):
        return str(val)
    if isinstance(val, (pd.Timestamp, dt.datetime, dt.date)):
        return val.isoformat()
    if isinstance(val, np.datetime64):
        try:
            return pd.to_datetime(val).isoformat()
        except Exception:
            return val
    return val


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_period_dtype(out[col]):
            out[col] = out[col].astype(str)
            continue
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
            continue
        if out[col].dtype == object:
            out[col] = out[col].apply(_json_safe_value)
    return out.to_dict("records")


def _di_daily_accuracy_backtest(
    original_df: pd.DataFrame,
    forecast_month: pd.Timestamp,
    distribution: pd.DataFrame,
) -> pd.DataFrame:
    if (
        original_df is None
        or original_df.empty
        or distribution is None
        or distribution.empty
        or "Distribution_Pct" not in distribution.columns
    ):
        return pd.DataFrame()

    d = original_df.copy()
    d["date"] = pd.to_datetime(d.get("date"), errors="coerce")
    d["volume"] = pd.to_numeric(d.get("volume"), errors="coerce")
    d = d.dropna(subset=["date", "volume"])
    if d.empty:
        return pd.DataFrame()

    daily = d.groupby("date", as_index=False)["volume"].sum()
    daily["month_start"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    daily["day"] = daily["date"].dt.day
    month_totals = daily.groupby("month_start")["volume"].transform("sum")
    daily["actual_pct"] = np.where(month_totals > 0, daily["volume"] / month_totals * 100.0, np.nan)

    months_used = _di_match_months(daily, forecast_month)
    if not months_used:
        return pd.DataFrame()

    pred_base = pd.to_numeric(distribution["Distribution_Pct"], errors="coerce").fillna(0.0).tolist()
    if not pred_base:
        return pd.DataFrame()

    rows = []
    for month_start in months_used:
        md = daily[daily["month_start"] == month_start].copy()
        if md.empty:
            continue
        days_in_month = calendar.monthrange(month_start.year, month_start.month)[1]
        pred = (pred_base[:days_in_month] + [0.0] * max(0, days_in_month - len(pred_base)))[:days_in_month]
        pred_total = float(np.nansum(pred))
        if pred_total > 0:
            pred = [v / pred_total * 100.0 for v in pred]

        md["pred_pct"] = md["day"].apply(lambda day: pred[int(day) - 1] if 1 <= int(day) <= len(pred) else np.nan)
        valid = md.dropna(subset=["actual_pct", "pred_pct"]).copy()
        valid = valid[valid["actual_pct"] > 0]
        if valid.empty:
            continue

        ratio = (valid["pred_pct"] / valid["actual_pct"]) * 100.0
        total_days = int(ratio.shape[0])

        def _within(lower: float, upper: float) -> int:
            return int(((ratio >= lower) & (ratio <= upper)).sum())

        count5 = _within(94.5, 105.5)
        count7 = _within(92.5, 107.5)
        count10 = _within(90.5, 110.5)
        mape = float((np.abs(valid["pred_pct"] - valid["actual_pct"]) / valid["actual_pct"]).mean() * 100.0)

        rows.append(
            {
                "Month_Year": pd.Timestamp(month_start).strftime("%b-%y"),
                "Total_Days": total_days,
                "Count_Within(+/-5%)": count5,
                "Accuracy(+/-5%)": round(count5 / total_days * 100.0, 1) if total_days else None,
                "Accuracy(+/-7%)": round(count7 / total_days * 100.0, 1) if total_days else None,
                "Accuracy(+/-10%)": round(count10 / total_days * 100.0, 1) if total_days else None,
                "MAPE%": round(mape, 1) if np.isfinite(mape) else None,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_sort"] = pd.to_datetime(out["Month_Year"], format="%b-%y", errors="coerce")
    out = out.sort_values("_sort").drop(columns=["_sort"])
    return out


def _di_build_analysis(
    original_df: pd.DataFrame,
    forecast_month: pd.Timestamp,
    distribution: pd.DataFrame,
    monthly_forecast: float,
    holidays_df: Optional[pd.DataFrame],
    interval_df: pd.DataFrame,
    *,
    group_label: str = "Forecast Group",
) -> dict:
    analysis: dict = {"tables": [], "charts": []}
    orig = _di_prepare_original_data(original_df)
    if orig.empty:
        analysis["error"] = "Original data not available for analysis."
        return analysis

    weekday_name = forecast_month.strftime("%A")
    first_days, matching_months = _di_matching_months_table(orig, forecast_month)
    matching_months = matching_months[-3:] if matching_months else []
    matching_df = first_days[first_days["Month_Year"].isin(matching_months)].copy() if not first_days.empty else pd.DataFrame()

    seq_table = _di_sequential_mapping_table(orig, matching_months, "Volume%")
    holidays_safe = holidays_df if isinstance(holidays_df, pd.DataFrame) and not holidays_df.empty else pd.DataFrame()
    table1 = _di_single_month_table(orig, matching_months, "Volume%", holidays_safe)

    available_months = _di_sort_month_str(orig["Month_Year"].dropna().unique().tolist())
    last_available = available_months[-1] if available_months else None
    one_year_ago = forecast_month.replace(year=forecast_month.year - 1)
    two_years_ago = forecast_month.replace(year=forecast_month.year - 2)
    one_year_str = one_year_ago.strftime("%b-%y")
    two_year_str = two_years_ago.strftime("%b-%y")

    table2 = _di_single_month_table(orig, [two_year_str], "Volume%", holidays_safe) if two_year_str in available_months else pd.DataFrame()
    table3_month = one_year_str if one_year_str in available_months else last_available
    table3 = _di_single_month_table(orig, [table3_month], "Volume%", holidays_safe) if table3_month else pd.DataFrame()
    table4 = _di_single_month_table(orig, [last_available], "Volume%", holidays_safe) if last_available else pd.DataFrame()

    holiday_info = ""
    if holidays_df is not None and not holidays_df.empty:
        month_holidays = _di_get_holidays_in_month(holidays_df, forecast_month.year, forecast_month.month)
        if not month_holidays.empty and "holiday_name" in month_holidays.columns:
            names = ", ".join(sorted({str(v) for v in month_holidays["holiday_name"].dropna().tolist()}))
            if names:
                holiday_info = f"Holidays in {forecast_month.strftime('%B %Y')}: {names}"

    year1_impact = _di_impact_analysis_table(orig, one_year_ago.year, one_year_ago.month, holidays_safe, "Volume%")
    year2_impact = _di_impact_analysis_table(orig, two_years_ago.year, two_years_ago.month, holidays_safe, "Volume%")

    base_values = []
    if not table1.empty:
        day_cols = [c for c in table1.columns if c.isdigit()]
        avg_row = table1[table1["Month_Year"] == "Avg"]
        if not avg_row.empty:
            base_values = avg_row[day_cols].iloc[0].tolist()
        else:
            base_values = table1[day_cols].mean(axis=0, skipna=True).tolist()
    forecast_rows, normalized_vals = _di_forecast_rows(base_values, year1_impact, forecast_month)
    if distribution is not None and not distribution.empty and "Distribution_Pct" in distribution.columns:
        final_vals = distribution["Distribution_Pct"].tolist()
        for idx, col in enumerate([c for c in forecast_rows.columns if c.isdigit()]):
            if idx < len(final_vals):
                forecast_rows.loc[forecast_rows["Row"] == "Final", col] = round(final_vals[idx], 2)
        forecast_rows.loc[forecast_rows["Row"] == "Final", "Sum"] = round(sum(final_vals), 2)
        normalized_vals = final_vals

    forecast_vertical = _di_forecast_vertical_table(normalized_vals, forecast_month)
    weekday_chart = _di_weekday_pattern_chart_data(orig, matching_months, "Volume%")

    monthly_tables = _di_weekly_vertical_tables(orig, matching_months, "Volume%")
    interval_section1, interval_section2, interval_chart = _di_interval_section_tables(interval_df)
    daily_accuracy = _di_daily_accuracy_backtest(original_df, forecast_month, distribution)

    analysis["tables"] = [
        (f"Original Data as per Selected {group_label}", orig.head(200)),
        (f"Months where 1st day is {weekday_name}", matching_df),
        ("Daily Distribution - Last 3 Historical Months Falling on Same Weekday", seq_table),
        (f"Table 1: Historical Months Starting on {weekday_name}", table1),
        (f"Table 2: {two_year_str} (2 Years Ago)", table2),
        (f"Table 3: {table3_month or 'N/A'}", table3),
        (f"Table 4: {last_available or 'N/A'} (Latest Data Available)", table4),
        (f"Holiday Impact Analysis for {one_year_str}", year1_impact),
        (f"Year-2 Analysis: {two_year_str} (Reference Only)", year2_impact),
        ("Forecast Rows", forecast_rows),
        ("Avg Distribution for Forecast Month", forecast_vertical),
        ("Daily Accuracy (Backtest on matching months)", daily_accuracy),
        ("Daily Distribution Pattern (Last 3 Months)", interval_section1),
        ("Average Daily Distribution Pattern by Weekday", interval_section2),
    ]
    analysis["charts"] = [
        ("Weekday Pattern Comparison", weekday_chart),
        ("Average Volume Share by Interval and Weekday", interval_chart),
    ]
    analysis["monthly_tables"] = monthly_tables
    analysis["holiday_info"] = holiday_info
    analysis["monthly_forecast"] = monthly_forecast
    return analysis


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    lookup: dict[str, str] = {}
    for c in df.columns:
        base = str(c).strip().lower()
        variants = {
            base,
            base.replace(" ", "_"),
            base.replace(" ", ""),
            base.replace("-", "_"),
            base.replace("-", ""),
            base.replace("_", ""),
            re.sub(r"[^a-z0-9]", "", base),
        }
        for v in variants:
            lookup.setdefault(v, c)

    for nm in candidates:
        key = str(nm).strip().lower()
        candidates_norm = {
            key,
            key.replace(" ", "_"),
            key.replace(" ", ""),
            key.replace("-", "_"),
            key.replace("-", ""),
            key.replace("_", ""),
            re.sub(r"[^a-z0-9]", "", key),
        }
        for cand in candidates_norm:
            col = lookup.get(cand)
            if col:
                return col
    return None


def _di_build_forecasts(
    transform_df: pd.DataFrame,
    interval_df: pd.DataFrame,
    forecast_month: pd.Timestamp,
    group_value: Optional[str],
    model_value: Optional[str],
    month_value: Optional[str],
    group_level: str = "forecast_group",
    distribution_override: Optional[pd.DataFrame] = None,
    holidays_df: Optional[pd.DataFrame] = None,
    original_df: Optional[pd.DataFrame] = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    if transform_df.empty:
        return "Load a transformed forecast first.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

    tf = transform_df.copy()
    tf_cols = {c.lower(): c for c in tf.columns}
    group_level_norm = str(group_level or "forecast_group").strip().lower()
    fg_col = None
    if group_level_norm == "business_area":
        fg_col = tf_cols.get("business_area")
    if not fg_col:
        fg_col = tf_cols.get("forecast_group") or tf_cols.get("queue_name") or tf_cols.get("category")
    model_col = tf_cols.get("model")

    if fg_col and group_value:
        tf = tf[tf[fg_col] == group_value]
    if model_col and model_value:
        tf = tf[tf[model_col] == model_value]

    val_col = None
    for cand in [
        "Final_Forecast_Post_Transformations",
        "Final_Forecast",
        "Forecast_Marketing Campaign 3",
        "Forecast_Marketing Campaign 2",
        "Forecast_Marketing Campaign 1",
    ]:
        if cand in tf.columns:
            val_col = cand
            break
    if not val_col:
        return "No forecast value column found.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

    if forecast_month is not None:
        month_num = forecast_month.month
        year_val = forecast_month.year
        if "Year" in tf.columns:
            tf["Year"] = pd.to_numeric(tf["Year"], errors="coerce")
        if "Month" in tf.columns:
            month_series = tf["Month"].astype(str).str.strip()
            month_num_series = pd.to_numeric(month_series, errors="coerce")
            if month_num_series.isna().any():
                month_num_series = month_num_series.fillna(month_series.apply(_month_name_to_num))
            tf["_month_num"] = month_num_series
            tf = tf[tf["_month_num"] == month_num]
        if "Year" in tf.columns:
            tf = tf[tf["Year"] == year_val]
        if "Month_Year" in tf.columns:
            month_year_series = tf["Month_Year"].astype(str).str.strip()
            month_year_dt = pd.to_datetime(month_year_series, format="%b-%y", errors="coerce")
            month_year_dt = month_year_dt.fillna(
                pd.to_datetime(month_year_series, format="%b-%Y", errors="coerce")
            )
            if month_year_dt.isna().any():
                month_year_dt = month_year_dt.fillna(
                    month_year_series.apply(lambda v: pd.to_datetime(v, errors="coerce"))
                )
            tf["_month_year_dt"] = month_year_dt
            tf = tf[(tf["_month_year_dt"].dt.year == year_val) & (tf["_month_year_dt"].dt.month == month_num)]
    elif month_value and "Month_Year" in tf.columns:
        tf = tf[tf["Month_Year"].astype(str) == str(month_value)]

    if tf.empty:
        return "No matching row for the selected filters.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

    forecast_val = pd.to_numeric(tf[val_col], errors="coerce").dropna()
    if forecast_val.empty:
        return "Selected forecast value is missing.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}
    monthly_forecast = float(forecast_val.iloc[0])

    if original_df is None:
        original_df = pd.DataFrame()
    if distribution_override is not None and not distribution_override.empty:
        distribution = distribution_override.copy()
        dist_msg = "Using edited distribution."
    else:
        if original_df.empty:
            return "Original data missing.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}
        daily = (
            original_df.groupby("date", as_index=False)["volume"]
            .sum()
            .dropna(subset=["date"])
        )
        daily["month_start"] = daily["date"].dt.to_period("M").dt.to_timestamp()
        daily["day"] = daily["date"].dt.day
        daily_totals = daily.groupby("month_start")["volume"].transform("sum")
        daily["volume_pct"] = np.where(daily_totals > 0, daily["volume"] / daily_totals * 100, np.nan)
        distribution, dist_msg = _di_compute_distribution(daily, forecast_month)

    distribution = _di_normalize_distribution(distribution)
    dist_vals = pd.to_numeric(distribution.get("Distribution_Pct"), errors="coerce").fillna(0.0).tolist()
    weights, dist_total = _normalize_weights(dist_vals)
    if not weights:
        n = int(len(distribution))
        if n > 0:
            weights = [1.0 / n] * n
            dist_msg = f"{dist_msg} Distribution was empty; using equal split."
        else:
            weights = []
    display_pct = (
        _round_to_step_with_total([w * 100 for w in weights], step=0.1, total=100.0)
        if weights
        else []
    )
    if display_pct:
        distribution["Distribution_Pct"] = display_pct
    dist_msg = f"{dist_msg} Normalized (sum {sum(display_pct) if display_pct else 0.0:.1f}%)."

    daily_tbl = distribution.copy()
    target_total = int(round(float(monthly_forecast)))
    raw_daily = [w * target_total for w in weights] if weights else [0.0] * len(daily_tbl)
    daily_vals = (
        _round_to_step_with_total(raw_daily, step=1.0, total=target_total)
        if weights
        else [0.0] * len(daily_tbl)
    )
    daily_tbl["Daily_Forecast"] = [int(round(v)) for v in daily_vals]

    interval_ratio, overall_ratio, interval_msg = _di_interval_ratios(interval_df)
    interval_rows = []
    if not interval_ratio.empty or not overall_ratio.empty:
        for _, row in daily_tbl.iterrows():
            wd = row["Weekday"]
            dv = float(row["Daily_Forecast"])
            dist = interval_ratio[interval_ratio["weekday"] == wd][["interval", "ratio"]].copy()
            if dist.empty:
                dist = overall_ratio.copy()
            if dist.empty:
                continue
            total = dist["ratio"].sum()
            if total <= 0:
                continue
            dist["ratio"] = dist["ratio"] / total
            if dv <= 0:
                dist["Interval_Forecast"] = 0
            else:
                raw_ivl = (dist["ratio"] * dv).tolist()
                ivl_vals = _round_to_step_with_total(raw_ivl, step=1.0, total=dv)
                dist["Interval_Forecast"] = [int(round(v)) for v in ivl_vals]
            dist["Date"] = row["Date"]
            dist["Weekday"] = wd
            interval_rows.append(dist[["Date", "Weekday", "interval", "Interval_Forecast"]])

    interval_tbl = pd.concat(interval_rows, ignore_index=True) if interval_rows else pd.DataFrame()
    if not interval_tbl.empty:
        interval_tbl = interval_tbl.rename(columns={"interval": "Interval"})
        interval_tbl["Interval"] = interval_tbl["Interval"].apply(_normalize_interval_label)

    analysis = _di_build_analysis(
        original_df,
        forecast_month,
        distribution,
        monthly_forecast,
        holidays_df,
        interval_df,
        group_label="Business Area" if group_level_norm == "business_area" else "Forecast Group",
    )

    meta = {
        "monthly_forecast": monthly_forecast,
        "distribution_sum": float(distribution["Distribution_Pct"].sum()) if not distribution.empty else 0.0,
        "interval_msg": interval_msg,
        "dist_msg": dist_msg,
    }
    status = f"Daily/interval forecast ready | monthly {monthly_forecast:,.0f}"
    if analysis.get("error"):
        status = f"{status} | {analysis['error']}"
    return status, distribution, daily_tbl, interval_tbl, meta, analysis


def run_daily_interval(
    transform_payload: Any,
    interval_payload: Any,
    forecast_month: str,
    group_value: Optional[str] = None,
    model_value: Optional[str] = None,
    distribution_override: Optional[Any] = None,
    original_data: Optional[Any] = None,
    holidays: Optional[Any] = None,
    group_level: str = "forecast_group",
) -> dict:
    transform_df = df_from_payload(transform_payload)
    interval_df = df_from_payload(interval_payload)
    if transform_df.empty:
        return {"status": "Load a transformed forecast first.", "results": {}}

    try:
        forecast_month_dt = pd.to_datetime(forecast_month).to_period("M").to_timestamp()
    except Exception:
        return {"status": "Invalid forecast month.", "results": {}}

    orig_df = df_from_payload(original_data)
    orig_df = _di_normalize_original_data(orig_df, group_value, group_level)
    if orig_df.empty:
        try:
            exports_dir = Path(__file__).resolve().parent.parent.parent / "exports"
            path = exports_dir / "original_data.csv"
            if path.exists():
                orig_df = pd.read_csv(path)
                orig_df = _di_normalize_original_data(orig_df, group_value, group_level)
        except Exception:
            orig_df = pd.DataFrame()
    if interval_df.empty and not orig_df.empty:
        ivl_col = _pick_col(
            orig_df,
            ("interval", "time", "interval_start", "start_time", "timeslot", "time_slot"),
        )
        if ivl_col:
            interval_df = orig_df.copy()

    dist_override_df = df_from_payload(distribution_override)
    holidays_df = df_from_payload(holidays)
    if not holidays_df.empty and "holiday_date" in holidays_df.columns:
        holidays_df["holiday_date"] = pd.to_datetime(holidays_df["holiday_date"], errors="coerce")

    status, dist_df, daily_tbl, interval_tbl, meta, analysis = _di_build_forecasts(
        transform_df,
        interval_df,
        forecast_month_dt,
        group_value,
        model_value,
        None,
        group_level=group_level,
        distribution_override=dist_override_df if not dist_override_df.empty else None,
        holidays_df=holidays_df if not holidays_df.empty else None,
        original_df=orig_df,
    )

    if dist_df.empty and daily_tbl.empty:
        return {"status": status, "results": {}, "meta": meta}

    dist_records = _json_safe_records(dist_df)
    daily_records = _json_safe_records(daily_tbl)
    interval_records = _json_safe_records(interval_tbl)
    results = {
        "distribution": dist_records,
        "daily": daily_records,
        "interval": interval_records,
        "analysis": _serialize_analysis(analysis),
    }
    return {
        "status": status,
        "results": results,
        "meta": meta,
        "interval_summary": _interval_summary_text(interval_tbl),
    }


def _serialize_analysis(analysis: dict) -> dict:
    if not analysis:
        return {}
    tables = []
    for title, df in analysis.get("tables", []):
        if df is None or df.empty:
            continue
        tables.append({"title": title, "rows": _json_safe_records(df)})
    monthly_tables = []
    for title, df in analysis.get("monthly_tables", []):
        if df is None or df.empty:
            continue
        monthly_tables.append({"title": title, "rows": _json_safe_records(df)})
    charts = []
    for title, chart in analysis.get("charts", []):
        if chart is None:
            continue
        charts.append({"title": title, "chart": chart})
    return {
        "tables": tables,
        "monthly_tables": monthly_tables,
        "charts": charts,
        "holiday_info": analysis.get("holiday_info", ""),
        "monthly_forecast": analysis.get("monthly_forecast"),
    }
