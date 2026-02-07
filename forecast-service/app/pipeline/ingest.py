from __future__ import annotations

import io
import re
from typing import Optional, Sequence

import pandas as pd


def _normalize_sheet(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def parse_upload_any(
    filename: str,
    content: bytes,
    preferred_sheets: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, str]:
    if not filename:
        return pd.DataFrame(), "No filename supplied."
    try:
        lower = filename.lower()
        sheet_note = ""
        if lower.endswith(".csv"):
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        else:
            xl = pd.ExcelFile(io.BytesIO(content))
            sheet_map = {_normalize_sheet(name): name for name in xl.sheet_names}
            selected = None
            for pref in preferred_sheets or []:
                key = _normalize_sheet(pref)
                if key in sheet_map:
                    selected = sheet_map[key]
                    break
            if not selected and xl.sheet_names:
                selected = xl.sheet_names[0]
            df = xl.parse(selected) if selected else pd.DataFrame()
            if selected:
                sheet_note = f" (sheet '{selected}')"
        msg = f"Loaded {len(df):,} rows from {filename}{sheet_note}."
        return df, msg
    except Exception as exc:
        return pd.DataFrame(), f"Failed to read {filename}: {exc}"
