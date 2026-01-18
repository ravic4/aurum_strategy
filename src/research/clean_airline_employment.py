"""
Clean airline_employment.xlsx (single-file pipeline)

Location:
  src/research/clean_airline_employment.py

Input:
  src/research/airline_employment.xlsx

Outputs:
  src/research/data/cleaned_airline_employment.csv
  src/research/data/cleaned_airline_employment.parquet (optional)

Run from project root:
  python -m src.research.clean_airline_employment
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd


# ----------------------------
# Paths (ALL local to research/)
# ----------------------------
RESEARCH_DIR = Path(__file__).resolve().parent
INPUT_XLSX = RESEARCH_DIR / "airline_employment.xlsx"

OUTPUT_DIR = RESEARCH_DIR / "data"
OUTPUT_CSV = OUTPUT_DIR / "cleaned_airline_employment.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "cleaned_airline_employment.parquet"


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clean_airline_employment")


# ----------------------------
# Helpers
# ----------------------------
def snake_case(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s\-]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_").lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen = {}
    cols = []
    for c in df.columns:
        base = snake_case(str(c)) or "col"
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        cols.append(base)
    df = df.copy()
    df.columns = cols
    return df


def looks_like_date_series(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.7


def coerce_numeric(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace("%", "", regex=False)
    s = s.replace({"": pd.NA, "na": pd.NA, "n/a": pd.NA, "-": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def infer_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    report = {}
    df = df.copy()

    for col in df.columns:
        s = df[col]

        if s.dtype == object or pd.api.types.is_string_dtype(s):
            if looks_like_date_series(s):
                parsed = pd.to_datetime(s, errors="coerce")
                if parsed.notna().mean() >= 0.6:
                    df[col] = parsed
                    report[col] = "datetime"
                    continue

            num = coerce_numeric(s)
            if s.notna().sum() > 0 and num.notna().sum() / s.notna().sum() >= 0.7:
                df[col] = num
                report[col] = "numeric"
            else:
                df[col] = s.str.strip()
                report[col] = "text"
        else:
            report[col] = str(s.dtype)

    return df, report


def read_best_sheet(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    best_df, best_rows = None, -1

    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet).dropna(how="all").dropna(axis=1, how="all")
        if len(df) > best_rows:
            best_df, best_rows = df, len(df)

    if best_df is None:
        raise ValueError("No usable data found in Excel file.")

    return best_df


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    log.info(f"Reading: {INPUT_XLSX}")

    df = read_best_sheet(INPUT_XLSX)
    log.info(f"Raw shape: {df.shape}")

    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = normalize_columns(df)

    df, type_report = infer_types(df)
    df = df.drop_duplicates()

    # sort if any datetime column exists
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if date_cols:
        df = df.sort_values(date_cols[0])

    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Saved CSV → {OUTPUT_CSV}")

    try:
        df.to_parquet(OUTPUT_PARQUET, index=False)
        log.info(f"Saved Parquet → {OUTPUT_PARQUET}")
    except Exception:
        log.warning("Parquet skipped (install pyarrow to enable)")

    log.info("Column type inference:")
    for k, v in type_report.items():
        log.info(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
