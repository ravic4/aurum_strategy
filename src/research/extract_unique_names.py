"""
Extract unique entity names from airline_employment.xlsx (robust)

Input:
  src/research/airline_employment.xlsx

Outputs:
  src/research/data/unique_entity_names.csv
  src/research/data/name_extraction_diagnostics.csv

Run from project root:
  python -m src.research.extract_unique_names
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd


# ----------------------------
# Paths
# ----------------------------
RESEARCH_DIR = Path(__file__).resolve().parent
INPUT_XLSX = RESEARCH_DIR / "airline_employment.xlsx"
DATA_DIR = RESEARCH_DIR / "data"
OUT_NAMES = DATA_DIR / "unique_entity_names.csv"
OUT_DIAG = DATA_DIR / "name_extraction_diagnostics.csv"


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("extract_unique_names")


def read_best_sheet(path: Path) -> pd.DataFrame:
    """Pick the sheet with the most non-empty rows (as before)."""
    xls = pd.ExcelFile(path)
    best_df = None
    best_rows = -1
    best_sheet = None

    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if len(df) > best_rows:
            best_rows = len(df)
            best_df = df
            best_sheet = sheet

    if best_df is None:
        raise ValueError("No usable data found in Excel.")

    log.info(f"Using sheet: {best_sheet} (rows={best_rows}, cols={best_df.shape[1]})")
    return best_df


def score_name_likeness(series: pd.Series) -> Tuple[float, Dict[str, Any]]:
    """
    Score how likely a column is to be "entity names" based on values (not header).
    Higher is better.
    """
    s = series.copy()

    # Convert to string safely
    s_str = s.astype("string").str.strip()
    nonempty = s_str.replace("", pd.NA).dropna()

    if nonempty.empty:
        return 0.0, {
            "nonempty": 0,
            "unique": 0,
            "unique_ratio": 0.0,
            "avg_len": 0.0,
            "numeric_parse_ratio": 1.0,
            "date_parse_ratio": 1.0,
        }

    # Basic stats
    nonempty_n = int(nonempty.shape[0])
    unique_n = int(nonempty.nunique(dropna=True))
    unique_ratio = unique_n / max(1, nonempty_n)

    avg_len = float(nonempty.str.len().mean())

    # How numeric is it?
    num = pd.to_numeric(nonempty.str.replace(",", "", regex=False), errors="coerce")
    numeric_parse_ratio = float(num.notna().mean())

    # How date-like is it?
    dt = pd.to_datetime(nonempty, errors="coerce", infer_datetime_format=True)
    date_parse_ratio = float(dt.notna().mean())

    # Heuristic scoring:
    # - prefer many non-empty values
    # - prefer decent uniqueness (names repeat but not like "US" 1 value)
    # - prefer reasonable text length (>= 3)
    # - penalize columns that are mostly numeric or mostly dates
    score = 0.0
    score += min(1.0, nonempty_n / 100.0) * 0.35
    score += min(1.0, unique_ratio * 2.0) * 0.25
    score += min(1.0, avg_len / 15.0) * 0.25
    score += (1.0 - numeric_parse_ratio) * 0.10
    score += (1.0 - date_parse_ratio) * 0.05

    diagnostics = {
        "nonempty": nonempty_n,
        "unique": unique_n,
        "unique_ratio": round(unique_ratio, 4),
        "avg_len": round(avg_len, 2),
        "numeric_parse_ratio": round(numeric_parse_ratio, 4),
        "date_parse_ratio": round(date_parse_ratio, 4),
    }
    return float(score), diagnostics


def pick_best_name_column(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best_col: Optional[str] = None
    best_score = -1.0

    for col in df.columns:
        score, diag = score_name_likeness(df[col])
        row = {"column": str(col), "score": round(score, 6), **diag}
        rows.append(row)

        if score > best_score:
            best_score = score
            best_col = col

    diag_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    if best_col is None or best_score <= 0:
        raise ValueError("Could not find any column with non-empty name-like values.")

    log.info(f"Best name column: {best_col} (score={best_score:.4f})")
    return str(best_col), diag_df


def extract_unique_names(df: pd.DataFrame, name_col: str) -> pd.Series:
    s = df[name_col].astype("string").str.strip().replace("", pd.NA).dropna()

    # Remove obviously junk rows (pure numbers / pure dates)
    num = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

    keep = ~(num.notna() & (num.astype("Int64").notna()))  # penalize numeric
    # If something is clearly a date, drop it too
    keep = keep & ~(dt.notna())

    s2 = s[keep]

    # Unique + sorted
    return pd.Series(sorted(s2.unique()))


def main() -> int:
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_XLSX}")

    DATA_DIR.mkdir(exist_ok=True)

    df = read_best_sheet(INPUT_XLSX)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    name_col, diag_df = pick_best_name_column(df)
    diag_df.to_csv(OUT_DIAG, index=False)
    log.info(f"Wrote diagnostics → {OUT_DIAG}")

    names = extract_unique_names(df, name_col)

    out_df = pd.DataFrame(
        {
            "source_name": names,
            "notes": "",
            "final_ticker": "",
            "ticker_status": "",
        }
    )

    out_df.to_csv(OUT_NAMES, index=False)
    log.info(f"Wrote {len(out_df)} names → {OUT_NAMES}")

    # Helpful terminal preview
    log.info("Preview (first 15 names):")
    for x in out_df["source_name"].head(15).tolist():
        log.info(f"  - {x}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
