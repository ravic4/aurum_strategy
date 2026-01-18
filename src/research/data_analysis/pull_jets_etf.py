"""
Pull JETS ETF (U.S. Global Jets ETF) historical data from Financial Modeling Prep API.
Converts daily data to monthly and saves both versions.

Outputs:
  src/research/data/jets_etf_daily.csv
  src/research/data/jets_etf_monthly.csv

Run from project root:
  python -m src.research.data_analysis.pull_jets_etf

Env:
  FMP_API_KEY=...
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ----------------------------
# Paths
# ----------------------------
RESEARCH_DIR = Path(__file__).resolve().parent.parent  # Go up to research folder
DATA_DIR = RESEARCH_DIR / "data"

JETS_DAILY_CSV = DATA_DIR / "jets_etf_daily.csv"
JETS_MONTHLY_CSV = DATA_DIR / "jets_etf_monthly.csv"


# ----------------------------
# Config
# ----------------------------
FMP_API_KEY = os.getenv("FMP_API_KEY") or os.getenv("apikey") or os.getenv("API_KEY")

# FMP Stable endpoint for full EOD OHLCV
FMP_BASE = "https://financialmodelingprep.com"
FMP_ENDPOINT = "/stable/historical-price-eod/full"

SYMBOL = "JETS"


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_jets_etf")


# ----------------------------
# Helpers
# ----------------------------
def fmp_get_prices(symbol: str, timeout: int = 30) -> Any:
    """
    Calls FMP API to get full historical price data.
    No date restrictions - gets all available data.
    """
    url = f"{FMP_BASE}{FMP_ENDPOINT}"
    params: Dict[str, Any] = {
        "symbol": symbol,
        "apikey": FMP_API_KEY,
    }

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def normalize_price_payload(payload: Any, symbol: str) -> pd.DataFrame:
    """
    Normalize FMP API response to DataFrame.
    """
    if isinstance(payload, dict) and "historical" in payload and isinstance(payload["historical"], list):
        rows = payload["historical"]
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Standardize date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Add symbol
    df["symbol"] = symbol

    # Keep typical OHLCV fields if present
    keep = ["symbol", "date", "open", "high", "low", "close", "volume", "vwap", "change", "changePercent"]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy() if cols else df

    # Sort ascending by date
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def convert_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily data to monthly using last trading day of each month.
    """
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")

    # Get last trading day of each month
    monthly = df.groupby("year_month").last().reset_index()

    # Drop the helper column
    monthly = monthly.drop("year_month", axis=1)

    return monthly


def main() -> int:
    if not FMP_API_KEY:
        raise EnvironmentError("Missing FMP_API_KEY in your environment/.env")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Pulling {SYMBOL} ETF data from FMP...")

    # Fetch data
    try:
        payload = fmp_get_prices(SYMBOL)
    except Exception as e:
        log.error(f"Failed to fetch data: {e}")
        return 1

    # Normalize to DataFrame
    df_daily = normalize_price_payload(payload, SYMBOL)

    if df_daily.empty:
        log.error(f"No data returned for {SYMBOL}")
        return 1

    # Save daily data
    df_daily.to_csv(JETS_DAILY_CSV, index=False)
    log.info(f"Saved daily data → {JETS_DAILY_CSV}")
    log.info(f"  Rows: {len(df_daily)}")
    log.info(f"  Date range: {df_daily['date'].min().date()} to {df_daily['date'].max().date()}")

    # Convert to monthly
    df_monthly = convert_to_monthly(df_daily)
    df_monthly.to_csv(JETS_MONTHLY_CSV, index=False)
    log.info(f"Saved monthly data → {JETS_MONTHLY_CSV}")
    log.info(f"  Rows: {len(df_monthly)}")
    log.info(f"  Date range: {df_monthly['date'].min().date()} to {df_monthly['date'].max().date()}")

    # Print sample
    print("\n--- Daily Data Sample (first 5 rows) ---")
    print(df_daily.head().to_string(index=False))

    print("\n--- Monthly Data Sample (first 10 rows) ---")
    print(df_monthly.head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
