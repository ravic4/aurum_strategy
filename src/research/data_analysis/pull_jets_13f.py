"""
Pull 13F filings data for JETS ETF from Financial Modeling Prep API.
Shows institutional holdings and ownership changes over time.

Outputs:
  src/research/data/jets_13f_holdings.csv

Run from project root:
  python -m src.research.data_analysis.pull_jets_13f

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

JETS_13F_CSV = DATA_DIR / "jets_13f_holdings.csv"


# ----------------------------
# Config
# ----------------------------
FMP_API_KEY = os.getenv("FMP_API_KEY") or os.getenv("apikey") or os.getenv("API_KEY")

FMP_BASE = "https://financialmodelingprep.com"

SYMBOL = "JETS"


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_jets_13f")


# ----------------------------
# API Endpoints
# ----------------------------
def get_13f_holders(symbol: str, timeout: int = 30) -> Any:
    """
    Get list of institutional holders from 13F filings.
    Endpoint: /api/v3/institutional-holder/{symbol}
    """
    url = f"{FMP_BASE}/api/v3/institutional-holder/{symbol}"
    params = {"apikey": FMP_API_KEY}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_13f_asset_allocation(symbol: str, timeout: int = 30) -> Any:
    """
    Get 13F asset allocation data.
    Endpoint: /stable/institutional-ownership/symbol-ownership
    """
    url = f"{FMP_BASE}/stable/institutional-ownership/symbol-ownership"
    params = {"symbol": symbol, "apikey": FMP_API_KEY}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_etf_holder(symbol: str, timeout: int = 30) -> Any:
    """
    Get ETF holder information.
    Endpoint: /api/v3/etf-holder/{symbol}
    """
    url = f"{FMP_BASE}/api/v3/etf-holder/{symbol}"
    params = {"apikey": FMP_API_KEY}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_etf_info(symbol: str, timeout: int = 30) -> Any:
    """
    Get ETF information including holdings.
    Endpoint: /api/v3/etf-info
    """
    url = f"{FMP_BASE}/api/v3/etf-info"
    params = {"symbol": symbol, "apikey": FMP_API_KEY}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> int:
    if not FMP_API_KEY:
        raise EnvironmentError("Missing FMP_API_KEY in your environment/.env")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Pulling 13F/institutional data for {SYMBOL} ETF...")

    all_data = []

    # Try institutional holders endpoint
    log.info("Fetching institutional holders (13F)...")
    try:
        holders = get_13f_holders(SYMBOL)
        if holders and isinstance(holders, list):
            df_holders = pd.DataFrame(holders)
            df_holders["data_type"] = "institutional_holder"
            all_data.append(df_holders)
            log.info(f"  Found {len(holders)} institutional holders")
        else:
            log.info("  No institutional holder data returned")
    except Exception as e:
        log.warning(f"  Institutional holders endpoint failed: {e}")

    # Try symbol ownership endpoint
    log.info("Fetching symbol ownership data...")
    try:
        ownership = get_13f_asset_allocation(SYMBOL)
        if ownership and isinstance(ownership, list):
            df_ownership = pd.DataFrame(ownership)
            df_ownership["data_type"] = "symbol_ownership"
            all_data.append(df_ownership)
            log.info(f"  Found {len(ownership)} ownership records")
        else:
            log.info("  No symbol ownership data returned")
    except Exception as e:
        log.warning(f"  Symbol ownership endpoint failed: {e}")

    # Try ETF holder endpoint
    log.info("Fetching ETF holdings...")
    try:
        etf_holdings = get_etf_holder(SYMBOL)
        if etf_holdings and isinstance(etf_holdings, list):
            df_etf = pd.DataFrame(etf_holdings)
            df_etf["data_type"] = "etf_holdings"
            all_data.append(df_etf)
            log.info(f"  Found {len(etf_holdings)} ETF holdings")
        else:
            log.info("  No ETF holdings data returned")
    except Exception as e:
        log.warning(f"  ETF holder endpoint failed: {e}")

    # Try ETF info endpoint
    log.info("Fetching ETF info...")
    try:
        etf_info = get_etf_info(SYMBOL)
        if etf_info and isinstance(etf_info, list):
            df_info = pd.DataFrame(etf_info)
            df_info["data_type"] = "etf_info"
            all_data.append(df_info)
            log.info(f"  Found ETF info")
            print("\n--- ETF Info ---")
            print(df_info.to_string(index=False))
        else:
            log.info("  No ETF info returned")
    except Exception as e:
        log.warning(f"  ETF info endpoint failed: {e}")

    # Combine and save
    if all_data:
        # Save each type separately for clarity
        for df in all_data:
            data_type = df["data_type"].iloc[0]
            output_file = DATA_DIR / f"jets_{data_type}.csv"
            df.to_csv(output_file, index=False)
            log.info(f"Saved → {output_file}")

            print(f"\n--- {data_type.upper()} (first 10 rows) ---")
            print(df.head(10).to_string(index=False))

        log.info("Done!")
        return 0
    else:
        log.error("No data retrieved from any endpoint")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
