"""
Pull EOD stock prices for manually-mapped tickers.

Input (manual mapping):
  src/research/data/unique_entity_names.csv
    - must include: final_ticker, exchange
    - optional: source_name, index, ticker_status, notes

Outputs:
  src/research/data/tickers_to_pull.csv
  src/research/data/prices/prices_<TICKER>.csv      (per ticker)
  src/research/data/prices_eod_long.csv             (combined)
  src/research/data/prices_eod_long.parquet         (combined, best)

Run from project root:
  python -m src.research.pull_prices_from_mapping

Env:
  FMP_API_KEY=...

Optional env:
  PRICE_FROM=2000-01-01
  PRICE_TO=2026-01-17
  FMP_SLEEP_SECONDS=0.15
  WRITE_PER_TICKER=1          # 1 or 0
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

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
RESEARCH_DIR = Path(__file__).resolve().parent
DATA_DIR = RESEARCH_DIR / "data"
MAPPING_CSV = DATA_DIR / "unique_entity_names.csv"

TICKERS_OUT = DATA_DIR / "tickers_to_pull.csv"
PRICES_DIR = DATA_DIR / "prices"
PRICES_LONG_CSV = DATA_DIR / "prices_eod_long.csv"
PRICES_LONG_PARQUET = DATA_DIR / "prices_eod_long.parquet"


# ----------------------------
# Config
# ----------------------------
FMP_API_KEY = os.getenv("FMP_API_KEY") or os.getenv("apikey") or os.getenv("API_KEY")
SLEEP_SECONDS = float(os.getenv("FMP_SLEEP_SECONDS", "0.15"))

PRICE_FROM = os.getenv("PRICE_FROM", "").strip()  # e.g., 2010-01-01
PRICE_TO = os.getenv("PRICE_TO", "").strip()      # e.g., 2026-01-17

WRITE_PER_TICKER = os.getenv("WRITE_PER_TICKER", "1").strip() not in {"0", "false", "False"}

# FMP Stable endpoint for full EOD OHLCV:
FMP_BASE = "https://financialmodelingprep.com"
FMP_ENDPOINT = "/stable/historical-price-eod/full"  # ?symbol=...&apikey=... :contentReference[oaicite:1]{index=1}


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_prices_from_mapping")


# ----------------------------
# Helpers
# ----------------------------
def must_have_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in mapping file: {missing}")


def clean_ticker(x: Any) -> str:
    return str(x).strip().upper()


def clean_exchange(x: Any) -> str:
    return str(x).strip().upper()


def fmp_get_prices(symbol: str, date_from: str = "", date_to: str = "", timeout: int = 30) -> Any:
    """
    Calls:
      https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=AAPL&apikey=...
    and optionally with from/to if provided (these params are widely used across FMP endpoints).
    """
    url = f"{FMP_BASE}{FMP_ENDPOINT}"
    params: Dict[str, Any] = {
        "symbol": symbol,
        "apikey": FMP_API_KEY,
    }
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def normalize_price_payload(payload: Any, symbol: str) -> pd.DataFrame:
    """
    Payload structure can vary by endpoint/version.
    We handle common patterns:
      - list[dict] with date/open/high/low/close/volume...
      - dict with key 'historical' -> list[dict]
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
    else:
        # last resort: try index-like
        pass

    # Add symbol
    df["symbol"] = symbol

    # Keep typical OHLCV fields if present
    keep = ["symbol", "date", "open", "high", "low", "close", "volume", "vwap", "change", "changePercent"]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy() if cols else df

    # Sort ascending
    if "date" in df.columns:
        df = df.sort_values("date")

    return df


def main() -> int:
    if not FMP_API_KEY:
        raise EnvironmentError("Missing FMP_API_KEY in your environment/.env")

    if not MAPPING_CSV.exists():
        raise FileNotFoundError(f"Missing mapping file: {MAPPING_CSV}")

    DATA_DIR.mkdir(exist_ok=True)
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(MAPPING_CSV)

    # Required columns based on what you said you added
    must_have_cols(mapping, ["final_ticker", "exchange"])

    # Filter to only rows with ticker + exchange filled
    mapping["final_ticker"] = mapping["final_ticker"].astype("string").fillna("").map(clean_ticker)
    mapping["exchange"] = mapping["exchange"].astype("string").fillna("").map(clean_exchange)

    universe = mapping[(mapping["final_ticker"] != "") & (mapping["exchange"] != "")].copy()

    if universe.empty:
        raise ValueError("No rows found with BOTH final_ticker and exchange filled. Nothing to pull.")

    # Deduplicate tickers (if multiple names map to same ticker)
    universe = universe.drop_duplicates(subset=["final_ticker", "exchange"]).reset_index(drop=True)

    # Save the universe list you’ll pull
    universe_out = universe.rename(columns={"final_ticker": "symbol"})[["symbol", "exchange"] + [c for c in ["source_name", "index", "ticker_status", "notes"] if c in universe.columns]]
    universe_out.to_csv(TICKERS_OUT, index=False)
    log.info(f"Saved tickers universe ({len(universe_out)}) → {TICKERS_OUT}")

    all_prices: List[pd.DataFrame] = []

    for i, row in universe_out.iterrows():
        symbol = row["symbol"]
        exch = row["exchange"]

        if i > 0:
            time.sleep(SLEEP_SECONDS)

        log.info(f"[{i+1}/{len(universe_out)}] Pulling {symbol} ({exch})")

        # Basic retry (network hiccups happen)
        payload = None
        for attempt in range(1, 4):
            try:
                payload = fmp_get_prices(symbol, date_from=PRICE_FROM, date_to=PRICE_TO)
                break
            except Exception as e:
                if attempt == 3:
                    log.warning(f"FAILED {symbol}: {e}")
                else:
                    time.sleep(0.8 * attempt)

        if payload is None:
            continue

        dfp = normalize_price_payload(payload, symbol)
        if dfp.empty:
            log.warning(f"No price data returned for {symbol} (may be delisted/unsupported or symbol mismatch).")
            continue

        # Attach exchange + any metadata columns you have
        dfp["exchange"] = exch
        for meta_col in ["index", "ticker_status", "source_name"]:
            if meta_col in row.index:
                dfp[meta_col] = row[meta_col]

        all_prices.append(dfp)

        if WRITE_PER_TICKER:
            out_path = PRICES_DIR / f"prices_{symbol}.csv"
            dfp.to_csv(out_path, index=False)

    if not all_prices:
        raise ValueError("No price data pulled for any ticker. Check symbols and API key/plan access.")

    prices_long = pd.concat(all_prices, ignore_index=True)

    # Write combined outputs
    prices_long.to_csv(PRICES_LONG_CSV, index=False)
    log.info(f"Wrote combined CSV → {PRICES_LONG_CSV} (rows={len(prices_long)})")

    try:
        prices_long.to_parquet(PRICES_LONG_PARQUET, index=False)
        log.info(f"Wrote combined Parquet → {PRICES_LONG_PARQUET}")
    except Exception as e:
        log.warning(f"Parquet skipped (install pyarrow to enable): {e}")

    # Quick summary
    n_syms = prices_long["symbol"].nunique() if "symbol" in prices_long.columns else 0
    dmin = prices_long["date"].min() if "date" in prices_long.columns else None
    dmax = prices_long["date"].max() if "date" in prices_long.columns else None
    log.info(f"Pulled symbols: {n_syms} | date range: {dmin} → {dmax}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
