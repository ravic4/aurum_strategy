"""
Find institutions that hold JETS ETF by searching SEC EDGAR 13F filings.
Uses SEC full-text search to find 13F filings mentioning "JETS" or the CUSIP.

JETS ETF CUSIP: 26922A107

Outputs:
  src/research/data/jets_institutional_holders.csv

Run from project root:
  python -m src.research.data_analysis.pull_jets_holders_sec

No API key required - uses SEC EDGAR free API.
"""

from __future__ import annotations

import os
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Any

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
RESEARCH_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = RESEARCH_DIR / "data"

OUTPUT_CSV = DATA_DIR / "jets_institutional_holders.csv"


# ----------------------------
# Config
# ----------------------------
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "RaviCapital/Aurum research@example.com")

# JETS ETF info
JETS_CUSIP = "26922A107"
JETS_SYMBOL = "JETS"

REQUEST_DELAY = 0.12  # SEC rate limit


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_jets_holders")


def get_headers() -> Dict[str, str]:
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


def search_edgar_fulltext(query: str, form_types: str = "13F-HR", start: int = 0, timeout: int = 30) -> Dict:
    """
    Use SEC EDGAR full-text search API.
    """
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": query,
        "forms": form_types,
        "dateRange": "custom",
        "startdt": "2020-01-01",
        "enddt": "2026-12-31",
        "start": start,
    }

    r = requests.get(url, params=params, headers=get_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_filing_document(url: str, timeout: int = 30) -> str:
    """Fetch a filing document."""
    r = requests.get(url, headers=get_headers(), timeout=timeout)
    r.raise_for_status()
    return r.text


def parse_13f_for_jets(content: str) -> List[Dict]:
    """
    Parse 13F filing content to find JETS holdings.
    Look for JETS symbol or CUSIP.
    """
    holdings = []

    # Look for lines containing JETS or the CUSIP
    lines = content.split('\n')

    for i, line in enumerate(lines):
        line_upper = line.upper()
        if JETS_SYMBOL in line_upper or JETS_CUSIP.replace("-", "") in line_upper.replace("-", ""):
            # Try to extract value and shares from surrounding context
            context = '\n'.join(lines[max(0, i-2):min(len(lines), i+3)])

            # Try to find numbers (shares, value)
            numbers = re.findall(r'[\d,]+', context)
            numbers = [int(n.replace(',', '')) for n in numbers if n.replace(',', '').isdigit()]

            holdings.append({
                "line": line.strip(),
                "context": context[:500],
                "numbers_found": numbers[:5] if numbers else [],
            })

    return holdings


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Searching SEC EDGAR for 13F filings containing JETS ETF...")
    log.info(f"JETS CUSIP: {JETS_CUSIP}")

    all_results = []

    # Search for JETS in 13F filings
    search_queries = [
        f'"{JETS_SYMBOL}"',
        f'"{JETS_CUSIP}"',
        '"U S Global Jets"',
        '"US Global Jets"',
    ]

    for query in search_queries:
        log.info(f"\nSearching for: {query}")
        time.sleep(REQUEST_DELAY)

        try:
            results = search_edgar_fulltext(query, form_types="13F-HR")
            hits = results.get("hits", {}).get("hits", [])
            total = results.get("hits", {}).get("total", {}).get("value", 0)

            log.info(f"  Found {total} filings")

            for hit in hits[:50]:  # Limit to first 50
                source = hit.get("_source", {})
                all_results.append({
                    "query": query,
                    "company": source.get("display_names", [""])[0] if source.get("display_names") else "",
                    "cik": source.get("ciks", [""])[0] if source.get("ciks") else "",
                    "form": source.get("form", ""),
                    "filed": source.get("file_date", ""),
                    "file_num": source.get("file_num", ""),
                    "accession": source.get("adsh", ""),
                })

        except Exception as e:
            log.warning(f"  Search failed: {e}")

    if all_results:
        df = pd.DataFrame(all_results)

        # Remove duplicates based on accession number
        df = df.drop_duplicates(subset=["accession"])
        df = df.sort_values("filed", ascending=False)

        df.to_csv(OUTPUT_CSV, index=False)
        log.info(f"\nSaved {len(df)} unique filings → {OUTPUT_CSV}")

        print("\n--- Institutions Filing 13F with JETS Holdings ---")
        print(df[["company", "cik", "filed", "form"]].head(30).to_string(index=False))

        # Summary stats
        print(f"\n--- Summary ---")
        print(f"Total unique institutions: {df['cik'].nunique()}")
        print(f"Date range: {df['filed'].min()} to {df['filed'].max()}")

        # Top filers
        top_filers = df.groupby("company").size().sort_values(ascending=False).head(10)
        print(f"\nTop institutions by filing frequency:")
        for company, count in top_filers.items():
            print(f"  {company}: {count} filings")

    else:
        log.warning("No results found")
        return 1

    log.info("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
