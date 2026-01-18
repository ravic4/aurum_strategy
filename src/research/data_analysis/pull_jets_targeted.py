"""
Targeted search for JETS ETF filings using SEC full-text search.
More efficient than scanning all ETF Series Solutions filings.

Outputs to: src/research/data/13F Filings/

Run from project root:
  python -m src.research.data_analysis.pull_jets_targeted
"""

from __future__ import annotations

import os
import logging
import time
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime

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
OUTPUT_DIR = DATA_DIR / "13F Filings"


# ----------------------------
# Config
# ----------------------------
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "RaviCapital/Aurum research@example.com")
REQUEST_DELAY = 0.15


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_jets_targeted")


def get_headers() -> Dict[str, str]:
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


def search_sec_fulltext(query: str, forms: str, start_date: str = "2015-01-01", timeout: int = 30) -> List[Dict]:
    """Use SEC full-text search API."""
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": query,
        "forms": forms,
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": "2026-12-31",
    }

    results = []
    try:
        r = requests.get(url, params=params, headers=get_headers(), timeout=timeout)
        r.raise_for_status()
        data = r.json()

        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {}).get("value", 0)

        log.info(f"  Found {total} results")

        for hit in hits:
            source = hit.get("_source", {})
            results.append({
                "company": source.get("display_names", [""])[0] if source.get("display_names") else "",
                "cik": source.get("ciks", [""])[0] if source.get("ciks") else "",
                "form": source.get("form", ""),
                "file_date": source.get("file_date", ""),
                "accession": source.get("adsh", ""),
            })
    except Exception as e:
        log.warning(f"Search failed: {e}")

    return results


def fetch_filing_content(cik: str, accession: str, filename: str, timeout: int = 60) -> str:
    """Fetch a specific filing document."""
    cik_padded = cik.lstrip("0").zfill(10)
    accession_clean = accession.replace("-", "")

    url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession_clean}/{filename}"

    try:
        r = requests.get(url, headers=get_headers(), timeout=timeout)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass

    return ""


def get_filing_index(cik: str, accession: str, timeout: int = 30) -> List[Dict]:
    """Get list of files in a filing."""
    cik_padded = cik.lstrip("0").zfill(10)
    accession_clean = accession.replace("-", "")

    url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession_clean}/index.json"

    try:
        r = requests.get(url, headers=get_headers(), timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            return data.get("directory", {}).get("item", [])
    except Exception:
        pass

    return []


def parse_nport_holdings(xml_content: str) -> pd.DataFrame:
    """Parse N-PORT XML."""
    holdings = []

    rep_pd_match = re.search(r'<repPdDate>(.*?)</repPdDate>', xml_content)
    report_date = rep_pd_match.group(1) if rep_pd_match else ""

    pattern = r'<invstOrSec>(.*?)</invstOrSec>'
    matches = re.findall(pattern, xml_content, re.DOTALL)

    for match in matches:
        holding = {}

        name = re.search(r'<name>(.*?)</name>', match)
        ticker = re.search(r'<ticker value="(.*?)"', match)
        cusip = re.search(r'<cusip>(.*?)</cusip>', match)
        balance = re.search(r'<balance>(.*?)</balance>', match)
        valUSD = re.search(r'<valUSD>(.*?)</valUSD>', match)
        pctVal = re.search(r'<pctVal>(.*?)</pctVal>', match)
        invCountry = re.search(r'<invCountry>(.*?)</invCountry>', match)

        if name:
            holding['name'] = name.group(1)
            holding['ticker'] = ticker.group(1) if ticker else ''
            holding['cusip'] = cusip.group(1) if cusip else ''
            holding['shares'] = float(balance.group(1)) if balance else 0
            holding['value_usd'] = float(valUSD.group(1)) if valUSD else 0
            holding['pct_portfolio'] = float(pctVal.group(1)) if pctVal else 0
            holding['country'] = invCountry.group(1) if invCountry else ''
            holding['report_date'] = report_date
            holdings.append(holding)

    return pd.DataFrame(holdings)


def get_quarter_from_date(date_str: str) -> str:
    """Convert date to quarter string."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}_Q{quarter}"
    except:
        return ""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Searching SEC EDGAR for JETS ETF filings since 2017...")

    # Search for JETS ETF filings
    search_terms = [
        ('"U.S. Global Jets ETF"', "NPORT-P,N-CSR,N-CSRS,N-Q"),
        ('"US Global Jets"', "NPORT-P"),
    ]

    all_filings = []

    for query, forms in search_terms:
        log.info(f"\nSearching: {query} in {forms}")
        results = search_sec_fulltext(query, forms, start_date="2017-01-01")
        all_filings.extend(results)
        time.sleep(REQUEST_DELAY)

    # Deduplicate
    seen = set()
    unique_filings = []
    for f in all_filings:
        key = f.get("accession", "")
        if key and key not in seen:
            seen.add(key)
            unique_filings.append(f)

    log.info(f"\nTotal unique filings: {len(unique_filings)}")

    # Sort by date
    unique_filings.sort(key=lambda x: x.get("file_date", ""), reverse=True)

    # Process each filing
    all_holdings = []
    filing_index = []
    processed_quarters = set()

    for i, filing in enumerate(unique_filings):
        time.sleep(REQUEST_DELAY)

        cik = filing.get("cik", "").lstrip("0")
        accession = filing.get("accession", "")
        file_date = filing.get("file_date", "")
        form = filing.get("form", "")

        log.info(f"[{i+1}/{len(unique_filings)}] {form} from {file_date}...")

        if not cik or not accession:
            continue

        # Get filing files
        items = get_filing_index(cik, accession)
        if not items:
            continue

        # Find primary_doc.xml or jets*.htm
        holdings_df = pd.DataFrame()
        report_date = file_date

        for item in items:
            filename = item.get("name", "")

            # N-PORT XML
            if filename == "primary_doc.xml":
                time.sleep(REQUEST_DELAY)
                content = fetch_filing_content(cik, accession, filename)

                if content and "U.S. Global Jets" in content:
                    holdings_df = parse_nport_holdings(content)
                    if not holdings_df.empty:
                        report_date = holdings_df["report_date"].iloc[0] if "report_date" in holdings_df.columns and holdings_df["report_date"].iloc[0] else file_date
                        log.info(f"  Found {len(holdings_df)} holdings for {report_date}")
                        break

        if holdings_df.empty:
            continue

        quarter = get_quarter_from_date(report_date)
        if not quarter:
            continue

        if quarter in processed_quarters:
            log.info(f"  Skipping {quarter} (already have)")
            continue

        processed_quarters.add(quarter)

        # Add metadata
        holdings_df["filing_date"] = file_date
        holdings_df["accession"] = accession
        holdings_df["quarter"] = quarter
        holdings_df["form_type"] = form

        # Save
        quarter_file = OUTPUT_DIR / f"jets_holdings_{quarter}.csv"
        holdings_df.to_csv(quarter_file, index=False)
        log.info(f"  Saved: {quarter_file.name}")

        all_holdings.append(holdings_df)
        filing_index.append({
            "filing_date": file_date,
            "report_date": report_date,
            "quarter": quarter,
            "form_type": form,
            "accession": accession,
            "num_holdings": len(holdings_df),
            "total_value": holdings_df["value_usd"].sum() if "value_usd" in holdings_df.columns else 0,
        })

    # Save combined
    if all_holdings:
        combined_df = pd.concat(all_holdings, ignore_index=True)
        combined_df.to_csv(OUTPUT_DIR / "jets_holdings_all.csv", index=False)

        index_df = pd.DataFrame(filing_index)
        index_df = index_df.sort_values("report_date")
        index_df.to_csv(OUTPUT_DIR / "jets_filings_index.csv", index=False)

        print("\n--- JETS Filings Index ---")
        print(index_df.to_string(index=False))

        quarters = sorted(index_df["quarter"].unique())
        print(f"\nQuarters covered ({len(quarters)}): {quarters}")

        log.info(f"\nTotal rows: {len(combined_df)}")
        log.info(f"Quarters: {len(quarters)}")

    log.info("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
