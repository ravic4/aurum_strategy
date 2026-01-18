"""
Pull all JETS ETF N-PORT filings (quarterly holdings) from 2020 to present.
Downloads and parses holdings from each filing.

Outputs to: src/research/data/13F Filings/
  - jets_holdings_YYYY_QX.csv (per quarter)
  - jets_holdings_all.csv (combined)
  - jets_filings_index.csv (filing metadata)

Run from project root:
  python -m src.research.data_analysis.pull_jets_13f_history
"""

from __future__ import annotations

import os
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Any
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

# ETF Series Solutions (JETS issuer) CIK
JETS_ISSUER_CIK = "1540305"

# JETS series ID for filtering
JETS_SERIES_ID = "S000048544"

REQUEST_DELAY = 0.12  # SEC rate limit


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_jets_13f_history")


def get_headers() -> Dict[str, str]:
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


def get_all_filings(cik: str, timeout: int = 30) -> Dict:
    """Get all filings for a CIK."""
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    r = requests.get(url, headers=get_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_filing_xml(cik: str, accession: str, timeout: int = 60) -> str:
    """Fetch the N-PORT XML filing."""
    cik_padded = cik.zfill(10)
    accession_clean = accession.replace("-", "")

    # Get filing index to find XML file
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession_clean}/index.json"

    try:
        r = requests.get(index_url, headers=get_headers(), timeout=timeout)
        if r.status_code == 200:
            index = r.json()
            items = index.get("directory", {}).get("item", [])

            # Find primary_doc.xml
            for item in items:
                name = item.get("name", "")
                if name == "primary_doc.xml":
                    xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession_clean}/{name}"
                    r2 = requests.get(xml_url, headers=get_headers(), timeout=timeout)
                    if r2.status_code == 200:
                        return r2.text
    except Exception as e:
        log.warning(f"Failed to fetch XML: {e}")

    return ""


def parse_nport_holdings(xml_content: str) -> pd.DataFrame:
    """Parse N-PORT XML to extract holdings using regex."""
    holdings = []

    # Check if this is JETS (look for series ID or name)
    if "U.S. Global Jets" not in xml_content and JETS_SERIES_ID not in xml_content:
        return pd.DataFrame()

    # Extract report period
    rep_pd_match = re.search(r'<repPdDate>(.*?)</repPdDate>', xml_content)
    report_date = rep_pd_match.group(1) if rep_pd_match else ""

    # Find all invstOrSec blocks
    pattern = r'<invstOrSec>(.*?)</invstOrSec>'
    matches = re.findall(pattern, xml_content, re.DOTALL)

    for match in matches:
        holding = {}

        # Extract fields
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
    """Convert date to quarter string (e.g., '2024_Q3')."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}_Q{quarter}"
    except:
        return ""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Fetching all JETS ETF N-PORT filings from 2020 to present...")
    log.info(f"Output directory: {OUTPUT_DIR}")

    # Get all filings for ETF Series Solutions
    log.info(f"\nFetching filings for CIK {JETS_ISSUER_CIK}...")
    filings_data = get_all_filings(JETS_ISSUER_CIK)

    company_name = filings_data.get("name", "")
    log.info(f"Company: {company_name}")

    # Extract filing info
    recent = filings_data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    # Filter to N-PORT filings from 2020+
    nport_filings = []
    for i in range(len(forms)):
        form = forms[i] if i < len(forms) else ""
        date = dates[i] if i < len(dates) else ""
        accession = accessions[i] if i < len(accessions) else ""

        if "NPORT" in form and date >= "2020-01-01":
            nport_filings.append({
                "form": form,
                "filing_date": date,
                "accession": accession,
            })

    log.info(f"Found {len(nport_filings)} N-PORT filings since 2020")

    # Process each filing
    all_holdings = []
    filing_index = []
    jets_count = 0

    for i, filing in enumerate(nport_filings):
        time.sleep(REQUEST_DELAY)

        accession = filing["accession"]
        filing_date = filing["filing_date"]

        log.info(f"[{i+1}/{len(nport_filings)}] Processing {filing_date} ({accession})...")

        # Fetch XML
        xml_content = fetch_filing_xml(JETS_ISSUER_CIK, accession)

        if not xml_content:
            log.info("  Could not fetch XML")
            continue

        # Check if this is JETS
        if "U.S. Global Jets" not in xml_content:
            continue

        jets_count += 1

        # Parse holdings
        holdings_df = parse_nport_holdings(xml_content)

        if holdings_df.empty:
            log.info("  No holdings parsed")
            continue

        # Get report date and quarter
        report_date = holdings_df["report_date"].iloc[0] if "report_date" in holdings_df.columns else filing_date
        quarter = get_quarter_from_date(report_date)

        log.info(f"  JETS filing for {report_date} ({quarter}): {len(holdings_df)} holdings")

        # Add metadata
        holdings_df["filing_date"] = filing_date
        holdings_df["accession"] = accession
        holdings_df["quarter"] = quarter

        # Save individual quarter file
        if quarter:
            quarter_file = OUTPUT_DIR / f"jets_holdings_{quarter}.csv"
            holdings_df.to_csv(quarter_file, index=False)
            log.info(f"  Saved: {quarter_file.name}")

        # Add to combined
        all_holdings.append(holdings_df)

        # Track filing
        filing_index.append({
            "filing_date": filing_date,
            "report_date": report_date,
            "quarter": quarter,
            "accession": accession,
            "num_holdings": len(holdings_df),
            "total_value": holdings_df["value_usd"].sum() if "value_usd" in holdings_df.columns else 0,
        })

    log.info(f"\nProcessed {jets_count} JETS filings out of {len(nport_filings)} total N-PORT filings")

    # Save combined holdings
    if all_holdings:
        combined_df = pd.concat(all_holdings, ignore_index=True)
        combined_file = OUTPUT_DIR / "jets_holdings_all.csv"
        combined_df.to_csv(combined_file, index=False)
        log.info(f"\nSaved combined holdings: {combined_file}")
        log.info(f"  Total rows: {len(combined_df)}")

        # Save filing index
        index_df = pd.DataFrame(filing_index)
        index_df = index_df.sort_values("report_date", ascending=False)
        index_file = OUTPUT_DIR / "jets_filings_index.csv"
        index_df.to_csv(index_file, index=False)
        log.info(f"Saved filing index: {index_file}")

        print("\n--- JETS N-PORT Filings Index ---")
        print(index_df.to_string(index=False))

        # Show quarters covered
        quarters = index_df["quarter"].unique()
        print(f"\nQuarters covered: {sorted(quarters)}")
    else:
        log.warning("No JETS holdings found")
        return 1

    log.info("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
