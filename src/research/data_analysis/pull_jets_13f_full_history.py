"""
Pull ALL JETS ETF filings from SEC EDGAR since 2017 (ETF inception was April 2015).
Searches for N-PORT, N-CSR, N-CSRS, and other fund filings.

Outputs to: src/research/data/13F Filings/
  - jets_holdings_YYYY_QX.csv (per quarter)
  - jets_holdings_all.csv (combined)
  - jets_filings_index.csv (filing metadata)

Run from project root:
  python -m src.research.data_analysis.pull_jets_13f_full_history
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

REQUEST_DELAY = 0.12  # SEC rate limit


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_jets_full_history")


def get_headers() -> Dict[str, str]:
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


def get_all_filings_with_older(cik: str, timeout: int = 30) -> List[Dict]:
    """Get all filings including older ones from archive files."""
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    r = requests.get(url, headers=get_headers(), timeout=timeout)
    r.raise_for_status()
    data = r.json()

    all_filings = []

    # Get recent filings
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    for i in range(len(forms)):
        all_filings.append({
            "form": forms[i] if i < len(forms) else "",
            "filing_date": dates[i] if i < len(dates) else "",
            "accession": accessions[i] if i < len(accessions) else "",
        })

    # Get older filings from archive files
    older_files = data.get("filings", {}).get("files", [])
    for file_info in older_files:
        filename = file_info.get("name", "")
        if filename:
            time.sleep(REQUEST_DELAY)
            archive_url = f"https://data.sec.gov/submissions/{filename}"
            try:
                r2 = requests.get(archive_url, headers=get_headers(), timeout=timeout)
                if r2.status_code == 200:
                    archive_data = r2.json()
                    forms = archive_data.get("form", [])
                    dates = archive_data.get("filingDate", [])
                    accessions = archive_data.get("accessionNumber", [])

                    for i in range(len(forms)):
                        all_filings.append({
                            "form": forms[i] if i < len(forms) else "",
                            "filing_date": dates[i] if i < len(dates) else "",
                            "accession": accessions[i] if i < len(accessions) else "",
                        })
                    log.info(f"  Loaded {len(forms)} filings from {filename}")
            except Exception as e:
                log.warning(f"  Failed to load {filename}: {e}")

    return all_filings


def fetch_filing_index(cik: str, accession: str, timeout: int = 60) -> Dict:
    """Fetch the filing index to find all documents."""
    cik_padded = cik.zfill(10)
    accession_clean = accession.replace("-", "")

    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession_clean}/index.json"

    try:
        r = requests.get(index_url, headers=get_headers(), timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    return {}


def fetch_filing_content(cik: str, accession: str, filename: str, timeout: int = 60) -> str:
    """Fetch a specific filing document."""
    cik_padded = cik.zfill(10)
    accession_clean = accession.replace("-", "")

    url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession_clean}/{filename}"

    try:
        r = requests.get(url, headers=get_headers(), timeout=timeout)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass

    return ""


def is_jets_filing(content: str) -> bool:
    """Check if filing content is for JETS ETF."""
    jets_indicators = [
        "U.S. Global Jets",
        "US Global Jets",
        "U.S. GLOBAL JETS",
        "US GLOBAL JETS",
        "S000048544",  # JETS series ID
    ]
    content_upper = content.upper()
    return any(ind.upper() in content_upper for ind in jets_indicators)


def parse_nport_holdings(xml_content: str) -> pd.DataFrame:
    """Parse N-PORT XML to extract holdings using regex."""
    holdings = []

    # Extract report period
    rep_pd_match = re.search(r'<repPdDate>(.*?)</repPdDate>', xml_content)
    report_date = rep_pd_match.group(1) if rep_pd_match else ""

    # Find all invstOrSec blocks
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


def parse_ncsr_holdings(html_content: str, filing_date: str) -> pd.DataFrame:
    """Parse N-CSR/N-CSRS filings (semi-annual/annual reports) for holdings."""
    holdings = []

    # These are HTML/text documents with schedule of investments
    # Look for tables with holdings data

    # Try to find JETS section
    jets_section = None
    jets_markers = ["U.S. Global Jets", "US Global Jets", "JETS ETF"]

    for marker in jets_markers:
        if marker in html_content:
            # Find the section
            start_idx = html_content.find(marker)
            if start_idx != -1:
                # Look for the schedule of investments table
                jets_section = html_content[start_idx:start_idx + 50000]
                break

    if not jets_section:
        return pd.DataFrame()

    # Extract holdings from table rows
    # Look for patterns like: Company Name | shares | value
    # This is complex because HTML structure varies

    # Try regex patterns for common table formats
    patterns = [
        r'<td[^>]*>([A-Za-z][^<]+)</td>\s*<td[^>]*>([^<]*)</td>\s*<td[^>]*>\$?([\d,]+)',
        r'>([A-Z][A-Za-z\s&,\.]+(?:Inc|Corp|Co|Ltd|LLC|PLC)?)[^<]*</.*?>([\d,]+)\s*(?:shares?)?\s*\$?([\d,]+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, jets_section, re.IGNORECASE | re.DOTALL)
        if matches:
            for match in matches:
                name = match[0].strip()
                if len(name) > 3 and not name.startswith('<'):
                    shares_str = match[1].replace(',', '').strip()
                    value_str = match[2].replace(',', '').strip()

                    try:
                        shares = float(shares_str) if shares_str else 0
                        value = float(value_str) if value_str else 0

                        if value > 1000:  # Filter noise
                            holdings.append({
                                'name': name,
                                'ticker': '',
                                'cusip': '',
                                'shares': shares,
                                'value_usd': value,
                                'pct_portfolio': 0,
                                'country': '',
                                'report_date': filing_date,
                            })
                    except ValueError:
                        continue

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

    log.info("Fetching ALL JETS ETF filings from SEC EDGAR since 2017...")
    log.info(f"Output directory: {OUTPUT_DIR}")

    # Get all filings including archived ones
    log.info(f"\nFetching all filings for CIK {JETS_ISSUER_CIK}...")
    all_filings = get_all_filings_with_older(JETS_ISSUER_CIK)

    log.info(f"Total filings found: {len(all_filings)}")

    # Filter to relevant form types and dates >= 2017
    relevant_forms = ["NPORT", "N-PORT", "N-CSR", "N-CSRS", "N-Q", "485"]
    filtered_filings = []

    for f in all_filings:
        form = f.get("form", "")
        date = f.get("filing_date", "")

        if date >= "2017-01-01":
            if any(rf in form.upper() for rf in ["NPORT", "N-PORT", "N-CSR", "N-Q"]):
                filtered_filings.append(f)

    log.info(f"Relevant filings since 2017: {len(filtered_filings)}")

    # Sort by date
    filtered_filings.sort(key=lambda x: x.get("filing_date", ""), reverse=True)

    # Process each filing
    all_holdings = []
    filing_index = []
    processed_quarters = set()

    for i, filing in enumerate(filtered_filings):
        time.sleep(REQUEST_DELAY)

        accession = filing["accession"]
        filing_date = filing["filing_date"]
        form_type = filing["form"]

        log.info(f"[{i+1}/{len(filtered_filings)}] {form_type} from {filing_date} ({accession})...")

        # Get filing index
        index_data = fetch_filing_index(JETS_ISSUER_CIK, accession)
        if not index_data:
            continue

        items = index_data.get("directory", {}).get("item", [])

        # Find the main document
        holdings_df = pd.DataFrame()
        report_date = filing_date

        for item in items:
            filename = item.get("name", "")

            # Check for JETS in filename
            if "jets" in filename.lower():
                log.info(f"  Found JETS file: {filename}")

            # Try XML files first (N-PORT)
            if filename == "primary_doc.xml" or filename.endswith(".xml"):
                time.sleep(REQUEST_DELAY)
                content = fetch_filing_content(JETS_ISSUER_CIK, accession, filename)

                if content and is_jets_filing(content):
                    holdings_df = parse_nport_holdings(content)
                    if not holdings_df.empty:
                        report_date = holdings_df["report_date"].iloc[0] if "report_date" in holdings_df.columns else filing_date
                        log.info(f"  JETS N-PORT: {len(holdings_df)} holdings for {report_date}")
                        break

            # Try HTML files for N-CSR
            elif filename.endswith(".htm") or filename.endswith(".html"):
                if "jets" in filename.lower():
                    time.sleep(REQUEST_DELAY)
                    content = fetch_filing_content(JETS_ISSUER_CIK, accession, filename)

                    if content and is_jets_filing(content):
                        holdings_df = parse_ncsr_holdings(content, filing_date)
                        if not holdings_df.empty:
                            log.info(f"  JETS N-CSR: {len(holdings_df)} holdings")
                            break

        if holdings_df.empty:
            continue

        # Get quarter
        quarter = get_quarter_from_date(report_date if report_date else filing_date)

        if not quarter:
            continue

        # Skip if we already have this quarter
        if quarter in processed_quarters:
            log.info(f"  Skipping {quarter} (already have)")
            continue

        processed_quarters.add(quarter)

        # Add metadata
        holdings_df["filing_date"] = filing_date
        holdings_df["accession"] = accession
        holdings_df["quarter"] = quarter
        holdings_df["form_type"] = form_type

        # Save individual quarter file
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
            "form_type": form_type,
            "accession": accession,
            "num_holdings": len(holdings_df),
            "total_value": holdings_df["value_usd"].sum() if "value_usd" in holdings_df.columns else 0,
        })

    log.info(f"\nProcessed {len(filing_index)} unique JETS quarterly filings")

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

        print("\n--- JETS Filings Index ---")
        print(index_df.to_string(index=False))

        # Show quarters covered
        quarters = sorted(index_df["quarter"].unique())
        print(f"\nQuarters covered ({len(quarters)}): {quarters}")
    else:
        log.warning("No JETS holdings found")
        return 1

    log.info("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
