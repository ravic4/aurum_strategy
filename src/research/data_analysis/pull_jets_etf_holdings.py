"""
Find JETS ETF holdings from SEC filings.

JETS ETF is managed by U.S. Global Investors, Inc.
ETFs file N-PORT (quarterly holdings) and N-CEN filings with SEC.

Outputs:
  src/research/data/jets_etf_holdings.csv

Run from project root:
  python -m src.research.data_analysis.pull_jets_etf_holdings
"""

from __future__ import annotations

import os
import logging
import time
import re
import xml.etree.ElementTree as ET
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

OUTPUT_CSV = DATA_DIR / "jets_etf_holdings.csv"
FILINGS_CSV = DATA_DIR / "jets_etf_filings.csv"


# ----------------------------
# Config
# ----------------------------
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "RaviCapital/Aurum research@example.com")

# U.S. Global Investors (JETS ETF issuer)
# CIK for U.S. Global Jets ETF trust
JETS_ISSUER_CIK = "0001592900"  # U.S. Global Jets ETF

REQUEST_DELAY = 0.12


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_jets_holdings")


def get_headers() -> Dict[str, str]:
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


def get_company_filings(cik: str, timeout: int = 30) -> Dict:
    """Get all filings for a CIK."""
    cik_padded = cik.lstrip("0").zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    r = requests.get(url, headers=get_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def search_for_jets_etf_cik() -> str:
    """Search SEC for JETS ETF issuer CIK."""
    log.info("Searching for JETS ETF issuer CIK...")

    # Try company search
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        "company": "U.S. Global Jets",
        "CIK": "",
        "type": "",
        "owner": "include",
        "count": "40",
        "action": "getcompany",
        "output": "atom",
    }

    try:
        r = requests.get(url, params=params, headers=get_headers(), timeout=30)
        r.raise_for_status()

        # Parse response
        if "CIK" in r.text:
            # Extract CIK from response
            matches = re.findall(r'CIK=(\d+)', r.text)
            if matches:
                return matches[0]
    except Exception as e:
        log.warning(f"Company search failed: {e}")

    return ""


def get_nport_filings(cik: str) -> List[Dict]:
    """Get N-PORT filings (quarterly holdings reports) for an ETF."""
    filings_data = get_company_filings(cik)

    filings = filings_data.get("filings", {}).get("recent", {})
    nport_filings = []

    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    for i in range(len(forms)):
        form = forms[i] if i < len(forms) else ""
        if "NPORT" in form or "N-PORT" in form:
            nport_filings.append({
                "form": form,
                "filing_date": dates[i] if i < len(dates) else "",
                "accession": accessions[i] if i < len(accessions) else "",
                "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
            })

    return nport_filings


def fetch_nport_holdings(cik: str, accession: str, timeout: int = 60) -> pd.DataFrame:
    """Fetch and parse N-PORT XML for holdings."""
    cik_padded = cik.lstrip("0").zfill(10)
    accession_clean = accession.replace("-", "")

    # N-PORT filings have primary_doc.xml
    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession_clean}"

    # Try to get filing index first
    index_url = f"{base_url}/index.json"

    try:
        r = requests.get(index_url, headers=get_headers(), timeout=timeout)
        if r.status_code == 200:
            index = r.json()
            items = index.get("directory", {}).get("item", [])

            # Find the XML file with holdings
            for item in items:
                name = item.get("name", "")
                if name.endswith(".xml") and "primary" not in name.lower():
                    xml_url = f"{base_url}/{name}"
                    log.info(f"  Fetching: {name}")

                    r2 = requests.get(xml_url, headers=get_headers(), timeout=timeout)
                    if r2.status_code == 200:
                        return parse_nport_xml(r2.text)
    except Exception as e:
        log.warning(f"  Failed to fetch N-PORT: {e}")

    return pd.DataFrame()


def parse_nport_xml(xml_content: str) -> pd.DataFrame:
    """Parse N-PORT XML to extract holdings."""
    holdings = []

    try:
        # Remove namespace for easier parsing
        xml_content = re.sub(r'xmlns[^"]*"[^"]*"', '', xml_content)
        root = ET.fromstring(xml_content)

        # Find all investment holdings
        for inv in root.iter():
            if 'invst' in inv.tag.lower() or 'holding' in inv.tag.lower():
                holding = {}
                for child in inv:
                    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if child.text and child.text.strip():
                        holding[tag] = child.text.strip()
                if holding and len(holding) > 1:
                    holdings.append(holding)

        # Also try to find securities directly
        for sec in root.iter():
            tag = sec.tag.split('}')[-1] if '}' in sec.tag else sec.tag
            if tag in ['name', 'title', 'cusip', 'isin', 'ticker']:
                # Get parent element's data
                pass

    except ET.ParseError as e:
        log.warning(f"XML parse error: {e}")

    return pd.DataFrame(holdings)


def search_edgar_for_jets_issuer():
    """Search EDGAR for the JETS ETF trust/issuer."""
    log.info("Searching EDGAR for JETS ETF trust...")

    searches = [
        "U.S. Global Jets",
        "US Global Jets ETF",
        "JETS ETF",
    ]

    results = []

    for search in searches:
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": f'"{search}"',
            "forms": "N-PORT,NPORT-P,N-CEN",
        }

        try:
            time.sleep(REQUEST_DELAY)
            r = requests.get(url, params=params, headers=get_headers(), timeout=30)
            if r.status_code == 200:
                data = r.json()
                hits = data.get("hits", {}).get("hits", [])
                log.info(f"  '{search}': {len(hits)} results")

                for hit in hits[:20]:
                    source = hit.get("_source", {})
                    results.append({
                        "search": search,
                        "company": source.get("display_names", [""])[0] if source.get("display_names") else "",
                        "cik": source.get("ciks", [""])[0] if source.get("ciks") else "",
                        "form": source.get("form", ""),
                        "filed": source.get("file_date", ""),
                        "accession": source.get("adsh", ""),
                    })
        except Exception as e:
            log.warning(f"  Search failed: {e}")

    return pd.DataFrame(results).drop_duplicates(subset=["accession"])


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Finding JETS ETF holdings from SEC filings...")

    # Search for JETS ETF filings
    filings_df = search_edgar_for_jets_issuer()

    if filings_df.empty:
        log.warning("No filings found via search, trying direct CIK lookup...")

        # Try known CIKs for U.S. Global Investors funds
        possible_ciks = [
            "1592900",  # U.S. Global Jets ETF
            "0001592900",
            "1654672",  # US Global Investors Funds
            "0001654672",
        ]

        for cik in possible_ciks:
            log.info(f"Trying CIK: {cik}")
            try:
                time.sleep(REQUEST_DELAY)
                data = get_company_filings(cik)
                name = data.get("name", "")
                log.info(f"  Found: {name}")

                if "jet" in name.lower() or "global" in name.lower():
                    nport = get_nport_filings(cik)
                    if nport:
                        log.info(f"  Found {len(nport)} N-PORT filings")
                        filings_df = pd.DataFrame(nport)
                        filings_df["cik"] = cik
                        filings_df["company"] = name
                        break
            except Exception as e:
                log.warning(f"  Failed: {e}")

    if not filings_df.empty:
        filings_df.to_csv(FILINGS_CSV, index=False)
        log.info(f"\nSaved filings list → {FILINGS_CSV}")

        print("\n--- JETS ETF Related Filings ---")
        print(filings_df.head(20).to_string(index=False))

        # Try to get holdings from most recent N-PORT
        nport_filings = filings_df[filings_df["form"].str.contains("PORT", case=False, na=False)]

        if not nport_filings.empty:
            latest = nport_filings.iloc[0]
            cik = str(latest.get("cik", "")).lstrip("0")
            accession = latest.get("accession", "")

            if cik and accession:
                log.info(f"\nFetching holdings from latest N-PORT filing...")
                log.info(f"  CIK: {cik}, Accession: {accession}")

                holdings = fetch_nport_holdings(cik, accession)

                if not holdings.empty:
                    holdings.to_csv(OUTPUT_CSV, index=False)
                    log.info(f"Saved holdings → {OUTPUT_CSV}")
                    print("\n--- JETS ETF Holdings ---")
                    print(holdings.head(30).to_string(index=False))
                else:
                    log.info("Could not parse holdings from filing")
    else:
        log.warning("No JETS ETF filings found")
        return 1

    log.info("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
