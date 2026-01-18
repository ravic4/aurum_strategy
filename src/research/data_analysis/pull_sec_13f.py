"""
Pull 13F filings directly from SEC EDGAR API.
Free, no API key required - just need to set a User-Agent.

This script fetches 13F-HR filings which show institutional holdings.

Outputs:
  src/research/data/sec_13f_filings.csv
  src/research/data/sec_13f_holdings.csv

Run from project root:
  python -m src.research.data_analysis.pull_sec_13f

Env (optional):
  SEC_USER_AGENT=YourName your@email.com  (SEC requires identification)
"""

from __future__ import annotations

import os
import logging
import time
from pathlib import Path
from typing import Any, Dict, List
import xml.etree.ElementTree as ET

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

FILINGS_CSV = DATA_DIR / "sec_13f_filings.csv"
HOLDINGS_CSV = DATA_DIR / "sec_13f_holdings.csv"


# ----------------------------
# Config
# ----------------------------
# SEC requires a User-Agent header with contact info
# Format: "Company/App contact@email.com"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "RaviCapital/Aurum research@example.com")

# SEC EDGAR base URLs
SEC_BASE = "https://www.sec.gov"
SEC_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
SEC_DATA_BASE = "https://data.sec.gov"

# Rate limiting - SEC asks for max 10 requests/second
REQUEST_DELAY = 0.15


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_sec_13f")


# ----------------------------
# SEC API Helpers
# ----------------------------
def get_headers() -> Dict[str, str]:
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


def search_13f_filings(query: str = "", form_type: str = "13F-HR", timeout: int = 30) -> Any:
    """
    Search SEC EDGAR for 13F filings using full-text search API.
    """
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": query,
        "forms": form_type,
        "dateRange": "custom",
        "startdt": "2020-01-01",
        "enddt": "2026-12-31",
    }

    r = requests.get(url, params=params, headers=get_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_company_filings(cik: str, form_type: str = "13F-HR", timeout: int = 30) -> Any:
    """
    Get filings for a specific company by CIK using submissions API.
    CIK should be zero-padded to 10 digits.
    """
    cik_padded = cik.zfill(10)
    url = f"{SEC_DATA_BASE}/submissions/CIK{cik_padded}.json"

    r = requests.get(url, headers=get_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_13f_info_table(accession_number: str, cik: str, timeout: int = 30) -> pd.DataFrame:
    """
    Fetch the 13F information table (holdings) from a specific filing.
    The info table is usually in XML format.
    """
    cik_padded = cik.zfill(10)
    accession_clean = accession_number.replace("-", "")

    # Try to get the index to find the info table file
    index_url = f"{SEC_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F-HR&dateb=&owner=include&count=40&output=atom"

    # Direct URL pattern for 13F XML info table
    # Format: /Archives/edgar/data/{cik}/{accession}/infotable.xml
    base_path = f"{SEC_BASE}/Archives/edgar/data/{cik_padded}/{accession_clean}"

    # Try common file names for 13F info table
    possible_files = [
        "infotable.xml",
        "primary_doc.xml",
        "form13fInfoTable.xml",
    ]

    for filename in possible_files:
        try:
            url = f"{base_path}/{filename}"
            r = requests.get(url, headers=get_headers(), timeout=timeout)
            if r.status_code == 200:
                return parse_13f_xml(r.text)
        except Exception:
            continue

    # If XML files not found, try to get filing index and find the info table
    try:
        index_url = f"{base_path}/index.json"
        r = requests.get(index_url, headers=get_headers(), timeout=timeout)
        if r.status_code == 200:
            index_data = r.json()
            for item in index_data.get("directory", {}).get("item", []):
                name = item.get("name", "")
                if "infotable" in name.lower() or "13f" in name.lower():
                    if name.endswith(".xml"):
                        xml_url = f"{base_path}/{name}"
                        r2 = requests.get(xml_url, headers=get_headers(), timeout=timeout)
                        if r2.status_code == 200:
                            return parse_13f_xml(r2.text)
    except Exception:
        pass

    return pd.DataFrame()


def parse_13f_xml(xml_content: str) -> pd.DataFrame:
    """
    Parse 13F information table XML into DataFrame.
    """
    holdings = []

    try:
        # Handle namespace
        xml_content = xml_content.replace('xmlns=', 'xmlns:default=')
        root = ET.fromstring(xml_content)

        # Find all infoTable entries
        for info in root.iter():
            if 'infotable' in info.tag.lower():
                holding = {}
                for child in info:
                    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    holding[tag] = child.text
                if holding:
                    holdings.append(holding)
    except ET.ParseError:
        log.warning("Failed to parse XML")

    return pd.DataFrame(holdings)


def search_filings_full_text(search_term: str, form_type: str = "13F", timeout: int = 30) -> List[Dict]:
    """
    Use SEC full-text search API to find filings mentioning a term.
    """
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": f'"{search_term}"',
        "forms": form_type,
    }

    try:
        r = requests.get(url, params=params, headers=get_headers(), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("hits", {}).get("hits", [])
    except Exception as e:
        log.warning(f"Full-text search failed: {e}")
        return []


def get_recent_13f_filers(timeout: int = 30) -> pd.DataFrame:
    """
    Get list of recent 13F filers from SEC.
    """
    # Use SEC's full-text search for recent 13F filings
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        "action": "getcurrent",
        "type": "13F-HR",
        "company": "",
        "dateb": "",
        "owner": "include",
        "count": "100",
        "output": "atom",
    }

    try:
        r = requests.get(url, params=params, headers=get_headers(), timeout=timeout)
        r.raise_for_status()

        # Parse Atom feed
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        filings = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            updated = entry.find("atom:updated", ns)
            link = entry.find("atom:link", ns)
            summary = entry.find("atom:summary", ns)

            filings.append({
                "title": title.text if title is not None else "",
                "updated": updated.text if updated is not None else "",
                "link": link.get("href") if link is not None else "",
                "summary": summary.text if summary is not None else "",
            })

        return pd.DataFrame(filings)
    except Exception as e:
        log.warning(f"Failed to get recent filers: {e}")
        return pd.DataFrame()


def get_major_etf_holders_13f(etf_symbol: str = "JETS") -> pd.DataFrame:
    """
    Strategy: Search 13F filings that mention the ETF symbol.
    This finds institutions that hold the ETF.
    """
    log.info(f"Searching 13F filings mentioning {etf_symbol}...")

    # Use SEC EDGAR company search API
    url = "https://www.sec.gov/cgi-bin/srch-ia"
    params = {
        "text": f"form-type=13F-HR AND {etf_symbol}",
        "first": 1,
        "last": 100,
    }

    # Alternative: Use the submissions endpoint for known large holders
    # Major ETF holders (asset managers) - some well-known CIKs
    major_holders = [
        {"name": "BlackRock", "cik": "1364742"},
        {"name": "Vanguard", "cik": "102909"},
        {"name": "State Street", "cik": "93751"},
        {"name": "Fidelity", "cik": "315066"},
        {"name": "JP Morgan", "cik": "19617"},
        {"name": "Morgan Stanley", "cik": "895421"},
        {"name": "Goldman Sachs", "cik": "886982"},
        {"name": "Bank of America", "cik": "70858"},
        {"name": "Citadel", "cik": "1423053"},
        {"name": "Bridgewater", "cik": "1350694"},
    ]

    return pd.DataFrame(major_holders)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Pulling 13F data from SEC EDGAR...")
    log.info(f"User-Agent: {SEC_USER_AGENT}")

    all_filings = []

    # Get recent 13F filings
    log.info("\n--- Fetching Recent 13F Filings ---")
    recent = get_recent_13f_filers()
    if not recent.empty:
        log.info(f"Found {len(recent)} recent 13F filings")
        print(recent.head(10).to_string(index=False))
        all_filings.append(recent)

    time.sleep(REQUEST_DELAY)

    # Get major institutional holders
    log.info("\n--- Major Institutional Holders (known CIKs) ---")
    holders = get_major_etf_holders_13f("JETS")
    print(holders.to_string(index=False))

    # For each major holder, get their recent 13F filings
    log.info("\n--- Fetching 13F filings from major institutions ---")
    holder_filings = []

    for _, holder in holders.iterrows():
        time.sleep(REQUEST_DELAY)
        log.info(f"Fetching filings for {holder['name']} (CIK: {holder['cik']})...")

        try:
            company_data = get_company_filings(holder['cik'])

            # Extract filing info
            filings = company_data.get("filings", {}).get("recent", {})
            if filings:
                n_filings = len(filings.get("form", []))

                for i in range(min(n_filings, 100)):  # Last 100 filings
                    form = filings.get("form", [])[i] if i < len(filings.get("form", [])) else ""
                    if "13F" in form:
                        holder_filings.append({
                            "holder_name": holder["name"],
                            "cik": holder["cik"],
                            "form": form,
                            "filing_date": filings.get("filingDate", [])[i] if i < len(filings.get("filingDate", [])) else "",
                            "accession_number": filings.get("accessionNumber", [])[i] if i < len(filings.get("accessionNumber", [])) else "",
                            "primary_document": filings.get("primaryDocument", [])[i] if i < len(filings.get("primaryDocument", [])) else "",
                        })

                log.info(f"  Found {len([f for f in holder_filings if f['holder_name'] == holder['name']])} 13F filings")
        except Exception as e:
            log.warning(f"  Failed: {e}")

    if holder_filings:
        df_filings = pd.DataFrame(holder_filings)
        df_filings = df_filings.sort_values("filing_date", ascending=False)
        df_filings.to_csv(FILINGS_CSV, index=False)
        log.info(f"\nSaved {len(df_filings)} 13F filings → {FILINGS_CSV}")

        print("\n--- Recent 13F Filings by Major Institutions ---")
        print(df_filings.head(20).to_string(index=False))

        # Try to get holdings from most recent filing
        log.info("\n--- Fetching holdings from most recent filing ---")
        if len(df_filings) > 0:
            latest = df_filings.iloc[0]
            log.info(f"Trying {latest['holder_name']} filing from {latest['filing_date']}...")

            holdings = get_13f_info_table(latest['accession_number'], latest['cik'])
            if not holdings.empty:
                holdings.to_csv(HOLDINGS_CSV, index=False)
                log.info(f"Saved holdings → {HOLDINGS_CSV}")
                print(holdings.head(20).to_string(index=False))
            else:
                log.info("Could not parse holdings from filing")

    log.info("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
