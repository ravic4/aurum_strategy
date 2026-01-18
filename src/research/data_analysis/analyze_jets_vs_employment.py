"""
Analyze JETS ETF quarterly holdings against airline employment and stock price data.

Compares:
1. JETS ETF total value/AUM vs aggregate employment changes
2. Individual holding weights vs company employment changes
3. JETS ETF returns vs employment changes (lagged analysis)

Outputs to: src/research/data/jets_employment_analysis/

Run from project root:
  python -m src.research.data_analysis.analyze_jets_vs_employment
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ----------------------------
# Paths
# ----------------------------
RESEARCH_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = RESEARCH_DIR / "data"
JETS_DIR = DATA_DIR / "13F Filings"
OUTPUT_DIR = DATA_DIR / "jets_employment_analysis"

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analyze_jets_employment")


def load_jets_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load JETS ETF filings index and combined holdings."""
    index_df = pd.read_csv(JETS_DIR / "jets_filings_index.csv")
    holdings_df = pd.read_csv(JETS_DIR / "jets_holdings_all.csv")

    # Parse dates
    index_df["report_date"] = pd.to_datetime(index_df["report_date"])
    index_df["filing_date"] = pd.to_datetime(index_df["filing_date"])
    holdings_df["report_date"] = pd.to_datetime(holdings_df["report_date"])

    return index_df, holdings_df


def load_employment_data() -> pd.DataFrame:
    """Load merged prices and employment data."""
    df = pd.read_excel(DATA_DIR / "prices_employment_merged.xlsx")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_jets_price_data() -> pd.DataFrame:
    """Load JETS ETF price data."""
    df = pd.read_csv(DATA_DIR / "jets_etf_monthly.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_quarter_from_date(dt: datetime) -> str:
    """Convert date to quarter string."""
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}_Q{quarter}"


def aggregate_employment_by_quarter(emp_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate employment data to quarterly level."""
    emp_df = emp_df.copy()
    emp_df["quarter"] = emp_df["date"].apply(lambda x: get_quarter_from_date(x))

    # Get last month of each quarter for each company
    quarterly = emp_df.groupby(["symbol", "quarter"]).agg({
        "date": "max",
        "close": "last",
        "total_employees": "last",
        "full_time_employees": "last",
        "part_time_employees": "last",
        "source_name": "first",
    }).reset_index()

    # Calculate quarter-over-quarter changes
    quarterly = quarterly.sort_values(["symbol", "quarter"])
    quarterly["emp_change"] = quarterly.groupby("symbol")["total_employees"].diff()
    quarterly["emp_pct_change"] = quarterly.groupby("symbol")["total_employees"].pct_change() * 100
    quarterly["price_return"] = quarterly.groupby("symbol")["close"].pct_change() * 100

    return quarterly


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")

    # Load all data
    jets_index, jets_holdings = load_jets_data()
    emp_data = load_employment_data()
    jets_prices = load_jets_price_data()

    log.info(f"JETS filings: {len(jets_index)} quarters")
    log.info(f"Employment data: {len(emp_data)} rows, {emp_data['symbol'].nunique()} companies")
    log.info(f"JETS prices: {len(jets_prices)} months")

    # Aggregate employment to quarterly
    emp_quarterly = aggregate_employment_by_quarter(emp_data)
    log.info(f"Quarterly employment: {len(emp_quarterly)} rows")

    # =========================================================================
    # ANALYSIS 1: JETS AUM vs Aggregate Industry Employment
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS 1: JETS AUM vs Aggregate Employment")
    log.info("=" * 60)

    # Aggregate total employment across all companies by quarter
    agg_emp = emp_quarterly.groupby("quarter").agg({
        "total_employees": "sum",
        "emp_change": "sum",
    }).reset_index()
    agg_emp.columns = ["quarter", "total_industry_emp", "industry_emp_change"]

    # Calculate percent change
    agg_emp = agg_emp.sort_values("quarter")
    agg_emp["industry_emp_pct_change"] = agg_emp["total_industry_emp"].pct_change() * 100

    # Merge with JETS data
    jets_emp = jets_index.merge(agg_emp, on="quarter", how="inner")
    jets_emp = jets_emp.sort_values("quarter")

    # Calculate JETS AUM percent change
    jets_emp["aum_pct_change"] = jets_emp["total_value"].pct_change() * 100

    print("\n--- JETS AUM vs Industry Employment by Quarter ---")
    print(jets_emp[["quarter", "total_value", "aum_pct_change", "total_industry_emp", "industry_emp_pct_change"]].to_string(index=False))

    # Correlation
    valid = jets_emp.dropna(subset=["aum_pct_change", "industry_emp_pct_change"])
    if len(valid) > 3:
        corr, pval = stats.pearsonr(valid["aum_pct_change"], valid["industry_emp_pct_change"])
        print(f"\nCorrelation (JETS AUM % change vs Employment % change): {corr:.4f} (p={pval:.4f})")

    # =========================================================================
    # ANALYSIS 2: JETS Holdings vs Individual Company Employment
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS 2: Holdings Weight vs Company Employment")
    log.info("=" * 60)

    # Map JETS holdings tickers to employment data symbols
    ticker_map = {
        "ALK": "ALK",
        "ALGT": "ALGT",
        "AAL": "AA",  # American Airlines
        "DAL": "DAL",
        "UAL": "UAL",
        "LUV": "LUV",
        "JBLU": "JBLU",
        "SKYW": "SKYW",
        "ULCC": "ULCC",
        "SAVE": None,  # Spirit - may not be in employment data
        "SNCY": None,  # Sun Country
    }

    # Get holdings for companies we have employment data for
    holdings_with_emp = []

    for quarter in jets_holdings["quarter"].unique():
        q_holdings = jets_holdings[jets_holdings["quarter"] == quarter]
        q_emp = emp_quarterly[emp_quarterly["quarter"] == quarter]

        for _, holding in q_holdings.iterrows():
            ticker = holding.get("ticker", "")
            if ticker in ticker_map and ticker_map[ticker]:
                emp_symbol = ticker_map[ticker]
                emp_row = q_emp[q_emp["symbol"] == emp_symbol]

                if not emp_row.empty:
                    holdings_with_emp.append({
                        "quarter": quarter,
                        "ticker": ticker,
                        "name": holding.get("name", ""),
                        "weight": holding.get("pct_portfolio", 0),
                        "value_usd": holding.get("value_usd", 0),
                        "emp_symbol": emp_symbol,
                        "total_employees": emp_row["total_employees"].iloc[0],
                        "emp_pct_change": emp_row["emp_pct_change"].iloc[0],
                        "price_return": emp_row["price_return"].iloc[0],
                    })

    holdings_emp_df = pd.DataFrame(holdings_with_emp)

    if not holdings_emp_df.empty:
        print("\n--- Holdings with Employment Data ---")
        print(holdings_emp_df.head(20).to_string(index=False))

        # Correlation: weight vs employment
        valid = holdings_emp_df.dropna(subset=["weight", "emp_pct_change"])
        if len(valid) > 5:
            corr, pval = stats.pearsonr(valid["weight"], valid["emp_pct_change"])
            print(f"\nCorrelation (Holding Weight vs Emp % change): {corr:.4f} (p={pval:.4f})")

        # Save
        holdings_emp_df.to_csv(OUTPUT_DIR / "jets_holdings_with_employment.csv", index=False)
        log.info(f"Saved: jets_holdings_with_employment.csv")

    # =========================================================================
    # ANALYSIS 3: JETS ETF Returns vs Employment (Lagged)
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS 3: JETS Returns vs Employment (Lagged)")
    log.info("=" * 60)

    # Calculate JETS quarterly returns from price data
    jets_prices["quarter"] = jets_prices["date"].apply(lambda x: get_quarter_from_date(x))
    jets_quarterly = jets_prices.groupby("quarter").agg({
        "close": "last",
        "date": "max",
    }).reset_index()
    jets_quarterly = jets_quarterly.sort_values("quarter")
    jets_quarterly["jets_return"] = jets_quarterly["close"].pct_change() * 100

    # Merge with aggregate employment
    jets_emp_returns = jets_quarterly.merge(agg_emp, on="quarter", how="inner")
    jets_emp_returns = jets_emp_returns.sort_values("quarter")

    # Add lagged employment changes
    for lag in range(1, 5):
        jets_emp_returns[f"emp_lag_{lag}"] = jets_emp_returns["industry_emp_pct_change"].shift(lag)

    print("\n--- JETS Returns vs Employment Changes ---")
    print(jets_emp_returns[["quarter", "jets_return", "industry_emp_pct_change", "emp_lag_1", "emp_lag_2"]].dropna().to_string(index=False))

    # Lagged correlations
    print("\n--- Lagged Correlations: JETS Return vs Employment Change ---")
    lag_results = []
    for lag in range(0, 5):
        col = "industry_emp_pct_change" if lag == 0 else f"emp_lag_{lag}"
        valid = jets_emp_returns.dropna(subset=["jets_return", col])
        if len(valid) > 3:
            corr, pval = stats.pearsonr(valid["jets_return"], valid[col])
            lag_results.append({"lag": lag, "correlation": corr, "p_value": pval, "n": len(valid)})
            print(f"  Lag {lag}: r = {corr:.4f}, p = {pval:.4f}, n = {len(valid)}")

    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("GENERATING VISUALIZATIONS")
    log.info("=" * 60)

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. JETS AUM vs Employment Time Series
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    jets_emp_sorted = jets_emp.sort_values("report_date")
    ax1.plot(jets_emp_sorted["report_date"], jets_emp_sorted["total_value"] / 1e9,
             marker="o", linewidth=2, color="blue", label="JETS AUM")
    ax1.set_ylabel("JETS AUM ($B)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.legend(loc="upper left")
    ax1.set_title("JETS ETF AUM vs Airline Industry Employment")

    ax1b = ax1.twinx()
    ax1b.plot(jets_emp_sorted["report_date"], jets_emp_sorted["total_industry_emp"] / 1e6,
              marker="s", linewidth=2, color="green", label="Total Employment")
    ax1b.set_ylabel("Total Employees (M)", color="green")
    ax1b.tick_params(axis="y", labelcolor="green")
    ax1b.legend(loc="upper right")

    # Percent changes
    ax2.bar(jets_emp_sorted["report_date"] - pd.Timedelta(days=15),
            jets_emp_sorted["aum_pct_change"], width=25, alpha=0.7, label="AUM % Change", color="blue")
    ax2.bar(jets_emp_sorted["report_date"] + pd.Timedelta(days=15),
            jets_emp_sorted["industry_emp_pct_change"], width=25, alpha=0.7, label="Emp % Change", color="green")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax2.set_ylabel("Quarterly % Change")
    ax2.set_xlabel("Date")
    ax2.legend()

    plt.tight_layout()
    fig1.savefig(OUTPUT_DIR / "jets_aum_vs_employment.png", dpi=150)
    log.info("Saved: jets_aum_vs_employment.png")

    # 2. Scatter: AUM change vs Employment change
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    valid = jets_emp.dropna(subset=["aum_pct_change", "industry_emp_pct_change"])
    ax2.scatter(valid["industry_emp_pct_change"], valid["aum_pct_change"],
                s=100, alpha=0.7, edgecolors="black")

    # Add labels
    for _, row in valid.iterrows():
        ax2.annotate(row["quarter"], (row["industry_emp_pct_change"], row["aum_pct_change"]),
                     fontsize=8, alpha=0.7)

    # Trend line
    if len(valid) > 2:
        z = np.polyfit(valid["industry_emp_pct_change"], valid["aum_pct_change"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid["industry_emp_pct_change"].min(), valid["industry_emp_pct_change"].max(), 100)
        ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        corr, _ = stats.pearsonr(valid["industry_emp_pct_change"], valid["aum_pct_change"])
        ax2.set_title(f"JETS AUM Change vs Employment Change (r = {corr:.3f})")

    ax2.set_xlabel("Industry Employment % Change")
    ax2.set_ylabel("JETS AUM % Change")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    fig2.savefig(OUTPUT_DIR / "jets_aum_vs_employment_scatter.png", dpi=150)
    log.info("Saved: jets_aum_vs_employment_scatter.png")

    # 3. JETS Returns vs Employment (lagged correlation bar chart)
    if lag_results:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        lag_df = pd.DataFrame(lag_results)
        colors = ["green" if r >= 0 else "red" for r in lag_df["correlation"]]
        bars = ax3.bar(lag_df["lag"], lag_df["correlation"], color=colors, edgecolor="black")

        # Add significance markers
        for i, row in lag_df.iterrows():
            marker = "*" if row["p_value"] < 0.05 else ""
            ax3.text(row["lag"], row["correlation"] + 0.02, f"{row['correlation']:.3f}{marker}",
                     ha="center", fontsize=10)

        ax3.axhline(y=0, color="black", linewidth=0.5)
        ax3.set_xlabel("Lag (Quarters)")
        ax3.set_ylabel("Correlation Coefficient")
        ax3.set_title("JETS ETF Return vs Lagged Employment Change")
        ax3.set_xticks(range(5))
        ax3.set_ylim(-1, 1)

        plt.tight_layout()
        fig3.savefig(OUTPUT_DIR / "jets_return_vs_employment_lag.png", dpi=150)
        log.info("Saved: jets_return_vs_employment_lag.png")

    plt.close("all")

    # Save summary results
    jets_emp.to_csv(OUTPUT_DIR / "jets_vs_employment_quarterly.csv", index=False)
    jets_emp_returns.to_csv(OUTPUT_DIR / "jets_returns_vs_employment.csv", index=False)

    if lag_results:
        pd.DataFrame(lag_results).to_csv(OUTPUT_DIR / "lagged_correlation_results.csv", index=False)

    log.info("\n" + "=" * 60)
    log.info("ANALYSIS COMPLETE")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
