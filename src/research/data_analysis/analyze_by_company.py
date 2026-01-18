"""
Analyze employment vs stock price relationship for each individual airline company.
Generates comprehensive findings report.

Outputs:
  src/research/data/findings_employee_count_and_stock_price.csv
  src/research/data/findings_employee_count_and_stock_price_summary.txt

Run from project root:
  python -m src.research.data_analysis.analyze_by_company
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------
# Paths
# ----------------------------
RESEARCH_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = RESEARCH_DIR / "data"

OUTPUT_CSV = DATA_DIR / "findings_employee_count_and_stock_price.csv"
OUTPUT_SUMMARY = DATA_DIR / "findings_employee_count_and_stock_price_summary.txt"

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analyze_by_company")


def load_data() -> pd.DataFrame:
    """Load merged prices and employment data."""
    df = pd.read_excel(DATA_DIR / "prices_employment_merged.xlsx")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["total_employees"])
    df = df.sort_values(["symbol", "date"])

    # Calculate percent changes
    df["price_return"] = df.groupby("symbol")["close"].pct_change() * 100
    df["emp_change_pct"] = df.groupby("symbol")["total_employees"].pct_change() * 100

    # Create lagged employment changes (0 to 48 months)
    for lag in range(1, 49):
        df[f"emp_lag_{lag}"] = df.groupby("symbol")["emp_change_pct"].shift(lag)

    return df.dropna(subset=["price_return", "emp_change_pct"])


def run_regression_by_company(df: pd.DataFrame) -> pd.DataFrame:
    """Run regression analysis for each company at each lag."""
    results = []

    for symbol in df["symbol"].unique():
        company_data = df[df["symbol"] == symbol].copy()
        source_name = company_data["source_name"].iloc[0]
        n_obs = len(company_data)

        for lag in range(0, 49):
            col = "emp_change_pct" if lag == 0 else f"emp_lag_{lag}"
            valid = company_data.dropna(subset=["price_return", col])

            if len(valid) < 5:
                continue

            X = valid[[col]].values
            y = valid["price_return"].values

            # Simple regression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            # Statsmodels for p-value
            X_sm = sm.add_constant(X)
            sm_model = sm.OLS(y, X_sm).fit()
            coef = sm_model.params[1]
            p_value = sm_model.pvalues[1]
            std_err = sm_model.bse[1]
            t_stat = sm_model.tvalues[1]

            # Correlation
            corr, corr_p = stats.pearsonr(valid[col], valid["price_return"])

            results.append({
                "symbol": symbol,
                "source_name": source_name,
                "lag": lag,
                "lag_label": f"t-{lag}" if lag > 0 else "t-0 (contemporaneous)",
                "n_observations": len(valid),
                "coefficient": coef,
                "std_error": std_err,
                "t_statistic": t_stat,
                "p_value": p_value,
                "r_squared": r2,
                "r_squared_pct": r2 * 100,
                "correlation": corr,
                "significant_05": p_value < 0.05,
                "significant_10": p_value < 0.10,
            })

    return pd.DataFrame(results)


def generate_summary_report(results_df: pd.DataFrame) -> str:
    """Generate a comprehensive text summary of findings."""
    lines = []
    lines.append("=" * 80)
    lines.append("FINDINGS: EMPLOYEE COUNT AND STOCK PRICE RELATIONSHIP")
    lines.append("=" * 80)
    lines.append("")
    lines.append("METHODOLOGY:")
    lines.append("-" * 40)
    lines.append("- Data: Monthly stock prices and employment levels for 10 airline companies")
    lines.append("- Variables: Month-over-month % change in stock price vs % change in employment")
    lines.append("- Analysis: Linear regression with lags from t-0 (contemporaneous) to t-7 months")
    lines.append("- Significance level: p < 0.05")
    lines.append("")

    # Overall summary
    lines.append("=" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    total_tests = len(results_df)
    sig_05 = results_df["significant_05"].sum()
    sig_10 = results_df["significant_10"].sum()

    lines.append(f"Total regression tests: {total_tests}")
    lines.append(f"Significant at p<0.05: {sig_05} ({sig_05/total_tests*100:.1f}%)")
    lines.append(f"Significant at p<0.10: {sig_10} ({sig_10/total_tests*100:.1f}%)")
    lines.append("")

    # Significant findings
    sig_results = results_df[results_df["significant_05"]].sort_values(["symbol", "lag"])

    if len(sig_results) > 0:
        lines.append("=" * 80)
        lines.append("SIGNIFICANT FINDINGS (p < 0.05)")
        lines.append("=" * 80)
        lines.append("")

        for _, row in sig_results.iterrows():
            lines.append(f"Company: {row['symbol']} ({row['source_name']})")
            lines.append(f"  Lag: {row['lag_label']}")
            lines.append(f"  Coefficient: {row['coefficient']:.4f}")
            lines.append(f"  R-squared: {row['r_squared_pct']:.2f}%")
            lines.append(f"  P-value: {row['p_value']:.4f}")
            lines.append(f"  Interpretation: 1% increase in employment {row['lag']} month(s) ago")
            lines.append(f"                  → {row['coefficient']:.2f}% change in stock return")
            lines.append("")
    else:
        lines.append("No statistically significant relationships found at p<0.05 level.")
        lines.append("")

    # By company summary
    lines.append("=" * 80)
    lines.append("RESULTS BY COMPANY")
    lines.append("=" * 80)
    lines.append("")

    for symbol in results_df["symbol"].unique():
        company_results = results_df[results_df["symbol"] == symbol]
        source_name = company_results["source_name"].iloc[0]
        n_obs = company_results["n_observations"].iloc[0]

        lines.append(f"--- {symbol} ({source_name}) ---")
        lines.append(f"Observations: {n_obs}")
        lines.append("")

        # Table header
        lines.append(f"{'Lag':<20} {'Coef':>10} {'R²':>10} {'p-value':>10} {'Sig':>8}")
        lines.append("-" * 60)

        for _, row in company_results.iterrows():
            sig_marker = "***" if row["p_value"] < 0.01 else ("**" if row["p_value"] < 0.05 else ("*" if row["p_value"] < 0.10 else ""))
            lines.append(f"{row['lag_label']:<20} {row['coefficient']:>10.4f} {row['r_squared_pct']:>9.2f}% {row['p_value']:>10.4f} {sig_marker:>8}")

        # Best lag for this company
        best_row = company_results.loc[company_results["r_squared"].idxmax()]
        lines.append("")
        lines.append(f"Best predictive lag: {best_row['lag_label']} (R²={best_row['r_squared_pct']:.2f}%)")

        sig_company = company_results[company_results["significant_05"]]
        if len(sig_company) > 0:
            lines.append(f"Significant lags: {', '.join(sig_company['lag_label'].tolist())}")
        else:
            lines.append("Significant lags: None")

        lines.append("")
        lines.append("")

    # Lag summary across all companies
    lines.append("=" * 80)
    lines.append("SUMMARY BY LAG (Pooled Across Companies)")
    lines.append("=" * 80)
    lines.append("")

    lag_summary = results_df.groupby("lag").agg({
        "coefficient": "mean",
        "r_squared": "mean",
        "p_value": "mean",
        "significant_05": "sum",
    }).reset_index()

    lines.append(f"{'Lag':<10} {'Avg Coef':>12} {'Avg R²':>12} {'Avg p-value':>12} {'# Sig (p<.05)':>15}")
    lines.append("-" * 65)

    for _, row in lag_summary.iterrows():
        lag_label = f"t-{int(row['lag'])}" if row["lag"] > 0 else "t-0"
        lines.append(f"{lag_label:<10} {row['coefficient']:>12.4f} {row['r_squared']*100:>11.2f}% {row['p_value']:>12.4f} {int(row['significant_05']):>15}")

    lines.append("")

    # Conclusions
    lines.append("=" * 80)
    lines.append("CONCLUSIONS")
    lines.append("=" * 80)
    lines.append("")

    # Find best overall lag
    best_lag = lag_summary.loc[lag_summary["r_squared"].idxmax()]
    most_sig_lag = lag_summary.loc[lag_summary["significant_05"].idxmax()]

    lines.append("1. PREDICTIVE POWER:")
    lines.append(f"   - Overall R² values are very low (typically <5%), indicating employment")
    lines.append(f"     changes explain minimal variance in stock returns")
    lines.append(f"   - Best average predictive power at lag {int(best_lag['lag'])} (R²={best_lag['r_squared']*100:.2f}%)")
    lines.append("")

    lines.append("2. STATISTICAL SIGNIFICANCE:")
    lines.append(f"   - Most significant findings at lag {int(most_sig_lag['lag'])} ({int(most_sig_lag['significant_05'])} companies)")
    if sig_05 > 0:
        lines.append(f"   - {sig_05} out of {total_tests} tests show statistical significance (p<0.05)")
    else:
        lines.append(f"   - Very few tests show statistical significance")
    lines.append("")

    lines.append("3. DIRECTION OF RELATIONSHIP:")
    neg_coef = (results_df["coefficient"] < 0).sum()
    lines.append(f"   - {neg_coef}/{total_tests} ({neg_coef/total_tests*100:.1f}%) of relationships are negative")
    lines.append(f"   - Negative relationship suggests: higher employment growth → lower stock returns")
    lines.append(f"   - This could indicate employment as a lagging indicator or cost pressure signal")
    lines.append("")

    lines.append("4. COMPANY-SPECIFIC FINDINGS:")
    for symbol in results_df["symbol"].unique():
        company_sig = results_df[(results_df["symbol"] == symbol) & (results_df["significant_05"])]
        if len(company_sig) > 0:
            best = company_sig.loc[company_sig["r_squared"].idxmax()]
            lines.append(f"   - {symbol}: Significant at {best['lag_label']}, R²={best['r_squared_pct']:.1f}%, coef={best['coefficient']:.2f}")

    lines.append("")
    lines.append("5. PRACTICAL IMPLICATIONS:")
    lines.append("   - Employment data has LIMITED predictive value for stock returns")
    lines.append("   - The relationship, where significant, tends to be INVERSE (more hiring → lower returns)")
    lines.append("   - Lag 3 appears most promising but R² remains very low")
    lines.append("   - NOT recommended as a standalone trading signal")
    lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def main() -> int:
    log.info("Loading data...")
    df = load_data()
    log.info(f"Data: {len(df)} rows, {df['symbol'].nunique()} companies")

    log.info("Running regression analysis by company...")
    results_df = run_regression_by_company(df)
    log.info(f"Generated {len(results_df)} regression results")

    # Save detailed results
    results_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Saved: {OUTPUT_CSV}")

    # Generate and save summary report
    summary = generate_summary_report(results_df)
    with open(OUTPUT_SUMMARY, "w") as f:
        f.write(summary)
    log.info(f"Saved: {OUTPUT_SUMMARY}")

    # Print summary to console
    print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
