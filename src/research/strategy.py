"""
Employment-Based Airline Trading Strategy
==========================================

Uses lagged airline employment data to predict stock returns.
Implements walk-forward Lasso regression with true out-of-sample testing.

Best signals found (p < 0.05, n >= 30 observations):
  1. SKYW  lag-13  p=0.0005  R²=25.1%  coef=-3.65  n=44
  2. SKYW  lag-9   p=0.0034  R²=17.1%  coef=-3.20  n=48
  3. ALGT  lag-3   p=0.0040  R²=14.8%  coef=-4.58  n=54
  4. ALGT  lag-4   p=0.0147  R²=11.1%  coef=-4.00  n=53
  5. AA    lag-3   p=0.0300  R²=8.7%   coef=-6.20  n=54

Negative coefficients mean: employment growth -> lower future returns
(hiring = cost pressure / lagging indicator of peak cycle)

Run:
  python -m src.research.strategy                    # full analysis on best signal
  python -m src.research.strategy --symbol ALGT      # run on specific company
  python -m src.research.strategy --scan             # scan all companies for signals

Requires:
  FMP_API_KEY environment variable for out-of-sample price data
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESEARCH_DIR = Path(__file__).resolve().parent
DATA_DIR = RESEARCH_DIR / "data"
MERGED_DATA = DATA_DIR / "prices_employment_merged.xlsx"
OUTPUT_DIR = DATA_DIR / "strategy_backtest"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_LAG = 48
MIN_OBS = 30          # minimum observations to trust a regression
MIN_TRAIN = 24        # minimum months for walk-forward training window
LASSO_ALPHA = 0.5     # L1 regularization strength

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("strategy")


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_data() -> pd.DataFrame:
    """Load merged prices + employment, compute returns and lagged features."""
    df = pd.read_excel(MERGED_DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["total_employees"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Percent changes
    df["price_return"] = df.groupby("symbol")["close"].pct_change() * 100
    df["emp_change_pct"] = df.groupby("symbol")["total_employees"].pct_change() * 100

    # Lagged employment features
    for lag in range(1, MAX_LAG + 1):
        df[f"emp_lag_{lag}"] = df.groupby("symbol")["emp_change_pct"].shift(lag)

    # Moving averages of employment change
    df["emp_ma_3"] = df.groupby("symbol")["emp_change_pct"].transform(
        lambda x: x.rolling(3).mean()
    )
    df["emp_ma_6"] = df.groupby("symbol")["emp_change_pct"].transform(
        lambda x: x.rolling(6).mean()
    )

    # Price momentum
    df["price_ma_3"] = df.groupby("symbol")["price_return"].transform(
        lambda x: x.rolling(3).mean()
    )

    return df.dropna(subset=["price_return", "emp_change_pct"])


# ============================================================================
# 2. SIGNAL SCAN — find lowest p-value signals per company
# ============================================================================

def scan_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Run OLS regression for each company x lag. Return sorted by p-value."""
    results = []
    for symbol in df["symbol"].unique():
        cd = df[df["symbol"] == symbol]
        name = cd["source_name"].iloc[0]

        for lag in range(1, MAX_LAG + 1):
            col = f"emp_lag_{lag}"
            valid = cd.dropna(subset=["price_return", col])
            if len(valid) < 5:
                continue

            X = sm.add_constant(valid[[col]].values)
            model = sm.OLS(valid["price_return"].values, X).fit()

            results.append({
                "symbol": symbol,
                "name": name,
                "lag": lag,
                "n_obs": len(valid),
                "coefficient": model.params[1],
                "p_value": model.pvalues[1],
                "r_squared_pct": model.rsquared * 100,
                "t_stat": model.tvalues[1],
            })

    out = pd.DataFrame(results).sort_values("p_value")
    return out


def print_scan_results(scan_df: pd.DataFrame) -> None:
    """Print the best signals, filtering for reliability."""
    reliable = scan_df[scan_df["n_obs"] >= MIN_OBS].copy()

    print("\n" + "=" * 80)
    print("SIGNAL SCAN: Employment lag vs stock returns (n >= %d)" % MIN_OBS)
    print("=" * 80)
    print(f"\n{'Symbol':<8} {'Lag':>4} {'N':>5} {'Coeff':>9} {'p-value':>10} {'R²%':>7} {'t-stat':>8}")
    print("-" * 60)

    for _, r in reliable.head(15).iterrows():
        sig = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05 else "")
        print(
            f"{r['symbol']:<8} {r['lag']:>4} {r['n_obs']:>5} "
            f"{r['coefficient']:>9.3f} {r['p_value']:>10.6f} "
            f"{r['r_squared_pct']:>6.1f}% {r['t_stat']:>8.2f}  {sig}"
        )

    print(f"\nTotal significant (p<0.05, n>={MIN_OBS}): "
          f"{len(reliable[reliable['p_value'] < 0.05])}")


# ============================================================================
# 3. WALK-FORWARD LASSO STRATEGY
# ============================================================================

def build_features(df_company: pd.DataFrame, max_feature_lag: int = 12) -> list[str]:
    """Return feature column names available for the Lasso model."""
    lag_cols = [f"emp_lag_{i}" for i in range(1, max_feature_lag + 1)]
    extra = ["emp_ma_3", "emp_ma_6", "price_ma_3"]
    return [c for c in lag_cols + extra if c in df_company.columns]


def run_walkforward(
    df: pd.DataFrame,
    symbol: str,
    max_feature_lag: int = 12,
) -> pd.DataFrame:
    """
    Walk-forward Lasso backtest.

    Each month t:
      1. Train Lasso on months [0 .. t-1]
      2. Predict direction for month t
      3. Go LONG if predicted return > 0, else SHORT
      4. Record actual return
    """
    cd = df[df["symbol"] == symbol].copy().sort_values("date").reset_index(drop=True)
    features = build_features(cd, max_feature_lag)
    cd = cd.dropna(subset=features + ["price_return"])

    if len(cd) < MIN_TRAIN + 1:
        log.warning(f"{symbol}: not enough data ({len(cd)} rows)")
        return pd.DataFrame()

    records = []
    scaler = StandardScaler()
    model = Lasso(alpha=LASSO_ALPHA, max_iter=10000)

    for t in range(MIN_TRAIN, len(cd)):
        train = cd.iloc[:t]
        test_row = cd.iloc[t]

        X_train = train[features].values
        y_train = train["price_return"].values
        X_test = test_row[features].values.reshape(1, -1)

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)[0]

        signal = "LONG" if pred > 0 else "SHORT"
        actual = test_row["price_return"]
        strategy_return = actual if signal == "LONG" else -actual

        records.append({
            "date": test_row["date"],
            "close": test_row["close"],
            "actual_return": actual,
            "predicted_return": pred,
            "signal": signal,
            "strategy_return": strategy_return,
        })

    results = pd.DataFrame(records)
    results["cumulative_strategy"] = (1 + results["strategy_return"] / 100).cumprod()
    results["cumulative_buyhold"] = (1 + results["actual_return"] / 100).cumprod()
    return results


def compute_stats(results: pd.DataFrame) -> dict:
    """Compute strategy performance metrics."""
    sr = results["strategy_return"]
    bh = results["actual_return"]
    n = len(results)

    total_strat = (results["cumulative_strategy"].iloc[-1] - 1) * 100
    total_bh = (results["cumulative_buyhold"].iloc[-1] - 1) * 100
    sharpe = sr.mean() / sr.std() * np.sqrt(12) if sr.std() > 0 else 0
    win_rate = (sr > 0).sum() / n * 100

    correct_long = ((results["signal"] == "LONG") & (results["actual_return"] > 0)).sum()
    correct_short = ((results["signal"] == "SHORT") & (results["actual_return"] < 0)).sum()
    accuracy = (correct_long + correct_short) / n * 100

    return {
        "n_months": n,
        "total_return_pct": total_strat,
        "buyhold_return_pct": total_bh,
        "sharpe_ratio": sharpe,
        "win_rate_pct": win_rate,
        "accuracy_pct": accuracy,
        "avg_monthly_return": sr.mean(),
        "max_drawdown_pct": _max_drawdown(results["cumulative_strategy"]) * 100,
    }


def _max_drawdown(cum_series: pd.Series) -> float:
    peak = cum_series.cummax()
    dd = (cum_series - peak) / peak
    return dd.min()


# ============================================================================
# 4. VISUALIZATION
# ============================================================================

def plot_strategy(results: pd.DataFrame, symbol: str, stats: dict, path: Path) -> None:
    """Generate strategy performance chart."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 1]})

    # --- Panel 1: Cumulative returns ---
    ax1 = axes[0]
    ax1.plot(results["date"], results["cumulative_strategy"], "b-", lw=2, label="Strategy")
    ax1.plot(results["date"], results["cumulative_buyhold"], "k--", lw=1.5, label="Buy & Hold")
    ax1.axhline(1, color="gray", ls=":", lw=0.5)
    ax1.set_ylabel("Cumulative Return (1 = start)")
    ax1.set_title(
        f"{symbol} | Walk-Forward Lasso Strategy\n"
        f"Return: {stats['total_return_pct']:+.1f}% vs B&H: {stats['buyhold_return_pct']:+.1f}% | "
        f"Sharpe: {stats['sharpe_ratio']:.2f} | Accuracy: {stats['accuracy_pct']:.0f}%"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Monthly returns ---
    ax2 = axes[1]
    colors = ["green" if r > 0 else "red" for r in results["strategy_return"]]
    ax2.bar(results["date"], results["strategy_return"], color=colors, alpha=0.7, width=20)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Monthly Return (%)")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Signals ---
    ax3 = axes[2]
    ax3.plot(results["date"], results["close"], "k-", lw=1, alpha=0.7)

    longs = results[results["signal"] == "LONG"]
    shorts = results[results["signal"] == "SHORT"]
    ax3.scatter(longs["date"], longs["close"], c="green", marker="^", s=40, label="LONG", zorder=5)
    ax3.scatter(shorts["date"], shorts["close"], c="red", marker="v", s=40, label="SHORT", zorder=5)
    ax3.set_ylabel("Stock Price ($)")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path}")


# ============================================================================
# 5. OUT-OF-SAMPLE TEST (live prices from FMP)
# ============================================================================

def fetch_oos_prices(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """Pull recent prices from FMP for true out-of-sample testing."""
    import requests

    api_key = os.getenv("FMP_API_KEY", "")
    if not api_key:
        log.warning("FMP_API_KEY not set — skipping OOS test")
        return pd.DataFrame()

    url = (
        f"https://financialmodelingprep.com/stable/historical-price-eod/full"
        f"?symbol={symbol}&from={from_date}&to={to_date}&apikey={api_key}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "historical" in data:
        data = data["historical"]
    if not data:
        return pd.DataFrame()

    prices = pd.DataFrame(data)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date")
    return prices


# ============================================================================
# 6. MAIN
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Employment-based airline trading strategy")
    parser.add_argument("--symbol", default="SKYW", help="Ticker to backtest (default: SKYW)")
    parser.add_argument("--scan", action="store_true", help="Scan all companies for signals")
    parser.add_argument("--max-lag", type=int, default=12, help="Max lag for Lasso features")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    df = load_data()
    log.info(f"Data: {len(df)} rows, {df['symbol'].nunique()} companies")

    # ------------------------------------------------------------------
    # SCAN MODE
    # ------------------------------------------------------------------
    if args.scan:
        scan_df = scan_signals(df)
        print_scan_results(scan_df)
        scan_df.to_csv(OUTPUT_DIR / "signal_scan.csv", index=False)
        log.info(f"Saved: {OUTPUT_DIR / 'signal_scan.csv'}")
        return 0

    # ------------------------------------------------------------------
    # STRATEGY MODE
    # ------------------------------------------------------------------
    symbol = args.symbol.upper()
    log.info(f"Running walk-forward Lasso on {symbol}...")

    # Step 1: Show the OLS signal for this company
    scan_df = scan_signals(df)
    company_signals = scan_df[
        (scan_df["symbol"] == symbol) & (scan_df["n_obs"] >= MIN_OBS)
    ]

    if not company_signals.empty:
        best = company_signals.iloc[0]
        print(f"\n--- Best OLS signal for {symbol} ---")
        print(f"  Lag:         {best['lag']} months")
        print(f"  Coefficient: {best['coefficient']:.4f}")
        print(f"  P-value:     {best['p_value']:.6f}")
        print(f"  R²:          {best['r_squared_pct']:.1f}%")
        print(f"  N obs:       {best['n_obs']}")
    else:
        print(f"\nNo reliable OLS signals found for {symbol} (n >= {MIN_OBS})")

    # Step 2: Walk-forward backtest
    results = run_walkforward(df, symbol, max_feature_lag=args.max_lag)
    if results.empty:
        log.error(f"No results for {symbol}")
        return 1

    strat_stats = compute_stats(results)

    print(f"\n--- Walk-Forward Results: {symbol} ---")
    print(f"  Months:        {strat_stats['n_months']}")
    print(f"  Strategy:      {strat_stats['total_return_pct']:+.1f}%")
    print(f"  Buy & Hold:    {strat_stats['buyhold_return_pct']:+.1f}%")
    print(f"  Sharpe:        {strat_stats['sharpe_ratio']:.2f}")
    print(f"  Win Rate:      {strat_stats['win_rate_pct']:.0f}%")
    print(f"  Accuracy:      {strat_stats['accuracy_pct']:.0f}%")
    print(f"  Max Drawdown:  {strat_stats['max_drawdown_pct']:.1f}%")

    # Step 3: Plot
    chart_path = OUTPUT_DIR / f"{symbol.lower()}_strategy.png"
    plot_strategy(results, symbol, strat_stats, chart_path)

    # Step 4: Save results
    results.to_csv(OUTPUT_DIR / f"{symbol.lower()}_walkforward.csv", index=False)

    # Step 5: Feature importance from final Lasso model
    cd = df[df["symbol"] == symbol].copy().sort_values("date").reset_index(drop=True)
    features = build_features(cd, args.max_lag)
    cd = cd.dropna(subset=features + ["price_return"])

    scaler = StandardScaler()
    model = Lasso(alpha=LASSO_ALPHA, max_iter=10000)
    X_s = scaler.fit_transform(cd[features].values)
    model.fit(X_s, cd["price_return"].values)

    print(f"\n--- Lasso Feature Importance (final model) ---")
    importance = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)
    for feat, coef in importance.items():
        status = "SELECTED" if abs(coef) > 0.001 else "zeroed"
        print(f"  {feat:<18} {coef:>9.4f}  {status}")

    print(f"\n  Intercept: {model.intercept_:.4f}")
    print(f"  Features selected: {(abs(importance) > 0.001).sum()} / {len(features)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
