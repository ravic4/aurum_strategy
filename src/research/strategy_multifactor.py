"""
Multi-Factor Risk-Managed Airline Trading Strategy
====================================================

Combines three signal pillars with quantum-inspired optimization:

  Pillar 1: Employment lags (monthly) — Lasso regression on BLS data
  Pillar 2: News sentiment (daily)   — FinBERT on Alpaca News API
  Pillar 3: Fundamentals (quarterly) — FMP financial ratios & scores

Risk management:
  - QAOA-inspired simulated annealing for factor weight optimization
  - Sentiment gate: override LONG → CASH if 7-day sentiment < -0.3
  - Fundamental filter: reduce position 50% if Piotroski < 4
  - Intra-month stop-loss: flatten if drawdown > 10%

Run:
  python -m src.research.strategy_multifactor --symbol SKYW
  python -m src.research.strategy_multifactor --symbol SKYW --no-sentiment  # skip FinBERT

Requires:
  FMP_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY in .env
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
RESEARCH_DIR = Path(__file__).resolve().parent
DATA_DIR = RESEARCH_DIR / "data"
OUTPUT_DIR = DATA_DIR / "strategy_backtest"

FMP_KEY = os.getenv("FMP_API_KEY", "").strip()
ALP_KEY = os.getenv("ALPACA_API_KEY", "").strip().strip('"')
ALP_SEC = os.getenv("ALPACA_SECRET_KEY", "").strip().strip('"')

MAX_LAG = 12
LASSO_ALPHA = 0.5
MIN_TRAIN = 24
STOP_LOSS_PCT = 10.0
HOLD_DAYS = 3   # hold position for N trading days after entry, then go to cash
ENTRY_DELAY = 1  # wait N trading days after signal before entering (day-0 is contra)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("multifactor")


# ============================================================================
# PILLAR 1: EMPLOYMENT SIGNAL (Lasso)
# ============================================================================

def load_employment_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "prices_employment_merged.xlsx")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["total_employees"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    df["price_return"] = df.groupby("symbol")["close"].pct_change() * 100
    df["emp_change_pct"] = df.groupby("symbol")["total_employees"].pct_change() * 100

    for lag in range(1, MAX_LAG + 1):
        df[f"emp_lag_{lag}"] = df.groupby("symbol")["emp_change_pct"].shift(lag)

    df["emp_ma_3"] = df.groupby("symbol")["emp_change_pct"].transform(lambda x: x.rolling(3).mean())
    df["price_ma_3"] = df.groupby("symbol")["price_return"].transform(lambda x: x.rolling(3).mean())

    return df.dropna(subset=["price_return", "emp_change_pct"])


def get_employment_signals(df: pd.DataFrame, symbol: str, train_end: str) -> pd.DataFrame:
    """Train Lasso on data before train_end, predict on all months after."""
    features = [f"emp_lag_{i}" for i in range(1, MAX_LAG + 1)] + ["emp_ma_3", "price_ma_3"]
    cd = df[df["symbol"] == symbol].dropna(subset=features).copy()

    train = cd[cd["date"] < train_end]
    test = cd[cd["date"] >= train_end]

    if len(train) < MIN_TRAIN or len(test) == 0:
        return pd.DataFrame()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features].values)
    model = Lasso(alpha=LASSO_ALPHA, max_iter=10000)
    model.fit(X_train, train["price_return"].values)

    X_test = scaler.transform(test[features].values)
    preds = model.predict(X_test)

    signals = test[["date"]].copy()
    signals["emp_signal"] = preds
    signals["emp_direction"] = np.where(preds > 0, 1, -1)

    # Feature importance for logging
    importance = pd.Series(model.coef_, index=features)
    selected = importance[abs(importance) > 0.001]
    log.info(f"Lasso selected {len(selected)}/{len(features)} features")

    return signals.reset_index(drop=True)


# ============================================================================
# PILLAR 2: NEWS SENTIMENT (FinBERT via Alpaca)
# ============================================================================

_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
        log.info("Loading FinBERT model...")
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, use_safetensors=True
        )
        _finbert_pipeline = hf_pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            device=-1,
        )
        log.info("FinBERT loaded.")
    return _finbert_pipeline


def fetch_news(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch news articles from Alpaca for a symbol in a date range."""
    headers = {"APCA-API-KEY-ID": ALP_KEY, "APCA-API-SECRET-KEY": ALP_SEC}
    all_news = []
    page_token = None

    for _ in range(20):  # max pages
        params = {
            "symbols": symbol,
            "start": start,
            "end": end,
            "limit": 50,
            "include_content": "false",
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(
            "https://data.alpaca.markets/v1beta1/news",
            headers=headers, params=params, timeout=30,
        )
        data = resp.json()
        articles = data.get("news", [])
        if not articles:
            break

        for a in articles:
            dt = pd.to_datetime(a["created_at"])
            if dt.tzinfo is not None:
                dt = dt.tz_localize(None)
            all_news.append({
                "date": dt.normalize(),
                "headline": a.get("headline", ""),
                "summary": a.get("summary", ""),
            })

        page_token = data.get("next_page_token")
        if not page_token:
            break

    if not all_news:
        return pd.DataFrame(columns=["date", "headline", "summary", "sentiment"])

    return pd.DataFrame(all_news)


def score_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Run FinBERT on headlines, return daily sentiment scores."""
    if news_df.empty:
        return pd.DataFrame(columns=["date", "sentiment_7d", "sentiment_30d"])

    finbert = _get_finbert()

    texts = news_df["headline"].tolist()
    # FinBERT returns [{label: positive/negative/neutral, score: float}]
    batch_size = 32
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = finbert(batch, truncation=True, max_length=128)
        for result in results:
            score_map = {r["label"]: r["score"] for r in result}
            # Net sentiment: positive - negative (range: -1 to +1)
            net = score_map.get("positive", 0) - score_map.get("negative", 0)
            scores.append(net)

    news_df = news_df.copy()
    news_df["sentiment"] = scores

    # Aggregate to daily mean
    daily = news_df.groupby("date")["sentiment"].mean().reset_index()
    daily = daily.sort_values("date")

    # Rolling averages
    daily["sentiment_7d"] = daily["sentiment"].rolling(7, min_periods=1).mean()
    daily["sentiment_30d"] = daily["sentiment"].rolling(30, min_periods=1).mean()

    return daily


# ============================================================================
# PILLAR 3: FUNDAMENTALS (FMP)
# ============================================================================

def fetch_fundamentals(symbol: str) -> pd.DataFrame:
    """Pull financial ratios and scores from FMP."""
    # Ratios
    r1 = requests.get(
        f"https://financialmodelingprep.com/stable/ratios",
        params={"symbol": symbol, "apikey": FMP_KEY, "limit": 20},
        timeout=15,
    )
    ratios = r1.json() if r1.status_code == 200 else []

    # Financial scores
    r2 = requests.get(
        f"https://financialmodelingprep.com/stable/financial-scores",
        params={"symbol": symbol, "apikey": FMP_KEY},
        timeout=15,
    )
    scores_data = r2.json() if r2.status_code == 200 else []

    if not ratios:
        return pd.DataFrame()

    rows = []
    for r in ratios:
        row = {
            "date": pd.to_datetime(r.get("date")),
            "pe_ratio": r.get("priceToEarningsRatio"),
            "peg_ratio": r.get("priceToEarningsGrowthRatio"),
            "price_to_sales": r.get("priceToSalesRatio"),
            "debt_to_equity": r.get("debtToEquityRatio"),
            "current_ratio": r.get("currentRatio"),
            "gross_margin": r.get("grossProfitMargin"),
            "operating_margin": r.get("operatingProfitMargin"),
            "net_margin": r.get("netProfitMargin"),
            "roe": r.get("returnOnEquity"),
            "interest_coverage": r.get("interestCoverageRatio"),
            "revenue_per_share": r.get("revenuePerShare"),
            "eps": r.get("netIncomePerShare"),
        }
        rows.append(row)

    fund_df = pd.DataFrame(rows).sort_values("date")

    # Add Piotroski score (from scores endpoint — single latest value)
    piotroski = None
    if scores_data:
        s = scores_data[0] if isinstance(scores_data, list) else scores_data
        piotroski = s.get("piotroskiScore")

    fund_df["piotroski"] = piotroski

    return fund_df


def get_fundamental_score(fund_df: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """Get the most recent fundamental data as of a given date."""
    if fund_df.empty:
        return {"fund_score": 0.0, "piotroski": 5, "pe_ratio": 15.0}

    valid = fund_df[fund_df["date"] <= as_of]
    if valid.empty:
        valid = fund_df.head(1)

    latest = valid.iloc[-1]

    # Composite fundamental score: higher = more attractive
    # Normalize each metric to [0, 1] range, then average
    scores = []

    pe = latest.get("pe_ratio")
    if pe and pe > 0:
        scores.append(min(1.0, 20.0 / pe))  # lower PE = better

    margin = latest.get("operating_margin")
    if margin and margin > 0:
        scores.append(min(1.0, margin / 0.25))  # higher margin = better

    peg = latest.get("peg_ratio")
    if peg and peg > 0:
        scores.append(min(1.0, 1.0 / peg))  # lower PEG = better

    fund_score = np.mean(scores) if scores else 0.5

    return {
        "fund_score": fund_score,
        "piotroski": latest.get("piotroski", 5),
        "pe_ratio": pe if pe else 15.0,
    }


# ============================================================================
# QUANTUM-INSPIRED OPTIMIZATION (QAOA Simulated Annealing)
# ============================================================================

def qaoa_optimize(
    factor_returns: np.ndarray,
    factor_names: list[str],
    n_iterations: int = 2000,
    temp_start: float = 10.0,
    temp_end: float = 0.01,
    risk_aversion: float = 1.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Quantum-inspired factor weight optimization via simulated annealing.

    Solves the QUBO problem: minimize -μᵀw + λ wᵀΣw
    where w ∈ {0, 0.25, 0.5, 0.75, 1.0}^n (discretized weights)

    This mimics QAOA's exploration of the solution landscape by:
    1. Starting at high "temperature" (wide quantum superposition analog)
    2. Gradually cooling (measurement/collapse analog)
    3. Accepting worse solutions probabilistically (quantum tunneling analog)

    Returns: optimal weight vector (normalized to sum=1)
    """
    rng = np.random.RandomState(seed)
    n_factors = factor_returns.shape[1]

    if n_factors == 0 or len(factor_returns) < 3:
        return np.ones(max(n_factors, 1)) / max(n_factors, 1)

    # Expected returns and covariance (the "Hamiltonian")
    mu = np.mean(factor_returns, axis=0)
    sigma = np.cov(factor_returns.T) if n_factors > 1 else np.array([[np.var(factor_returns)]])

    # Ensure sigma is 2D
    if sigma.ndim == 1:
        sigma = sigma.reshape(1, 1)

    # Discretized weight space (quantum-like discrete states)
    # Minimum 0.1 ensures no factor is fully ignored
    weight_levels = np.array([0.1, 0.25, 0.5, 0.75, 1.0])

    def energy(w):
        """QUBO energy function (lower = better)."""
        w_sum = w.sum()
        if w_sum == 0:
            return 1e6
        w_norm = w / w_sum
        ret = mu @ w_norm
        risk = w_norm @ sigma @ w_norm
        return -ret + risk_aversion * risk

    # Initialize: equal weights
    current = np.full(n_factors, 0.5)
    current_e = energy(current)
    best = current.copy()
    best_e = current_e

    # Annealing schedule (geometric cooling — models decoherence)
    temps = np.geomspace(temp_start, temp_end, n_iterations)

    for i, temp in enumerate(temps):
        # Propose neighbor (quantum tunneling analog — discrete jump)
        candidate = current.copy()
        flip_idx = rng.randint(n_factors)
        candidate[flip_idx] = rng.choice(weight_levels)

        candidate_e = energy(candidate)
        delta = candidate_e - current_e

        # Metropolis acceptance (Boltzmann distribution — quantum thermal state analog)
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current = candidate
            current_e = candidate_e

            if current_e < best_e:
                best = current.copy()
                best_e = current_e

    # Normalize
    if best.sum() > 0:
        best = best / best.sum()
    else:
        best = np.ones(n_factors) / n_factors

    log.info(f"QAOA optimization: {dict(zip(factor_names, np.round(best, 3)))}")
    return best


# ============================================================================
# STRATEGY ENGINE
# ============================================================================

def run_strategy(
    symbol: str,
    train_end: str = "2025-01-01",
    use_sentiment: bool = True,
) -> pd.DataFrame:
    """
    Full multi-factor strategy with daily execution.

    1. Monthly: Lasso employment signal → direction
    2. Daily: FinBERT sentiment → risk gate
    3. Quarterly: FMP fundamentals → position sizing
    4. QAOA: optimal factor combination
    """

    log.info(f"{'='*60}")
    log.info(f"MULTI-FACTOR STRATEGY: {symbol}")
    log.info(f"Train period: before {train_end}")
    log.info(f"{'='*60}")

    # --- Load data ---
    log.info("Loading employment data...")
    emp_df = load_employment_data()

    log.info("Computing employment signals...")
    emp_signals = get_employment_signals(emp_df, symbol, train_end)
    if emp_signals.empty:
        log.error("No employment signals generated")
        return pd.DataFrame()

    log.info(f"Employment signals: {len(emp_signals)} months")

    # --- Daily prices ---
    log.info("Loading daily prices...")
    daily = pd.read_csv(DATA_DIR / "prices_eod_long.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    daily_sym = daily[daily["symbol"] == symbol].sort_values("date").reset_index(drop=True)
    daily_sym["daily_return"] = daily_sym["close"].pct_change() * 100

    test_start = pd.to_datetime(train_end)
    daily_test = daily_sym[daily_sym["date"] >= test_start].copy()

    if daily_test.empty:
        log.error("No daily test data")
        return pd.DataFrame()

    log.info(f"Daily test data: {len(daily_test)} days ({daily_test['date'].min().date()} to {daily_test['date'].max().date()})")

    # --- Fundamentals ---
    log.info("Fetching FMP fundamentals...")
    fund_df = fetch_fundamentals(symbol)
    log.info(f"Fundamentals: {len(fund_df)} periods")

    # --- Sentiment ---
    sentiment_daily = pd.DataFrame()
    if use_sentiment:
        log.info("Fetching Alpaca news & running FinBERT...")
        start_str = test_start.strftime("%Y-%m-%d")
        end_str = daily_test["date"].max().strftime("%Y-%m-%d")
        news_df = fetch_news(symbol, start_str, end_str)
        log.info(f"News articles: {len(news_df)}")

        if not news_df.empty:
            sentiment_daily = score_sentiment(news_df)
            log.info(f"Sentiment days: {len(sentiment_daily)}")
    else:
        log.info("Sentiment disabled (--no-sentiment)")

    # --- Build factor matrix for QAOA ---
    # Align monthly employment signals with factor scores
    factor_history = []
    for _, sig in emp_signals.iterrows():
        month_date = sig["date"]
        fund_info = get_fundamental_score(fund_df, month_date)

        # Get sentiment around this month
        sent_score = 0.0
        if not sentiment_daily.empty:
            nearby = sentiment_daily[
                (sentiment_daily["date"] >= month_date - pd.Timedelta(days=30))
                & (sentiment_daily["date"] <= month_date)
            ]
            if not nearby.empty:
                sent_score = nearby["sentiment_7d"].iloc[-1]

        factor_history.append({
            "employment": sig["emp_signal"],
            "sentiment": sent_score,
            "fundamental": fund_info["fund_score"],
        })

    factor_df = pd.DataFrame(factor_history)
    factor_names = ["employment", "sentiment", "fundamental"]

    # QAOA optimization — use prior knowledge that employment has the proven edge
    # Build factor return history from the training period for better optimization
    log.info("Running QAOA-inspired factor optimization...")

    # Use walk-forward: train QAOA on historical employment signal performance
    cd_full = emp_df[emp_df["symbol"] == symbol].copy()
    features_full = [f"emp_lag_{i}" for i in range(1, MAX_LAG + 1)] + ["emp_ma_3", "price_ma_3"]
    cd_full = cd_full.dropna(subset=features_full + ["price_return"])
    train_cd = cd_full[cd_full["date"] < train_end]

    if len(train_cd) >= MIN_TRAIN:
        # Simulate employment signal returns during training
        scaler_t = StandardScaler()
        model_t = Lasso(alpha=LASSO_ALPHA, max_iter=10000)
        train_factor_returns = []
        for t in range(MIN_TRAIN, len(train_cd)):
            tr = train_cd.iloc[:t]
            te = train_cd.iloc[t]
            X_tr = scaler_t.fit_transform(tr[features_full].values)
            model_t.fit(X_tr, tr["price_return"].values)
            pred = model_t.predict(scaler_t.transform(te[features_full].values.reshape(1, -1)))[0]
            actual = te["price_return"]
            # employment factor: predicted return
            # sentiment proxy: momentum (price_ma_3) since we don't have historical sentiment
            # fundamental proxy: mean-reversion signal
            train_factor_returns.append({
                "employment": pred,
                "sentiment": te.get("price_ma_3", 0),
                "fundamental": 0.5,  # neutral baseline
            })

        train_factor_df = pd.DataFrame(train_factor_returns)
        raw_weights = qaoa_optimize(
            train_factor_df[factor_names].values,
            factor_names,
            risk_aversion=1.0,
        )
        # Enforce minimum 40% on employment — it has the proven p-value edge
        factor_weights = raw_weights.copy()
        if factor_weights[0] < 0.4:
            deficit = 0.4 - factor_weights[0]
            factor_weights[0] = 0.4
            # Redistribute deficit proportionally from other factors
            others_sum = factor_weights[1:].sum()
            if others_sum > 0:
                factor_weights[1:] -= deficit * (factor_weights[1:] / others_sum)
            factor_weights = np.clip(factor_weights, 0.05, 1.0)
            factor_weights /= factor_weights.sum()
    else:
        factor_weights = np.array([0.6, 0.2, 0.2])

    log.info(f"Optimal weights: emp={factor_weights[0]:.2f}, sent={factor_weights[1]:.2f}, fund={factor_weights[2]:.2f}")

    # --- Execute daily strategy ---
    log.info("Executing daily strategy with risk management...")

    # Assign monthly signal to each trading day
    daily_test = daily_test.copy()
    daily_test["signal"] = None
    daily_test["position_size"] = 0.0
    daily_test["risk_gate"] = ""

    for i, sig in emp_signals.iterrows():
        sig_date = sig["date"]

        # Skip ENTRY_DELAY days, then hold for HOLD_DAYS
        after_signal = daily_test[daily_test["date"] >= sig_date]
        if len(after_signal) <= ENTRY_DELAY:
            continue
        eligible = after_signal.iloc[ENTRY_DELAY : ENTRY_DELAY + HOLD_DAYS]
        if eligible.empty:
            continue

        mask = daily_test.index.isin(eligible.index)

        direction = sig["emp_direction"]  # +1 or -1
        fund_info = get_fundamental_score(fund_df, sig_date)

        # ---- POSITION SIZING from QAOA-weighted risk factors ----
        sent_score = factor_history[i]["sentiment"]
        fund_score = fund_info["fund_score"]

        # Base conviction from employment signal strength (0 to 1)
        emp_conviction = min(1.0, abs(sig["emp_signal"]) / 5.0)

        # Sentiment modifier: agree with direction = boost, disagree = reduce
        sent_agrees = (sent_score > 0 and direction == 1) or (sent_score < 0 and direction == -1)
        sent_modifier = 1.0 + (0.3 if sent_agrees else -0.3) * factor_weights[1]

        # Fundamental modifier: strong fundamentals = boost long, weak = boost short
        fund_modifier = 1.0 + (fund_score - 0.5) * direction * factor_weights[2]

        # Final position size: conviction * modifiers (not multiplied by weight again)
        size = np.clip(emp_conviction * sent_modifier * fund_modifier, 0.2, 1.0)

        # Risk gate 1: Piotroski filter
        gate = "OK"
        if fund_info["piotroski"] is not None and fund_info["piotroski"] < 4:
            size *= 0.5
            gate = "PIOTROSKI_REDUCED"

        daily_test.loc[mask, "signal"] = "LONG" if direction == 1 else "SHORT"
        daily_test.loc[mask, "position_size"] = size
        daily_test.loc[mask, "risk_gate"] = gate

    # Days without a signal = CASH
    daily_test["signal"] = daily_test["signal"].fillna("CASH")
    daily_test["position_size"] = daily_test["position_size"].fillna(0.0)

    # Apply daily sentiment gate + stop-loss
    daily_test["strategy_return"] = 0.0
    daily_test["active_signal"] = daily_test["signal"]
    cumulative_month_return = 0.0
    current_month = None
    stopped_out = False

    for idx in daily_test.index:
        row = daily_test.loc[idx]
        if pd.isna(row["daily_return"]) or row["signal"] == "CASH":
            daily_test.at[idx, "active_signal"] = "CASH"
            daily_test.at[idx, "strategy_return"] = 0.0
            continue

        day = row["date"]
        month_key = day.to_period("M")

        # Reset stop-loss at month boundary
        if month_key != current_month:
            current_month = month_key
            cumulative_month_return = 0.0
            stopped_out = False

        # Check stop-loss
        if stopped_out:
            daily_test.at[idx, "active_signal"] = "CASH"
            daily_test.at[idx, "risk_gate"] = "STOP_LOSS"
            daily_test.at[idx, "strategy_return"] = 0.0
            continue

        # Sentiment gate (daily)
        if not sentiment_daily.empty and use_sentiment:
            sent_row = sentiment_daily[sentiment_daily["date"] <= day]
            if not sent_row.empty:
                recent_sent = sent_row.iloc[-1]["sentiment_7d"]
                if recent_sent < -0.3 and row["signal"] == "LONG":
                    daily_test.at[idx, "active_signal"] = "CASH"
                    daily_test.at[idx, "risk_gate"] = "SENTIMENT_GATE"
                    daily_test.at[idx, "strategy_return"] = 0.0
                    continue

        # Compute return
        size = row["position_size"]
        if row["signal"] == "LONG":
            strat_ret = row["daily_return"] * size
        else:
            strat_ret = -row["daily_return"] * size

        daily_test.at[idx, "strategy_return"] = strat_ret

        # Track intra-month drawdown for stop-loss
        cumulative_month_return += strat_ret
        if cumulative_month_return < -STOP_LOSS_PCT:
            stopped_out = True
            daily_test.at[idx, "risk_gate"] = "STOP_LOSS_TRIGGERED"

    # --- Compute cumulative returns ---
    valid = daily_test.dropna(subset=["daily_return"]).copy()
    valid["cum_strategy"] = (1 + valid["strategy_return"] / 100).cumprod()
    valid["cum_buyhold"] = (1 + valid["daily_return"] / 100).cumprod()

    return valid


# ============================================================================
# METRICS & CHARTS
# ============================================================================

def compute_metrics(results: pd.DataFrame) -> dict:
    sr = results["strategy_return"]
    n = len(results)
    total_strat = (results["cum_strategy"].iloc[-1] - 1) * 100
    total_bh = (results["cum_buyhold"].iloc[-1] - 1) * 100
    sharpe = sr.mean() / sr.std() * np.sqrt(252) if sr.std() > 0 else 0

    # Beta vs buy & hold
    bh = results["daily_return"]
    valid_mask = bh.notna() & sr.notna()
    if valid_mask.sum() > 5:
        beta, _, r_val, p_val, _ = stats.linregress(bh[valid_mask], sr[valid_mask])
    else:
        beta, r_val, p_val = 0, 0, 1

    # Max drawdown
    peak = results["cum_strategy"].cummax()
    dd = (results["cum_strategy"] - peak) / peak
    max_dd = dd.min() * 100

    # Win rate
    win_rate = (sr > 0).sum() / n * 100 if n > 0 else 0

    # Risk gates triggered
    gates = results["risk_gate"].value_counts()

    return {
        "n_days": n,
        "total_return_pct": total_strat,
        "buyhold_return_pct": total_bh,
        "sharpe": sharpe,
        "beta": beta,
        "r_squared": r_val ** 2,
        "beta_pvalue": p_val,
        "max_drawdown_pct": max_dd,
        "win_rate_pct": win_rate,
        "avg_daily_return": sr.mean(),
        "risk_gates": gates.to_dict(),
    }


def plot_results(results: pd.DataFrame, symbol: str, metrics: dict, path: Path):
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # --- Panel 1: Rolling Beta (60-day) ---
    ax1 = fig.add_subplot(gs[0, 0])
    window = 60
    rolling_beta = []
    rolling_dates = []
    for i in range(window, len(results)):
        chunk = results.iloc[i - window : i]
        bh = chunk["daily_return"]
        sr = chunk["strategy_return"]
        if bh.std() > 0 and sr.std() > 0:
            b, _, _, _, _ = stats.linregress(bh, sr)
            rolling_beta.append(b)
            rolling_dates.append(chunk.iloc[-1]["date"])

    ax1.plot(rolling_dates, rolling_beta, "b-", lw=1.5)
    ax1.axhline(1, color="gray", ls="--", lw=1, label="Beta = 1")
    ax1.axhline(0, color="black", ls=":", lw=0.5)
    ax1.axhline(metrics["beta"], color="red", ls="-", lw=1.5, label=f"Overall: {metrics['beta']:.2f}")

    # Shade signals
    _shade_signals(ax1, results)

    ax1.set_title("Rolling Beta (60-day)\nStrategy vs Buy & Hold", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Beta")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Rolling Sharpe (60-day) ---
    ax2 = fig.add_subplot(gs[0, 1])
    rolling_sharpe = []
    rolling_sharpe_dates = []
    for i in range(window, len(results)):
        chunk = results.iloc[i - window : i]
        s = chunk["strategy_return"]
        sh = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
        rolling_sharpe.append(sh)
        rolling_sharpe_dates.append(chunk.iloc[-1]["date"])

    ax2.plot(rolling_sharpe_dates, rolling_sharpe, "b-", lw=1.5)
    ax2.axhline(0, color="black", ls=":", lw=0.5)
    ax2.axhline(2, color="green", ls="--", lw=1, alpha=0.5, label="Sharpe = 2")
    ax2.axhline(metrics["sharpe"], color="red", ls="-", lw=1.5, label=f"Overall: {metrics['sharpe']:.2f}")

    _shade_signals(ax2, results)

    ax2.set_title("Rolling Sharpe (60-day, annualized)\nMulti-Factor Strategy", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Annualized Sharpe")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Cumulative Returns ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(results["date"], results["cum_strategy"], "b-", lw=2, label="Multi-Factor Strategy")
    ax3.plot(results["date"], results["cum_buyhold"], "k--", lw=1.5, label="Buy & Hold")
    ax3.axhline(1, color="gray", ls=":", lw=0.5)

    _shade_signals(ax3, results)

    # Mark risk gates
    gates = results[results["risk_gate"].isin(["STOP_LOSS", "STOP_LOSS_TRIGGERED", "SENTIMENT_GATE"])]
    if not gates.empty:
        ax3.scatter(gates["date"], gates["cum_strategy"], c="orange", marker="x", s=30, zorder=5, label="Risk Gate Active")

    ax3.set_title(
        f"Cumulative Returns\nStrategy: {metrics['total_return_pct']:+.1f}% vs B&H: {metrics['buyhold_return_pct']:+.1f}%",
        fontsize=13, fontweight="bold",
    )
    ax3.set_ylabel("Cumulative Return")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Daily Returns Distribution ---
    ax4 = fig.add_subplot(gs[1, 1])
    sr = results["strategy_return"]
    bh = results["daily_return"]

    ax4.hist(sr, bins=50, alpha=0.6, color="blue", label="Strategy", edgecolor="black", lw=0.3)
    ax4.hist(bh, bins=50, alpha=0.4, color="gray", label="Buy & Hold", edgecolor="black", lw=0.3)
    ax4.axvline(sr.mean(), color="blue", ls="--", lw=1.5, label=f"Strat Mean: {sr.mean():.3f}%")
    ax4.axvline(bh.mean(), color="gray", ls="--", lw=1.5, label=f"B&H Mean: {bh.mean():.3f}%")

    ax4.set_title(
        f"Daily Returns Distribution\nWin Rate: {metrics['win_rate_pct']:.0f}% | Max DD: {metrics['max_drawdown_pct']:.1f}%",
        fontsize=13, fontweight="bold",
    )
    ax4.set_xlabel("Daily Return (%)")
    ax4.set_ylabel("Frequency")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        f"{symbol} — Multi-Factor Risk-Managed Strategy (Daily, 2025 OOS)\n"
        f"Pillars: Employment + Sentiment + Fundamentals | QAOA-Optimized Weights",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path}")


def _shade_signals(ax, results):
    """Shade LONG/SHORT/CASH periods on a chart."""
    signals = results[["date", "active_signal"]].copy()
    signals["change"] = signals["active_signal"] != signals["active_signal"].shift()
    boundaries = signals[signals["change"]].index.tolist() + [signals.index[-1]]

    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        sig = results.loc[start_idx, "active_signal"]
        start_date = results.loc[start_idx, "date"]
        end_date = results.loc[end_idx, "date"]

        if sig == "LONG":
            ax.axvspan(start_date, end_date, alpha=0.08, color="green")
        elif sig == "SHORT":
            ax.axvspan(start_date, end_date, alpha=0.08, color="red")
        elif sig == "CASH":
            ax.axvspan(start_date, end_date, alpha=0.08, color="yellow")


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-factor risk-managed strategy")
    parser.add_argument("--symbol", default="SKYW")
    parser.add_argument("--train-end", default="2025-01-01", help="Train/test cutoff date")
    parser.add_argument("--no-sentiment", action="store_true", help="Skip FinBERT (faster)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    symbol = args.symbol.upper()

    results = run_strategy(
        symbol=symbol,
        train_end=args.train_end,
        use_sentiment=not args.no_sentiment,
    )

    if results.empty:
        log.error("No results")
        return 1

    metrics = compute_metrics(results)

    print(f"\n{'='*60}")
    print(f"RESULTS: {symbol} Multi-Factor Strategy (Daily)")
    print(f"{'='*60}")
    print(f"  Trading days:    {metrics['n_days']}")
    print(f"  Strategy Return: {metrics['total_return_pct']:+.1f}%")
    print(f"  Buy & Hold:      {metrics['buyhold_return_pct']:+.1f}%")
    print(f"  Sharpe Ratio:    {metrics['sharpe']:.2f}")
    print(f"  Beta:            {metrics['beta']:.2f}")
    print(f"  R²:              {metrics['r_squared']:.4f}")
    print(f"  Beta p-value:    {metrics['beta_pvalue']:.6f}")
    print(f"  Win Rate:        {metrics['win_rate_pct']:.0f}%")
    print(f"  Max Drawdown:    {metrics['max_drawdown_pct']:.1f}%")

    if metrics["risk_gates"]:
        print(f"\n  Risk Gates Triggered:")
        for gate, count in metrics["risk_gates"].items():
            if gate and gate != "OK":
                print(f"    {gate}: {count} days")

    chart_path = OUTPUT_DIR / f"{symbol.lower()}_multifactor.png"
    plot_results(results, symbol, metrics, chart_path)

    results.to_csv(OUTPUT_DIR / f"{symbol.lower()}_multifactor.csv", index=False)
    log.info(f"Saved: {OUTPUT_DIR / f'{symbol.lower()}_multifactor.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
