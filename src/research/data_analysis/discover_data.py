import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from pathlib import Path

path = Path(__file__).parent / "data" / "prices_employment_merged.xlsx"
df = pd.read_excel(path)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

# Forward return (next period close-to-close)
df["ret_fwd_1p"] = df.groupby("symbol")["close"].pct_change().shift(-1)

# Employment diffs / pct changes
for col in ["full_time_employees", "part_time_employees", "total_employees"]:
    df[f"{col}_diff"] = df.groupby("symbol")[col].diff()
    df[f"{col}_pctchg"] = df.groupby("symbol")[col].pct_change(fill_method=None)

# Feature set
features = [
    "volume", "vwap", "changePercent",
    "full_time_employees_diff", "part_time_employees_diff", "total_employees_diff",
    "full_time_employees_pctchg", "part_time_employees_pctchg", "total_employees_pctchg",
]

model_df = df.dropna(subset=features + ["ret_fwd_1p"]).copy()
model_df = model_df.sort_values("date").reset_index(drop=True)

X = model_df[features].astype(float)
y = model_df["ret_fwd_1p"].astype(float)

# Time split by date ordering
tscv = TimeSeriesSplit(n_splits=5)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNet(alpha=0.01, l1_ratio=0.3, random_state=0))
])

r2s = []
splits = list(tscv.split(X))
for tr, te in splits:
    pipe.fit(X.iloc[tr], y.iloc[tr])
    pred = pipe.predict(X.iloc[te])
    r2s.append(r2_score(y.iloc[te], pred))

print("fold R2:", [round(v, 4) for v in r2s], "mean:", round(np.mean(r2s), 4))

# Label randomization test on last fold
tr, te = splits[-1]
pipe.fit(X.iloc[tr], y.iloc[tr])
real_r2 = r2_score(y.iloc[te], pipe.predict(X.iloc[te]))

shuf = []
for _ in range(200):
    y_shuf = y.iloc[tr].sample(frac=1.0, replace=False, random_state=None).values
    pipe.fit(X.iloc[tr], y_shuf)
    shuf.append(r2_score(y.iloc[te], pipe.predict(X.iloc[te])))

print("last-fold real R2:", round(real_r2, 4))
print("last-fold shuffled R2 mean/std:", round(np.mean(shuf),4), round(np.std(shuf),4))
print("p(shuffled >= real):", round(np.mean(np.array(shuf) >= real_r2), 4))

# Permutation importance (last fold)
pipe.fit(X.iloc[tr], y.iloc[tr])
pi = permutation_importance(pipe, X.iloc[te], y.iloc[te], n_repeats=30, random_state=0)
imp = pd.Series(pi.importances_mean, index=features).sort_values(ascending=False)
print("\nPermutation importance:\n", imp)

# PCA
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=min(5, X.shape[1]), random_state=0)
pca.fit(Xs)
print("\nPCA explained variance:", np.round(pca.explained_variance_ratio_, 3))

# IsolationForest anomalies
iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=0)
iso.fit(Xs)
model_df["anomaly_score"] = -iso.decision_function(Xs)
print("\nTop anomalies:")
print(model_df.sort_values("anomaly_score", ascending=False)[
    ["date","symbol","source_name","close","volume","total_employees","total_employees_pctchg","ret_fwd_1p","anomaly_score"]
].head(10).to_string(index=False))
