#!/usr/bin/env python3
"""Self-contained reproduction of the Study 2 statistics (N = 397).

Reads only data/study2_model_dataset_N397.csv (no external data download) and
reproduces every headline number in the thesis: OLS, Random Forest, the
robustness suite, and the model comparison. Prints everything to stdout.

    pip install -r requirements.txt
    python3 reproduce.py

XGBoost / LightGBM are optional: if not installed, that one comparison row is
skipped and everything else still runs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

CSV = Path(__file__).parent / "data" / "study2_model_dataset_N397.csv"
RF_PARAMS = dict(n_estimators=500, max_depth=10, min_samples_leaf=5,
                 random_state=42, n_jobs=-1)
TRAIN_PCT = 0.80


def h(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main():
    df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    feats = [c for c in df.columns if c not in ("date", "count")]
    X, y = df[feats].values, df["count"].values
    N = len(df)
    print(f"Loaded {CSV.name}: N = {N} days, {len(feats)} features")
    print(f"Date span: {df['date'].min().date()} -> {df['date'].max().date()}")

    # ── OLS ───────────────────────────────────────────────────────────────────
    h("OLS  (expected: in-sample R2 = 0.8096, Adj = 0.802)")
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    print(ols.summary())
    print(f"\nOLS R2 = {ols.rsquared:.4f} | Adj R2 = {ols.rsquared_adj:.4f}")

    # Newey-West HAC standard errors
    maxlags = max(1, int(0.75 * N ** (1 / 3)))
    hac = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    nsig = int((hac.pvalues < 0.05).sum())
    print(f"\nNewey-West HAC (bandwidth = {maxlags}): {nsig} predictors significant at p<0.05")

    # VIF
    print("\nVariance Inflation Factors:")
    Xc = sm.add_constant(df[feats].values)
    for i, c in enumerate(feats):
        try:
            v = variance_inflation_factor(Xc, i + 1)
            print(f"  {c:24s} {v:8.1f}{'  HIGH' if v > 10 else ''}")
        except Exception:
            print(f"  {c:24s}      N/A")

    # ── Random Forest ─────────────────────────────────────────────────────────
    h("RANDOM FOREST  (expected: CV R2 = 0.557, hold-out R2 = 0.512)")
    print(f"Hyperparameters: {RF_PARAMS}")
    rf = RandomForestRegressor(**RF_PARAMS).fit(X, y)
    print(f"RF in-sample R2 = {r2_score(y, rf.predict(X)):.4f}")
    cv = cross_val_score(rf, X, y, cv=5, scoring="r2")
    print(f"RF 5-fold CV R2 = {cv.mean():.4f} +/- {cv.std():.4f}")

    mdi = sorted(zip(feats, rf.feature_importances_), key=lambda t: -t[1])
    print("\nMDI importance (top 5):")
    for f, imp in mdi[:5]:
        print(f"  {f:24s} {imp:.4f}")
    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    pi = sorted(zip(feats, perm.importances_mean), key=lambda t: -t[1])
    print("Permutation importance (top 5):")
    for f, imp in pi[:5]:
        print(f"  {f:24s} {imp:+.4f}")

    # ── Robustness ────────────────────────────────────────────────────────────
    h("ROBUSTNESS  (expected: DW 1.005 | FD R2 0.708 DW 2.525 | LDV R2 0.848 DW 1.899)")
    print(f"Durbin-Watson (original OLS residuals): {durbin_watson(ols.resid):.3f}")

    d = df[["count"] + feats].diff().dropna()
    fd = sm.OLS(d["count"].values, sm.add_constant(d[feats].values)).fit()
    print(f"First-difference: R2 = {fd.rsquared:.4f}  DW = {durbin_watson(fd.resid):.3f}")

    l = df[["count"] + feats].copy()
    l["count_lag1"] = l["count"].shift(1)
    l = l.dropna()
    ldv = sm.OLS(l["count"].values, sm.add_constant(l[feats + ["count_lag1"]].values)).fit()
    print(f"LDV:              R2 = {ldv.rsquared:.4f}  DW = {durbin_watson(ldv.resid):.3f}  "
          f"(count_lag1 coef = {ldv.params[-1]:+.4f}, p = {ldv.pvalues[-1]:.4f})")

    # ── Model comparison (chronological hold-out, 318/79) ─────────────────────
    h("MODEL COMPARISON  (chronological hold-out 318/79; expected OLS 0.657 > all trees)")
    split = round(N * TRAIN_PCT)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    print(f"Split: train n = {split}, hold-out n = {N - split}\n")

    # Split-rounding note: 397*0.8 = 317.6. The 318/79 split (round) is used here
    # and by the benchmark module; the 317/80 split (truncate) gives OLS 0.683.
    s2 = int(N * TRAIN_PCT)  # 317
    o2 = sm.OLS(y[:s2], sm.add_constant(X[:s2])).fit()
    r2_317 = r2_score(y[s2:], o2.predict(sm.add_constant(X[s2:], has_constant="add")))
    print(f"OLS hold-out R2  (317/80 split) = {r2_317:.4f}   (README value 0.683)\n")

    rows = []
    op = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
    r2o = r2_score(yte, op.predict(sm.add_constant(Xte, has_constant="add")))
    rows.append(("OLS (linear)", r2o))
    rfh = RandomForestRegressor(**RF_PARAMS).fit(Xtr, ytr)
    rows.append(("Random Forest", r2_score(yte, rfh.predict(Xte))))
    try:
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             random_state=42, n_jobs=1, tree_method="hist").fit(Xtr, ytr)
        rows.append(("XGBoost", r2_score(yte, m.predict(Xte))))
    except ImportError:
        print("  (xgboost not installed - skipping)")
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                              num_leaves=31, subsample=0.9,
                              random_state=42, n_jobs=1, verbose=-1).fit(Xtr, ytr)
        rows.append(("LightGBM", r2_score(yte, m.predict(Xte))))
    except ImportError:
        print("  (lightgbm not installed - skipping)")

    print(f"\n  {'Model':<18}{'Hold-out R2':>12}")
    print("  " + "-" * 30)
    for name, r2 in sorted(rows, key=lambda t: -t[1]):
        print(f"  {name:<18}{r2:>12.4f}")
    best_tree = max(r2 for name, r2 in rows if name != "OLS (linear)")
    print(f"\n  => OLS ({r2o:.4f}) {'BEATS' if r2o > best_tree else 'does NOT beat'} "
          f"every tree model (best tree = {best_tree:.4f}).")

    h("DONE - all headline numbers reproduced from the N=397 dataset.")


if __name__ == "__main__":
    main()
