#!/usr/bin/env python3
"""Study 2 reproducibility appendix — core analysis runner.

Reproduces the exact published Study 2 pipeline (N = 397 effective days) and
prints every intermediate result to stdout so the raw console output can be
captured verbatim as a reproducibility log. It calls the SAME functions used
by the paper (src.models.fit_ols / fit_random_forest / robustness_suite /
statistical_rigor and src.benchmark.run_benchmark); nothing is recomputed by
hand.

It covers, in order:
  1. Data load + merge + exclusion down to N = 397 (raw row counts printed).
  2. OLS: full summary() (coefficients, p-values, R², standard errors),
     Newey-West HAC standard errors, and VIF.
  3. Random Forest: hyperparameters, 5-fold CV, chronological hold-out,
     MDI and permutation importance (source data for Fig 5.2).
  4. Robustness: first-difference and LDV specifications with R² and
     Durbin-Watson; weather-removal sensitivity.
  5. Effect sizes + chronological 80/20 hold-out (statistical_rigor) and the
     benchmark same-split hold-out (OLS 0.657 / RF).

Run from the repo root:
    python3 appendix_study2_reproducibility/reproduce_core.py

Requires the RSI data pinned at bf2cfc45 (paper state). Expected: 397 rows,
OLS in-sample R² = 0.8096, hold-out R² = 0.6834 (317/80) and 0.657 (318/79),
RF CV R² = 0.557, RF hold-out R² = 0.512. See DATA_PROVENANCE.md.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.benchmark import run_benchmark
from src.config import load_config
from src.data_loader import load_all_data
from src.feature_engineering import build_features
from src.models import (
    fit_ols,
    fit_random_forest,
    robustness_suite,
    statistical_rigor,
)
from src.report import Reporter


def banner(title: str) -> None:
    print("\n" + "#" * 80)
    print(f"# {title}")
    print("#" * 80)


def main() -> None:
    cfg = load_config()
    rpt = Reporter(cfg)

    # ── 1. DATA LOAD + MERGE + EXCLUSION → N = 397 ────────────────────────────
    banner("ITEM 1 — SOURCE DATA: LOAD, MERGE, EXCLUSION TO N = 397")
    data = load_all_data(cfg, rpt)
    daily, feature_cols = build_features(data["daily"], data["route_col"], rpt)

    model_df = daily[["date", "count"] + feature_cols].dropna().copy()
    print(f"\n>>> Effective model rows after merge + dropna (exclusions applied): "
          f"N = {len(model_df)}")
    print(f">>> Date span: {model_df['date'].min().date()} → "
          f"{model_df['date'].max().date()}")
    print(f">>> Feature columns ({len(feature_cols)}): {feature_cols}")

    # ── 2. OLS (summary, coefficients, p, R², std errors) ─────────────────────
    banner("ITEM 2 — OLS REGRESSION: FULL summary(), COEFFICIENTS, P-VALUES, R²")
    ols_result = fit_ols(model_df, feature_cols, rpt)

    # ── 3. RANDOM FOREST (hyperparams, CV, MDI + permutation importance) ──────
    banner("ITEM 3 — RANDOM FOREST: HYPERPARAMETERS, 5-FOLD CV, IMPORTANCES")
    rf_params = cfg.get("model", {}).get("random_forest")
    print(f">>> RF hyperparameters (from config/settings.yaml): {rf_params}")
    rf_result = fit_random_forest(model_df, feature_cols, rpt, rf_params=rf_params)

    # RF chronological hold-out on both split-roundings of 397 x 0.8 = 317.6.
    # The paper's headline RF hold-out R² = 0.512 is the 318/79 benchmark split
    # (see the BENCHMARK section below). The 317/80 split is shown here only to
    # mirror the OLS split-rounding (OLS: 0.683 at 317/80, 0.657 at 318/79).
    print("\n--- Random Forest chronological hold-out (split-rounding sensitivity) ---")
    sorted_df = model_df.sort_values("date").reset_index(drop=True)
    for label, split in [("317/80 (int)", int(len(sorted_df) * 0.80)),
                         ("318/79 (round, = paper)", round(len(sorted_df) * 0.80))]:
        tr, ho = sorted_df.iloc[:split], sorted_df.iloc[split:]
        rf_ho = RandomForestRegressor(**(rf_params or {}))
        rf_ho.fit(tr[feature_cols].values, tr["count"].values)
        rf_pred = rf_ho.predict(ho[feature_cols].values)
        rf_ho_r2 = r2_score(ho["count"].values, rf_pred)
        rf_ho_mae = mean_absolute_error(ho["count"].values, rf_pred)
        print(f"  [{label}] train n={len(tr)} hold-out n={len(ho)}: "
              f"RF hold-out R² = {rf_ho_r2:.4f}, MAE = {rf_ho_mae:.1f}")

    # ── 4 + 5. ROBUSTNESS (HAC, VIF, first-difference, LDV, DW) ───────────────
    banner("ITEM 4/5 — ROBUSTNESS: HAC, VIF, FIRST-DIFFERENCE, LDV, DURBIN-WATSON")
    robustness_suite(model_df, ols_result, feature_cols, rpt)

    # ── EFFECT SIZES + CHRONOLOGICAL HOLD-OUT (statistical_rigor) ─────────────
    banner("EFFECT SIZES + CHRONOLOGICAL HOLD-OUT (standardised β, Cohen's f²)")
    statistical_rigor(model_df, ols_result, feature_cols, rpt)

    # ── BENCHMARK: same-split hold-out (OLS 0.657 / RF) + baselines ───────────
    banner("BENCHMARK — SAME-SPLIT CHRONOLOGICAL HOLD-OUT (OLS 0.657, baselines)")
    run_benchmark(
        {"model_df": model_df, "feature_cols": feature_cols,
         "route_col": data["route_col"], "cfg": cfg},
        rpt,
    )

    banner("END OF CORE REPRODUCIBILITY RUN")


if __name__ == "__main__":
    main()
