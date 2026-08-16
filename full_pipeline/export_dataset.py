#!/usr/bin/env python3
"""Export the merged Study 2 modelling dataset (N = 397) as CSV.

Writes two files to appendix_study2_reproducibility/data/:
  - study2_model_dataset_N397.csv : the exact 397-row estimation table
    (date, count, and all 16 model features) after merge + exclusion.
  - study2_daily_full_preexclusion.csv : the merged daily panel BEFORE the
    final feature dropna, so the excluded rows are inspectable.

This is the analysis-ready dataset. The raw upstream repositories (AI-camera
people-flow, RSI trend data, survey) are pinned by commit in DATA_PROVENANCE.md;
this CSV lets a reviewer re-run the models directly without cloning them.

Run from the repo root:
    python3 appendix_study2_reproducibility/export_dataset.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.data_loader import load_all_data
from src.feature_engineering import build_features
from src.report import Reporter

OUT_DIR = REPO_ROOT / "appendix_study2_reproducibility" / "data"


def main() -> None:
    cfg = load_config()
    rpt = Reporter(cfg)
    data = load_all_data(cfg, rpt)
    daily, feature_cols = build_features(data["daily"], data["route_col"], rpt)

    cols = ["date", "count"] + feature_cols
    full = daily[cols].copy()
    model_df = full.dropna().copy()

    full_path = OUT_DIR / "study2_daily_full_preexclusion.csv"
    model_path = OUT_DIR / "study2_model_dataset_N397.csv"
    full.to_csv(full_path, index=False)
    model_df.to_csv(model_path, index=False)

    print(f"Full merged daily panel (pre-exclusion): {len(full)} rows -> {full_path}")
    print(f"Model estimation dataset (post-exclusion): {len(model_df)} rows -> {model_path}")
    print(f"Rows excluded by final feature dropna: {len(full) - len(model_df)}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Date span: {model_df['date'].min().date()} -> {model_df['date'].max().date()}")


if __name__ == "__main__":
    main()
