# DHDE Study 2: Reproducibility Appendix

Supplementary appendix to the PhD thesis of **Amil Khanzada** (University of
Fukui). Reproduces every published Study 2 statistic (Fukui tourism demand
forecasting, N = 397 effective days): OLS, Random Forest, robustness suite,
and model comparison. Japanese: [`README_JA.md`](README_JA.md).

## Quick start (one command, no data download)

```bash
pip install -r requirements.txt
python3 reproduce.py
```

That reads `data/study2_model_dataset_N397.csv` and prints all headline numbers.
`xgboost` / `lightgbm` are optional (their comparison rows are skipped if absent);
everything else needs only numpy, pandas, statsmodels, scikit-learn.

## Expected output (verified)

| Quantity | Value |
|---|---|
| OLS in-sample *R²* | 0.8096 (Adj 0.802) |
| OLS hold-out *R²* (318/79 split) | 0.657 |
| OLS hold-out *R²* (317/80 split) | 0.683 |
| RF 5-fold CV *R²* | 0.557 ± 0.131 |
| RF hold-out *R²* | 0.512 |
| Newey-West HAC bandwidth | 5 |
| Durbin-Watson (original / first-diff / LDV) | 1.005 / 2.525 / 1.899 |
| First-difference *R²* / LDV *R²* | 0.708 / 0.848 |
| Model comparison | **OLS beats RF, XGBoost and LightGBM on the hold-out** |

The two hold-out values (0.657 / 0.683) are the same model on two roundings of
the 80/20 split of 397 (318/79 vs 317/80).

## Data

- `data/study2_model_dataset_N397.csv`: the 397-row estimation table (date,
  count, 16 features). This is all `reproduce.py` needs.
- `data/study2_daily_full_preexclusion.csv`: the 427-row merged panel before
  exclusions, so the 30 excluded rows are inspectable (17 sensor-outage days +
  13 missing-RSI / lag-warm-up days).
- Raw upstream repositories (AI-camera people-flow, route-search RSI, weather,
  survey) are pinned by commit in [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md)
  (RSI at `bf2cfc45`).

## Full original pipeline (optional)

`full_pipeline/` holds the exact analysis code used for the paper
(`code/src/`), the runners (`reproduce_core.py`, `model_comparison.py`,
`export_dataset.py`), and their raw console logs (`logs/`). Those scripts ran
inside the main analysis repo against the pinned raw data; the logs are the
verbatim output. For a quick check, the top-level `reproduce.py` is sufficient.

Software: Python 3.12; random seed 42.
