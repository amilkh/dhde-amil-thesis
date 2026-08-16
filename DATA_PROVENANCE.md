# Data Provenance (DHDE / Fukui Tourism, Study 2)

The exact upstream dataset states that reproduce the published Study 2
statistics. Each source repository carries an annotated tag `meas-v2-data`
at the paper-correct commit.

## Repo to commit table

| Upstream repo | Paper-correct commit | Notes |
| --- | --- | --- |
| `fukui-kanko-trend-data` (route-search RSI) | `bf2cfc45` | 2026-02-12 data. This is the one that must be checked out away from HEAD. |
| `fukui-kanko-trend-report` (superproject) | `8bbab300` | 2026-03-11 |
| `fukui-kanko-people-flow-data` (AI-camera counts) | `ca79a526` | data through 2026-03-10 |
| `opendata` (merged surveys) | `c782c518` | 2026-02-18 |
| `fukui-kanko-survey` | `30f8aa1c` | 2026-03-10 |

Notes:

- Only the RSI repo needs a checkout away from HEAD; the other four HEADs are
  already paper-correct. The tag pins them against future drift.
- Upstream rewrote the history of `fukui-kanko-trend-data`, so `bf2cfc45` is no
  longer on their `main`. The paper-correct state is preserved on the `amilkh`
  fork and by the `meas-v2-data` tag.

## Fork mirrors (tag `meas-v2-data`)

All five source repositories are mirrored under `github.com/amilkh/` carrying
the `meas-v2-data` annotated tag, independent of upstream deletions or rewrites:

| Fork | Tagged commit |
| --- | --- |
| `github.com/amilkh/fukui-kanko-trend-data` | `bf2cfc45` |
| `github.com/amilkh/fukui-kanko-trend-report` | `8bbab300` |
| `github.com/amilkh/fukui-kanko-people-flow-data` | `ca79a526` |
| `github.com/amilkh/opendata` | `c782c518` |
| `github.com/amilkh/fukui-kanko-survey` | `30f8aa1c` |

Restore, per repo:

```bash
git clone https://github.com/amilkh/<repo>.git
cd <repo> && git checkout meas-v2-data
```

## Verification

With the RSI repo pinned at `bf2cfc45`, the pipeline yields **N = 397**,
OLS in-sample *R²* = 0.8096, hold-out *R²* = 0.6834 (317/80), MAE = 1,793.
With drifted (post-February) RSI data you instead see ~418 rows and
*R²* ≈ 0.79, which signals the pin is not applied.

For a data-free check, `reproduce.py` in this repository runs against the
bundled `data/study2_model_dataset_N397.csv` and reproduces every headline
number without any upstream checkout.
