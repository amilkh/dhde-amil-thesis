# 別紙：研究2 再現性パッケージ（DHDE／福井県観光需要）

福井大学 博士論文（カンザダ・アミル）の別紙資料です。学位審査委員会からの
再現性に関するご依頼（2026年8月）に対応するもので、**本文は変更していません**。

本パッケージは、公表済みの研究2の統計量（有効 N = 397 日）を、論文作成時と
同一のコード（`src/models.py`, `src/benchmark.py`）でそのまま再現します。手計算は
一切行っていません。詳細な対応表は英語版 [`README.md`](README.md) をご参照ください。

## 委員ご依頼項目との対応

| # | ご依頼項目 | 収録場所 |
|---|---|---|
| 1 | 元データ＋マージ・除外処理（N=397 まで） | 生JMA気象CSV（`data/jma_raw/`）、マージ・除外コード（`code/src/`）、N=397 データセットCSV（`data/`）、出所（`DATA_PROVENANCE.md`）。ログ：`logs/01_core_analysis.log` |
| 2 | OLS：`summary()`（係数・p値・R²・Newey-West HAC標準誤差・VIF） | ログ `logs/01_core_analysis.log` |
| 3 | ランダムフォレスト：ハイパーパラメータ、交差検証、chronological hold-out、重要度（MDI・permutation） | ログ `logs/01_core_analysis.log` |
| 4 | ロバストネス：first-difference・LDV、R²・Durbin-Watson | ログ `logs/01_core_analysis.log` |
| 5 | モデル比較：RF・XGBoost・LightGBM 対 OLS（hold-out） | ログ `logs/02_model_comparison.log` |

## 再現された主要数値（論文と一致）

N=397、OLS 決定係数（in-sample）0.8096、hold-out 0.6834（317/80分割）／0.657
（318/79分割）、RF 5分割CV 0.557±0.131、RF hold-out 0.512、first-difference R²
0.708／DW 2.525、LDV R² 0.848／DW 1.899、Newey-West バンド幅 5、Cohen's f² 4.25。
モデル比較では、**OLS が全ての木系モデルを hold-out で上回る**ことを確認。

## 除外日数について

収集期間は 427 日、推定に用いた有効日数は 397 日で、**除外は計 30 日**です
（内訳：センサー障害 17 日＋RSI欠測・ラグ確保分）。「17」はセンサー障害のみの
部分集合です。除外前の全行は `data/study2_daily_full_preexclusion.csv` に収録して
おり、除外内容を確認できます。

## 実行方法

```bash
cd ~/active/hokuriku-tourism-ai-governance
python3 appendix_study2_reproducibility/reproduce_core.py       # 項目1-4
python3 appendix_study2_reproducibility/model_comparison.py     # 項目5
python3 appendix_study2_reproducibility/export_dataset.py       # N=397 CSV 出力
```

RSIデータは論文時点のコミット `bf2cfc45` に固定する必要があります
（`DATA_PROVENANCE.md` 参照）。固定しない場合、数値が約418行・R²≈0.79 に変動します。
