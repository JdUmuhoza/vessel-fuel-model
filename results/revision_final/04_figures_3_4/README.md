# Revised manuscript Figures 3 and 4

**Folder:** `results/revision_final/04_figures_3_4/`
**Script:** `experiments/revision/s04_figures_3_4.py`
**Generated:** 2026-08-17 · **Seed:** 42 · **Git commit:** `e125867a8f7d3908d9daf52ea4c4b0864bd09a06`

Both figures belong to Section 5.3, *Architecture-level predictive performance*. Nothing was
overwritten: this is a new folder, and no `.tex` file was modified.

---

## 1. Source configuration

Everything is rebuilt from the frozen configuration of
`results/revision_final/01_full_data_602020/`:

| Item | Value |
|---|---|
| Protocol | full-data chronological 60/20/20 on `train.csv` (530,706 records) |
| Partition sizes | train 318,423 · validation 106,141 · internal test 106,142 · dev_in 18,368 · dev_out 18,368 |
| Fitting / early-stopping split | 286,580 / 31,843 (last 10 % of the training block) |
| Calibrated physics | WBM = k · F_phys, **k = 0.393586** fitted on the training block only |
| Residual learner | **Huber** |
| S-GBM | WBM + full Huber residual (λ = 1) |
| C-GBM | WBM + **0.70** · Huber residual |
| BBM | **Huber fitted directly on measured power** — the folder-01 definition |
| Primary criterion | MAPE; sMAPE, MAE, RMSE, R² complementary |

**BBM definition — read this before comparing folders.** Folder 01 defines the BBM as the
Huber learner applied to the raw target, giving dev_in 18.820 % / dev_out 21.930 % MAPE.
That is the definition used here, because it is the analysis being retained. The
independently selected ExtraTrees BBM (dev_in 8.447 % / dev_out 19.348 %) comes from folder
25, a **later alternative experiment**, and is deliberately not used in these figures.
The two are not interchangeable.

**No leakage.** No model was refitted, retuned or reselected on dev_in or dev_out; λ was not
reselected on either partition. dev_in and dev_out enter only at evaluation.

---

## 2. Numerical verification — run before any file was written

The script recomputes every metric from the frozen predictions and asserts them against both
`01_full_data_602020/final_dev_in_dev_out.csv` and the Section 5.3 values, aborting instead
of plotting if any check fails. Full output: `verification_recomputed_metrics.csv`.

**Result: 8/8 MAPE values, 4/4 ΔMAPE values and k reproduce. Maximum deviation 0.000 pp.**

| Partition | Architecture | recomputed MAPE | folder 01 | Section 5.3 | Δ vs folder 01 | Δ vs Section 5.3 |
|---|---|---|---|---|---|---|
| dev_in | WBM | 8.847667 | 8.847667 | 8.848 | 0.000000 | 0.000 |
| dev_in | BBM | 18.820018 | 18.820018 | 18.820 | 0.000000 | 0.000 |
| dev_in | S-GBM | 8.788294 | 8.788294 | 8.788 | 0.000000 | 0.000 |
| dev_in | C-GBM | 8.237556 | 8.237556 | 8.238 | 0.000000 | 0.000 |
| dev_out | WBM | 9.213677 | 9.213677 | 9.214 | 0.000000 | 0.000 |
| dev_out | BBM | 21.930367 | 21.930367 | 21.930 | 0.000000 | 0.000 |
| dev_out | S-GBM | 9.544853 | 9.544853 | 9.545 | 0.000000 | 0.000 |
| dev_out | C-GBM | 8.732392 | 8.732392 | 8.732 | 0.000000 | 0.000 |

ΔMAPE (dev_out − dev_in), recomputed: WBM **+0.366**, BBM **+3.110**, S-GBM **+0.757**,
C-GBM **+0.495** pp. k recomputed = 0.393586, identical to the frozen value.

---

## 3. Figure 3 — `fig_regime_performance_representative.{pdf,svg,png}`

**Plotting data:** `fig3_plotting_data.csv` (8 rows: 2 partitions × 4 architectures, with
MAPE, sMAPE, MAE, RMSE, R², n).

Two panels, both grouped by architecture with solid bars for dev_in and hatched bars for
dev_out:

- **(a) MAPE (%)** — the primary criterion, with ΔMAPE printed above each pair.
- **(b) R²** — a complementary criterion, deliberately included because it **reorders the
  models**: S-GBM attains the highest R² on dev_out (0.9383) while C-GBM leads on MAPE
  (8.732 %).

| | MAPE dev_in | MAPE dev_out | ΔMAPE | R² dev_in | R² dev_out |
|---|---|---|---|---|---|
| WBM | 8.848 | 9.214 | +0.366 | 0.9392 | 0.9110 |
| BBM | 18.820 | 21.930 | +3.110 | 0.9156 | 0.8045 |
| S-GBM | 8.788 | 9.545 | +0.757 | 0.9575 | 0.9383 |
| C-GBM | 8.238 | 8.732 | +0.495 | 0.9553 | 0.9366 |

**What the figure supports.** C-GBM has the lowest MAPE on both benchmark partitions. The
BBM has the highest error on both and degrades most. Panel (b) shows the ordering is
criterion-dependent, so the figure does not present C-GBM as universally superior.

**What the figure does not support.** ΔMAPE is a descriptive change between two partitions,
not a robustness statistic: the smallest ΔMAPE belongs to the WBM (+0.366 pp), and the
C-GBM–WBM MAPE difference on dev_out is not statistically resolved elsewhere in the
revision. The annotation is therefore a description of the two bars beneath it and must not
be captioned as establishing robustness.

---

## 4. Figure 4 — `fig_pred_vs_actual_shrinkage.{pdf,svg,png}`

**Plotting data:** `fig4_plotting_data.csv` (73,472 rows = 4 panels × 18,368 records:
partition, architecture, measured_power_kw, predicted_power_kw).

2 × 2 hexbin of predicted against measured shaft power, from the frozen predictions:
WBM/dev_in, C-GBM/dev_in (top row); WBM/dev_out, C-GBM/dev_out (bottom row).

- identical axis limits on all four panels, derived from the combined measured-and-predicted
  range, equal aspect, shared axes;
- 1:1 perfect-prediction diagonal (dashed red) on every panel;
- identical hexbin representation everywhere: `gridsize = 45`, same extent, `mincnt = 1`,
  one shared logarithmic colour scale and a single colour bar;
- each panel labelled with its architecture and partition.

Annotations, **all recomputed from these predictions** (no value reused from the
CatBoost/λ = 0.25 experiment):

| Panel | MAPE (%) | R² | slope | intercept (kW) |
|---|---|---|---|---|
| WBM · dev_in | 8.848 | 0.9392 | 1.0227 | −146 |
| C-GBM · dev_in | 8.238 | 0.9553 | 1.0047 | +214 |
| WBM · dev_out | 9.214 | 0.9110 | 0.9375 | +660 |
| C-GBM · dev_out | 8.732 | 0.9366 | 0.9222 | +1280 |

**Interpretation.** In domain, C-GBM sits closest to the diagonal (slope 1.0047). On dev_out
both models regress toward the mean (slopes 0.9375 and 0.9222 with positive offsets), i.e.
both under-predict high-power operation on the shifted partition — a shared limitation, not
a C-GBM advantage. This is a predictive-calibration diagnostic only.

---

## 5. Scope — what these figures must not be used to claim

Per the revision decision, the old Section 5.4 physical-consistency, D_phys, speed-exponent
and operating-regime analyses are **removed**, and Figures 3 and 4 move into Section 5.3.
Neither figure computes or plots any of the following, and neither may be cited for:

- physical consistency of any architecture;
- preservation of a speed–power exponent;
- superiority argued from D_phys;
- universal robustness across operating regimes;
- robustness to multiple independent shift conditions.

Both are predictive diagnostics on dev_in and dev_out under a single compound benchmark
shift.

Note on the file name `fig_regime_performance_representative`: it is kept for continuity
with the existing `\includegraphics` path, but the figure is an **architecture-level**
comparison. It contains no per-regime stratification, and the caption should not describe
one.

---

## 6. Folder contents

| File | Contents |
|---|---|
| `fig_regime_performance_representative.{pdf,svg,png}` | Figure 3 |
| `fig_pred_vs_actual_shrinkage.{pdf,svg,png}` | Figure 4 |
| `fig3_plotting_data.csv` | exact Figure 3 plotting data (8 rows) |
| `fig4_plotting_data.csv` | exact Figure 4 plotting data (73,472 rows) |
| `verification_recomputed_metrics.csv` | recomputed metrics vs folder 01 vs Section 5.3, with pass flags, slopes and intercepts |
| `run_metadata.json` | frozen configuration, sample sizes, leakage flags, scope, seed, commit |
| `README.md` | this report |

Reproduce with:

```bash
python -m experiments.revision.s04_figures_3_4
```
