# Paper values not reproduced by the code

Paper: the manuscript version of 2026-11-20 (`paper_values.json` is extracted from it).
Results: `results/revision_final/`. Comparison: `python -m reproduce.compare --results results`,
which checks 476 table cells, each bound to one value of one result file (`table_map.json`),
at the precision printed in the paper.

## Tables that reproduce

| Paper label | Cells | Source |
|---|---|---|
| `tab:decision-results` | 32 | `34_final_decision_uncertainty/decision_results.csv` |
| `tab:uncertainty-results` | 23 | `34_final_decision_uncertainty/uncertainty_results.csv` |
| `tab:benchmarks` (record counts) | 6 | `FINAL_MANUSCRIPT_RESULTS/T01_data_protocol.csv`, `30_fuelcast_wind_and_wbm/partA_lambda_selection.csv` |
| `tab:ml-model-selection` | 108 | `24_full_data_602020/ml_selection_validation_full_data.csv` |
| `tab:overall-performance` | 36 | `25_bbm_selection_full_data/frozen_all_partitions.csv` |
| `tab:friedman-nemenyi-tests` | 112 | `38_architecture_friedman_nemenyi/{friedman,ranks,wilcoxon}.csv` |
| `tab:block-bootstrap-mape` | 24 | `39_block_bootstrap_huber/block_bootstrap_mape_differences.csv` |
| `tab:multigas-results` | 50 | `35_final_multigas_cii/multigas_results.csv` |
| `tab:cii-results` (intensity) | 10 | `35_final_multigas_cii/cii_results.csv` |
| `tab:mlp-sensitivity` | 32 | `43_mlp_architecture_sensitivity/`, `41_table6_selection_check/` |
| `tab:hyperparameters` (selected iterations) | 3 | `24_full_data_602020/ml_selection_validation_full_data.csv` |
| `tab:wbm-params` (fitted k) | 1 | `FINAL_MANUSCRIPT_RESULTS/T00_provenance.csv` |

Not compared, by design: input and configuration tables (`tab:features`, `tab:bbm-backends`,
`tab:cii-assumptions`, `tab:emission-factors`, the fixed rows of `tab:wbm-params` and
`tab:hyperparameters`), the CII rating letters, and the cells printed as an inequality
(`<0.001` in `tab:friedman-nemenyi-tests` and `tab:uncertainty-results`).

## Not reproduced

### 1. `tab:fuelcast-performance`, Poseidon and Triton (kept by the author)

| Cell | Paper | Code (`30_fuelcast_wind_and_wbm/partA_manuscript_table_mape.csv`) |
|---|---|---|
| Poseidon S-GBM, MAPE in / out / delta | 6.557 / 7.576 / +1.019 | 5.937 / 7.076 / +1.139 |
| Poseidon C-GBM, MAPE out / delta / lambda* | 6.076 / +0.139 / 0.75 | 7.076 / +1.139 / 1.00 |
| Triton C-GBM, MAPE in / out / delta / lambda* | 9.605 / 12.758 / +3.153 / 0.80 | 10.765 / 14.539 / +3.774 / 1.00 |

The code selects lambda* = 1.00 for Poseidon and Triton (C-GBM equals S-GBM) and 0.44 for Ceto.
Every WBM and BBM row and every Ceto row match. The printed values appear only in
`32_fuelcast_figure_manuscript_values/`, whose script states that the figure is drawn from a
table supplied by the author. The author decided on 2026-11-20 to keep the printed values;
the quick check therefore reports these ten cells as mismatches.

### Corrected in the paper on 2026-11-20

Three cells found by this check were corrected in the manuscript and now reproduce: the
`tab:decision-results` WBM confidence-interval upper bounds (15.86 to 15.85 in-domain, 27.68 to
27.67 shifted; the printed values had been rounded twice) and the `tab:uncertainty-results`
WBM shifted interval width (7203 to 7202 kW; the printed value was the internal-test width).

## Full rerun of 2026-11-19

Every final stage (`rf32` to `rf60`, except `rf47`, whose values are author-supplied) was run
again in its own process into a fresh results folder (`VFM_RESULTS_ROOT`, `PYTHONHASHSEED=0`).
All exited 0 except `rf43`, which exits 2 on its own check P3 (Part B hull geometry at the grid
bound for Poseidon and Triton), the same verdict as the saved run. Against
`results/revision_final/`: 123 CSV files identical, 15 numerically equal, 15 differing only in
timing columns (`fit_seconds`, scaling exponents, projected hours), with every exclusion
verdict unchanged.

### Stale saved FuelCast outputs (release only, not quoted in the paper)

`29_fuelcast_huber_final/` and `FINAL_MANUSCRIPT_RESULTS/T16*.csv` predate the FuelCast loader
fix (shift on `hs`, Ceto lambda* = 0.31); the rerun gives the wind shift and Ceto lambda* = 0.44,
matching folder 30. No value of these files appears in the paper. `rf43` reads folder 29 as its
pre-fix comparison, so the release must decide how folder 29 is shipped (Task 5).
