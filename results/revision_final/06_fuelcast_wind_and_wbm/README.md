# FuelCast Section 5.4 — after the loader fix, with a fuller physics prior

**Folder:** `results/revision_final/06_fuelcast_wind_and_wbm/`
**Scripts:** `experiments/revision/s06_fuelcast_wind_and_wbm.py`,
`hm_grid_boundary_probe.py`
**Generated:** 2026-08-17 · **Seed:** 42 · **Git commit:** `e125867a8f7d3908d9daf52ea4c4b0864bd09a06`
**Supersedes:** an earlier FuelCast run with CatBoost (not released) and folder 05 (pre-fix feature set). Both left intact.
Shifts results unchanged. No `.tex` file modified.

---

## 1. The loader defect this round corrects

`_FC_MAP` in `experiments/benchmark_measured/common.py` requested three columns by the
wrong case:

| Requested | Actual in the parquet | Presence |
|---|---|---|
| `Weather_WindSpeed10m` | `Weather_WindSpeed10M` | 99.83–99.85 % |
| `Weather_WindDirection10m` | `Weather_WindDirection10M` | 99.83–99.85 % |
| `Weather_Temperature2m` | `Weather_Temperature2M` | 99.83–99.85 % |

`load_fuelcast_vessel` filled unknown names with NaN without raising, and the
`notna().mean() > 0.95` filter then dropped them. **Every earlier FuelCast run was
therefore wind-blind and air-temperature-blind.** Fixes applied:

- three names corrected; `heading` (`Ship_Heading`) and `wave_ang`
  (`Weather_WaveDirection`) added for the physics prior only — neither is an ML feature;
- the loader now **raises** on an unknown non-optional source column, so a name typo can
  never silently NaN-fill again (`_FC_OPTIONAL` lists the columns genuinely absent for
  some vessels: `Ship_DraftAft`, `Ship_DraftFore`, `Ship_SpeedThroughWater`,
  `Ship_Heading`);
- rows with a NaN time index are dropped before the temporal ordering (Triton and Ceto
  carry NaN there; Poseidon does not). Row counts are unchanged (60,317 / 18,247 / 6,905),
  so this guard changed no result.

FuelCast ships no position and no timestamp — `index` is a bare row counter, so an
independent Open-Meteo lookup is impossible. It is also unnecessary: the `Weather_*`
columns *are* an Open-Meteo join performed by the dataset authors (the variable names are
Open-Meteo's own), and the coordinates were stripped afterwards.

### A second consequence: the shift variable changed

`fuelcast_shift_splits` defaults to `shift_var="wind10_kn"` and falls back to `hs` when
that column is unusable. Pre-fix it always fell back to `hs`; post-fix it uses wind, the
intended design. **So folder 05 and Part A below do not share partitions**, and their
numbers are not a clean A/B on the feature set alone:

| Vessel | folder 05 (hs) | Part A (wind) |
|---|---|---|
| Poseidon | dev_in 9,762 / dev_out 8,334 | dev_in 12,976 / dev_out 5,120 |
| Triton | 4,854 / 621 | 5,043 / 432 |
| Ceto | 1,443 / 629 | 1,558 / 514 |

Part A2 is a control that holds the split at `hs` so the feature-set effect can be read on
its own. It reproduces folder 05's partition sizes exactly (`partitions_match = True` for
all three vessels).

---

## 2. Part A — final Section 5.4 table (recommended)

Cube-law prior, wind-aware features, wind-quantile shift. `partA_manuscript_table_mape.csv`

| Vessel | Architecture | MAPE_in (%) | MAPE_out (%) | ΔMAPE (pp) | λ* |
|---|---|---|---|---|---|
| **Poseidon** | WBM-lite (cube law) | 19.586 | 12.815 | −6.771 | – |
| | BBM (Huber on target) | 10.042 | 7.651 | −2.391 | – |
| | S-GBM (Huber residual) | **5.937** | **7.076** | +1.139 | – |
| | C-GBM (Huber residual) | **5.937** | **7.076** | +1.139 | **1.00** |
| **Triton** | WBM-lite (cube law) | 13.977 | 17.574 | +3.597 | – |
| | BBM (Huber on target) | **10.292** | 14.843 | +4.551 | – |
| | S-GBM (Huber residual) | 10.765 | **14.539** | +3.774 | – |
| | C-GBM (Huber residual) | 10.765 | **14.539** | +3.774 | **1.00** |
| **Ceto** | WBM-lite (cube law) | 31.089 | 21.899 | −9.190 | – |
| | BBM (Huber on target) | 39.828 | 43.171 | +3.343 | – |
| | S-GBM (Huber residual) | 34.261 | 20.605 | −13.656 | – |
| | C-GBM (Huber residual) | **17.507** | **19.441** | +1.934 | **0.44** |

Supporting metrics: `partA_all_metrics.csv`. λ curves: `partA_lambda_curves_validation.csv`.

## 3. Part A2 — feature-only control (hs split, matched to folder 05)

`partA2_control_hs_split_table_mape.csv`, `feature_set_diff_vs_05_fuelcast_huber.csv`

| Vessel | features 29 → 30 | recovered | λ* 29 → 30 | C-GBM MAPE_in 29 → 30 | MAPE_out 29 → 30 |
|---|---|---|---|---|---|
| Poseidon | 8 → 11 | `wind10_kn`, `wind10_ang`, `temp` | 1.00 → 1.00 | 6.570 → **5.723** | 7.052 → **6.887** |
| Triton | 10 → 13 | same | 1.00 → 1.00 | 11.417 → **11.121** | 10.812 → **10.607** |
| Ceto | 10 → 13 | same | **0.31 → 0.44** | 20.210 → **18.633** | 16.962 → **16.505** |

Recovering wind improves every vessel on both partitions and raises Ceto's selected λ from
0.31 to 0.44. The two benchmark vessels stay at the λ = 1.00 boundary.

## 4. Part B — fuller Holtrop–Mennen prior (Poseidon, Triton)

`partB_manuscript_table_mape.csv`, `partB_selection.csv`, `partB_geometry_grid_validation.csv`

Prior: calm-water Holtrop–Mennen + Blendermann wind + STAWAVE-1 waves + the current effect
on speed over ground. **Schultz fouling disabled** (`months_since_cleaning = 0`) — no
FuelCast vessel carries a dry-docking clock. Vessel-relative wind and wave angles are formed
from `Ship_Heading`; apparent wind is the true 10 m wind vector minus the vessel velocity.
Ceto is excluded: no heading and no speed through water.

Hull particulars are unknown, so geometry is treated as a prior hyperparameter and selected
on the training-side validation tail by the MAPE of the k-scaled prior alone (270 candidates
over L, beam ratio, C_B and draught; k fitted on the fitting portion only). Poseidon's
draught is not measured, so T is a grid parameter; Triton's measured per-record draught and
trim are used.

| Vessel | Architecture | MAPE_in (%) | MAPE_out (%) | ΔMAPE (pp) | λ* |
|---|---|---|---|---|---|
| **Poseidon** | WBM (Holtrop–Mennen) | 20.898 | 14.949 | −5.950 | – |
| | BBM (Huber on target) | 10.042 | 7.651 | −2.391 | – |
| | S-GBM (Huber residual) | 7.259 | 8.750 | +1.492 | – |
| | C-GBM (Huber residual) | 7.259 | 8.750 | +1.492 | **1.00** |
| **Triton** | WBM (Holtrop–Mennen) | 13.675 | 17.244 | +3.569 | – |
| | BBM (Huber on target) | 10.292 | 14.843 | +4.551 | – |
| | S-GBM (Huber residual) | 11.436 | 15.755 | +4.319 | – |
| | C-GBM (Huber residual) | 11.365 | 15.734 | +4.369 | **0.99** |

### The fuller prior does not help

`prior_comparison_cube_vs_hm.csv`

| Vessel | cube MAPE_in | HM MAPE_in | cube MAPE_out | HM MAPE_out | HM better in / out |
|---|---|---|---|---|---|
| Poseidon | **19.586** | 20.898 | **12.815** | 14.949 | no / no |
| Triton | 13.977 | **13.675** | 17.574 | **17.244** | yes / yes (≈0.3 pp) |

With residual learning on top, the HM-based hybrids are **worse** than the cube-law-based
hybrids on both vessels (Poseidon S-GBM 7.259 vs 5.937; Triton 11.436 vs 10.765).

### Why — the geometry search leaves the physical envelope

The selected geometry sat on **every** bound of the grid (check P3 FAIL, kept as a finding).
The boundary probe (`hm_grid_boundary_probe`, 750 candidates, Poseidon) resolves what that means:

| Quantity | Value |
|---|---|
| Best inside the original grid | **19.342 %** validation MAPE (L 180 m, C_B 0.60, T 5.0 m) |
| Best over the extended grid | 15.839 % — at L 300 m, B 30 m, T 3.0 m |
| That optimum's ratios | **L/B = 10.0, T/B = 0.100** |
| Physically plausible? | **No** (screen: L/B ∈ [4, 8], T/B ≥ 0.25, L ≤ 200 m) |
| Gain from extending *within* a plausible envelope | **−0.10 pp** |
| Cube-law prior, same validation tail | 17.620 % |

The 3.5 pp apparent gain is bought entirely by driving the hull to a 300 × 30 × 3 m form
that cannot be this vessel. Inside a plausible envelope, extending the grid buys nothing.
**The Holtrop–Mennen prior is shape-limited on FuelCast, not size-limited** — no admissible
hull makes it fit better, and it does not even reach the cube law's 17.620 %.

The mechanism is visible in the raw data: these are offshore/construction vessels with five
generator sets and two boilers, median speed over ground 6.6 kn (Poseidon) and 0.03 kn
(Ceto), yet propeller shaft power reaching 33.9 MW. Power of that order at those speeds is
not overcoming hull resistance — it is dynamic positioning, thrusters, and hotel and mission
load. The target `Consumer_Total_MomentaryFuel` is *total* vessel fuel, so a hull-resistance
prior scaled by a single multiplicative k models only a minority component of it, and a
multiplicative k cannot represent a load that persists at zero speed. That limitation
applies to the cube law equally; it is a property of the benchmark, not of Holtrop–Mennen.

---

## 5. Verification

`checks.csv` — **10 / 11 PASS, 1 FAIL (the FAIL is the reported finding, not a defect).**

| ID | Check | Verdict |
|---|---|---|
| E1 | λ = 0 reproduces the prior, every partition/prior/metric | **PASS** 168/168 |
| E2 | λ = 1 reproduces S-GBM, every partition/prior/metric | **PASS** 168/168 |
| F1 | wind recovered into the feature set on all three vessels | **PASS** 11/13/13 features |
| L1 | no learner subsampled | **PASS** |
| L2 | fitting portion + validation tail = training block | **PASS** |
| L3 | geometry **and** λ selected on the validation tail only | **PASS** 270 candidates, none on dev |
| L4 | Shifts λ* = 0.70 not imposed anywhere | **PASS** λ* = 1.00, 1.00, 0.44 / 1.00, 0.99 |
| P1 | fouling disabled and disclosed | **PASS** |
| P2 | Ceto excluded from the HM prior (no heading, no STW) | **PASS** |
| P3 | selected geometry interior to the grid | **FAIL** — on all four bounds; resolved by `hm_grid_boundary_probe` as shape-limited |
| S1 | partition change vs folder 05 recorded; A2 control matches | **PASS** |

---

## 6. Recommendation for Section 5.4

Keep **Part A** as Table 7: the cube-law prior with the corrected feature set and the
intended wind-quantile shift. Report the Holtrop–Mennen attempt as a short paragraph plus
the boundary probe, concluding that the richer prior does not improve on the cube law
because the FuelCast target is total vessel fuel dominated by non-propulsive load, and that
this is why FuelCast remains a boundary analysis rather than a replication of the primary
benchmark.

Statements to update from the submitted manuscript: λ* is 1.00 / 1.00 / **0.44** (published
0.95 / 1.00 / 0.42); the primary-benchmark contrast reads λ* = 0.70, not 0.25; the residual
learner is Huber, not CatBoost; and for Ceto the claim that the prior "remains the most
accurate model under shift" is false — C-GBM is 19.441 % against the prior's 21.899 %.

---

## 7. Folder contents

| File | Contents |
|---|---|
| `partA_manuscript_table_mape.csv` | **recommended Table 7** — MAPE and λ* |
| `partA_all_metrics.csv`, `partA_lambda_selection.csv`, `partA_lambda_curves_validation.csv` | Part A supporting data |
| `partA2_control_hs_split_*.csv` | feature-only control on the folder-05 split |
| `feature_set_diff_vs_05_fuelcast_huber.csv` | features recovered, λ* and C-GBM MAPE before/after |
| `partB_manuscript_table_mape.csv`, `partB_all_metrics.csv`, `partB_selection.csv`, `partB_lambda_curves_validation.csv` | Holtrop–Mennen prior results |
| `partB_geometry_grid_validation.csv` | 270-candidate geometry search |
| `partB_geometry_boundary_probe.csv`, `..._summary.csv` | 750-candidate extended probe with plausibility screen |
| `prior_comparison_cube_vs_hm.csv` | prior-alone and hybrid comparison |
| `endpoint_identity.csv`, `checks.csv` | verification |
| `run_metadata.json` | configuration, bug record, supersession |
| `s06_fuelcast_wind_and_wbm.py`, `hm_grid_boundary_probe.py` | generating scripts |

```bash
python -m experiments.revision.s06_fuelcast_wind_and_wbm
python -m experiments.revision.hm_grid_boundary_probe
```
