# Section 5.5 — decision-level performance and predictive uncertainty

**Folder:** `results/revision_final/07_decision_uncertainty/`
**Script:** `experiments/revision/s07_decision_uncertainty.py`
**Generated:** 2026-08-17 · **Seed:** 42 · **Git commit:** `e125867a8f7d3908d9daf52ea4c4b0864bd09a06`
**Verification: 9 / 9 PASS.** Nothing overwritten; no `.tex` file modified; FuelCast not used.

---

## 1. Inputs and model definitions

| Item | Value |
|---|---|
| Benchmark | **Shifts marine (primary)**; `data/external/shifts_vessel/v2/.../real_data/train.csv`, `dev_in.csv`, `dev_out.csv` |
| Reference configuration | `results/revision_final/01_full_data_602020/` (`final_dev_in_dev_out.csv` used for the consistency check) |
| Protocol | chronological 60/20/20 on the full file: train 318,423 · validation 106,141 · internal test 106,142; dev_in 18,368; dev_out 18,368 |
| WBM | k · F_phys, **k = 0.393586** fitted on the training block |
| BBM | **Huber fitted directly on measured shaft power** |
| S-GBM | k · F_phys + f_Huber(x) — λ = 1 |
| C-GBM | k · F_phys + **0.70** · f_Huber(x) |
| λ* | **0.70, imported from Section 5.2 and frozen** |
| Excluded | CatBoost, ExtraTrees, ridge-to-physics, bounded residual |

**Huber has two distinct roles**, both used here: it is the standalone BBM (fitted on the
measured target *y*) and it is the residual learner inside S-GBM and C-GBM (fitted on
y − k·F_phys). These are two separate fits of the same estimator, never the same model
reused.

**Confirmation that λ was not re-selected.** λ = 0.70 is a module constant. There is no
grid, no sweep and no argmin over λ anywhere in this script. It was not informed by dev_in,
dev_out, the internal test block, the decision experiment or the uncertainty experiment.
Check F2 asserts the value; check F1 shows the resulting predictions reproduce Section 5.3
exactly, which is only possible if λ is unchanged.

---

## 2. Section 5.3 consistency check — PASS

`section53_consistency_check.csv`. All 12 rows within tolerance (MAPE ≤ 0.001,
R² ≤ 0.0001, ΔMAPE ≤ 0.001), and every MAPE also matches folder 01 to < 1e-4.

| Partition | Architecture | MAPE recomputed / 5.3 | R² recomputed / 5.3 |
|---|---|---|---|
| dev_in | WBM | 8.848 / 8.848 | 0.9392 / 0.9392 |
| dev_in | BBM | 18.820 / 18.820 | 0.9156 / 0.9156 |
| dev_in | S-GBM | 8.788 / 8.788 | 0.9575 / 0.9575 |
| dev_in | C-GBM | 8.238 / 8.238 | 0.9553 / 0.9553 |
| dev_out | WBM | 9.214 / 9.214 | 0.9110 / 0.9110 |
| dev_out | BBM | 21.930 / 21.930 | 0.8045 / 0.8045 |
| dev_out | S-GBM | 9.545 / 9.545 | 0.9383 / 0.9383 |
| dev_out | C-GBM | 8.732 / 8.732 | 0.9366 / 0.9366 |

ΔMAPE: WBM **+0.366**, BBM **+3.110**, S-GBM **+0.757**, C-GBM **+0.495** pp — all four
reproduce. λ verified as exactly 0.70.

---

## 3. Decision protocol — and one point that needed a decision

The paper's Section 4.3 fits the architectures on the **interquartile speed band of the
training data** so that the low- and high-speed cells used for evaluation are speeds the
model never saw; that is what makes the measured outcomes genuine counterfactuals. Using
the fully-trained Section 5.3 models would destroy that property, because those speeds are
in their training data.

Both are therefore reported, clearly labelled:

* **Primary — `mid-band fit (published)`.** Architectures refitted on the training
  interquartile speed band **[14.70, 19.18] kn**, n = 159,216, k = 0.406155. λ stays 0.70.
  This is the published protocol and the counterfactual claim holds.
* **Sensitivity — `full-training fit`.** The Section 5.3 models evaluated on the same
  decision groups. **Not counterfactual**; included only to show how much the conclusion
  depends on the fitting window.

Everything else is imported from `regret_core` so it is provably the published code:
economics (SFOC 170 g/kWh, fuel 600 USD/t, charter 15,000 USD/day), condition bins
(draught tercile × wind tercile), speed cells (1 kn, ≥ 5 records, ≥ 4 cells per bin), and
the unit cost `c(v) = (P·SFOC·p_fuel + p_charter/24)/v`.

Decisions are scored **only against measured power**: within each speed cell the measured
median power gives the realised cost, the model's role is limited to choosing which cell,
and regret is 100·(J_meas[chosen] − min J_meas)/min J_meas.

**Bootstrap:** B = 2000 paired replicates. The resampling unit is the rows inside a speed
cell, drawn with replacement at the cell's own size, and the **same** resample is applied
to the measured curve and to every architecture, so comparisons are paired. Note that this
resamples the counterfactual as well as the predictions, which is why the intervals are
wide.

---

## 4. Decision results — primary (mid-band fit)

`decision_results.csv`

| Partition | Architecture | Groups | Mean regret (%) | 95 % CI | Median | p90 | Match rate | Agreement with WBM | Groups differing |
|---|---|---|---|---|---|---|---|---|---|
| dev_in | WBM | 9 | **2.757** | [0.90, 15.86] | 0.000 | 10.129 | 0.556 | 1.000 | 0 |
| dev_in | BBM | 9 | 32.509 | [26.12, 48.48] | 42.524 | 65.363 | 0.333 | 0.444 | 5 |
| dev_in | S-GBM | 9 | 5.282 | [0.96, 16.08] | 0.000 | 18.626 | 0.556 | 0.667 | 3 |
| dev_in | C-GBM | 9 | 5.282 | [0.90, 16.01] | 0.000 | 18.626 | 0.556 | 0.667 | 3 |
| dev_out | WBM | 9 | 15.749 | [6.14, 27.68] | 0.000 | 50.739 | 0.556 | 1.000 | 0 |
| dev_out | BBM | 9 | 52.302 | [37.10, 58.41] | 45.570 | 98.466 | 0.222 | 0.333 | 6 |
| dev_out | S-GBM | 9 | 9.508 | [5.24, 29.58] | 0.000 | 20.717 | 0.556 | 0.667 | 3 |
| dev_out | C-GBM | 9 | **8.194** | [5.58, 28.07] | 0.000 | 20.717 | **0.667** | 0.778 | 2 |

**Shrinkage changes no in-domain decision.** On dev_in the C-GBM and S-GBM mean, median,
p90 and match rate are identical to four decimals, and the paired difference is exactly
0.0000 pp. With 1-kn cells and 9 groups, scaling the residual by 0.70 never moved an
argmin in domain. On dev_out it moved two, and those two are what separate 8.194 from
9.508.

**WBM–C-GBM agreement:** 0.667 on dev_in (3 of 9 groups differ), 0.778 on dev_out (2 of 9
differ). Per-group choices, selected speeds and regrets: `decision_group_details.csv`.

### Sensitivity (full-training fit, not counterfactual)

| Partition | WBM | BBM | S-GBM | C-GBM |
|---|---|---|---|---|
| dev_in | 2.757 | 32.509 | 7.303 | 7.077 |
| dev_out | 15.749 | 52.302 | 17.472 | 9.508 |

The WBM and BBM rows are unchanged (their fits do not depend on the band in a way that
alters any pick). The grey boxes get **worse** in domain when fitted on the whole training
block, and on dev_out the S-GBM degrades sharply (9.508 → 17.472) while C-GBM does not.
The direction of the C-GBM-versus-WBM comparison flips between protocols on dev_in, which
is itself a reason to treat the decision evidence as fragile.

---

## 5. Paired bootstrap — primary

`decision_pairwise_bootstrap.csv`

| Partition | Comparison | Δ mean regret (pp) | 95 % CI | p | Resolved |
|---|---|---|---|---|---|
| dev_in | C-GBM − WBM | +2.525 | [−5.37, +5.58] | 0.848 | **no** |
| dev_in | C-GBM − BBM | −27.227 | [−43.27, −15.54] | 0.0005 | **yes** |
| dev_in | C-GBM − S-GBM | 0.000 | [−3.94, +5.09] | 1.000 | **no** |
| dev_out | C-GBM − WBM | −7.555 | [−14.46, +11.88] | 0.692 | **no** |
| dev_out | C-GBM − BBM | −44.108 | [−48.69, −19.36] | 0.0005 | **yes** |
| dev_out | C-GBM − S-GBM | −1.315 | [−8.38, +6.00] | 0.654 | **no** |

**Two of six resolve, both against the BBM.** No comparison of C-GBM against the
calibrated physics model or against the unconstrained hybrid is statistically resolved on
either partition.

---

## 6. Uncertainty results

`uncertainty_results.csv`, `conformal_calibration_details.csv`

Calibration uses the **validation block only** (n = 106,141). No dev_in or dev_out record
enters the calibration. The radius is computed once per architecture and then applied
unchanged — verified by checks L1 and L2.

The quantile level is the finite-sample split-conformal value ⌈(n+1)(1−α)⌉/n rather than
the plain (1−α) quantile used previously. Both are reported: the largest resulting change
in any radius is **0.24 kW** (check L3), so the correction is immaterial here. It is
documented rather than applied silently.

| Partition | Architecture | n | Coverage | Coverage error | Mean width (kW) | Radius (kW) |
|---|---|---|---|---|---|---|
| internal test | WBM | 106,142 | 0.9272 | +0.0272 | 7203.5 | 3615.77 |
| internal test | BBM | 106,142 | 0.8277 | −0.0723 | 5755.9 | 2986.82 |
| internal test | S-GBM | 106,142 | 0.8358 | −0.0642 | 5162.1 | 2581.15 |
| internal test | C-GBM | 106,142 | 0.8715 | −0.0285 | 5645.9 | 2823.35 |
| dev_in | WBM | 18,368 | **0.9188** | +0.0188 | 7200.8 | 3615.77 |
| dev_in | BBM | 18,368 | 0.8658 | −0.0342 | 5728.6 | 2986.82 |
| dev_in | S-GBM | 18,368 | 0.8855 | −0.0145 | 5160.4 | 2581.15 |
| dev_in | C-GBM | 18,368 | **0.8997** | **−0.0003** | 5642.9 | 2823.35 |
| dev_out | WBM | 18,368 | **0.9296** | +0.0296 | 7202.5 | 3615.77 |
| dev_out | BBM | 18,368 | **0.7721** | −0.1279 | 5767.5 | 2986.82 |
| dev_out | S-GBM | 18,368 | 0.9086 | +0.0086 | 5162.3 | 2581.15 |
| dev_out | C-GBM | 18,368 | 0.9235 | +0.0235 | 5646.7 | 2823.35 |

### Answers (`uncertainty_questions.csv`)

**1. Any architecture ≥ 0.90 on both held-out partitions?** **Only the WBM**
(0.9188 and 0.9296). C-GBM reaches 0.8997 on dev_in — short of nominal by 3 parts in
10,000 — and 0.9235 on dev_out, so it fails the strict test on a margin that is smaller
than the resolution at which coverage is worth quoting. S-GBM and BBM fail clearly.

**2. Coverage change dev_in → dev_out.** WBM **+0.0107**, S-GBM **+0.0231**,
C-GBM **+0.0238**, BBM **−0.0937**. Coverage *improves* under shift for all three
physics-anchored architectures and collapses only for the pure black box.

**3. Does C-GBM preserve nominal calibration under shift?** Not in the strict sense. It is
essentially exactly nominal in domain (−0.0003) and becomes **conservative** under shift
(+0.0235), so its intervals remain trustworthy in the sense that matters operationally —
they do not under-cover — but the nominal level is not preserved, it is exceeded. Stability
across the shift should not be reported as preserved calibration.

**4. Does better MAPE imply better coverage?** **No.** On dev_in the two orderings happen
to agree, but on dev_out the best MAPE is C-GBM (8.732) while the smallest coverage error
is S-GBM (+0.0086 against C-GBM's +0.0235), and the WBM has the *worst* MAPE of the three
grey/white models yet the only coverage above nominal on both partitions. Point accuracy
and interval validity are distinct properties.

**5. Are widths effectively fixed?** **Yes.** Relative width change from dev_in to
dev_out is below 1 % for every architecture (WBM +1.7 kW, S-GBM +1.9, C-GBM +3.8, BBM
+39.0 kW on a 5729 kW base = 0.68 %). The radius is frozen, so the intervals cannot widen
adaptively. **Coverage differences between partitions therefore reflect changes in the
prediction-error distribution, not adaptive interval widening.**

**Conformal interpretation.** Split conformal guarantees marginal coverage only under
exchangeability between calibration and evaluation data. This protocol deliberately
violates it: the calibration block precedes the evaluation sets in time, and dev_out is a
compound distribution shift. The numbers above therefore characterise the **empirical
transfer of a frozen calibration** to new conditions. No distribution-free guarantee is
claimed under shift.

---

## 7. Old versus recomputed

`old_vs_recomputed_results.csv`. Old values were treated purely as quantities to check.

| Quantity | Reproduces? | Detail |
|---|---|---|
| WBM regret, dev_in | **PASS** | 2.757 vs 2.757 (difference −0.000) |
| WBM regret, dev_out | **PASS** | 15.749 vs 15.749 (difference −0.000) |
| BBM regret | DIFFERS | dev_in 39.576 → 32.509; dev_out 26.384 → **52.302** |
| S-GBM regret | DIFFERS | dev_in 6.317 → 5.282; dev_out 14.818 → 9.508 |
| C-GBM regret | DIFFERS | dev_in 2.916 → 5.282; dev_out 14.047 → 8.194 |
| All 8 coverage values | DIFFERS | e.g. WBM dev_in 0.832 → 0.919; C-GBM dev_out 0.848 → 0.923 |

**Why, precisely.**

* The **WBM rows reproduce exactly**, which is the strongest available evidence that the
  decision protocol implementation is unchanged: the WBM depends only on the physics core
  and a scale factor, so it is the one architecture whose decisions the model-selection
  round could not alter.
* Every **learned** architecture differs because the learner changed. The old numbers were
  produced with the HistGB/CatBoost-era representative learner; the current BBM, S-GBM and
  C-GBM all use Huber. The old C-GBM additionally used **λ = 0.25**, not 0.70. Those are
  different models, so different regret is the expected result, not a discrepancy.
* Every **coverage** value differs because the calibration set changed. The old radii came
  from the 40,000-record protocol's 15 % validation tail (≈ 6,000 records); the current
  radii come from the 106,141-record validation block of the 60/20/20 protocol. A larger,
  differently-distributed calibration set gives a different quantile, and the coverage
  follows. The old WBM coverage of 0.832 versus 0.919 now is entirely attributable to
  this, since the WBM predictions themselves are essentially unchanged.

**No implementation error was found in the published bootstrap or binning.** The exact
reproduction of both WBM regret values, using economics and binning imported from
`regret_core` rather than reimplemented, indicates the previous machinery was correct; the
differences are attributable to the model configuration and the calibration set, both of
which changed deliberately.

---

## 8. Verification checks — 9 / 9 PASS

`checks.csv`

| ID | Check | Verdict |
|---|---|---|
| F1 | Section 5.3 MAPE, R² and ΔMAPE reproduced | **PASS** 12/12 |
| F2 | λ exactly 0.70, never re-selected | **PASS** |
| F3 | no obsolete learner or constraint | **PASS** Huber only |
| L1 | conformal radius calibrated on the validation block only | **PASS** n = 106,141 |
| L2 | radius frozen before dev evaluation | **PASS** |
| L3 | finite-sample quantile change immaterial | **PASS** max 0.24 kW |
| D1 | decisions scored against measured power only | **PASS** |
| D2 | published economics and binning imported, not reimplemented | **PASS** |
| D3 | decision-group count recorded as an evidence limit | **PASS** 9 groups |

---

## 9. Limitations of the decision experiment

1. **Nine decision groups per partition.** Four of six paired comparisons do not resolve,
   and the bootstrap intervals span an order of magnitude. This is a diagnostic of
   decision sensitivity, not evidence of operational superiority.
2. **The bootstrap resamples the counterfactual.** Because the measured curve is resampled
   along with the predictions, the intervals include uncertainty in the ground truth
   itself. That is the right thing to do, and it is why the CIs are wide.
3. **Protocol dependence.** The C-GBM-versus-WBM ordering on dev_in flips between the
   mid-band and full-training fits. A conclusion that depends on the fitting window is not
   robust.
4. **Coarse action space.** Speeds are binned to 1 kn, so a shrinkage of 0.70 versus 1.0
   is invisible in domain: it changed no argmin at all on dev_in.
5. **Single vessel, single corridor.** dev_out is one compound shift on one anonymised
   vessel.

---

## 10. Defensible interpretation

The prediction-level differences in Section 5.3 **do not** translate into statistically
resolved differences in decision quality, except against the pure black box. The black box
is unambiguously the worst decision-maker (32.5 % and 52.3 % mean regret, match rate 0.33
and 0.22, and a 90 % interval that covers only 77 % of outcomes under shift) and both
resolved comparisons are against it. Among the three physics-anchored architectures, the
calibrated white-box model gives the lowest in-domain regret while the constrained hybrid
gives the lowest shifted regret and the highest match rate (0.667), but neither difference
is resolved, and the in-domain ordering is not stable across fitting protocols.

On uncertainty, the honest summary is that a frozen calibration transfers better when the
model is anchored to physics: the three anchored architectures keep or improve coverage
across the shift while the black box loses 9.4 points. Only the WBM holds ≥ 0.90 on both
partitions, and it does so by carrying the widest interval (7201 kW against C-GBM's
5643 kW, 27.6 % wider). C-GBM achieves the smallest in-domain coverage error of the four
(−0.0003) at a 22 % narrower interval than the WBM, which is a real efficiency result and
is what the section should report — not "preserved nominal coverage".

---

## 11. Folder contents

| File | Contents |
|---|---|
| `section53_consistency_check.csv` | 12-row reproduction check against Section 5.3 and folder 01 |
| `decision_results.csv` | both protocols × 2 partitions × 4 architectures, with CI, median, p90, match rate, WBM agreement |
| `decision_pairwise_bootstrap.csv` | 12 paired comparisons (6 per protocol) |
| `decision_group_details.csv` | per-group speed grid, measured optimum, each architecture's choice and regret |
| `uncertainty_results.csv` | coverage, error, width, radius on internal test, dev_in, dev_out |
| `conformal_calibration_details.csv` | calibration set, quantile level, both radii, freeze flags |
| `uncertainty_questions.csv` | machine-evaluated answers to questions 1–5 |
| `old_vs_recomputed_results.csv` | 16-row old-versus-new comparison with PASS/DIFFERS |
| `checks.csv`, `run_metadata.json` | verification and full configuration |
| `s07_decision_uncertainty.py` | complete reproducible script |

```bash
python -m experiments.revision.s07_decision_uncertainty
```
