# Section 5.6 — multi-gas emission and CII propagation, verified

**Folder:** `results/revision_final/08_multigas_cii/`
**Script:** `experiments/revision/s08_multigas_cii.py`
**Generated:** 2026-08-17 · **Seed:** 42 · **Git commit:** `e125867a8f7d3908d9daf52ea4c4b0864bd09a06`
**Checks: 9 / 12 PASS.** The three FAILs are substantive findings, not defects — each is
reported below and none was patched away.

`main.tex` not modified. No existing folder overwritten. FuelCast not loaded.

---

## 1. Configuration

| Item | Value |
|---|---|
| Benchmark | Shifts marine, primary; in-domain and distribution-shifted benchmark partitions |
| Protocol | chronological 60/20/20 on the full `train.csv` |
| WBM | k · F_phys, **k = 0.393586** |
| BBM | **Huber fitted directly on measured shaft power** |
| S-GBM | WBM + full Huber residual (λ = 1) |
| C-GBM | WBM + **0.70** · Huber residual |
| λ* | **0.70, imported and frozen** — no grid, no argmin, not informed by any partition |
| Excluded | CatBoost, ExtraTrees, ridge-to-physics, bounded residual, λ = 0.25 |
| Fuel | MDO |
| Emission factors | CO₂ 3.206 t/t · CH₄ 0.010 kg/t · N₂O 0.160 kg/t |
| GWP₁₀₀ | CH₄ 29.8 · N₂O 273 (AR6) |
| Engine | MCR 45,000 kW, SFOC_MCR 170 g/kWh |

### Section 5.3 reproduction — PASS

`section53_consistency_check.csv`: 8/8 MAPE reproduce to ≤ 0.001 and match folder 01 to
< 1e-4. k = 0.393586, λ = 0.70 verified.

| Partition | WBM | BBM | S-GBM | C-GBM |
|---|---|---|---|---|
| in-domain | 8.848 | 18.820 | 8.788 | 8.238 |
| distribution-shifted | 9.214 | 21.930 | 9.545 | 8.732 |

---

## 2. Part A — power-to-fuel audit

`fuel_conversion_audit.csv`, `time_base_audit.csv`

**Chain, with units at every step**

| Step | Equation | In | Out |
|---|---|---|---|
| engine load | ℓ = P / MCR | kW / kW | – |
| SFOC | SFOC(ℓ) = 170 · (0.455ℓ² − 0.710ℓ + 1.280) / 1.025 | – | g/kWh |
| fuel rate | ṁ = SFOC(ℓ) · P / 10⁶ | g/kWh × kW | t/h |
| cumulative fuel | m = Σ ṁ_i · Δt | t/h × h | t |

The 1.025 divisor re-anchors the polynomial so SFOC(1) = SFOC_MCR. n = 18,368 observations
per partition. Median fuel rate 3.916 t/h (in-domain) and 2.614 t/h (shifted); median
engine load 0.506 and 0.318.

### The finding: Δt is assumed, not measured

`time_id` and `timeSinceDryDock` have **identical consecutive differences**, which fixes the
unit as **minutes**. The records are **not** uniformly spaced:

| Partition | Span | Gap median / mean / max (min) | Coverage if 1 min each |
|---|---|---|---|
| in-domain | 1,197.7 d | 26 / 93.9 / 38,763 | **1.06 %** |
| distribution-shifted | 201.0 d | 1 / 15.8 / 28,456 | **6.35 %** |

Three candidate integrations:

| Assumption | in-domain fuel (t) | shifted fuel (t) |
|---|---|---|
| **uniform 1 h per record** | **64,620.7** | **54,220.0** |
| uniform 1 min per record | 1,077.0 | 903.7 |
| gap to next record | 82,665.6 | 14,622.9 |

**The published totals are exactly the uniform-1-hour convention** — reproduced to 0.05 t
(check A3 PASS). But that convention matches neither the 1-minute time unit nor the observed
gaps, and the two partitions have different sampling density, so:

* **the old totals are numerically reproducible but not dimensionally defensible as physical
  cumulative fuel** (check A2 FAIL). They are Σ(rate × 1 h), a normalisation;
* **the two partitions' totals are not comparable with each other** — under gap-weighting
  the shifted total would be 3.7× smaller, not roughly equal;
* **ratios and attained CII survive**, because a uniform Δt cancels (checks R1, C1).

The measured-power reference is **measured shaft power passed through the identical
deterministic SFOC and emission chain**. It is *not* measured fuel; Shifts contains no fuel
measurement. Labelled as such in every output row.

---

## 3. Part B — multi-gas propagation

`multigas_results.csv`, `emission_factor_audit.csv`, `gas_composition_audit.csv`

| Gas | Factor | Unit | Equation | Load dependent | GWP₁₀₀ |
|---|---|---|---|---|---|
| CO₂ | 3.206 | t / t fuel | E = fuel × 3.206 | no | 1 |
| CH₄ | 0.010 | kg / t fuel | E = fuel × 0.010 × (1 + g·max(0, 0.5−ℓ)/0.5), **g = 0.0** | **no** | 29.8 |
| N₂O | 0.160 | kg / t fuel | E = fuel × 0.160 | no | 273 |

**CO₂e** = CO₂ + (29.8·CH₄ + 273·N₂O)/1000.

**Is proportionality real?** Yes, and it is now established rather than assumed. The CH₄
slip term *is* implemented as load dependent, but for MDO the gain `ch4_lowload_gain` is
**0.0**, so the slip factor collapses to a constant and all three gases are strictly
proportional to fuel (check B1 PASS). Consequently the composition is identical for every
series to machine precision — CO₂e/fuel spread across the 10 series is 3.9 × 10⁻¹⁶ t/t
(check B2 PASS). The old proportionality statement therefore stands, but only because
g = 0 for this fuel; it would fail for LNG, where `ch4_lowload_gain = 1.5`.

### Final table (uniform 1 h convention, n = 18,368 per partition)

| Partition | Series | Fuel (t) | CO₂ (t) | CH₄ (kg) | N₂O (kg) | CO₂e (t) |
|---|---|---|---|---|---|---|
| in-domain | Measured-power reference | 64,620.7 | 207,173.8 | 646.2 | 10,339.3 | 210,015.7 |
| in-domain | WBM | 65,549.5 | 210,151.8 | 655.5 | 10,487.9 | 213,034.5 |
| in-domain | BBM | 64,472.0 | 206,697.4 | 644.7 | 10,315.5 | 209,532.7 |
| in-domain | S-GBM | 65,554.0 | 210,166.3 | 655.5 | 10,488.6 | 213,049.2 |
| in-domain | C-GBM | 65,554.6 | 210,168.0 | 655.5 | 10,488.7 | 213,051.0 |
| shifted | Measured-power reference | 54,220.0 | 173,829.2 | 542.2 | 8,675.2 | 176,213.7 |
| shifted | WBM | 53,072.5 | 170,150.4 | 530.7 | 8,491.6 | 172,484.4 |
| shifted | BBM | 57,218.0 | 183,441.0 | 572.2 | 9,154.9 | 185,957.3 |
| shifted | S-GBM | 54,673.3 | 175,282.5 | 546.7 | 8,747.7 | 177,686.9 |
| shifted | C-GBM | 54,197.5 | 173,757.1 | 542.0 | 8,671.6 | 176,140.6 |

### Relative cumulative differences (`multigas_relative_errors.csv`)

Identical for fuel, CO₂, CH₄, N₂O and CO₂e, by proportionality. **Invariant to the interval
convention.**

| Partition | WBM | BBM | S-GBM | C-GBM |
|---|---|---|---|---|
| in-domain | +1.437 % | **−0.230 %** | +1.444 % | +1.445 % |
| shifted | −2.116 % | +5.529 % | +0.836 % | **−0.041 %** |

These are **relative differences of a propagated cumulative quantity against the
measured-power reference**. They are not gas-specific prediction errors: no stack
measurement of any gas exists in this dataset.

**A caution that must go in the text.** In domain the **BBM** has the smallest cumulative
difference (−0.230 %) despite by far the worst MAPE (18.820 % against C-GBM's 8.238 %). A
cumulative sum lets signed per-record errors cancel, so agreement of a fuel budget is not
evidence of predictive skill. Reporting the budget error without this caveat would invite
exactly the wrong inference.

---

## 4. Part C — CII

Full formula and input provenance: **`cii_formula_audit.md`**. Summary of the audit:

* attained CII = (CO₂ [t] × 10⁶) / (DWT × Σ stw·Δt) g CO₂/(DWT·nm);
* required CII = a·DWT^(−c) × (1 − Z), Z = 0.11 for 2026;
* DWT = **90,813.766 t**, derived as an assumed-hull displacement of 110,748.5 t × a
  deadweight coefficient of 0.82 — **doubly assumed**;
* distance is a **speed-through-water proxy**; the dataset has no distance or position
  channel;
* the accumulation period is **not a calendar year** — the partitions cover 1.06 % and
  6.35 % of their spans, so this is a sample-mean intensity, not a regulatory attained CII;
* **ship class is unknown** (check C2 FAIL); both reference lines are retained **only as a
  sensitivity analysis**, never as a classification of the anonymised vessel;
* attained CII is **invariant to the uniform interval** — 20/20 values identical between the
  1 h and 1 min conventions (check C1 PASS).

Required CII: bulk carrier **3.4806**, container ship **6.6435**.

| Partition | Series | Attained CII | Bulk ratio / rating | Container ratio / rating |
|---|---|---|---|---|
| in-domain | Measured-power ref. | 7.1866 | 2.0648 / **E** | 1.0818 / **D** |
| in-domain | WBM | 7.2899 | 2.0944 / E | 1.0973 / D |
| in-domain | BBM | 7.1701 | 2.0600 / E | 1.0793 / D |
| in-domain | S-GBM | 7.2904 | 2.0946 / E | 1.0974 / D |
| in-domain | C-GBM | 7.2905 | 2.0946 / E | 1.0974 / D |
| shifted | Measured-power ref. | 6.7500 | 1.9393 / **E** | 1.0160 / **C** |
| shifted | WBM | 6.6072 | 1.8983 / E | 0.9945 / C |
| shifted | BBM | **7.1233** | 2.0466 / E | **1.0722 / D** |
| shifted | S-GBM | 6.8065 | 1.9555 / E | 1.0245 / C |
| shifted | C-GBM | 6.7472 | 1.9385 / E | 1.0156 / C |

**New finding — architecture does change a rating** (check C3 PASS). On the shifted
partition under the container-ship line the **BBM lands on D** while the measured-power
reference and all three physics-anchored architectures land on **C**. Distance is identical
across series (it depends only on measured speed), so the rating change is driven purely by
the CO₂ difference. In the other three type × partition combinations no architecture changes
the rating.

---

## 5. Part D — should uncertainty be propagated here?

1. **Is it already implemented correctly?** Partly. `vessel_fuel.emissions` provides
   `co2e_interval`, and Section 5.5 now supplies verified split-conformal radii for all four
   frozen architectures. Nothing in this folder propagates them, and the previous
   probabilistic-CII machinery was built on obsolete models. So: the pieces exist, the
   composition does not.
2. **Is it scientifically useful in the main paper?** For CO₂e, only marginally. Because
   gases are strictly proportional to fuel and the SFOC map is deterministic and monotone in
   power, a conformal interval on power maps to a CO₂e interval by a fixed transformation —
   it adds no information beyond Section 5.5's interval, restated in different units. It
   would be a units conversion presented as a result.
3. **Can it be derived consistently?** Yes for a **per-record** CO₂e interval: push
   `prediction ± q` through the SFOC and emission chain. **No** for a defensible interval on
   *cumulative* CO₂e or on attained CII, because that requires aggregating interval
   endpoints across 18,368 records whose temporal dependence is unmodelled, on top of a
   cumulative quantity that §2 has already shown is not a physical total. Doing it naively
   would produce a confident-looking number resting on two unjustified steps.
4. **Recommendation.** Keep the **per-record CO₂e interval as one sentence plus the
   transformation** in the main text, pointing back to Section 5.5 for the interval itself.
   Move any cumulative or CII interval to **future work**, conditional on a dataset with a
   real time base. Do **not** report probabilistic CII ratings in this revision — the old
   manuscript already withdrew them, and that withdrawal is correct.

---

## 6. Part E — old versus recomputed

`old_vs_recomputed_results.csv`, 20 entries.

| Old claim | Recomputed | Verdict |
|---|---|---|
| Measured-power ref. fuel, in-domain 64,620.7 t | 64,620.65 | **PASS** |
| Measured-power ref. fuel, shifted 54,220.0 t | 54,219.98 | **PASS** |
| WBM fuel, in-domain 64,030.7 t | 65,549.53 | **FAIL** |
| BBM fuel, in-domain 64,680.0 t | 64,472.04 | **FAIL** |
| S-GBM fuel, in-domain 64,754.4 t | 65,554.05 | **FAIL** |
| C-GBM fuel, in-domain 64,209.9 t | 65,554.59 | **FAIL** |
| WBM fuel, shifted 51,875.5 t | 53,072.48 | **FAIL** |
| BBM fuel, shifted 54,471.1 t | 57,218.03 | **FAIL** |
| S-GBM fuel, shifted 54,119.6 t | 54,673.26 | **FAIL** |
| C-GBM fuel, shifted 52,440.0 t | 54,197.48 | **FAIL** |
| CO₂ share ≈ 98.647 % | 98.646822 % | **PASS** |
| N₂O share ≈ 1.344 % | 1.344009 % | **PASS** |
| CH₄ share ≈ 0.009 % | 0.009169 % | **PASS** |
| CO₂e ≈ 3.250 t per t fuel | 3.249978 | **PASS** |
| C-GBM fuel error, in-domain −0.64 % | **+1.445 %** | **FAIL** |
| C-GBM fuel error, shifted −3.28 % | **−0.041 %** | **FAIL** |
| All bulk ratings E, in-domain | E | **PASS** |
| All container ratings D, in-domain | D | **PASS** |
| All bulk ratings E, shifted | E | **PASS** |
| All container ratings C, shifted | **C / D** | **FAIL** |

**Why each failure occurs.**

* The **measured-power reference reproduces exactly**, which localises the cause: the SFOC
  chain, the emission factors and the 1-hour convention are unchanged from the published
  implementation. Only the model predictions moved.
* Every **model** fuel total and the two **C-GBM budget errors** fail because the
  architectures changed: the old numbers used the CatBoost/HistGB-era learner and λ = 0.25,
  the current ones use Huber and λ = 0.70. These are different models; different totals are
  the expected consequence, not a discrepancy.
* The **container-ship rating on the shifted partition** fails because the current BBM
  over-predicts CO₂ by +5.53 % there, enough to cross the C/D boundary at ratio 1.07. The
  old BBM did not. This is a real change in the result, and it strengthens rather than
  weakens the section.
* All four **composition** claims reproduce, because they follow from the emission factors
  alone and those did not change.

---

## 7. Terminology

Throughout these outputs: **in-domain benchmark partition** and **distribution-shifted
benchmark partition**. Where the nature of the shift is discussed it is described as the
benchmark's **compound operational and environmental distribution shift**. The phrases
"wind-shifted condition" and "controlled wind shift" are not used.

---

## 8. Minimum defensible content for Section 5.6

**Retain.**

1. The emission-factor table with the explicit statement that `ch4_lowload_gain = 0` for
   MDO, so all gases are proportional to fuel and the composition is identical across
   architectures **by construction** — pre-empting the obvious referee question.
2. The **relative** cumulative differences (invariant to the interval), with the caveat that
   the BBM's small in-domain budget error coexists with the worst MAPE, so budget agreement
   is not predictive skill.
3. The CII table as a **two-class sensitivity analysis**, with the BBM's C→D rating change
   on the shifted partition as the decision-relevant result.
4. One paragraph on the time base: interval assumed, ratios and attained CII invariant,
   absolute tonnes not a physical total.

**Change or remove.**

5. **Absolute cumulative tonnes**: either drop the column or relabel it explicitly as
   "propagated fuel at a nominal 1 h per record" and stop describing it as cumulative
   consumption. Do not compare the two partitions' totals.
6. All old numeric values for the model rows and the two C-GBM budget errors are obsolete.
7. "All container ratings are C on the shifted partition" is now false.
8. Any probabilistic CII result stays out (Part D).
9. State that attained CII here is a sample-mean intensity over non-contiguous records, not
   a regulatory annual CII, and that DWT and ship class are assumptions.

---

## 9. Folder contents

| File | Contents |
|---|---|
| `section53_consistency_check.csv` | 8-row reproduction check |
| `time_base_audit.csv` | span, gaps, coverage, fuel-rate medians per partition |
| `fuel_conversion_audit.csv` | three integration assumptions × 2 partitions, with totals and distances |
| `emission_factor_audit.csv` | factor, unit, equation, load dependence, GWP, source per gas |
| `multigas_results.csv` | final fuel/CO₂/CH₄/N₂O/CO₂e table with series classification |
| `gas_composition_audit.csv` | per-series shares and CO₂e per tonne fuel |
| `multigas_relative_errors.csv` | relative cumulative differences vs the measured-power reference |
| `cii_results.csv` | attained/required CII and rating, 5 series × 2 types × 2 partitions |
| `cii_interval_invariance.csv` | 20-row proof that attained CII is interval-invariant |
| `cii_formula_audit.md` | formula, inputs, provenance, defensibility |
| `old_vs_recomputed_results.csv` | 20-entry PASS/FAIL audit of the old section |
| `checks.csv`, `run_metadata.json` | verification and configuration |
| `s08_multigas_cii.py` | complete reproducible script |

```bash
python -m experiments.revision.s08_multigas_cii
```
