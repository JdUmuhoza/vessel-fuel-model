# CII formula and input audit

Independent audit of the attained-CII calculation as implemented, with the provenance of
every input. Source of the constants and helper functions:
`experiments/benchmark_measured/run_cii_probabilistic.py`; recomputation:
`experiments/revision/s08_multigas_cii.py`.

---

## 1. Implemented formula

**Attained operational CII**

```
CII_attained = (E_CO2 [t] x 1e6) / (DWT [t] x D [nm])            [g CO2 / (DWT-t . nm)]
```

with

```
E_CO2 = sum_i  fuel_i x 3.206                                     [t]
fuel_i = SFOC(l_i) x P_i / 1e6 x dt                               [t]
D      = sum_i  stw_i x dt                                        [nm]     (kn x h = nm)
```

**Required CII (reference line)**

```
CII_ref(DWT)      = a x DWT^(-c)
CII_required(year)= CII_ref x (1 - Z_year)
```

**Rating boundaries** — the attained/required ratio is compared with four
type-specific boundaries `d1..d4`; A is below `d1`, B in `[d1, d2)`, C in `[d2, d3)`,
D in `[d3, d4)`, E at or above `d4`.

---

## 2. Inputs, values, and where each comes from

| Input | Value used | Provenance | Status |
|---|---|---|---|
| CO2 mass | Σ fuel × 3.206 t/t | Fourth IMO GHG Study MDO factor, applied to the propagated fuel trajectory | derived, auditable |
| Fuel | SFOC(ℓ)·P/1e6·Δt | quadratic SFOC load curve, SFOC_MCR = 170 g/kWh, MCR = 45,000 kW | derived from **assumed** engine particulars |
| Δt (interval) | **uniform 1 h per record, by convention** | not measured; see §4 | **assumption** |
| Distance D | Σ stw·Δt | speed through water, not speed over ground | see §5 |
| DWT | **90,813.766 t** | `L·B·T·C_B·ρ_sw/1000 = 110,748.5 t` displacement × deadweight coefficient **0.82** | **doubly assumed** (hull is itself assumed) |
| Reference line, bulk carrier | a = 4745.0, c = 0.622 | MARPOL Annex VI / MEPC.353(78) | correct for the type |
| Reference line, container ship | a = 1984.0, c = 0.489 | MARPOL Annex VI / MEPC.353(78) | correct for the type |
| Rating boundaries, bulk | d = 0.86, 0.94, 1.06, 1.18 | MEPC.354(78) | correct for the type |
| Rating boundaries, container | d = 0.83, 0.94, 1.07, 1.19 | MEPC.354(78) | correct for the type |
| Reduction factor Z | **0.11** (2026) | MARPOL Annex VI reg. 28 schedule: 5/7/9/11 % for 2023–2026 | correct, and uniform across types |
| Ship class | **unknown** | vessel anonymised by the Shifts benchmark | **not established** |

Required CII at DWT = 90,813.766 t and Z = 0.11:
**bulk carrier 3.4806**, **container ship 6.6435** g CO₂/(DWT·nm).

---

## 3. Does the Shifts dataset support a defensible attained CII?

**Partly — and not as a regulatory quantity.** Three separate problems, of different
severity.

**(a) Transport work is proxied, not measured.** The regulation defines the denominator as
deadweight × distance *travelled*. The dataset provides no distance channel and no
position, so distance is reconstructed as Σ stw·Δt. This uses **speed through water**, so
the reconstructed distance is the distance through the water column, not over ground; the
two differ by the along-track current integral. Using SOG would be closer to the
regulatory intent, but `diff_speed_overground` in Shifts is a differenced quantity, not an
absolute SOG, so the through-water proxy is the only one available.

**(b) The accumulation period is not a year.** Regulation 28 defines the attained CII over
a **calendar year** of operation. These partitions are non-contiguous samples covering
1.06 % (in-domain) and 6.35 % (distribution-shifted) of their own calendar spans (§4). The
computed quantity is therefore a **sample-mean carbon intensity over the retained records**,
not an annual attained CII.

**(c) Deadweight and class are assumed.** DWT is derived from an assumed hull, itself
chosen as representative for the vessel class, and then multiplied by an assumed
deadweight coefficient of 0.82 taken from the bulk-carrier range. The ship type is not
published. A pre-existing inconsistency is already disclosed in the source module: a
**bulk-carrier** deadweight coefficient is applied to a hull whose own reference-line check
does not indicate a bulk-carrier form.

**Consequence for the manuscript.** The two-class calculation must be presented strictly as
a **sensitivity analysis over an unknown reference line**, never as a regulatory
classification of the anonymised vessel. That is how it is reported in `cii_results.csv`
and in the README.

---

## 4. The interval problem, and what survives it

`time_id` and `timeSinceDryDock` have **identical consecutive differences**, which fixes
the time unit as **minutes**. The records are not uniformly spaced:

| Partition | n | Span | Gap median / mean / max | Coverage if 1 min each |
|---|---|---|---|---|
| in-domain | 18,368 | 1,197.7 d | 26 / 93.9 / 38,763 min | 1.06 % |
| distribution-shifted | 18,368 | 201.0 d | 1 / 15.8 / 28,456 min | 6.35 % |

Three candidate integrations give totals differing by nearly two orders of magnitude
(`fuel_conversion_audit.csv`). The published totals correspond exactly to a **uniform 1 h
per record**.

What this does and does not invalidate:

* **Absolute cumulative tonnes: not defensible** as a physical total. They are
  `Σ rate × 1 h`, a normalisation, not an integration over measured time. The two
  partitions also have different sampling density, so their totals are not comparable with
  each other.
* **Attained CII: invariant.** Both numerator and denominator scale linearly in a *uniform*
  Δt, so Δt cancels exactly. Verified numerically: all 20 attained-CII values are identical
  between the 1 h and 1 min conventions (`cii_interval_invariance.csv`). Attained CII is
  therefore well defined here as a sample-mean intensity, despite the interval ambiguity.
* **Relative differences between series: invariant.** A uniform Δt multiplies every series
  identically, so it cancels in any ratio against the measured-power reference.

---

## 5. Verification results

| Check | Verdict |
|---|---|
| Attained CII invariant to a uniform interval | **PASS** 20/20 identical |
| Vessel class established by the benchmark | **FAIL** — anonymised; two-class sensitivity retained |
| Architecture changes at least one rating | **PASS** — see below |

**Architecture-dependent rating change (new finding).** On the distribution-shifted
partition under the container-ship reference line, the **BBM lands on rating D** (ratio
1.0722) while the measured-power reference, WBM, S-GBM and C-GBM all land on **C** (ratios
1.0160, 0.9945, 1.0245, 1.0156). Under the bulk-carrier line every series is E on both
partitions, and on the in-domain partition every series is D under the container line. So
one of four type × partition combinations shows a rating change, and it is driven by the
black box, not by the choice among the physics-anchored architectures.
