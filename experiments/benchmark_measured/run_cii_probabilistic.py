"""Probabilistic CII rating per architecture (Axis 3 decision layer, closing B2).

Closes a promise the paper makes twice and keeps nowhere else. Section
"Axis 3: uncertainty validity, CO2e, and compliance decisions" states that
a probabilistic CII rating, P(rating <= C) from the interval, "is reported
per architecture" -- no such table exists elsewhere in this repository. The
Conclusion further claims that architecture choice "sets how early in an
accumulation period a CII compliance rating can be called with confidence"
-- a quantitative claim with no supporting number anywhere in the paper.
This script produces both: results/cii_probabilistic.{csv,md} plus a
companion sensitivity sweep, results/cii_probabilistic_sweep.csv.

CO2-ONLY, NOT CO2e
------------------
The IMO Carbon Intensity Indicator regulates operational carbon intensity
in CO2 alone (``vessel_fuel.ghg_scope.gases_for("CII") == ("CO2",)``), the
single-gas subset of the multi-gas emission head's output. Every quantity
accumulated below is CO2 mass -- never CO2e, which would fold in CH4/N2O
through their GWP100 weights and compute a rating for a basket the CII
does not consume. (The Axis-3 prose that motivated this table talks about
"the CO2e interval" propagating to the rating; that phrasing is imprecise
for CII specifically and a later manuscript pass should tighten it -- see
this script's companion report.)

THE ANONYMISED-VESSEL PROBLEM -- read before trusting any number below
------------------------------------------------------------------------
The Shifts marine benchmark (Malinin et al., 2022) discloses only "one
merchant cargo vessel"; ship type and deadweight are withheld. The IMO
reference-line formula (required CII = a * DWT^(-c), coefficients keyed
by ship type) cannot be evaluated without both, so ``required_cii``
below is NOT this vessel's real regulatory requirement -- the inputs to
compute that requirement are simply not public. What is reported instead
is the consequence of EXPLICIT, STATED ASSUMPTIONS, carried in every
column header and in the results .md caption:

    ASSUMED ship type : BOTH bulk carrier and container ship, reported
                         side by side -- see "THE SHIP-TYPE PROBLEM"
                         below; this benchmark cannot identify which one
                         applies
    ASSUMED DWT       : derived from the vessel's own physics, not
                         chosen to make architectures differ -- see
                         "HOW THE CENTRAL DWT IS DERIVED" below
    ASSUMED year      : 2026 (current IMO reduction-factor phase; the
                         Shifts recording years pre-date the CII
                         regulation entirely, so this is an explicit
                         "if rated under today's schedule" hypothetical)

Never read ``required_cii``, ``rating``, or ``p_rating_le_C`` below as
this vessel's actual requirement or actual compliance status.

HOW THE CENTRAL DWT IS DERIVED (fix round 1 -- see task-13-report.md)
------------------------------------------------------------------------
An earlier version of this script picked DWT = 400,000 t because that is
where the five architectures happen to disagree -- a free parameter
tuned to manufacture the contrast the table exists to test, flagged by
the controller as the specific failure mode this project has a standing
rule against ("do not force the narrative"). This version derives the
central DWT from the SAME physics particulars this benchmark already
assumes for the Shifts vessel throughout (``CONFIG["particulars"]`` in
common.py: L=300 m, B=40 m, T=14.5 m, CB=0.62 -- the hull already used
for every ``physics_power()`` call in this package), via the one
relation that needs no ship-type regression: Archimedes' principle.

    displacement (t) = L * B * T * CB * rho_sw / 1000

using ``vessel_fuel.sw_density`` (UNESCO-83) at 12 degC, the same sea
surface temperature already hard-coded into ``physics_power()``'s
environment dict in common.py -- so the derivation reuses, rather than
adds to, this benchmark's existing assumptions. Deadweight is then
displacement times a deadweight coefficient (DWT / displacement), a
standard naval-architecture ratio. This module uses 0.82, the commonly
cited midpoint of the documented BULK-CARRIER range (~0.80-0.85);
fix round 2 (below) found the hull is more likely a container-ship
form, whose deadweight coefficient runs lower (~0.60-0.70), which would
lower DWT_CENTRAL_T further. The controller's fix-round-2 instruction
was explicit: keep the DWT derivation unchanged this round. This
residual inconsistency -- a bulk-carrier deadweight coefficient applied
to a hull the reference-line check below shows is not bulk-carrier-
shaped -- is disclosed here as an open item for future work, not
silently fixed.

This yields DWT_CENTRAL_T ~= 90,800 t (computed at runtime from
CONFIG["particulars"]; printed exactly at the top of every run) -- a
plausible mid-size bulk carrier (Panamax/Kamsarmax range) BY DISPLACEMENT
ALONE, motivated entirely by the vessel's own assumed hull, not by which
DWT makes the architectures disagree. Never read this, or any other
number in this file, as the anonymised vessel's real registered
particulars.

THE SHIP-TYPE PROBLEM (fix round 2 -- see task-13-report.md)
------------------------------------------------------------------------
Fix round 1 derived a central DWT from the vessel's own assumed hull,
but kept the bulk-carrier reference line inherited from
run_corridor_case_study.py without checking whether that hull is a bulk
carrier at all. It is not, on two independent checks against
``CONFIG["particulars"]`` and the vessel's own measured operating
profile (verified against ``benchmark_measured.common.load_shifts_splits()``
directly, not assumed):

    - CB = 0.62. Bulk carriers run roughly 0.80-0.85 (full, boxy hulls
      built to maximise cargo volume); container ships and other
      fine-formed cargo hulls run roughly 0.60-0.68 (slender hulls built
      for speed). 0.62 sits in the container-ship range, not the
      bulk-carrier one -- consistent with common.py's OWN comment
      on this dict, "representative large container-ship particulars",
      which this script had simply not reconciled with the reference
      line it applied.
    - Median operating point: 18.4 kn on 22.8 MW of shaft power
      (train/dev_in medians). A Panamax/Kamsarmax bulk carrier of
      comparable size runs roughly 14 kn on 8-10 MW; 18.4 kn on 22.8 MW
      is a fast, powerful profile characteristic of a container ship,
      not a bulk carrier.

Both checks point the same way, so the bulk-carrier reference line is
not defensible as the ONLY type assumption reported. This module now
carries BOTH plausible types side by side, using coefficients verified
(not guessed) against the primary IMO texts, fetched and read in full
on 2026-08-03:

    Reference line, CII_ref = a * DWT^(-c) -- International Maritime
    Organization (2021). Resolution MEPC.337(76), Annex 11, "2021
    Guidelines on the Reference Lines for Use with Operational Carbon
    Intensity Indicators (CII Reference Lines Guidelines, G2)", Table 1
    ("Parameters for determining the 2019 ship type specific reference
    lines"), page 3:
        bulk carrier   : a=4745.0,  c=0.622
        container ship : a=1984.0,  c=0.489

    Rating boundaries (applied to the attained/required ratio) --
    International Maritime Organization (2022). Resolution MEPC.354(78),
    Annex 16, "2022 Guidelines on the Operational Carbon Intensity
    Rating of Ships (CII Rating Guidelines, G4)", Table 1 ("dd vectors
    for determining the rating boundaries of ship types"), page 6:
        bulk carrier   : 0.86 / 0.94 / 1.06 / 1.18
        container ship : 0.83 / 0.94 / 1.07 / 1.19

The annual reduction-factor schedule (``CII_REDUCTION``: 5/7/9/11% for
2023-2026, MARPOL Annex VI regulation 28) was separately confirmed
uniform across ship types, so it needs no type-specific variant.

Both coefficient sets are carried through this script (``TYPE_PARAMS``
below). The main table reports every architecture-set combination under
EACH type -- TWENTY rows total, not ten, with the ``architecture``
column suffixed ``[bulk]``/``[container]`` so the fixed seven-column
interface (``architecture, set, attained_cii, required_cii, rating,
p_rating_le_C, months_to_confident_call``) needs no new column. The
sensitivity sweep gets a genuine ``ship_type`` column instead, since
that file is not interface-constrained.

THE FINDING: switching only the type assumption, holding the
physics-derived DWT and every other input fixed, moves EVERY
architecture on EVERY evaluation set from rating E (bulk carrier) to
rating C (container ship) -- a full two-notch swing, uniform across all
ten architecture-set combinations. This is a LARGER and much more
UNIFORM effect than the DWT sweep in fix round 1 produced (which needed
the assumed size to move by a factor of ~4-5x to change any rating at
all, and even then changed it non-monotonically and only for some
architectures, not all). The honest conclusion is not "architecture
choice changes the CII call"; it is that **the CII rating for this
anonymised vessel is not identifiable at all without first assuming a
ship type, and that single assumption moves the rating further than any
model architecture examined in this benchmark.**

Never read ``required_cii``, ``rating``, or ``p_rating_le_C`` below,
under either type, as this vessel's actual requirement or actual
compliance status.

REUSED HELPERS (per the controller's second resolution -- unchanged by
this fix round)
---------------------------------------------------------
- ``benchmark_measured.common``: ``CONFIG``, ``FEATS``, ``SEED``,
  ``MIN_PER_MONTH``, ``fit_scale``, ``load_shifts_splits``,
  ``physics_power`` -- identical loaders to every other script in this
  package.
- ``benchmark_measured.run_uq_emissions``: ``ALPHA``, ``MCR_KW``,
  ``power_to_fuel_mt_h`` -- the SAME split-conformal quantile
  construction published in ``results/uq_emissions.{csv,md}``
  (unweighted; see "why unweighted" below for why the covariate-shift
  weighted variant of that module is deliberately not reused here).
- ``vessel_fuel.fuel_to_co2e`` -- only its ``"co2_mt"`` key is read.
  ``vessel_fuel.sw_density`` -- the displacement derivation above.
  ``vessel_fuel.inference.predict_state``'s interval is explicitly NOT
  used (per the controller: an uncalibrated heuristic stand-in, not the
  interval the paper's uncertainty section publishes).

WHY THE UNWEIGHTED SPLIT INTERVAL, NOT THE WEIGHTED ONE
-----------------------------------------------------------
``results/uq_emissions.csv`` (Table "uq-coverage" in the manuscript)
reports both an unweighted split-conformal interval (calibrated once,
in-domain) and a covariate-shift-weighted variant retargeted at
dev_out. This module uses the SAME unweighted interval on BOTH
evaluation sets deliberately: the paper's own caveat is that an
interval calibrated in-domain silently under-covers under shift, so a
rating "called" from it is overconfident precisely when a
decision-support system needs caution most. Reusing the weighted
(shift-corrected) interval for dev_out would hide exactly that failure
mode. The unweighted interval's coverage on each set (already published
in ``results/uq_emissions.csv`` and Table "uq-coverage") is the numeric
backing for that caveat, repeated in the results .md written here.

ACCUMULATION-PERIOD CONSTRUCTION -- a second, disclosed simplification
------------------------------------------------------------------------
CII is an ANNUAL metric. Neither Shifts evaluation split is a single
continuous voyage-year: dev_in (18,368 records) is scattered across the
full ~4-year recording horizon (the benchmark's own in-domain holdout,
interleaved with train), dev_out spans a contiguous ~200-day
wind-shifted window. Reconstructing a literal calendar accumulation
from either would be either nonsensical (dev_in) or shorter than a
year (dev_out). Both are therefore normalised into the SAME nominal
one-year accumulation period, hour by hour, by reusing the convention
already implicit in ``run_ghg_breakdown.py`` / ``run_uq_emissions.py``
(every retained hourly record contributes one hour's worth of the rate
returned by ``power_to_fuel_mt_h``/``fuel_to_co2e``, with no further
time weighting) and truncating each set's time-sorted records to the
first ``round(12 * hours_per_month)`` of them. "Month t into the
accumulation period" is a NOMINAL month of retained-record hours, not
a calendar month -- the same simplification the rest of this benchmark
already relies on, not a new one introduced here.

CONFIDENCE-CALL CRITERION
--------------------------
At each nominal hour, the running attained CII and its standard error
are formed from the cumulative sum of per-hour CO2 point estimates and
the per-hour split-conformal half-width translated through the same
chain into CO2 mass, ASSUMING approximate hour-to-hour independence of
the conformal residual. That independence assumption is optimistic --
the source series is documented elsewhere in this benchmark to be
temporally autocorrelated -- so the reported ``months_to_confident_call``
should be read as an optimistic (early) bound on when a real,
autocorrelated residual series would resolve, not a conservative one.
P(rating <= C) = Phi((d3 * required_cii - attained_cii(t)) / SE(t))
using the standard normal CDF, the same normal-approximation convention
already used for ``p_compliant`` in
``experiments/deployment/run_corridor_case_study.py``, with ``d3`` and
``required_cii`` taken from whichever ship type is being evaluated. The
call is "confident" at the stated threshold ``P_CONF = 0.95`` once
P(rating<=C) >= 0.95 (confidently C-or-better) or <= 0.05 (confidently
worse than C) -- and only counted once that state PERSISTS through the
end of the nominal year (guards against one volatile early hour
flipping the call transiently). ``months_to_confident_call`` is NaN
when no such persistent crossing occurs inside the nominal year.

Run:  PYTHONPATH=. python experiments/benchmark_measured/run_cii_probabilistic.py
Outputs: experiments/benchmark_measured/results/cii_probabilistic.{csv,md},
         experiments/benchmark_measured/results/cii_probabilistic_sweep.csv
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark_measured.common import (  # noqa: E402
    CONFIG, FEATS, MIN_PER_MONTH, SEED,
    fit_scale, load_shifts_splits, physics_power,
)
from benchmark_measured.run_uq_emissions import (  # noqa: E402
    ALPHA, MCR_KW, power_to_fuel_mt_h,
)

from vessel_fuel import fuel_to_co2e, sw_density  # noqa: E402

# Written directly under results/ (checked into git), matching every other
# script in this package (run_uq_emissions.py, run_ghg_breakdown.py, ...).
# NOT common.OUT (outputs/benchmark_measured/, gitignored).
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "benchmark_measured" / "results"

MODELS = ["WBM", "BBM", "S-GBM", "C-GBM-lam0.25", "C-GBM-ridge"]

Z90 = 1.645  # two-sided 90% normal quantile; same convention as
             # experiments/deployment/run_corridor_case_study.py.

ASSUMED_YEAR = 2026  # ASSUMED "if rated under today's IMO schedule"

# ---- Central DWT, derived from the vessel's OWN assumed physics hull
# (CONFIG["particulars"]), not chosen to make architectures differ -- see
# "HOW THE CENTRAL DWT IS DERIVED" in the module docstring. Unchanged by
# fix round 2 per the controller's instruction.
_PART = CONFIG["particulars"]
_SST_C = 12.0  # same sea-surface temperature already used by physics_power()
DISPLACEMENT_T = _PART["L"] * _PART["B"] * _PART["T"] * _PART["CB"] * float(sw_density(_SST_C)) / 1000.0
DEADWEIGHT_COEFF = 0.82  # documented bulk-carrier range ~0.80-0.85; midpoint
DWT_CENTRAL_T = DISPLACEMENT_T * DEADWEIGHT_COEFF

# Sensitivity sweep: the full plausible bulk-carrier size-class range, from
# Handysize (~10,000 t) to the largest ore carriers in commercial service
# (~500,000 t, beyond even the ~400,000 t "Valemax" VLOC class) -- spanning
# Handysize, Handymax/Supramax, Panamax/Kamsarmax, Capesize, and VLOC. Kept
# unchanged by fix round 2; the type dimension is crossed with it below.
DWT_SWEEP_T = sorted({
    10_000, 25_000, 50_000, 75_000, round(DWT_CENTRAL_T), 100_000, 150_000,
    200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000,
})

# IMO reference-line coefficients (a, c in CII_ref = a * DWT^(-c)) and rating
# boundaries (d1..d4, applied to attained/required), BOTH per ship type,
# sourced from the primary IMO texts -- see "THE SHIP-TYPE PROBLEM" above for
# the exact resolution/annex/table/page citations. The reduction-factor
# schedule below is confirmed uniform across ship types (MARPOL Annex VI
# regulation 28), so it is NOT part of TYPE_PARAMS.
TYPE_PARAMS: dict[str, dict[str, float]] = {
    "bulk carrier": {"a": 4745.0, "c": 0.622,
                     "d1": 0.86, "d2": 0.94, "d3": 1.06, "d4": 1.18},
    "container ship": {"a": 1984.0, "c": 0.489,
                       "d1": 0.83, "d2": 0.94, "d3": 1.07, "d4": 1.19},
}
SHIP_TYPES = ("bulk carrier", "container ship")
TYPE_SHORT = {"bulk carrier": "bulk", "container ship": "container"}

CII_REDUCTION = {2023: 0.05, 2024: 0.07, 2025: 0.09, 2026: 0.11}

HOURS_PER_MONTH = MIN_PER_MONTH / 60.0        # nominal hours per month
ACCUM_MONTHS = 12.0                           # CII's real accumulation period
N_YEAR_HOURS = int(round(ACCUM_MONTHS * HOURS_PER_MONTH))
P_CONF = 0.95                                 # stated confidence-call threshold


def cii_required(dwt: float, year: int, ship_type: str) -> float:
    """IMO CII reference line for one ship type, reduction-factor adjusted.

    Parameters
    ----------
    dwt : float
        ASSUMED deadweight (t) -- see module docstring; not a measured value.
    year : int
        Calendar year whose reduction factor applies (``CII_REDUCTION``).
    ship_type : str
        Key of :data:`TYPE_PARAMS` (``"bulk carrier"`` or ``"container ship"``).

    Returns
    -------
    float
        Required annual CII, g CO2 / (DWT-t . nm).

    References
    ----------
    International Maritime Organization (2021). Resolution MEPC.337(76),
    Annex 11, 2021 Guidelines on the Reference Lines for Use with
    Operational Carbon Intensity Indicators (CII Reference Lines
    Guidelines, G2), Table 1, page 3.
    """
    p = TYPE_PARAMS[ship_type]
    ref = p["a"] * (dwt ** (-p["c"]))
    return ref * (1.0 - CII_REDUCTION.get(year, 0.07))


def cii_rating(attained: float, required: float, ship_type: str) -> str:
    """Letter rating (A-E) from the attained/required ratio, per ship type.

    Parameters
    ----------
    attained : float
        Attained CII, same units as *required*.
    required : float
        Required CII from :func:`cii_required` for the SAME *ship_type*.
    ship_type : str
        Key of :data:`TYPE_PARAMS`.

    Returns
    -------
    str
        One of ``"A"``..``"E"``.

    References
    ----------
    International Maritime Organization (2022). Resolution MEPC.354(78),
    Annex 16, 2022 Guidelines on the Operational Carbon Intensity Rating
    of Ships (CII Rating Guidelines, G4), Table 1, page 6.
    """
    b = TYPE_PARAMS[ship_type]
    ratio = attained / required
    if ratio < b["d1"]:
        return "A"
    if ratio < b["d2"]:
        return "B"
    if ratio < b["d3"]:
        return "C"
    if ratio < b["d4"]:
        return "D"
    return "E"


def _trajectory(
    cum_co2_g: np.ndarray, cum_distance: np.ndarray, se_co2_g: np.ndarray,
    months_elapsed: np.ndarray, dwt: float, year: int, ship_type: str,
) -> tuple[float, float, str, float, float]:
    """Attained/required CII, rating, P(<=C), and months-to-confident-call.

    Reuses the (DWT- and type-independent) cumulative CO2 mass, cumulative
    distance, and cumulative CO2 standard-error trajectories for one
    (architecture, set) pair -- computed once in :func:`main` -- so
    sweeping many assumed deadweights or ship types never re-runs the
    model predictions.

    Parameters
    ----------
    cum_co2_g, cum_distance, se_co2_g : numpy.ndarray
        Per-hour cumulative CO2 mass (g), distance (nm), and CO2 mass
        standard error (g); independent of DWT and ship type.
    months_elapsed : numpy.ndarray
        Nominal months into the accumulation period, one per hour.
    dwt : float
        Assumed deadweight (t).
    year : int
        Calendar year for the reduction factor (see :func:`cii_required`).
    ship_type : str
        Key of :data:`TYPE_PARAMS`.

    Returns
    -------
    tuple
        ``(attained_final, required, rating, p_rating_le_C_final, months_to_confident_call)``.
    """
    required = cii_required(dwt, year, ship_type)
    threshold_c = TYPE_PARAMS[ship_type]["d3"] * required
    denom = np.maximum(dwt * cum_distance, 1e-9)
    attained = cum_co2_g / denom
    se_attained = se_co2_g / denom
    z = (threshold_c - attained) / np.maximum(se_attained, 1e-12)
    p_le_c = norm.cdf(z)

    confident = (p_le_c >= P_CONF) | (p_le_c <= 1.0 - P_CONF)
    uncertain_idx = np.where(~confident)[0]
    n = len(p_le_c)
    first_confident = 0 if uncertain_idx.size == 0 else uncertain_idx[-1] + 1
    months_to_confident = (
        float(months_elapsed[first_confident]) if first_confident < n else float("nan")
    )
    attained_final = float(attained[-1])
    return (attained_final, required, cii_rating(attained_final, required, ship_type),
            float(p_le_c[-1]), months_to_confident)


def main() -> None:
    print(f"[CII] central DWT derived from the vessel's own assumed hull "
          f"(CONFIG['particulars']): displacement={DISPLACEMENT_T:,.0f} t "
          f"x deadweight coeff {DEADWEIGHT_COEFF} = DWT_CENTRAL_T={DWT_CENTRAL_T:,.0f} t "
          f"-- NOT chosen to make architectures differ (see module docstring).")
    print(f"[CII] ASSUMPTION: ship type is UNKNOWN (Shifts anonymisation); "
          f"reporting BOTH {' and '.join(SHIP_TYPES)}, side by side, year={ASSUMED_YEAR}.")

    tr, din, dout = load_shifts_splits()
    n_fit = int(0.85 * len(tr))
    fit, cal = tr.iloc[:n_fit], tr.iloc[n_fit:]
    ph_all = physics_power(tr, "train")
    ph_fit, ph_cal = ph_all[:n_fit], ph_all[n_fit:]
    ph_in, ph_out = physics_power(din, "dev_in"), physics_power(dout, "dev_out")

    y_fit, y_cal = fit["power"].to_numpy(), cal["power"].to_numpy()
    X_fit = fit[FEATS].to_numpy()
    k = fit_scale(y_fit, ph_fit)

    print(f"[CII] observed profile check: median stw={din['stw'].median():.1f} kn, "
          f"median power={din['power'].median():,.0f} kW (dev_in); "
          f"CONFIG['particulars']['CB']={_PART['CB']} -- see 'THE SHIP-TYPE PROBLEM' "
          f"in the module docstring.")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    # Identical fits to run_uq_emissions.py / run_ghg_breakdown.py: fit and
    # calibrate on the training split ONLY (never dev_in/dev_out).
    res_fit = y_fit - k * ph_fit
    f_ml = HistGradientBoostingRegressor(random_state=SEED, max_iter=400).fit(X_fit, res_fit)
    bbm = HistGradientBoostingRegressor(random_state=SEED, max_iter=400).fit(X_fit, y_fit)
    sc = StandardScaler().fit(X_fit)
    rdg = Ridge(alpha=100.0).fit(sc.transform(X_fit), res_fit)

    def predict(model: str, df: pd.DataFrame, phys: np.ndarray) -> np.ndarray:
        X = df[FEATS].to_numpy()
        if model == "WBM":
            return k * phys
        if model == "BBM":
            return bbm.predict(X)
        if model == "S-GBM":
            return k * phys + f_ml.predict(X)
        if model == "C-GBM-lam0.25":
            return k * phys + 0.25 * f_ml.predict(X)
        if model == "C-GBM-ridge":
            return k * phys + rdg.predict(sc.transform(X))
        raise ValueError(model)

    # ---- DWT- and type-independent per-(architecture, set) trajectories.
    precomputed: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for m in MODELS:
        p_cal = predict(m, cal, ph_cal)
        q = float(np.quantile(np.abs(y_cal - p_cal), 1.0 - ALPHA))  # published split-conformal half-width, kW
        sigma_power = q / Z90

        for split_name, dset, phys in [("dev_in", din, ph_in), ("dev_out", dout, ph_out)]:
            n = min(len(dset), N_YEAR_HOURS)
            d, ph = dset.iloc[:n], phys[:n]

            p = predict(m, d, ph)
            load = np.clip(p / MCR_KW, 0.0, 1.0)
            fuel = power_to_fuel_mt_h(p)
            co2_point = np.asarray(
                fuel_to_co2e(fuel, load_fraction=load, fuel_type="MDO")["co2_mt"], dtype=float,
            )

            p_hi, p_lo = p + sigma_power, np.maximum(p - sigma_power, 0.0)
            load_hi, load_lo = np.clip(p_hi / MCR_KW, 0.0, 1.0), np.clip(p_lo / MCR_KW, 0.0, 1.0)
            co2_hi = np.asarray(
                fuel_to_co2e(power_to_fuel_mt_h(p_hi), load_fraction=load_hi,
                            fuel_type="MDO")["co2_mt"], dtype=float,
            )
            co2_lo = np.asarray(
                fuel_to_co2e(power_to_fuel_mt_h(p_lo), load_fraction=load_lo,
                            fuel_type="MDO")["co2_mt"], dtype=float,
            )
            sigma_co2_g = 0.5 * (co2_hi - co2_lo) * 1.0e6  # 1-sigma, grams, per retained hour

            distance_nm = d["stw"].to_numpy(dtype=float)   # 1 nominal hour * stw(kn)

            cum_co2_g = np.cumsum(co2_point) * 1.0e6
            cum_distance = np.cumsum(distance_nm)
            se_co2_g = np.sqrt(np.cumsum(sigma_co2_g ** 2))  # i.i.d.-hours simplification (disclosed)
            months_elapsed = np.arange(1, n + 1) / HOURS_PER_MONTH

            precomputed[(m, split_name)] = (cum_co2_g, cum_distance, se_co2_g, months_elapsed)

    # ---- Main table at the physics-derived central DWT: BOTH ship types,
    # side by side (twenty rows: 5 architectures x 2 sets x 2 types), the
    # seven-column interface preserved by suffixing the type onto
    # `architecture` rather than adding a column.
    rows: list[dict] = []
    for ship_type in SHIP_TYPES:
        for m in MODELS:
            for split_name in ("dev_in", "dev_out"):
                cum_co2_g, cum_distance, se_co2_g, months_elapsed = precomputed[(m, split_name)]
                attained_final, required, rating, p_le_c_final, months_to_confident = _trajectory(
                    cum_co2_g, cum_distance, se_co2_g, months_elapsed,
                    DWT_CENTRAL_T, ASSUMED_YEAR, ship_type,
                )
                rows.append({
                    "architecture": f"{m} [{TYPE_SHORT[ship_type]}]", "set": split_name,
                    "attained_cii": attained_final, "required_cii": required,
                    "rating": rating, "p_rating_le_C": p_le_c_final,
                    "months_to_confident_call": months_to_confident,
                })
                conf_str = (f"{months_to_confident:.2f} mo" if not np.isnan(months_to_confident)
                           else f"not reached in {ACCUM_MONTHS:.0f} mo")
                print(f"  [{ship_type:14s}] {m:14s} {split_name:8s} attained={attained_final:8.4f} "
                      f"rating={rating} P(<=C)={p_le_c_final:.4f}  confident_at={conf_str}")

    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "cii_probabilistic.csv", index=False)

    # ---- Sensitivity sweep: same architectures/sets, DWT_SWEEP_T deadweights,
    # crossed with BOTH ship types (a genuine ship_type column here, since
    # this companion file is not interface-constrained).
    sweep_rows: list[dict] = []
    for ship_type in SHIP_TYPES:
        for dwt in DWT_SWEEP_T:
            for m in MODELS:
                for split_name in ("dev_in", "dev_out"):
                    cum_co2_g, cum_distance, se_co2_g, months_elapsed = precomputed[(m, split_name)]
                    attained_final, required, rating, p_le_c_final, months_to_confident = _trajectory(
                        cum_co2_g, cum_distance, se_co2_g, months_elapsed,
                        float(dwt), ASSUMED_YEAR, ship_type,
                    )
                    sweep_rows.append({
                        "ship_type": ship_type, "dwt_t": dwt,
                        "architecture": m, "set": split_name,
                        "attained_cii": attained_final, "required_cii": required,
                        "rating": rating, "p_rating_le_C": p_le_c_final,
                        "months_to_confident_call": months_to_confident,
                    })
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(RESULTS / "cii_probabilistic_sweep.csv", index=False)
    print(f"\n[CII] sweep spans DWT in {DWT_SWEEP_T[0]:,} .. {DWT_SWEEP_T[-1]:,} t, "
          f"x {len(SHIP_TYPES)} ship types "
          f"({len(DWT_SWEEP_T)} points x {len(MODELS)} architectures x 2 sets x "
          f"{len(SHIP_TYPES)} types = {len(sweep_rows)} rows)")

    md = [
        "# Probabilistic CII rating per architecture (Shifts, seed 42)",
        "",
        "**ASSUMED reference line -- NOT the vessel's real requirement, AND "
        "the ship type itself is unidentifiable.** The Shifts marine "
        "benchmark discloses only \"one merchant cargo vessel\"; ship type "
        "and deadweight are withheld by its anonymisation. `required_cii`, "
        "`rating`, and `p_rating_le_C` below assume the "
        f"**{ASSUMED_YEAR}** IMO reduction schedule and are reported under "
        "**both** a bulk-carrier and a container-ship reference line "
        "(coefficients sourced from MEPC.337(76) Annex 11 Table 1 and "
        "MEPC.354(78) Annex 16 Table 1 -- see the module docstring for the "
        "exact page citations). Every number below is conditional on "
        "whichever type heads its row, never a measurement of this "
        "vessel's actual regulatory position.",
        "",
        f"**The DWT itself is derived, not chosen for contrast.** "
        f"Displacement = L x B x T x CB x rho_sw / 1000 = "
        f"{DISPLACEMENT_T:,.0f} t, from this benchmark's own assumed hull "
        "(`CONFIG[\"particulars\"]` in `common.py`: L=300 m, B=40 m, T=14.5 m, "
        "CB=0.62) and `vessel_fuel.sw_density` at 12 degC (the same "
        "sea-surface temperature `physics_power()` already uses). Deadweight "
        f"= displacement x {DEADWEIGHT_COEFF} (a documented bulk-carrier "
        "deadweight-coefficient midpoint, range ~0.80-0.85; kept unchanged "
        "this round even though the ship-type check below suggests a "
        "container-ship coefficient, ~0.60-0.70, would be more consistent "
        "-- a disclosed, not-yet-resolved tension), giving "
        f"**DWT_CENTRAL_T = {DWT_CENTRAL_T:,.0f} t**.",
        "",
        "**The ship type cannot be identified either, and it moves the "
        "rating further than architecture choice does.** "
        f"CONFIG[\"particulars\"][\"CB\"] = {_PART['CB']} (bulk carriers run "
        "~0.80-0.85; container ships ~0.60-0.68), and the vessel's own "
        "median operating point -- 18.4 kn on 22.8 MW of shaft power -- is "
        "a container-ship profile, not a bulk-carrier one (a comparable "
        "bulk carrier runs ~14 kn on 8-10 MW). An earlier version of this "
        "table used only the bulk-carrier line inherited from "
        "`run_corridor_case_study.py` without checking this; see "
        "`task-13-report.md`, \"Fix round 2\". The table below reports both "
        "types side by side: **every architecture on every evaluation set "
        "rates E under the bulk-carrier assumption and C under the "
        "container-ship assumption** -- a full two-notch, completely "
        "uniform swing driven by the type assumption alone, larger and "
        "more consistent than anything architecture choice produces.",
        "",
        "CII is CO2-only "
        "(`vessel_fuel.ghg_scope.gases_for(\"CII\") == (\"CO2\",)`); every "
        "accumulated quantity here is CO2 mass, never CO2e.",
        "",
        f"Interval: unweighted split-conformal (alpha={ALPHA}), the SAME "
        "interval published in `results/uq_emissions.csv` / Table "
        "\"uq-coverage\" -- applied to BOTH evaluation sets deliberately "
        "(module docstring, \"why unweighted\"). Its coverage collapses "
        "under shift (e.g. S-GBM 0.973 -> 0.863, BBM 0.976 -> 0.924; see "
        "`results/uq_emissions.csv`), which is exactly the failure mode "
        "this table exposes: a rating called from a too-narrow interval is "
        "overconfident precisely under shift.",
        "",
        f"Confidence-call threshold P_CONF={P_CONF}; accumulation capped at "
        f"the first nominal {ACCUM_MONTHS:.0f} months (CII is an annual "
        "metric; see \"accumulation-period construction\" in the module "
        "docstring). `months_to_confident_call` = NaN means the call never "
        "became -- and stayed -- confident inside that nominal year.",
        "",
        "## Main table (central DWT, both ship types side by side)",
        "",
        "Twenty rows: 5 architectures x 2 evaluation sets x 2 ship-type "
        "assumptions (`architecture` suffixed `[bulk]`/`[container]`; the "
        "seven-column interface is unchanged).",
        "",
        out.round(4).to_markdown(index=False),
        "",
        "## Sensitivity sweep -- DWT and ship type, is architecture choice EVER material?",
        "",
        f"Same architectures and sets, swept across DWT = "
        f"{DWT_SWEEP_T[0]:,}..{DWT_SWEEP_T[-1]:,} t (the full plausible "
        "bulk-carrier size-class range, Handysize through VLOC) crossed "
        "with both ship types. Full data (dwt_t x ship_type x architecture "
        "x set = "
        f"{len(DWT_SWEEP_T) * len(SHIP_TYPES) * len(MODELS) * 2} rows) in "
        "the companion `cii_probabilistic_sweep.csv`.",
        "",
        "**Type-robust finding:** the bulk-carrier-vs-container-ship gap "
        "(roughly two rating notches at the central DWT) holds throughout "
        "essentially the whole swept DWT range on the bulk-carrier side "
        "(ratings stay E up to ~250,000 t under bulk carrier while the "
        "same architectures are already C-to-borderline-B under container "
        "ship at the SAME DWT) -- the type assumption dominates the DWT "
        "assumption over most of the sweep, not only at the central "
        "estimate.",
        "",
        "**Size-robust finding (within EACH type):** for every swept DWT "
        "up to ~250,000 t under the bulk-carrier assumption, all five "
        "architectures agree completely (rating E, confident within the "
        "first hour or so); architecture choice is irrelevant there. Under "
        "the container-ship assumption the picture is closer to the "
        "boundary throughout, so architecture choice has more room to "
        "matter even at moderate DWTs -- see the companion CSV.",
        "",
        "**Size-specific finding (NOT robust -- holds only in a narrow "
        "band, and only clearly demonstrated on the bulk-carrier side):** "
        "only from roughly 300,000-500,000 t (Capesize/VLOC scale, well "
        "above the physics-derived central estimate) does the "
        "bulk-carrier attained/required ratio sit close enough to a "
        "rating boundary for architecture choice to change the call -- "
        "different letters, very different confident-call timing, or no "
        "persistent resolution at all within the nominal year -- and even "
        "the ranking of which architecture is fastest/slowest is not "
        "stable across that band. See `task-13-report.md` for the full "
        "numbers under both types. No single architecture is reported "
        "here as \"the fast one\" or \"the slow one\" in general.",
        "",
        sweep.round(4).to_markdown(index=False),
    ]
    (RESULTS / "cii_probabilistic.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'cii_probabilistic.csv'}, "
          f"{RESULTS / 'cii_probabilistic_sweep.csv'}, and cii_probabilistic.md")


if __name__ == "__main__":
    main()
