"""Boundary probe for the FuelCast Holtrop-Mennen prior geometry search.

s06 selected a geometry sitting on every bound of its grid (L max, CB min, beam
ratio max, draught min). That could mean the grid was simply too small, or that the
prior's *shape* is wrong and the search is trying to reshape the resistance curve by
driving the hull toward an unphysical extreme. This probe distinguishes the two by
extending the grid well past anything plausible for a 6-8 kn offshore vessel and
reporting how much validation MAPE actually improves.

Poseidon only: it has no measured draught, so the prior is a pure function of the
geometry parameters and evaluates vectorised in milliseconds.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .common_rev import REPO_ROOT, REV_FINAL, write_csv
from .s01_full_data_602020 import metrics7
from .s06_fuelcast_wind_and_wbm import (
    FIT_FRACTION, GRID_BRATIO, GRID_CB, GRID_L, GRID_T_NODRAFT, particulars, wbm_prior,
)

_E = REPO_ROOT / "experiments"
if str(_E) not in sys.path:
    sys.path.insert(0, str(_E))
from benchmark_measured.common import (  # noqa: E402
    fit_scale, fuelcast_shift_splits, load_fuelcast_vessel,
)

OUT = REV_FINAL / "06_fuelcast_wind_and_wbm"
VESSEL = "CPS_Poseidon"
# deliberately far beyond a plausible offshore-vessel envelope
EXT_L = [80.0, 120.0, 180.0, 240.0, 300.0, 400.0]
EXT_BRATIO = [4.5, 5.5, 6.5, 8.0, 10.0]
EXT_CB = [0.45, 0.50, 0.60, 0.70, 0.80]
EXT_T = [3.0, 5.0, 7.0, 9.0, 11.0]


def main() -> int:
    d, feats = load_fuelcast_vessel(VESSEL)
    tr, _, _, _ = fuelcast_shift_splits(d)
    y = tr["fuel_kgh"].to_numpy()
    n_fit = int(FIT_FRACTION * len(tr))
    fit_i, val_i = np.arange(n_fit), np.arange(n_fit, len(tr))
    mcr = float(np.ceil(np.nanpercentile(y, 99) / 0.18 / 1000.0) * 1000.0)

    rows = []
    for L in EXT_L:
        for br in EXT_BRATIO:
            for cb in EXT_CB:
                for T in EXT_T:
                    part = particulars(L, br, cb, T, mcr)
                    pr = wbm_prior(tr, part, False)
                    if not np.all(np.isfinite(pr)) or np.all(pr <= 0):
                        continue
                    k = fit_scale(y[fit_i], pr[fit_i])
                    m = metrics7(y[val_i], k * pr[val_i])
                    rows.append({"L": L, "beam_ratio": br, "B": L / br, "CB": cb,
                                 "T": T, "k": float(k),
                                 "in_original_grid": bool(
                                     L in GRID_L and br in GRID_BRATIO
                                     and cb in GRID_CB and T in GRID_T_NODRAFT),
                                 "validation_MAPE": m["MAPE"],
                                 "validation_RMSE": m["RMSE"],
                                 "validation_R2": m["R2"]})
    g = pd.DataFrame(rows).sort_values("validation_MAPE").reset_index(drop=True)
    # A lower validation MAPE bought by an impossible hull is not evidence that the
    # prior needed a bigger grid. Screen every candidate on slenderness and draught
    # ratio before reading anything off the ranking.
    g["L_over_B"] = g.L / g.B
    g["T_over_B"] = g["T"] / g.B
    g["physically_plausible"] = ((g.L_over_B >= 4.0) & (g.L_over_B <= 8.0)
                                 & (g.T_over_B >= 0.25) & (g.L <= 200.0))
    best_orig = g[g.in_original_grid].iloc[0]
    best_ext = g.iloc[0]
    best_plaus = g[g.physically_plausible].iloc[0]
    gain = float(best_orig.validation_MAPE - best_ext.validation_MAPE)
    gain_plaus = float(best_orig.validation_MAPE - best_plaus.validation_MAPE)

    summary = pd.DataFrame([{
        "vessel": VESSEL, "n_candidates_extended": int(len(g)),
        "n_candidates_original_grid": int(g.in_original_grid.sum()),
        "best_original_grid_MAPE": float(best_orig.validation_MAPE),
        "best_original_L": float(best_orig.L), "best_original_CB": float(best_orig.CB),
        "best_original_T": float(best_orig["T"]),
        "best_extended_MAPE": float(best_ext.validation_MAPE),
        "best_extended_L": float(best_ext.L), "best_extended_B": float(best_ext.B),
        "best_extended_CB": float(best_ext.CB), "best_extended_T": float(best_ext["T"]),
        "MAPE_gain_pp_from_extending": gain,
        "best_extended_L_over_B": float(best_ext.L / best_ext.B),
        "best_extended_T_over_B": float(best_ext["T"] / best_ext.B),
        "best_extended_physically_plausible": bool(best_ext.physically_plausible),
        "best_plausible_MAPE": float(best_plaus.validation_MAPE),
        "best_plausible_L": float(best_plaus.L), "best_plausible_B": float(best_plaus.B),
        "best_plausible_CB": float(best_plaus.CB),
        "best_plausible_T": float(best_plaus["T"]),
        "MAPE_gain_pp_within_plausible_envelope": gain_plaus,
        "plausibility_screen": "L/B in [4, 8], T/B >= 0.25, L <= 200 m",
        "extended_optimum_still_on_a_bound": bool(
            best_ext.L >= max(EXT_L) or best_ext.CB <= min(EXT_CB)
            or best_ext["T"] <= min(EXT_T) or best_ext.beam_ratio >= max(EXT_BRATIO)),
        "verdict": (
            "shape-limited: the unconstrained optimum improves validation MAPE only "
            "by leaving the physical envelope (L/B = "
            f"{best_ext.L / best_ext.B:.1f}, T/B = {best_ext['T'] / best_ext.B:.2f}), "
            "which cannot be the real vessel; inside a plausible envelope extending "
            f"the grid buys {gain_plaus:.2f} pp. The misfit is in the form of the "
            "prior, not the size of the assumed hull."
            if not bool(best_ext.physically_plausible) else
            "size-limited: the extended optimum is physically plausible and improves "
            f"validation MAPE by {gain:.2f} pp, so the original grid was too small")}])

    write_csv(g, OUT / "partB_geometry_boundary_probe.csv")
    write_csv(summary, OUT / "partB_geometry_boundary_probe_summary.csv")
    print(f"[probe] {len(g)} candidates; best in original grid "
          f"{best_orig.validation_MAPE:.3f}% -> best extended "
          f"{best_ext.validation_MAPE:.3f}% (gain {gain:.3f} pp)", flush=True)
    print(f"[probe] extended optimum: L={best_ext.L:.0f} B={best_ext.B:.1f} "
          f"CB={best_ext.CB:.2f} T={best_ext["T"]:.1f}", flush=True)
    print(f"[probe] {summary.verdict.iloc[0]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
