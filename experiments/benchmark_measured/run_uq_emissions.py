"""Phase D — shift-valid conformal UQ propagated to CO2e intervals.

D1: split-conformal 90% intervals per architecture, calibrated on a temporal
    tail of train; empirical coverage on dev_in AND dev_out (expect collapse).
D2: weighted (covariate-shift) conformal in the style of Tibshirani et al.
    (2019) / Barber et al. (2023): density-ratio weights from a logistic
    classifier between the calibration set and the shifted evaluation set;
    per-test-point weighted quantile. Report restored coverage + width cost.
D4: propagate the power interval through the anchored engine model
    (IMO SFOC at load = P/MCR) and the multi-gas emission head to a CO2e
    interval per hour on the shifted set. No emission-allowance pricing is
    computed here (out of scope for this paper).

Run:  PYTHONPATH=. python experiments/benchmark_measured/run_uq_emissions.py
Outputs: experiments/benchmark_measured/results/uq_emissions.{csv,md}
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark_measured.common import (  # noqa: E402
    CONFIG, FEATS, SEED,
    fit_scale, load_shifts_splits, physics_power,
)

# Written directly under results/ (checked into git), matching
# run_baseline_table.py / run_head_vs_regressor.py / run_ef_validation.py --
# NOT common.OUT (outputs/benchmark_measured/, gitignored).
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "benchmark_measured" / "results"
from vessel_fuel import fuel_to_co2e, co2e_interval, sfoc_curve  # noqa: E402

ALPHA = 0.10
MCR_KW = CONFIG["particulars"]["MCR"]
SFOC_MCR = CONFIG["particulars"]["sfoc_at_mcr"]


def power_to_fuel_mt_h(p_kw: np.ndarray) -> np.ndarray:
    """Measured/predicted shaft power -> fuel rate (t/h) via anchored SFOC."""
    load = np.clip(np.asarray(p_kw, dtype=float) / MCR_KW, 0.0, 1.0)
    sfoc = np.asarray(sfoc_curve(load, SFOC_MCR), dtype=float)
    return np.maximum(p_kw, 0.0) * sfoc / 1e6


def weighted_quantile_thresholds(scores_cal: np.ndarray, w_cal: np.ndarray,
                                 w_test: np.ndarray, alpha: float) -> np.ndarray:
    """Per-test-point weighted conformal threshold (Tibshirani et al. 2019).

    For each test point with weight w_t, the threshold is the smallest score s
    such that sum(w_cal[scores<=s]) >= (1-alpha) * (sum(w_cal) + w_t).
    """
    order = np.argsort(scores_cal)
    s_sorted = scores_cal[order]
    cw = np.cumsum(w_cal[order])
    total = cw[-1]
    need = (1.0 - alpha) * (total + w_test)          # per test point
    idx = np.searchsorted(cw, need, side="left")
    out = np.where(idx >= len(s_sorted), np.inf, s_sorted[np.minimum(idx, len(s_sorted) - 1)])
    return out


def main() -> None:
    tr, din, dout = load_shifts_splits()
    n_fit = int(0.85 * len(tr))
    fit, cal = tr.iloc[:n_fit], tr.iloc[n_fit:]
    ph_fit = physics_power(fit, "train")[:n_fit] if False else physics_power(tr, "train")[:n_fit]
    ph_all = physics_power(tr, "train")
    ph_fit, ph_cal = ph_all[:n_fit], ph_all[n_fit:]
    ph_in, ph_out = physics_power(din, "dev_in"), physics_power(dout, "dev_out")

    y_fit, y_cal = fit["power"].to_numpy(), cal["power"].to_numpy()
    X_fit, X_cal = fit[FEATS].to_numpy(), cal[FEATS].to_numpy()
    k = fit_scale(y_fit, ph_fit)

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler

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

    MODELS = ["WBM", "BBM", "S-GBM", "C-GBM-lam0.25", "C-GBM-ridge"]

    # density-ratio weights cal -> dev_out (wind features drive the shift)
    shift_feats = ["wind_kn", "hs", "stw"]
    Z = np.vstack([cal[shift_feats].to_numpy(), dout[shift_feats].to_numpy()])
    lbl = np.concatenate([np.zeros(len(cal)), np.ones(len(dout))])
    zsc = StandardScaler().fit(Z)
    clf = LogisticRegression(max_iter=1000).fit(zsc.transform(Z), lbl)

    def dens_ratio(df: pd.DataFrame) -> np.ndarray:
        p = clf.predict_proba(zsc.transform(df[shift_feats].to_numpy()))[:, 1]
        p = np.clip(p, 1e-3, 1 - 1e-3)
        return p / (1 - p)

    w_cal = dens_ratio(cal)
    rows: list[dict] = []
    for m in MODELS:
        p_cal = predict(m, cal, ph_cal)
        scores = np.abs(y_cal - p_cal)
        q_split = np.quantile(scores, 1.0 - ALPHA)

        for split_name, dset, phys in [("dev_in", din, ph_in), ("dev_out", dout, ph_out)]:
            y = dset["power"].to_numpy()
            p = predict(m, dset, phys)
            cov = float(np.mean(np.abs(y - p) <= q_split))
            row = {"model": m, "split": split_name, "method": "split",
                   "coverage": cov, "width_kw": 2.0 * q_split}
            rows.append(row)

        # weighted conformal targeted at the shifted set
        w_test = dens_ratio(dout)
        qs = weighted_quantile_thresholds(scores, w_cal, w_test, ALPHA)
        y, p = dout["power"].to_numpy(), predict(m, dout, ph_out)
        finite = np.isfinite(qs)
        cov_w = float(np.mean(np.abs(y - p) <= qs))
        rows.append({"model": m, "split": "dev_out", "method": "weighted",
                     "coverage": cov_w,
                     "width_kw": float(2.0 * np.mean(qs[finite])) if finite.any() else np.inf})

        # ---- D4: CO2e interval on dev_out (weighted intervals)
        lo_p, hi_p = np.maximum(p - qs, 0.0), p + qs
        fuel_lo, fuel_hi = power_to_fuel_mt_h(lo_p), power_to_fuel_mt_h(hi_p)
        load = np.clip(p / MCR_KW, 0.0, 1.0)
        c_lo, c_hi = co2e_interval(fuel_lo, fuel_hi, load_fraction=load, fuel_type="MDO")
        fuel_true = power_to_fuel_mt_h(y)
        co2e_true = fuel_to_co2e(fuel_true, load_fraction=np.clip(y / MCR_KW, 0, 1))["co2e_mt"]
        rows[-1].update({
            "co2e_true_t": float(np.sum(np.asarray(co2e_true))),
            "co2e_upper_t": float(np.sum(np.asarray(c_hi))),
        })

    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "uq_emissions.csv", index=False)
    md = ["# Conformal UQ under shift + CO2e propagation (Shifts, seed 42)", "",
          f"alpha={ALPHA} (target coverage {1-ALPHA:.0%}); weighted = covariate-shift",
          "conformal (density-ratio weights on wind/hs/stw, Tibshirani-2019 style).",
          "CO2e chain: power interval -> anchored IMO SFOC -> emission head (MDO, AR6).",
          f"dev_out hours = {len(dout)}.",
          "", out.round(3).to_markdown(index=False)]
    (RESULTS / "uq_emissions.md").write_text("\n".join(md), encoding="utf-8")
    print(out.round(3).to_string(index=False))
    print(f"\nwrote {RESULTS/'uq_emissions.csv'} and .md")


if __name__ == "__main__":
    main()
