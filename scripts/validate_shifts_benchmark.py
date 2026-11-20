"""Path B — real-data validation benchmark on SHIFTS (measured shaft power).

Target: measured main-engine shaft power (kW) of the Shifts merchant cargo vessel.
Four models, TEMPORAL holdout (train early, test late — no leakage), honest metrics
(MAE/RMSE/MAPE/R2) + split-conformal 90% intervals.

Models:
  baseline  : admiralty cube law  P = a * stw^3  (a fit on train)
  physics   : Holtrop resistance->power (vessel_fuel), per-row draft, ONE scale
              factor calibrated on train (grey-box; absorbs anon-particular error)
  pure_ml   : HistGradientBoosting on features
  hybrid    : physics_calibrated + HistGradientBoosting on the residual

The vessel is anonymous (Shifts) -> representative large-container particulars,
scale-calibrated. What is tested is the physics SHAPE (speed/draft/fouling), not
absolute hull size.

Run:  PYTHONPATH=. python scripts/validate_shifts_benchmark.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from vessel_fuel import fuel_components

REAL = pathlib.Path("data/external/shifts_vessel/v2/power_consumption_upload/real_data")
MIN_PER_MONTH = 60.0 * 24.0 * 30.44
SEED = 42
N_SUB = 40000  # subsample (physics is a per-row loop); keeps runtime sane

# representative large container-ship particulars (anon vessel; scale calibrated)
PART = dict(L=300.0, B=40.0, T=14.5, S=15200.0, CB=0.62, MCR=45000.0,
            sfoc_at_mcr=170.0, A_front=1200.0, A_lateral=8000.0, design_draft=14.5,
            eta_0=0.70, eta_wave_loss=0.03)


def load_shifts() -> pd.DataFrame:
    frames = []
    for f in ["train.csv", "dev_in.csv", "dev_out.csv"]:
        p = REAL / f
        if p.exists():
            frames.append(pd.read_csv(p))
    d = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "time_id": d["time_id"],
        "stw": d["stw"],
        "draft": 0.5 * (d["draft_aft_telegram"] + d["draft_fore_telegram"]),
        "trim": d["draft_aft_telegram"] - d["draft_fore_telegram"],
        "wind_kn": np.hypot(d["awind_ucomp_provider"], d["awind_vcomp_provider"]),
        "wind_ang": np.degrees(np.arctan2(d["awind_vcomp_provider"], d["awind_ucomp_provider"])),
        "cur_kn": np.hypot(d["rcurrent_ucomp"], d["rcurrent_vcomp"]),
        "cur_ang": np.degrees(np.arctan2(d["rcurrent_vcomp"], d["rcurrent_ucomp"])),
        "hs": d["comb_wind_swell_wave_height"],
        "months": d["timeSinceDryDock"] / MIN_PER_MONTH,
        "power": d["power"],
    })
    out = out[(out["stw"] > 4) & (out["power"] > 100)].dropna()
    return out


def physics_power(df: pd.DataFrame) -> np.ndarray:
    """Per-row Holtrop shaft power (kW), uncalibrated."""
    out = np.empty(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        env = {"wind_kn": r.wind_kn, "wind_angle_deg": r.wind_ang, "Hs": r.hs,
               "Tp": 8.0, "wave_angle_deg": 0.0, "current_kn": r.cur_kn,
               "current_angle_deg": r.cur_ang, "sst_c": 12.0, "depth_m": 2000.0}
        prm = dict(PART, T=float(r.draft), trim_m=float(r.trim),
                   months_since_cleaning=float(r.months))
        c = fuel_components(100.0, float(r.stw), env, prm)
        out[i] = float(c["shaft_power_kw"])
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE%": float(np.mean(np.abs(err / y)) * 100),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def conformal_cov(res_cal: np.ndarray, y: np.ndarray, p: np.ndarray, alpha=0.1) -> float:
    q = np.quantile(np.abs(res_cal), 1 - alpha)
    return float(np.mean(np.abs(y - p) <= q))


def main() -> None:
    df = load_shifts().sort_values("time_id").reset_index(drop=True)
    if len(df) > N_SUB:
        idx = np.linspace(0, len(df) - 1, N_SUB).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
    # 3-way TEMPORAL split: train (fit) / cal (conformal, out-of-sample) / test
    n_tr, n_cal = int(0.60 * len(df)), int(0.70 * len(df))
    tr, cal, te = df.iloc[:n_tr], df.iloc[n_tr:n_cal], df.iloc[n_cal:]
    ytr, ycal, yte = tr["power"].to_numpy(), cal["power"].to_numpy(), te["power"].to_numpy()
    feats = ["stw", "draft", "trim", "wind_kn", "wind_ang", "cur_kn", "cur_ang", "hs", "months"]
    Xtr, Xcal, Xte = tr[feats].to_numpy(), cal[feats].to_numpy(), te[feats].to_numpy()
    print(f"SHIFTS_CARGO_01 | rows={len(df)} (train {len(tr)} / cal {len(cal)} / test {len(te)}), temporal split by time_id")

    results = {}
    # physics power (uncalibrated) for each split
    ph_tr, ph_cal, ph_te = physics_power(tr), physics_power(cal), physics_power(te)

    # 1) baseline cube law  P = a*stw^3  (a fit on train)
    a = np.sum(ytr * tr["stw"].to_numpy() ** 3) / np.sum(tr["stw"].to_numpy() ** 6)
    results["baseline_cube"] = metrics(yte, a * te["stw"].to_numpy() ** 3)

    # 2) physics (Holtrop), single scale calibrated on train
    k = float(np.sum(ytr * ph_tr) / np.sum(ph_tr ** 2))  # least-squares scale
    results["physics_cal"] = metrics(yte, k * ph_te)
    results["physics_cal"]["_k"] = round(k, 3)

    # 3) pure ML (fit on train only)
    ml = HistGradientBoostingRegressor(random_state=SEED, max_iter=400).fit(Xtr, ytr)
    results["pure_ml"] = metrics(yte, ml.predict(Xte))

    # 4) hybrid = physics_cal + ML on residual (fit on train only)
    hyb = HistGradientBoostingRegressor(random_state=SEED, max_iter=400).fit(Xtr, ytr - k * ph_tr)
    p_hyb = k * ph_te + hyb.predict(Xte)
    results["hybrid"] = metrics(yte, p_hyb)

    # conformal: calibrate q on CAL (never seen in training), evaluate on test
    res_cal = ycal - (k * ph_cal + hyb.predict(Xcal))
    cov = conformal_cov(res_cal, yte, p_hyb, 0.1)

    print(f"\n{'model':16s} {'MAE(kW)':>9s} {'RMSE(kW)':>9s} {'MAPE%':>7s} {'R2':>7s}")
    for name, m in results.items():
        extra = f"  (k={m['_k']})" if "_k" in m else ""
        print(f"{name:16s} {m['MAE']:9.0f} {m['RMSE']:9.0f} {m['MAPE%']:7.1f} {m['R2']:7.3f}{extra}")
    print(f"\nhybrid 90% split-conformal empirical coverage on test: {cov:.3f} (target 0.90)")
    print(f"mean measured power (test): {yte.mean():.0f} kW")


if __name__ == "__main__":
    main()
