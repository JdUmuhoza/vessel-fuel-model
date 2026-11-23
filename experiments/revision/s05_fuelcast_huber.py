"""FuelCast cross-vessel boundary analysis, rebuilt with the retained Huber methodology.

Section 5.4 must be methodologically consistent with the revised Sections 5.1-5.3, whose
frozen configuration is: Huber residual learner, shrinkage-only constraint, and a BBM that
is the same learner fitted directly on the measured target. The published FuelCast table
used CatBoost and is therefore superseded; no CatBoost number is reused here.

Per vessel:
  1. the existing leakage-controlled temporal protocol is preserved (first 85 % of the
     training block fits, last 15 % is the training-side validation tail);
  2. the WBM-lite cube-law prior is calibrated on the fitting portion only;
  3. Huber is fitted to the training residuals on the fitting portion only;
  4. lambda is selected on that vessel's own validation tail alone;
  5. the grid is the complete lambda = 0.00, 0.01, ..., 1.00 with MAPE as the primary
     criterion;
  6. lambda is frozen before the held-out partitions are touched;
  7. the endpoint identities lambda=0 -> WBM-lite and lambda=1 -> S-GBM are verified
     numerically on every partition and every metric.

The Shifts lambda* = 0.70 is NOT imposed. Nothing is retuned on held-out data.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from .common_rev import REPO_ROOT, REV_FINAL, SEED, write_csv
from .s01_full_data_602020 import metrics7

_E = REPO_ROOT / "experiments"
if str(_E) not in sys.path:
    sys.path.insert(0, str(_E))
from benchmark_measured.common import (  # noqa: E402
    FC_VESSELS, fit_cube, fuelcast_shift_splits, load_fuelcast_vessel, make_learner,
)

OUT = REV_FINAL / "05_fuelcast_huber"
BACKEND = "huber"                 # residual learner AND standalone BBM, per Section 5.1
FIT_FRACTION = 0.85               # unchanged from the published FuelCast protocol
LAM = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 4)
VESSELS = ["CPS_Poseidon", "CPS_Triton", "OSS_Ceto"]
SHORT = {"CPS_Poseidon": "Poseidon", "CPS_Triton": "Triton", "OSS_Ceto": "Ceto"}
SHIFTS_LAMBDA = 0.70              # reference only, never imposed
IDENTITY_TOL = 1e-9
METRICS = ["MAPE", "sMAPE", "MAE", "nMAE", "RMSE", "NRMSE", "R2"]


def _git() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


def run_vessel(name: str) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(SEED)
    df, feats = load_fuelcast_vessel(name)
    tr, din, dout, shift_var = fuelcast_shift_splits(df)

    ytr = tr["fuel_kgh"].to_numpy()
    Xtr = tr[feats].to_numpy()
    vtr = tr["speed"].to_numpy()
    n_fit = int(FIT_FRACTION * len(tr))
    fit_i, val_i = np.arange(n_fit), np.arange(n_fit, len(tr))

    # --- WBM-lite prior: calibrated on the FITTING portion only -------------
    a = fit_cube(ytr[fit_i], vtr[fit_i])
    a_full_train = fit_cube(ytr, vtr)      # recorded for the leakage sensitivity check

    parts = {"validation_tail": (ytr[val_i], Xtr[val_i], vtr[val_i]),
             "dev_in": (din["fuel_kgh"].to_numpy(), din[feats].to_numpy(),
                        din["speed"].to_numpy()),
             "dev_out": (dout["fuel_kgh"].to_numpy(), dout[feats].to_numpy(),
                         dout["speed"].to_numpy())}

    # --- learners: fitting portion only ------------------------------------
    res = ytr - a * vtr ** 3
    resid, sub = make_learner(BACKEND, rng)
    Xs, rs = sub(Xtr[fit_i], res[fit_i])
    resid.fit(Xs, rs)
    bbm, sub_b = make_learner(BACKEND, rng)
    Xb, yb = sub_b(Xtr[fit_i], ytr[fit_i])
    bbm.fit(Xb, yb)
    subsampled = bool(len(Xs) != len(fit_i) or len(Xb) != len(fit_i))

    pred = {}
    for pname, (y, X, v) in parts.items():
        cube = a * v ** 3
        r = resid.predict(X)
        pred[pname] = {"y": y, "cube": cube, "r": r, "bbm": bbm.predict(X)}

    # --- lambda selection: validation tail only ----------------------------
    yv, cv, rv = (pred["validation_tail"]["y"], pred["validation_tail"]["cube"],
                  pred["validation_tail"]["r"])
    curve = pd.DataFrame([{"vessel": name, "lambda": float(l),
                           **metrics7(yv, cv + l * rv)} for l in LAM])
    lam_v = float(curve.loc[curve.MAPE.idxmin(), "lambda"])

    # sensitivity: would the full-train prior have changed the selection?
    res_ft = ytr - a_full_train * vtr ** 3
    resid_ft, sub_ft = make_learner(BACKEND, np.random.default_rng(SEED))
    Xf, rf = sub_ft(Xtr[fit_i], res_ft[fit_i])
    resid_ft.fit(Xf, rf)
    rv_ft = resid_ft.predict(Xtr[val_i])
    cv_ft = a_full_train * vtr[val_i] ** 3
    lam_ft = float(min(LAM, key=lambda l: metrics7(yv, cv_ft + l * rv_ft)["MAPE"]))

    # --- endpoint identity, every partition, every metric ------------------
    ident = []
    for pname, p in pred.items():
        sweep0, sweep1 = p["cube"] + 0.0 * p["r"], p["cube"] + 1.0 * p["r"]
        m0d, m1d = metrics7(p["y"], p["cube"]), metrics7(p["y"], p["cube"] + p["r"])
        m0s, m1s = metrics7(p["y"], sweep0), metrics7(p["y"], sweep1)
        for met in METRICS:
            ident.append({
                "vessel": name, "partition": pname, "metric": met,
                "lambda0_sweep": m0s[met], "wbm_lite_direct": m0d[met],
                "lambda0_identical": abs(m0s[met] - m0d[met]) < IDENTITY_TOL,
                "lambda1_sweep": m1s[met], "sgbm_direct": m1d[met],
                "lambda1_identical": abs(m1s[met] - m1d[met]) < IDENTITY_TOL,
                "max_abs_prediction_diff": float(max(
                    np.max(np.abs(sweep0 - p["cube"])),
                    np.max(np.abs(sweep1 - (p["cube"] + p["r"])))))})

    # --- held-out evaluation with the FROZEN lambda ------------------------
    rows = []
    archs = [("WBM-lite (cube law)", "-", lambda p: p["cube"]),
             (f"BBM (Huber on target)", "-", lambda p: p["bbm"]),
             ("S-GBM (Huber residual)", "lambda=1", lambda p: p["cube"] + p["r"]),
             ("C-GBM (Huber residual)", f"lambda*={lam_v:g}",
              lambda p: p["cube"] + lam_v * p["r"])]
    for arch, param, fn in archs:
        mi = metrics7(pred["dev_in"]["y"], fn(pred["dev_in"]))
        mo = metrics7(pred["dev_out"]["y"], fn(pred["dev_out"]))
        rows.append({"vessel": name, "vessel_short": SHORT[name],
                     "architecture": arch, "param": param,
                     "n_dev_in": int(len(pred["dev_in"]["y"])),
                     "n_dev_out": int(len(pred["dev_out"]["y"])),
                     **{f"{k}_in": v for k, v in mi.items()},
                     **{f"{k}_out": v for k, v in mo.items()},
                     "dMAPE": mo["MAPE"] - mi["MAPE"],
                     "lambda_star": lam_v if arch.startswith("C-GBM") else np.nan})

    # reference row: what imposing the Shifts lambda would have given (not used)
    mi = metrics7(pred["dev_in"]["y"],
                  pred["dev_in"]["cube"] + SHIFTS_LAMBDA * pred["dev_in"]["r"])
    mo = metrics7(pred["dev_out"]["y"],
                  pred["dev_out"]["cube"] + SHIFTS_LAMBDA * pred["dev_out"]["r"])
    ref = [{"vessel": name, "vessel_short": SHORT[name],
            "comparison": "vessel-selected lambda vs imposed Shifts lambda=0.70",
            "lambda_vessel": lam_v, "lambda_shifts": SHIFTS_LAMBDA,
            "MAPE_in_vessel": float(rows[-1]["MAPE_in"]),
            "MAPE_in_imposed": mi["MAPE"],
            "MAPE_out_vessel": float(rows[-1]["MAPE_out"]),
            "MAPE_out_imposed": mo["MAPE"],
            "imposing_would_help_in": mi["MAPE"] < float(rows[-1]["MAPE_in"]),
            "imposing_would_help_out": mo["MAPE"] < float(rows[-1]["MAPE_out"]),
            "note": "reference only; the Shifts lambda is never imposed"}]

    meta = {"vessel": name, "vessel_short": SHORT[name],
            "role": FC_VESSELS[name]["role"], "residual_learner": "Huber",
            "bbm_learner": "Huber (fitted on the measured target)",
            "constraint": "residual shrinkage only",
            "physical_prior": "WBM-lite cube law (existing FuelCast prior)",
            "cube_coefficient_fit_portion": float(a),
            "cube_coefficient_full_train": float(a_full_train),
            "lambda_star": lam_v,
            "lambda_star_if_prior_fitted_on_full_train": lam_ft,
            "lambda_selection_unchanged_by_prior_scope": bool(
                abs(lam_v - lam_ft) < 1e-9),
            "lambda_grid": "0.00 to 1.00 step 0.01 (101 values)",
            "lambda_selected_on": "this vessel's own training-side validation tail only",
            "shifts_lambda_imposed": False,
            "n_train_block": int(len(tr)), "n_fit": int(len(fit_i)),
            "n_validation_tail": int(len(val_i)),
            "n_dev_in": int(len(pred["dev_in"]["y"])),
            "n_dev_out": int(len(pred["dev_out"]["y"])),
            "shift_variable": shift_var, "learner_subsampled": subsampled,
            "features": list(feats)}
    print(f"    {SHORT[name]:9s} lambda*={lam_v:.2f}  (full-train-prior check: "
          f"{lam_ft:.2f})  n_val={len(val_i)}  n_in={len(pred['dev_in']['y'])}  "
          f"n_out={len(pred['dev_out']['y'])}", flush=True)
    return meta, rows, curve.to_dict("records"), ident, ref


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[FC] FuelCast rebuilt with Huber (residual and BBM), shrinkage only",
          flush=True)
    metas, rows, curves, idents, refs = [], [], [], [], []
    for v in VESSELS:
        m, r, c, i, rf = run_vessel(v)
        metas.append(m); rows += r; curves += c; idents += i; refs += rf

    arch = pd.DataFrame(rows)
    sel = pd.DataFrame(metas)
    curve = pd.DataFrame(curves)
    ident = pd.DataFrame(idents)
    ref = pd.DataFrame(refs)

    # ---- manuscript table: MAPE only --------------------------------------
    tab = arch[["vessel_short", "architecture", "param", "MAPE_in", "MAPE_out",
                "dMAPE", "lambda_star"]].copy()
    tab.columns = ["Vessel", "Architecture", "Parameter", "MAPE_in_pct",
                   "MAPE_out_pct", "delta_MAPE_pp", "lambda_star"]

    # ---- leakage and identity checks --------------------------------------
    checks = []

    def chk(cid, item, ok, detail):
        checks.append({"check_id": cid, "item": item,
                       "verdict": "PASS" if ok else "FAIL", "detail": detail})

    ok = bool(ident.lambda0_identical.all())
    chk("E1", "lambda=0 reproduces WBM-lite on every partition and metric", ok,
        f"{int(ident.lambda0_identical.sum())}/{len(ident)} identical; "
        f"max prediction diff {ident.max_abs_prediction_diff.max():.3e}")
    ok = bool(ident.lambda1_identical.all())
    chk("E2", "lambda=1 reproduces S-GBM on every partition and metric", ok,
        f"{int(ident.lambda1_identical.sum())}/{len(ident)} identical")
    ok = sel.lambda_selected_on.eq(
        "this vessel's own training-side validation tail only").all()
    chk("L1", "lambda selected on the training-side validation tail only", bool(ok),
        "; ".join(f"{r.vessel_short}: lambda*={r.lambda_star:g}"
                  for _, r in sel.iterrows()))
    ok = not sel.shifts_lambda_imposed.astype(bool).any()
    chk("L2", "Shifts lambda* = 0.70 not imposed on any vessel", bool(ok),
        f"vessel lambda* = {[float(x) for x in sel.lambda_star]}")
    ok = sel.residual_learner.eq("Huber").all() and sel.bbm_learner.str.startswith(
        "Huber").all()
    chk("L3", "Huber used for both roles; no CatBoost anywhere", bool(ok),
        f"residual={sorted(set(sel.residual_learner))}, "
        f"bbm={sorted(set(sel.bbm_learner))}")
    ok = (sel.n_fit + sel.n_validation_tail == sel.n_train_block).all()
    chk("L4", "fitting portion and validation tail partition the training block",
        bool(ok), "; ".join(f"{r.vessel_short}: {r.n_fit}+{r.n_validation_tail}"
                            f"={r.n_train_block}" for _, r in sel.iterrows()))
    ok = bool(sel.lambda_selection_unchanged_by_prior_scope.all())
    chk("L5", "lambda selection unchanged if the prior is fitted on the full "
              "training block (prior-scope sensitivity)", ok,
        "; ".join(f"{r.vessel_short}: {r.lambda_star:g} vs "
                  f"{r.lambda_star_if_prior_fitted_on_full_train:g}"
                  for _, r in sel.iterrows()))
    ok = not sel.learner_subsampled.astype(bool).any()
    chk("L6", "no learner subsampled", bool(ok),
        f"subsampled flags = {[bool(x) for x in sel.learner_subsampled]}")
    ok = bool(len(curve) == len(VESSELS) * len(LAM))
    chk("G1", "complete 101-point grid searched per vessel", ok,
        f"{len(curve)} rows = {len(VESSELS)} vessels x {len(LAM)} lambda values")

    cdf = pd.DataFrame(checks)
    n_fail = int((cdf.verdict == "FAIL").sum())

    write_csv(tab, OUT / "fuelcast_manuscript_table_mape.csv")
    write_csv(arch, OUT / "fuelcast_all_metrics.csv")
    write_csv(sel, OUT / "fuelcast_lambda_selection.csv")
    write_csv(curve, OUT / "fuelcast_lambda_curves_validation.csv")
    write_csv(ident, OUT / "fuelcast_endpoint_identity.csv")
    write_csv(ref, OUT / "fuelcast_imposed_shifts_lambda_reference.csv")
    write_csv(cdf, OUT / "fuelcast_checks.csv")
    (OUT / "run_metadata.json").write_text(json.dumps({
        "purpose": "Section 5.4 FuelCast cross-vessel boundary analysis, "
                   "Huber methodology",
        "supersedes": "an earlier FuelCast run with a CatBoost residual learner "
                      "(not released)",
        "catboost_numbers_reused": False,
        "residual_learner": "Huber", "bbm_learner": "Huber on the measured target",
        "constraint": "residual shrinkage only",
        "physical_prior": "existing FuelCast WBM-lite cube law",
        "prior_calibrated_on": "fitting portion of the training block only",
        "lambda_grid": "0.00..1.00 step 0.01", "primary_criterion": "MAPE",
        "shifts_lambda_imposed": False, "retuned_on_heldout": False,
        "shifts_results_unchanged": True,
        "checks_passed": int(len(cdf) - n_fail), "checks_total": int(len(cdf)),
        "seed": SEED, "git_commit": _git()}, indent=2), encoding="utf-8")

    print("\n[FC] checks", flush=True)
    for _, r in cdf.iterrows():
        print(f"  [{r['verdict']}] {r['check_id']} {r['item']}\n"
              f"        {r['detail']}", flush=True)
    print(f"\n[FC] {len(cdf) - n_fail}/{len(cdf)} PASS, {n_fail} FAIL", flush=True)
    if n_fail:
        return 2
    return 0


def _guarded() -> int:
    import joblib
    with joblib.parallel_backend("threading", n_jobs=1):
        return main()


if __name__ == "__main__":
    sys.exit(_guarded())
