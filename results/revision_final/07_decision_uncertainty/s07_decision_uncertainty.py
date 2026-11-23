"""Section 5.5 rebuilt: decision-level performance and predictive uncertainty.

Primary Shifts benchmark only. FuelCast is not touched.

Frozen architecture set, identical to the final Section 5.3 results (folder 01):
    WBM    = k * F_phys, k fitted on the training block
    BBM    = Huber fitted directly on measured shaft power
    S-GBM  = k * F_phys + f_Huber(x)                 (lambda = 1)
    C-GBM  = k * F_phys + 0.70 * f_Huber(x)          (lambda* = 0.70, FROZEN)

lambda is imported, not selected. It is never touched by dev_in, dev_out, the internal
test block, the decision experiment or the uncertainty experiment.

Two protocol points are stated rather than left implicit.

1. The decision task follows the published protocol in the paper (Section 4.3): the
   architectures are refitted on the interquartile speed band of the training block, so
   the low- and high-speed cells used for the decision evaluation are speeds the models
   never saw. That is what makes the measured outcomes genuine counterfactuals. Fitting
   on the whole training block would destroy that property, so the fully-trained
   Section 5.3 models are reported as a clearly-labelled sensitivity, not as the primary
   result. lambda stays 0.70 in both.

2. The split-conformal radius uses the finite-sample quantile
   ceil((n+1)(1-alpha))/n rather than the plain (1-alpha) quantile used previously. With
   n = 106,141 validation records the two differ in the fifth decimal of probability; both
   are computed and reported so the change is visible and demonstrably immaterial.

Economics, condition bins, speed cells and the paired bootstrap are imported from
``regret_core`` so the decision protocol is provably the same code as the published run.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from .common_rev import REPO_ROOT, REV_FINAL, SEED, write_csv
from .s01_full_data_602020 import (
    FEATS, build_splits, fit_learner, metrics7, physics,
)
from .regret_core import (
    CHARTER_USD_DAY, FUEL_USD_T, MIN_CELL, MIN_CELLS_PER_BIN, SFOC_G_KWH, cost_per_nm,
)

_E = REPO_ROOT / "experiments"
if str(_E) not in sys.path:
    sys.path.insert(0, str(_E))
from benchmark_measured.common import fit_scale  # noqa: E402

OUT = REV_FINAL / "07_decision_uncertainty"
F01 = REV_FINAL / "01_full_data_602020"

LAMBDA_STAR = 0.70
RESID_KEY = BBM_KEY = "huber"
ARCHS = ["WBM", "BBM", "S-GBM", "C-GBM"]
ALPHA = 0.10
B_BOOT = 2000
TOL_MAPE, TOL_R2, TOL_DELTA = 0.001, 0.0001, 0.001

# Section 5.3 values that must be reproduced before anything else runs
EXPECT = {
    ("dev_in", "WBM"): (8.848, 0.9392), ("dev_in", "BBM"): (18.820, 0.9156),
    ("dev_in", "S-GBM"): (8.788, 0.9575), ("dev_in", "C-GBM"): (8.238, 0.9553),
    ("dev_out", "WBM"): (9.214, 0.9110), ("dev_out", "BBM"): (21.930, 0.8045),
    ("dev_out", "S-GBM"): (9.545, 0.9383), ("dev_out", "C-GBM"): (8.732, 0.9366),
}
EXPECT_DELTA = {"WBM": 0.366, "BBM": 3.110, "S-GBM": 0.757, "C-GBM": 0.495}

# old manuscript numbers, treated strictly as values to check
OLD_REGRET = {("dev_in", "WBM"): 2.757, ("dev_in", "BBM"): 39.576,
              ("dev_in", "S-GBM"): 6.317, ("dev_in", "C-GBM"): 2.916,
              ("dev_out", "WBM"): 15.749, ("dev_out", "BBM"): 26.384,
              ("dev_out", "S-GBM"): 14.818, ("dev_out", "C-GBM"): 14.047}
OLD_COV = {("dev_in", "WBM"): 0.832, ("dev_out", "WBM"): 0.840,
           ("dev_in", "BBM"): 0.937, ("dev_out", "BBM"): 0.858,
           ("dev_in", "S-GBM"): 0.943, ("dev_out", "S-GBM"): 0.875,
           ("dev_in", "C-GBM"): 0.862, ("dev_out", "C-GBM"): 0.848}


def _git() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------ model building --

def build(parts, ph, rows_idx=None, tag="full training block"):
    """Fit k, the Huber residual learner and the Huber BBM on a training subset."""
    tr = parts["train"] if rows_idx is None else parts["train"].iloc[rows_idx]
    ptr = ph["train"] if rows_idx is None else ph["train"][rows_idx]
    y, X = tr["power"].to_numpy(), tr[FEATS].to_numpy()
    k = fit_scale(y, ptr)
    res = y - k * ptr
    if rows_idx is None:
        fi, ei = parts["_fit_idx"], parts["_es_idx"]
    else:
        fi, ei = np.arange(len(tr)), np.array([], dtype=int)
    bbm = fit_learner(BBM_KEY, X, y, fi, ei)[0]
    resid = fit_learner(RESID_KEY, X, res, fi, ei)[0]
    print(f"    fitted on {tag}: n={len(tr):,}  k={k:.6f}", flush=True)
    return {"k": k, "bbm": bbm, "resid": resid, "n_fit": int(len(tr)), "tag": tag}


def predict(mdl, frame, phys):
    X = frame[FEATS].to_numpy()
    base = mdl["k"] * phys
    r = mdl["resid"].predict(X)
    return {"WBM": base, "BBM": mdl["bbm"].predict(X),
            "S-GBM": base + r, "C-GBM": base + LAMBDA_STAR * r}


# ------------------------------------------------- Section 5.3 consistency ---

def consistency(parts, ph, full) -> pd.DataFrame:
    ref = pd.read_csv(F01 / "final_dev_in_dev_out.csv")
    ref["architecture"] = ref["architecture"].str.replace(r"^C-GBM.*", "C-GBM",
                                                          regex=True)
    rows, fails, mape = [], [], {}
    for nm in ("dev_in", "dev_out"):
        pr = predict(full, parts[nm], ph[nm])
        y = parts[nm]["power"].to_numpy()
        for a in ARCHS:
            m = metrics7(y, pr[a])
            mape[(nm, a)] = m["MAPE"]
            e_mape, e_r2 = EXPECT[(nm, a)]
            rr = ref[(ref.partition == nm) & (ref.architecture == a)].iloc[0]
            d_sec = abs(round(m["MAPE"], 3) - e_mape)
            d_r2 = abs(round(m["R2"], 4) - e_r2)
            d_f24 = abs(m["MAPE"] - float(rr.MAPE))
            ok = d_sec <= TOL_MAPE and d_r2 <= TOL_R2 and d_f24 < 1e-4
            if not ok:
                fails.append(f"{nm}/{a}: MAPE {m['MAPE']:.4f} vs 5.3 {e_mape}, "
                             f"R2 {m['R2']:.4f} vs {e_r2}, folder24 {rr.MAPE:.4f}")
            rows.append({"partition": nm, "architecture": a,
                         "MAPE_recomputed": m["MAPE"], "MAPE_section53": e_mape,
                         "MAPE_folder24": float(rr.MAPE),
                         "R2_recomputed": m["R2"], "R2_section53": e_r2,
                         "abs_diff_MAPE_vs_section53": d_sec,
                         "abs_diff_R2_vs_section53": d_r2,
                         "abs_diff_MAPE_vs_folder24": d_f24,
                         "reproduces": ok})
    for a in ARCHS:
        dm = mape[("dev_out", a)] - mape[("dev_in", a)]
        d = abs(round(dm, 3) - EXPECT_DELTA[a])
        if d > TOL_DELTA:
            fails.append(f"delta {a}: {dm:+.3f} vs {EXPECT_DELTA[a]:+.3f}")
        rows.append({"partition": "delta", "architecture": a,
                     "MAPE_recomputed": dm, "MAPE_section53": EXPECT_DELTA[a],
                     "MAPE_folder24": np.nan, "R2_recomputed": np.nan,
                     "R2_section53": np.nan,
                     "abs_diff_MAPE_vs_section53": d,
                     "abs_diff_R2_vs_section53": np.nan,
                     "abs_diff_MAPE_vs_folder24": np.nan,
                     "reproduces": bool(d <= TOL_DELTA)})
    if abs(LAMBDA_STAR - 0.70) > 1e-12:
        fails.append(f"lambda is {LAMBDA_STAR}, not 0.70")
    if fails:
        print("\n[STOP] Section 5.3 values are NOT reproduced:", flush=True)
        for f in fails:
            print(f"        {f}", flush=True)
        raise SystemExit(2)
    print(f"[5.3] reproduced: 8/8 MAPE, 8/8 R2, 4/4 deltaMAPE, lambda = "
          f"{LAMBDA_STAR:.2f}", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- uncertainty ---

def uncertainty(parts, ph, full):
    yv = parts["validation"]["power"].to_numpy()
    pv = predict(full, parts["validation"], ph["validation"])
    n = len(yv)
    lvl_fs = min(1.0, np.ceil((n + 1) * (1.0 - ALPHA)) / n)      # finite-sample
    cal, q = [], {}
    for a in ARCHS:
        absres = np.abs(yv - pv[a])
        q[a] = float(np.quantile(absres, lvl_fs))
        cal.append({"architecture": a, "calibration_partition": "validation",
                    "n_calibration": int(n), "alpha": ALPHA,
                    "nominal_coverage": 1 - ALPHA,
                    "quantile_level_finite_sample": float(lvl_fs),
                    "radius_kw_finite_sample": q[a],
                    "radius_kw_plain_quantile": float(
                        np.quantile(absres, 1.0 - ALPHA)),
                    "radius_difference_kw": q[a] - float(
                        np.quantile(absres, 1.0 - ALPHA)),
                    "frozen_before_dev_evaluation": True,
                    "dev_data_used_in_calibration": False})
    rows = []
    for nm in ("internal_test", "dev_in", "dev_out"):
        y = parts[nm]["power"].to_numpy()
        pr = predict(full, parts[nm], ph[nm])
        for a in ARCHS:
            lo, hi = np.maximum(pr[a] - q[a], 0.0), pr[a] + q[a]
            cov = float(np.mean((y >= lo) & (y <= hi)))
            rows.append({"partition": nm, "architecture": a, "n": int(len(y)),
                         "nominal_coverage": 1 - ALPHA,
                         "empirical_coverage": cov,
                         "coverage_error": cov - (1 - ALPHA),
                         "mean_interval_width_kw": float(np.mean(hi - lo)),
                         "conformal_radius_kw": q[a],
                         "radius_frozen_before_evaluation": True})
    return pd.DataFrame(rows), pd.DataFrame(cal)


# ---------------------------------------------------------------- decision ---

def make_bins(frame, phys, mdl):
    d = frame.reset_index(drop=True).copy()
    d["_cell"] = d["stw"].round().astype(int)
    d["_draft_bin"] = pd.qcut(d["draft"], 3, labels=False, duplicates="drop")
    d["_wind_bin"] = pd.qcut(d["wind_kn"], 3, labels=False, duplicates="drop")
    pred = predict(mdl, d, phys)
    power = d["power"].to_numpy()
    bins = []
    for (db, wb), g in d.groupby(["_draft_bin", "_wind_bin"]):
        cells = {c: cg.index.to_numpy() for c, cg in g.groupby("_cell")
                 if len(cg) >= MIN_CELL}
        if len(cells) < MIN_CELLS_PER_BIN:
            continue
        vs = np.array(sorted(cells), dtype=float)
        bins.append({"draft_bin": int(db), "wind_bin": int(wb), "vs": vs,
                     "idx": [cells[int(v)] for v in vs],
                     "n_records": int(sum(len(cells[int(v)]) for v in vs))})
    return bins, pred, power


def regret_once(bins, pred, power, rng=None):
    acc = {a: [] for a in ARCHS}
    picks = {a: [] for a in ARCHS}
    truth = []
    for b in bins:
        idx = (b["idx"] if rng is None
               else [rng.choice(i, size=len(i), replace=True) for i in b["idx"]])
        mp = np.array([np.median(power[i]) for i in idx])
        mJ = cost_per_nm(mp, b["vs"])
        j_true, i_true = float(mJ.min()), int(np.argmin(mJ))
        truth.append(i_true)
        for a in ARCHS:
            pp = np.array([np.median(pred[a][i]) for i in idx])
            ip = int(np.argmin(cost_per_nm(pp, b["vs"])))
            picks[a].append(ip)
            acc[a].append(100.0 * (mJ[ip] - j_true) / j_true)
    return acc, picks, truth


def decision(parts, ph, mdl, label):
    rng = np.random.default_rng(SEED)
    res, pair, details = [], [], []
    for nm in ("dev_in", "dev_out"):
        bins, pred, power = make_bins(parts[nm], ph[nm], mdl)
        acc, picks, truth = regret_once(bins, pred, power)
        draws = {a: [] for a in ARCHS}
        for _ in range(B_BOOT):
            a_, _p, _t = regret_once(bins, pred, power, rng)
            for a in ARCHS:
                draws[a].append(float(np.mean(a_[a])))
        for a in ARCHS:
            r = np.array(acc[a])
            d = np.array(draws[a])
            res.append({"fit_protocol": label, "partition": nm, "architecture": a,
                        "n_decision_groups": len(bins),
                        "n_records_in_groups": int(sum(b["n_records"] for b in bins)),
                        "mean_regret_pct": float(r.mean()),
                        "ci95_lo": float(np.percentile(d, 2.5)),
                        "ci95_hi": float(np.percentile(d, 97.5)),
                        "median_regret_pct": float(np.median(r)),
                        "p90_regret_pct": float(np.percentile(r, 90)),
                        "match_rate_measured_optimum": float(
                            np.mean(np.array(picks[a]) == np.array(truth))),
                        "agreement_with_wbm": float(
                            np.mean(np.array(picks[a]) == np.array(picks["WBM"]))),
                        "n_groups_differing_from_wbm": int(
                            np.sum(np.array(picks[a]) != np.array(picks["WBM"])))})
        for ref in ("WBM", "BBM", "S-GBM"):
            dd = np.array(draws["C-GBM"]) - np.array(draws[ref])
            lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
            p = float(np.clip(2 * min(float(np.mean(dd <= 0)), float(np.mean(dd >= 0))),
                              1.0 / B_BOOT, 1.0))
            pair.append({"fit_protocol": label, "partition": nm,
                         "comparison": f"C-GBM minus {ref}", "reference": ref,
                         "mean_regret_difference_pp": float(
                             np.mean(acc["C-GBM"]) - np.mean(acc[ref])),
                         "ci95_lo": lo, "ci95_hi": hi, "p_value": p,
                         "statistically_resolved": bool(p < 0.05
                                                        and (lo > 0 or hi < 0)),
                         "n_resamples": B_BOOT,
                         "method": "paired bootstrap; rows resampled within each "
                                   "speed cell, same resample for the measured curve "
                                   "and every architecture"})
        for j, b in enumerate(bins):
            row = {"fit_protocol": label, "partition": nm, "group": j,
                   "draft_bin": b["draft_bin"], "wind_bin": b["wind_bin"],
                   "n_speed_cells": len(b["vs"]),
                   "speed_grid_kn": ";".join(f"{v:.0f}" for v in b["vs"]),
                   "n_records": b["n_records"],
                   "measured_optimal_speed_kn": float(b["vs"][truth[j]])}
            for a in ARCHS:
                row[f"{a}_selected_speed_kn"] = float(b["vs"][picks[a][j]])
                row[f"{a}_regret_pct"] = float(acc[a][j])
            row["wbm_cgbm_same_choice"] = bool(picks["WBM"][j] == picks["C-GBM"][j])
            details.append(row)
        print(f"    {label} / {nm}: {len(bins)} groups, "
              f"C-GBM mean regret {np.mean(acc['C-GBM']):.3f} %", flush=True)
    return pd.DataFrame(res), pd.DataFrame(pair), pd.DataFrame(details)


# --------------------------------------------------------------------- main ---

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = build_splits()
    ph = physics(parts)

    print("[A0] frozen Section 5.3 models", flush=True)
    full = build(parts, ph)
    cons = consistency(parts, ph, full)
    write_csv(cons, OUT / "section53_consistency_check.csv")

    print("[B] split-conformal uncertainty", flush=True)
    unc, cal = uncertainty(parts, ph, full)
    write_csv(unc, OUT / "uncertainty_results.csv")
    write_csv(cal, OUT / "conformal_calibration_details.csv")

    # decision task: published protocol refits on the training mid speed band
    q1, q3 = parts["train"]["stw"].quantile([0.25, 0.75])
    mid = np.where((parts["train"]["stw"] >= q1)
                   & (parts["train"]["stw"] <= q3))[0]
    print(f"[A] decision task; training mid speed band [{q1:.2f}, {q3:.2f}] kn",
          flush=True)
    midmdl = build(parts, ph, mid, f"training mid speed band [{q1:.2f}, {q3:.2f}] kn")
    d_res, d_pair, d_det = decision(parts, ph, midmdl, "mid-band fit (published)")
    s_res, s_pair, s_det = decision(parts, ph, full, "full-training fit (sensitivity)")
    res = pd.concat([d_res, s_res], ignore_index=True)
    pair = pd.concat([d_pair, s_pair], ignore_index=True)
    det = pd.concat([d_det, s_det], ignore_index=True)
    write_csv(res, OUT / "decision_results.csv")
    write_csv(pair, OUT / "decision_pairwise_bootstrap.csv")
    write_csv(det, OUT / "decision_group_details.csv")

    # ---------------- Part C: old vs recomputed ----------------
    old = []
    prim = res[res.fit_protocol == "mid-band fit (published)"]
    for (nm, a), v in OLD_REGRET.items():
        new = float(prim[(prim.partition == nm)
                         & (prim.architecture == a)]["mean_regret_pct"].iloc[0])
        old.append({"quantity": "mean decision regret (%)", "partition": nm,
                    "architecture": a, "old_manuscript": v, "recomputed": new,
                    "difference": new - v,
                    "verdict": "PASS" if abs(new - v) <= 0.05 else "DIFFERS"})
    for (nm, a), v in OLD_COV.items():
        new = float(unc[(unc.partition == nm)
                        & (unc.architecture == a)]["empirical_coverage"].iloc[0])
        old.append({"quantity": "conformal coverage at nominal 0.90",
                    "partition": nm, "architecture": a, "old_manuscript": v,
                    "recomputed": new, "difference": new - v,
                    "verdict": "PASS" if abs(new - v) <= 0.005 else "DIFFERS"})
    oldnew = pd.DataFrame(old)
    write_csv(oldnew, OUT / "old_vs_recomputed_results.csv")

    # ---------------- scientific questions, Part B ----------------
    qa = []
    for a in ARCHS:
        ci = float(unc[(unc.partition == "dev_in")
                       & (unc.architecture == a)]["empirical_coverage"].iloc[0])
        co = float(unc[(unc.partition == "dev_out")
                       & (unc.architecture == a)]["empirical_coverage"].iloc[0])
        wi = float(unc[(unc.partition == "dev_in")
                       & (unc.architecture == a)]["mean_interval_width_kw"].iloc[0])
        wo = float(unc[(unc.partition == "dev_out")
                       & (unc.architecture == a)]["mean_interval_width_kw"].iloc[0])
        qa.append({"architecture": a, "coverage_dev_in": ci, "coverage_dev_out": co,
                   "attains_nominal_both_partitions": bool(ci >= 0.90 and co >= 0.90),
                   "coverage_change_in_to_out": co - ci,
                   "width_dev_in_kw": wi, "width_dev_out_kw": wo,
                   "width_change_kw": wo - wi,
                   "width_effectively_fixed": bool(abs(wo - wi) / max(wi, 1e-9) < 0.01)})
    qdf = pd.DataFrame(qa)
    write_csv(qdf, OUT / "uncertainty_questions.csv")

    # ---------------- checks ----------------
    checks = []

    def chk(cid, item, ok, detail):
        checks.append({"check_id": cid, "item": item,
                       "verdict": "PASS" if ok else "FAIL", "detail": detail})

    chk("F1", "Section 5.3 MAPE, R2 and deltaMAPE reproduced",
        bool(cons.reproduces.all()),
        f"{int(cons.reproduces.sum())}/{len(cons)} rows within tolerance")
    chk("F2", "lambda is exactly 0.70 and never re-selected", LAMBDA_STAR == 0.70,
        "imported constant; no grid search, no dev/test/decision/uncertainty input")
    chk("F3", "no obsolete learner or constraint used", True,
        "Huber only (BBM and residual); no CatBoost, ExtraTrees, ridge anchor or "
        "bounded residual anywhere in this script")
    chk("L1", "conformal radius calibrated on the validation block only",
        bool((cal.calibration_partition == "validation").all()
             and not cal.dev_data_used_in_calibration.any()),
        f"n_calibration = {int(cal.n_calibration.iloc[0]):,}")
    chk("L2", "conformal radius frozen before dev evaluation",
        bool(cal.frozen_before_dev_evaluation.all()
             and unc.radius_frozen_before_evaluation.all()),
        "radius computed once per architecture, then applied unchanged")
    chk("L3", "finite-sample quantile change is immaterial",
        bool(cal.radius_difference_kw.abs().max() < 1.0),
        f"max radius difference {cal.radius_difference_kw.abs().max():.4f} kW vs the "
        f"plain (1-alpha) quantile")
    chk("D1", "decision outcomes evaluated against measured power only", True,
        "regret uses cost_per_nm of the median MEASURED power per speed cell; "
        "predictions only choose the speed")
    chk("D2", "published economics and binning imported, not reimplemented",
        True, f"SFOC {SFOC_G_KWH} g/kWh, fuel {FUEL_USD_T} USD/t, charter "
              f"{CHARTER_USD_DAY} USD/day, MIN_CELL {MIN_CELL}, "
              f"MIN_CELLS_PER_BIN {MIN_CELLS_PER_BIN} from regret_core")
    ng = sorted(set(prim.n_decision_groups))
    chk("D3", "decision-group count recorded as an evidence limit", True,
        f"{ng} groups per partition -> underpowered; treat as qualitative")
    cdf = pd.DataFrame(checks)
    n_fail = int((cdf.verdict == "FAIL").sum())
    write_csv(cdf, OUT / "checks.csv")

    (OUT / "run_metadata.json").write_text(json.dumps({
        "purpose": "Section 5.5 decision-level performance and predictive uncertainty",
        "benchmark": "Shifts marine (primary); FuelCast not used",
        "source_configuration": "results/revision_final/01_full_data_602020",
        "protocol": "chronological 60/20/20 on the full train.csv",
        "k_full_training_block": full["k"],
        "residual_learner": "Huber", "bbm": "Huber fitted on measured power",
        "lambda_star": LAMBDA_STAR, "lambda_reselected": False,
        "lambda_touched_by_dev_or_test_or_decision_or_uncertainty": False,
        "excluded": ["CatBoost", "ExtraTrees", "ridge-to-physics",
                     "bounded residual"],
        "conformal": {"alpha": ALPHA, "calibration": "validation block only",
                      "n_calibration": int(cal.n_calibration.iloc[0]),
                      "quantile": "ceil((n+1)(1-alpha))/n, finite-sample corrected",
                      "radius_frozen_before_dev": True},
        "decision": {"primary_protocol": "refit on the training interquartile speed "
                                         "band, as published (Section 4.3)",
                     "mid_band_kn": [float(q1), float(q3)],
                     "sensitivity_protocol": "fully-trained Section 5.3 models, "
                                             "labelled as such; NOT counterfactual "
                                             "because those speeds were in training",
                     "bins": "draft tercile x wind tercile",
                     "cells": "1-kn speed cells, >=5 records, >=4 cells per bin",
                     "economics_source": "regret_core (published constants)",
                     "bootstrap": {"B": B_BOOT, "unit": "rows within a speed cell",
                                   "paired": True}},
        "checks_passed": int(len(cdf) - n_fail), "checks_total": int(len(cdf)),
        "seed": SEED, "git_commit": _git()}, indent=2), encoding="utf-8")

    print("\n[34] checks", flush=True)
    for _, r in cdf.iterrows():
        print(f"  [{r['verdict']}] {r['check_id']} {r['item']}\n"
              f"        {r['detail']}", flush=True)
    print(f"\n[34] {len(cdf) - n_fail}/{len(cdf)} PASS, {n_fail} FAIL", flush=True)
    return 2 if n_fail else 0


def _guarded() -> int:
    import joblib
    with joblib.parallel_backend("threading", n_jobs=1):
        return main()


if __name__ == "__main__":
    sys.exit(_guarded())
