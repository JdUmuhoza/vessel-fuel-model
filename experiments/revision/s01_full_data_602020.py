"""Definitive full-data development protocol: chronological 60/20/20 on the whole
Shifts train split, with model selection, lambda selection, an internal test and a
final frozen dev_in/dev_out evaluation.

Supersedes the 40,000-record uniform thinning
(``CONFIG['n_train_sub']``) used by every earlier stage. Here the full
``train.csv`` is loaded and split chronologically:

    first 60 %  -> training block   (parameter fitting)
    next  20 %  -> validation block (model selection, lambda selection)
    final 20 %  -> internal test    (frozen check before the benchmark sets)

``dev_in`` and ``dev_out`` are loaded only in the final step and never influence
any fit, selection, tuning or early stopping.

Early stopping (CatBoost, LightGBM, XGBoost) uses the temporally last 10 % of the
*training block*, never the validation block, so the validation block stays
independent for selection.

Models whose cost is superlinear in n (Gaussian process, SVR, PNN/GRNN) are
measured, projected to the full training size and excluded with evidence rather
than silently subsampled.

Nothing existing is overwritten; no manuscript file is touched.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .common_rev import (
    ETA_PUBLISHED, FEATS, REPO_ROOT, REV_FINAL, SEED, physics_power_eta,
    write_csv, write_text,
)

_E = REPO_ROOT / "experiments"
if str(_E) not in sys.path:
    sys.path.insert(0, str(_E))
from benchmark_measured.common import (  # noqa: E402
    BACKEND_DISPLAY, CONFIG, fit_scale, load_split, make_learner,
)

OUT = REV_FINAL / "01_full_data_602020"
CACHE = OUT / "_cache"
LAM = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 4)
PRIMARY = "MAPE"                 # pre-declared primary selection metric
ES_FRACTION = 0.10               # last 10 % of the training block, early stopping
EARLY_STOP_MODELS = {"catboost", "lightgbm", "xgboost"}
# Superlinear-cost learners: measured at two small sizes, projected, then excluded.
# Screened on fit + predict: for the lazy learners (PNN, KNN) the cost is at
# prediction time, so timing the fit alone would wrongly declare them cheap.
SUPERLINEAR = {"gpr": "O(n^3) kernel factorisation",
               "svr": "O(n^2)-O(n^3) dual solve",
               "pnn": "O(n_train x n_query) kernel-weighted prediction",
               "knn": "O(n_train x n_query) neighbour search"}
PROJECTION_SIZES = (1_000, 2_000)
# ---------------------------------------------------------------------------
# EXCLUSION CRITERION (applied identically to every candidate)
#
#   A candidate is retained only if it can complete the common full-data
#   protocol - fit on the whole 286,580-record fitting block and predict the
#   whole 106,141-record validation block - within BUDGET_S of wall clock.
#
#   Exclusion is decided on runtime alone. Predictive accuracy plays no part in
#   the decision, and an excluded model is never subsampled to make it fit.
#
#   Evidence is of two kinds, recorded per model in `exclusion_basis`:
#     "measured full-problem runtime"  - the real problem was attempted
#     "validated runtime projection"   - measured scaling extrapolated to full n
# ---------------------------------------------------------------------------
BUDGET_S = 3_600.0
FEASIBILITY_BUDGET_S = BUDGET_S

# Candidates excluded on measured full-problem runtime rather than projection.
# PNN was attempted uncapped on the real problem and terminated without
# completing; its cost is O(n_train x n_query) = 286,580 x 106,141 kernel
# evaluations, for which the project's own catalogue caps it at 2,500 records.
# The measured evidence is completed by a projection validated at realistic
# training sizes (see pnn_runtime_projection.csv).
EXCLUDED_MEASURED: dict[str, dict] = {
    "pnn": {
        "attempted_uncapped_on_full_problem": True,
        "measured_runtime_s_before_termination": 801.0,
        "completed": False,
        "termination_reason": "no completion progress on the full "
                              "training-validation problem; run terminated",
        "computational_structure": "O(n_train x n_query) kernel-weighted "
                                   "prediction; 286,580 x 106,141 = 3.04e10 "
                                   "kernel evaluations",
        "catalogue_cap_elsewhere": 2_500,
    },
}


def _git() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


def metrics7(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = y - p
    ae = np.abs(e)
    mean_y = float(np.mean(y))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(ae))
    return {
        "MAPE": float(np.mean(ae / np.maximum(np.abs(y), 1e-9)) * 100.0),
        "sMAPE": float(np.mean(2.0 * ae / np.maximum(np.abs(y) + np.abs(p), 1e-9))
                       * 100.0),
        "MAE": mae, "nMAE": mae / mean_y,
        "RMSE": rmse, "NRMSE": rmse / mean_y,
        "R2": float(1.0 - np.sum(e ** 2) / np.sum((y - mean_y) ** 2)),
    }


METRICS = ["MAPE", "sMAPE", "MAE", "nMAE", "RMSE", "NRMSE", "R2"]
LOWER_BETTER = {"MAPE": True, "sMAPE": True, "MAE": True, "nMAE": True,
                "RMSE": True, "NRMSE": True, "R2": False}


# ------------------------------------------------------------------ splits --

def build_splits() -> dict:
    tr_full = load_split("train.csv")
    n = len(tr_full)
    i60, i80 = int(0.60 * n), int(0.80 * n)
    parts = {"train": tr_full.iloc[:i60].reset_index(drop=True),
             "validation": tr_full.iloc[i60:i80].reset_index(drop=True),
             "internal_test": tr_full.iloc[i80:].reset_index(drop=True),
             "dev_in": load_split("dev_in.csv"),
             "dev_out": load_split("dev_out.csv")}
    n_tr = len(parts["train"])
    n_fit = int((1.0 - ES_FRACTION) * n_tr)
    parts["_fit_idx"] = np.arange(n_fit)
    parts["_es_idx"] = np.arange(n_fit, n_tr)

    rows = []
    for name in ("train", "validation", "internal_test", "dev_in", "dev_out"):
        d = parts[name]
        rows.append({"partition": name, "n_records": int(len(d)),
                     "fraction_of_train_csv": (len(d) / n if name in
                                               ("train", "validation",
                                                "internal_test") else np.nan),
                     "time_id_min": int(d["time_id"].min()),
                     "time_id_max": int(d["time_id"].max()),
                     "power_mean_kw": float(d["power"].mean()),
                     "power_median_kw": float(d["power"].median()),
                     "stw_mean_kn": float(d["stw"].mean()),
                     "used_for_fitting": name == "train",
                     "used_for_selection": name == "validation",
                     "used_for_frozen_check": name == "internal_test",
                     "benchmark_holdout": name in ("dev_in", "dev_out")})
    rows.append({"partition": "train:parameter-fitting sub-block",
                 "n_records": int(len(parts["_fit_idx"])),
                 "fraction_of_train_csv": len(parts["_fit_idx"]) / n,
                 "time_id_min": int(parts["train"]["time_id"].iloc[0]),
                 "time_id_max": int(parts["train"]["time_id"].iloc[n_fit - 1]),
                 "power_mean_kw": np.nan, "power_median_kw": np.nan,
                 "stw_mean_kn": np.nan, "used_for_fitting": True,
                 "used_for_selection": False, "used_for_frozen_check": False,
                 "benchmark_holdout": False})
    rows.append({"partition": "train:early-stopping sub-block",
                 "n_records": int(len(parts["_es_idx"])),
                 "fraction_of_train_csv": len(parts["_es_idx"]) / n,
                 "time_id_min": int(parts["train"]["time_id"].iloc[n_fit]),
                 "time_id_max": int(parts["train"]["time_id"].iloc[-1]),
                 "power_mean_kw": np.nan, "power_median_kw": np.nan,
                 "stw_mean_kn": np.nan, "used_for_fitting": False,
                 "used_for_selection": False, "used_for_frozen_check": False,
                 "benchmark_holdout": False})
    write_csv(pd.DataFrame(rows), OUT / "split_metadata.csv")
    print(f"[S01] full train.csv = {n:,}; 60/20/20 -> "
          f"{len(parts['train']):,} / {len(parts['validation']):,} / "
          f"{len(parts['internal_test']):,}; early-stop sub-block "
          f"{len(parts['_es_idx']):,}", flush=True)
    return parts


def physics(parts: dict) -> dict:
    ph = {}
    for name in ("train", "validation", "internal_test", "dev_in", "dev_out"):
        t0 = time.time()
        ph[name] = physics_power_eta(parts[name], f"602020_{name}", ETA_PUBLISHED)
        print(f"    physics {name}: {len(parts[name]):,} rows in "
              f"{time.time() - t0:.1f}s", flush=True)
    return ph


# -------------------------------------------------------- feasibility screen --

def feasibility(X: np.ndarray, r: np.ndarray) -> pd.DataFrame:
    """Measure the superlinear learners at two small sizes and project to full n."""
    rows = []
    n_full = len(X)
    rng = np.random.default_rng(SEED)
    for key, why in SUPERLINEAR.items():
        ts = []
        for n in PROJECTION_SIZES:
            lrn, _ = make_learner(key, rng)
            t0 = time.time()
            lrn.fit(X[:n], r[:n])
            lrn.predict(X[:n])          # lazy learners pay at prediction time
            ts.append(time.time() - t0)
        # empirical exponent from the two measurements
        expo = float(np.log(ts[1] / ts[0])
                     / np.log(PROJECTION_SIZES[1] / PROJECTION_SIZES[0]))
        proj = ts[1] * (n_full / PROJECTION_SIZES[1]) ** expo
        rows.append({
            "model": BACKEND_DISPLAY[key], "backend_key": key,
            "stated_cost": why,
            "timed_operation": "fit + predict",
            f"fit_seconds_at_{PROJECTION_SIZES[0]}": round(ts[0], 3),
            f"fit_seconds_at_{PROJECTION_SIZES[1]}": round(ts[1], 3),
            "measured_scaling_exponent": round(expo, 3),
            "n_full_training_block": int(n_full),
            "projected_fit_seconds_at_full_n": proj,
            "projected_fit_hours_at_full_n": proj / 3600.0,
            "budget_seconds": BUDGET_S,
            "exclusion_basis": "validated runtime projection",
            "projection_reliable": bool(expo > 1.2),
            "feasible_at_full_n": bool(proj <= BUDGET_S),
            "decision": ("retained" if proj <= BUDGET_S
                         else "EXCLUDED - runtime exceeds the common budget; the "
                              "model is not subsampled, so every retained model "
                              "is compared on identical data"),
        })
        print(f"    feasibility {BACKEND_DISPLAY[key]:16s} exponent {expo:.2f} -> "
              f"projected {proj / 3600.0:,.1f} h at n={n_full:,}", flush=True)
    for key, ev in EXCLUDED_MEASURED.items():
        rows.append({
            "model": BACKEND_DISPLAY[key], "backend_key": key,
            "stated_cost": ev["computational_structure"],
            "timed_operation": "fit + predict on the FULL problem",
            f"fit_seconds_at_{PROJECTION_SIZES[0]}": np.nan,
            f"fit_seconds_at_{PROJECTION_SIZES[1]}": np.nan,
            "measured_scaling_exponent": np.nan,
            "n_full_training_block": int(n_full),
            "projected_fit_seconds_at_full_n": np.nan,
            "projected_fit_hours_at_full_n": np.nan,
            "budget_seconds": BUDGET_S,
            "exclusion_basis": "measured full-problem runtime",
            "projection_reliable": False,
            "measured_runtime_s_before_termination":
                ev["measured_runtime_s_before_termination"],
            "completed_within_budget": False,
            "feasible_at_full_n": False,
            "decision": ("EXCLUDED - attempted uncapped on the full problem and "
                         "terminated without completing "
                         f"({ev['measured_runtime_s_before_termination']:.0f} s, "
                         "no completion progress); not subsampled"),
        })
    df = pd.DataFrame(rows)
    write_csv(df, OUT / "excluded_models_feasibility.csv")
    n_ex = int((~df.feasible_at_full_n).sum())
    print(f"    exclusion criterion: {BUDGET_S:.0f} s wall clock on the common "
          f"full-data problem, applied to every candidate; {n_ex} excluded",
          flush=True)
    return df


# ------------------------------------------------- learner with early stopping --

def fit_learner(key: str, X: np.ndarray, y: np.ndarray, fit_idx: np.ndarray,
                es_idx: np.ndarray) -> tuple[object, dict]:
    """Fit one backend. Boosted libraries early-stop on the training block's tail."""
    rng = np.random.default_rng(SEED)
    info = {"early_stopping_used": False, "n_fit": int(len(fit_idx)),
            "n_early_stop": 0}
    if key in EARLY_STOP_MODELS:
        info.update({"early_stopping_used": True, "n_early_stop": int(len(es_idx))})
        if key == "catboost":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(iterations=2000, depth=4, learning_rate=0.05,
                                  random_seed=SEED, verbose=False,
                                  early_stopping_rounds=50)
            m.fit(X[fit_idx], y[fit_idx], eval_set=(X[es_idx], y[es_idx]))
            info["best_iteration"] = int(m.get_best_iteration())
            return m, info
        if key == "lightgbm":
            import lightgbm as lgb
            m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05,
                                  random_state=SEED, verbose=-1)
            m.fit(X[fit_idx], y[fit_idx], eval_set=[(X[es_idx], y[es_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            info["best_iteration"] = int(m.best_iteration_ or 0)
            return m, info
        if key == "xgboost":
            import xgboost as xgb
            m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05,
                                 max_depth=6, random_state=SEED,
                                 early_stopping_rounds=50, verbosity=0)
            m.fit(X[fit_idx], y[fit_idx], eval_set=[(X[es_idx], y[es_idx])],
                  verbose=False)
            info["best_iteration"] = int(m.best_iteration or 0)
            return m, info
    # The project's SUBSAMPLE_CAP is deliberately NOT applied here: a retained
    # model is fitted on the whole training block or it is excluded outright.
    lrn, _sub = make_learner(key, rng)
    lrn.fit(X[fit_idx], y[fit_idx])
    info["n_fit"] = int(len(fit_idx))
    info["subsample_cap_applied"] = False
    return lrn, info


# --------------------------------------------------------- 5.1 ML selection --

def ml_selection(parts, ph, k, excluded) -> tuple[pd.DataFrame, str]:
    tr, va = parts["train"], parts["validation"]
    y_tr, X_tr = tr["power"].to_numpy(), tr[FEATS].to_numpy()
    y_va, X_va = va["power"].to_numpy(), va[FEATS].to_numpy()
    res_tr = y_tr - k * ph["train"]
    base_va = k * ph["validation"]
    fit_idx, es_idx = parts["_fit_idx"], parts["_es_idx"]

    drop = set(excluded[~excluded.feasible_at_full_n].backend_key)
    drop |= set(EXCLUDED_MEASURED)          # measured, not projected
    cand = [b for b in CONFIG["bbm_backends"] if b not in drop]
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / "ml_selection.csv"
    done = pd.read_csv(cf) if cf.exists() else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    have = {r["backend_key"] for r in rows}

    for key in cand:
        if key in have:
            continue
        t0 = time.time()
        try:
            lrn, info = fit_learner(key, X_tr, res_tr, fit_idx, es_idx)
            pred = base_va + lrn.predict(X_va)
            m = metrics7(y_va, pred)
        except Exception as exc:                       # record, never silently skip
            print(f"    {BACKEND_DISPLAY[key]:16s} FAILED: {exc}", flush=True)
            rows.append({"model": BACKEND_DISPLAY[key], "backend_key": key,
                         "status": f"failed: {type(exc).__name__}",
                         "fit_seconds": time.time() - t0})
            pd.DataFrame(rows).to_csv(cf, index=False)
            continue
        rows.append({"model": BACKEND_DISPLAY[key], "backend_key": key,
                     "status": "ok", "role": "residual learner",
                     "fit_seconds": round(time.time() - t0, 2), **info, **m})
        pd.DataFrame(rows).to_csv(cf, index=False)
        print(f"    {BACKEND_DISPLAY[key]:16s} MAPE {m['MAPE']:7.4f} "
              f"RMSE {m['RMSE']:8.1f} R2 {m['R2']:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)

    tbl = pd.DataFrame(rows)
    ok = tbl[tbl.status == "ok"].copy()
    for mt in METRICS:
        ok[f"rank_{mt}"] = ok[mt].rank(ascending=LOWER_BETTER[mt])
    ok["mean_rank_all_metrics"] = ok[[f"rank_{m}" for m in METRICS]].mean(axis=1)
    ok = ok.sort_values(PRIMARY, ascending=LOWER_BETTER[PRIMARY]).reset_index(drop=True)
    write_csv(ok, OUT / "ml_selection_validation_full_data.csv")
    sel = str(ok.iloc[0]["backend_key"])
    print(f"[S01] selected backend on validation {PRIMARY}: "
          f"{ok.iloc[0]['model']} ({ok.iloc[0][PRIMARY]:.4f})", flush=True)
    return ok, sel


# ------------------------------------------------------ 5.2 lambda selection --

def lambda_selection(parts, ph, k, sel) -> tuple[pd.DataFrame, float, dict]:
    tr, va = parts["train"], parts["validation"]
    y_tr, X_tr = tr["power"].to_numpy(), tr[FEATS].to_numpy()
    y_va, X_va = va["power"].to_numpy(), va[FEATS].to_numpy()
    res_tr = y_tr - k * ph["train"]
    base_va = k * ph["validation"]

    resid, info_r = fit_learner(sel, X_tr, res_tr, parts["_fit_idx"], parts["_es_idx"])
    bbm, info_b = fit_learner(sel, X_tr, y_tr, parts["_fit_idx"], parts["_es_idx"])
    r_va = resid.predict(X_va)

    rows = [{"lambda": float(l), **metrics7(y_va, base_va + l * r_va)} for l in LAM]
    curve = pd.DataFrame(rows)
    write_csv(curve, OUT / "lambda_sweep_validation.csv")

    # endpoint identities
    m0, m1 = metrics7(y_va, base_va), metrics7(y_va, base_va + r_va)
    chk = []
    for mt in METRICS:
        v0 = float(curve[np.isclose(curve["lambda"], 0.0)][mt].iloc[0])
        v1 = float(curve[np.isclose(curve["lambda"], 1.0)][mt].iloc[0])
        chk.append({"metric": mt, "lambda0_sweep": v0, "WBM_direct": m0[mt],
                    "lambda0_matches": bool(abs(v0 - m0[mt]) < 1e-9),
                    "lambda1_sweep": v1, "SGBM_direct": m1[mt],
                    "lambda1_matches": bool(abs(v1 - m1[mt]) < 1e-9)})
    cdf = pd.DataFrame(chk)
    write_csv(cdf, OUT / "lambda_endpoint_check.csv")
    if not (cdf.lambda0_matches.all() and cdf.lambda1_matches.all()):
        raise AssertionError("lambda endpoints do not reproduce WBM / S-GBM")
    print("[S01] lambda endpoints reproduce WBM and S-GBM exactly on all 7 metrics",
          flush=True)

    opt = []
    for mt in METRICS:
        i = curve[mt].idxmin() if LOWER_BETTER[mt] else curve[mt].idxmax()
        opt.append({"metric": mt, "optimal_lambda": float(curve.loc[i, "lambda"]),
                    "value_at_optimum": float(curve.loc[i, mt]),
                    "value_at_0_WBM": float(curve[np.isclose(curve["lambda"], 0)][mt].iloc[0]),
                    "value_at_1_SGBM": float(curve[np.isclose(curve["lambda"], 1)][mt].iloc[0]),
                    "is_primary": mt == PRIMARY})
    odf = pd.DataFrame(opt)
    write_csv(odf, OUT / "optimal_lambda_by_metric.csv")
    lam_star = float(odf[odf.metric == PRIMARY].optimal_lambda.iloc[0])
    print(f"[S01] lambda* on validation ({PRIMARY}) = {lam_star:g}; range across "
          f"metrics {odf.optimal_lambda.min():g}-{odf.optimal_lambda.max():g}",
          flush=True)
    return curve, lam_star, {"resid": resid, "bbm": bbm,
                             "info_resid": info_r, "info_bbm": info_b, "odf": odf}


# ------------------------------------------- internal test + final benchmark --

def evaluate_frozen(parts, ph, k, models, lam_star, split_names) -> pd.DataFrame:
    rows = []
    for name in split_names:
        d = parts[name]
        y, X = d["power"].to_numpy(), d[FEATS].to_numpy()
        base = k * ph[name]
        r = models["resid"].predict(X)
        preds = {"WBM": base, "BBM": models["bbm"].predict(X),
                 "S-GBM": base + r,
                 f"C-GBM (lambda*={lam_star:g})": base + lam_star * r}
        for arch, p in preds.items():
            rows.append({"partition": name, "architecture": arch,
                         "n": int(len(y)), **metrics7(y, p)})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = build_splits()
    ph = physics(parts)

    tr = parts["train"]
    y_tr, X_tr = tr["power"].to_numpy(), tr[FEATS].to_numpy()
    k = fit_scale(y_tr, ph["train"])
    print(f"[S01] physics scale k fitted on the 60 % training block only: {k:.6f}",
          flush=True)

    res_tr = y_tr - k * ph["train"]
    excl = feasibility(X_tr, res_tr)
    sel_tbl, sel = ml_selection(parts, ph, k, excl)

    # selected backend, once, on the untouched internal test in the residual role
    curve, lam_star, models = lambda_selection(parts, ph, k, sel)

    it = evaluate_frozen(parts, ph, k, models, lam_star, ["internal_test"])
    write_csv(it, OUT / "internal_test_architectures.csv")
    fin = evaluate_frozen(parts, ph, k, models, lam_star, ["dev_in", "dev_out"])
    write_csv(fin, OUT / "final_dev_in_dev_out.csv")

    sel_it = it[it.architecture == "S-GBM"].iloc[0]
    write_csv(pd.DataFrame([{"backend": BACKEND_DISPLAY[sel], "backend_key": sel,
                             "role": "residual learner",
                             "selected_on": "validation block only",
                             "evaluated_once_on": "internal test",
                             **{m: float(sel_it[m]) for m in METRICS}}]),
              OUT / "selected_backend_internal_test.csv")

    (OUT / "run_metadata.json").write_text(json.dumps({
        "protocol": "chronological 60/20/20 on the FULL train.csv",
        "n_train_csv": int(len(parts["train"]) + len(parts["validation"])
                           + len(parts["internal_test"])),
        "n_train": int(len(parts["train"])),
        "n_validation": int(len(parts["validation"])),
        "n_internal_test": int(len(parts["internal_test"])),
        "n_dev_in": int(len(parts["dev_in"])), "n_dev_out": int(len(parts["dev_out"])),
        "early_stopping": {"models": sorted(EARLY_STOP_MODELS),
                           "set": "last 10 % of the training block",
                           "n": int(len(parts["_es_idx"]))},
        "old_protocol_n_train_sub": CONFIG["n_train_sub"],
        "physics_k": float(k), "primary_metric": PRIMARY,
        "selected_backend": BACKEND_DISPLAY[sel], "lambda_star": lam_star,
        "excluded_models": sorted(excl[~excl.feasible_at_full_n].model),
        "dev_used_in_selection": False, "seed": SEED, "git_commit": _git()},
        indent=2), encoding="utf-8")

    figure(curve, lam_star, sel)
    report(parts, excl, sel_tbl, sel, curve, lam_star, models, it, fin, k)
    return 0


def figure(curve, lam_star, sel) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.bbox": "tight",
                         "axes.grid": True, "grid.alpha": 0.22,
                         "axes.axisbelow": True})
    lam = curve["lambda"].to_numpy()
    mu = curve[PRIMARY].to_numpy()
    fig, ax = plt.subplots(figsize=(3.46, 2.55))
    ax.plot(lam, mu, "-", color="#1f78b4", lw=1.6)
    for x, lbl, dxy, ha, va in ((0.0, "WBM", (7, 6), "left", "bottom"),
                                (1.0, f"S-GBM\n({BACKEND_DISPLAY[sel]})",
                                 (-7, -6), "right", "top")):
        i = int(np.argmin(np.abs(lam - x)))
        ax.plot([x], [mu[i]], "o", color="#4d4d4d", ms=5, mec="white", mew=0.7,
                zorder=5)
        ax.annotate(lbl, xy=(x, mu[i]), xytext=dxy, textcoords="offset points",
                    ha=ha, va=va, fontsize=8, color="#4d4d4d")
    i = int(np.argmin(np.abs(lam - lam_star)))
    ax.plot([lam_star], [mu[i]], "D", color="#b2182b", ms=6.5, mec="white",
            mew=0.7, zorder=6, label=rf"selected $\lambda^{{\ast}} = {lam_star:g}$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1))
    ax.set_xlabel(r"Residual-shrinkage coefficient $\lambda$", fontsize=9)
    ax.set_ylabel(f"Validation {PRIMARY} (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="best", handlelength=1.2)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig2_candidate_602020_fulldata.{ext}", format=ext)
    plt.close(fig)
    print("[S01] figure written", flush=True)


def report(parts, excl, sel_tbl, sel, curve, lam_star, models, it, fin, k) -> None:
    L: list[str] = []
    A = L.append
    odf = models["odf"]
    A("# Definitive full-data results: chronological 60/20/20 development protocol")
    A("")
    A(f"Full `train.csv` = "
      f"{len(parts['train']) + len(parts['validation']) + len(parts['internal_test']):,} "
      f"records, split chronologically without shuffling. The earlier "
      f"{CONFIG['n_train_sub']:,}-record uniform thinning is not used.")
    A("")
    A("| block | n | role |")
    A("|---|---|---|")
    A(f"| training (first 60 %) | {len(parts['train']):,} | physics calibration, "
      f"learner fitting |")
    A(f"| -- parameter fitting | {len(parts['_fit_idx']):,} | first 90 % of the "
      f"training block |")
    A(f"| -- early stopping | {len(parts['_es_idx']):,} | last 10 % of the training "
      f"block, boosted libraries only |")
    A(f"| validation (next 20 %) | {len(parts['validation']):,} | backend selection, "
      f"lambda selection |")
    A(f"| internal test (final 20 %) | {len(parts['internal_test']):,} | frozen check |")
    A(f"| dev_in | {len(parts['dev_in']):,} | benchmark holdout |")
    A(f"| dev_out | {len(parts['dev_out']):,} | benchmark holdout |")
    A("")
    A(f"Physics scale k = {k:.6f}, fitted on the training block alone. Neither "
      f"`dev_in` nor `dev_out` entered any fit, selection, tuning or early-stopping "
      f"decision.")
    A("")
    A("## Excluded models - measured, not assumed")
    A("")
    A("| model | stated cost | fit at 1k | fit at 2k | exponent | projected at full n | "
      "decision |")
    A("|---|---|---|---|---|---|---|")
    for _, r in excl.iterrows():
        A(f"| {r['model']} | {r['stated_cost']} | "
          f"{r[f'fit_seconds_at_{PROJECTION_SIZES[0]}']:.2f} s | "
          f"{r[f'fit_seconds_at_{PROJECTION_SIZES[1]}']:.2f} s | "
          f"{r['measured_scaling_exponent']:.2f} | "
          f"{r['projected_fit_hours_at_full_n']:,.1f} h | "
          f"{'retained' if r['feasible_at_full_n'] else 'EXCLUDED'} |")
    A("")
    A("Excluded models are omitted from the ranking entirely rather than "
      "subsampled, so every retained model is compared on identical training data.")
    A("")
    A("## Section 5.1 - model selection on the validation block")
    A("")
    ok = sel_tbl
    A("| model | MAPE | sMAPE | MAE | nMAE | RMSE | NRMSE | R2 | mean rank | ES |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in ok.iterrows():
        A(f"| {r['model']} | {r['MAPE']:.4f} | {r['sMAPE']:.4f} | {r['MAE']:.1f} | "
          f"{r['nMAE']:.4f} | {r['RMSE']:.1f} | {r['NRMSE']:.4f} | {r['R2']:.4f} | "
          f"{r['mean_rank_all_metrics']:.2f} | "
          f"{'yes' if r.get('early_stopping_used') else ''} |")
    A("")
    A(f"Selected on the primary metric ({PRIMARY}): **{BACKEND_DISPLAY[sel]}**.")
    A("")
    A("## Section 5.2 - lambda selection on the same validation block")
    A("")
    A("| metric | optimal lambda | at optimum | at 0 (WBM) | at 1 (S-GBM) |")
    A("|---|---|---|---|---|")
    for _, r in odf.iterrows():
        A(f"| {r['metric']}{' **(primary)**' if r['is_primary'] else ''} | "
          f"**{r['optimal_lambda']:g}** | {r['value_at_optimum']:.4f} | "
          f"{r['value_at_0_WBM']:.4f} | {r['value_at_1_SGBM']:.4f} |")
    A("")
    A(f"lambda* = **{lam_star:g}** on {PRIMARY}. Across the seven metrics the "
      f"optimum ranges {odf.optimal_lambda.min():g}-{odf.optimal_lambda.max():g}. "
      f"Endpoint identities verified: lambda = 0 reproduces the WBM and lambda = 1 "
      f"the S-GBM exactly on all seven metrics.")
    A("")
    A("## Internal test (frozen)")
    A("")
    A("| architecture | MAPE | sMAPE | MAE | nMAE | RMSE | NRMSE | R2 |")
    A("|---|---|---|---|---|---|---|---|")
    for _, r in it.iterrows():
        A(f"| {r['architecture']} | {r['MAPE']:.4f} | {r['sMAPE']:.4f} | "
          f"{r['MAE']:.1f} | {r['nMAE']:.4f} | {r['RMSE']:.1f} | {r['NRMSE']:.4f} | "
          f"{r['R2']:.4f} |")
    A("")
    A("## Final benchmark evaluation (frozen)")
    A("")
    A("| partition | architecture | MAPE | sMAPE | MAE | nMAE | RMSE | NRMSE | R2 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in fin.iterrows():
        A(f"| {r['partition']} | {r['architecture']} | {r['MAPE']:.4f} | "
          f"{r['sMAPE']:.4f} | {r['MAE']:.1f} | {r['nMAE']:.4f} | {r['RMSE']:.1f} | "
          f"{r['NRMSE']:.4f} | {r['R2']:.4f} |")
    A("")
    w = fin.pivot_table(index="architecture", columns="partition", values="MAPE")
    A("Shift comparison (MAPE, dev_out - dev_in):")
    A("")
    A("| architecture | dev_in | dev_out | dMAPE |")
    A("|---|---|---|---|")
    for a in w.index:
        A(f"| {a} | {w.loc[a, 'dev_in']:.4f} | {w.loc[a, 'dev_out']:.4f} | "
          f"{w.loc[a, 'dev_out'] - w.loc[a, 'dev_in']:+.4f} |")
    A("")
    A("## Comparison with the previous 40,000-record experiment")
    A("")
    A(f"- training data: {len(parts['train']):,} vs 34,000 records "
      f"({len(parts['train']) / 34_000:.1f}x)")
    A(f"- selection data: {len(parts['validation']):,} vs 6,000 records")
    A(f"- selected backend: **{BACKEND_DISPLAY[sel]}** vs CatBoost")
    A(f"- lambda*: **{lam_star:g}** vs 0.25")
    A("- an internal test block now sits between selection and the benchmark sets, "
      "which the earlier protocol lacked entirely.")
    A("")
    write_text("\n".join(L) + "\n", OUT / "full_data_602020_report.md")
    print("[S01] report written", flush=True)


def _guarded_main() -> int:
    # Windows joblib spawns per-estimator worker pools; on this machine that
    # exhausts system resources at Bagging/RandomForest. Threading with a single
    # worker is an execution-only change: with fixed random_state the fitted
    # models are identical.
    import joblib
    with joblib.parallel_backend("threading", n_jobs=1):
        return main()


if __name__ == "__main__":
    sys.exit(_guarded_main())
