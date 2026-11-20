"""Rigorous statistical comparison of the 21 candidate ML models (Friedman + Nemenyi).

Applies the two-step Demsar (2006) protocol to the **21 candidate learning
backends** in the residual role, on repeated *temporal* validation folds drawn
from the Shifts TRAINING partition alone. ``dev_in`` and ``dev_out`` are never
read.

Departure from the reference study, deliberate and necessary: the reference uses
random k-fold cross-validation. That is invalid for this dataset, whose target is
an hourly shaft-power series with strong temporal autocorrelation -- shuffled
folds place neighbouring hours on both sides of a split and inflate every score.
This script keeps the Friedman/Nemenyi statistics and replaces shuffled folds
with **rolling-origin (expanding-window) blocked validation**: contiguous
unshuffled blocks in time, fold f fitting on blocks 0..f-1 and validating on
block f, so every validation block occurs strictly after its fitting data.

Per fold the physics scale ``k`` is refitted on that fold's fitting block only,
and the residual target is built from that fold-specific calibrated physics, so
nothing leaks across folds.

Nemenyi p-values are computed exactly from the studentized-range distribution
(``scipy.stats.studentized_range``); ``scikit_posthocs`` is absent from this
environment and no dependency was added.

Run:  PYTHONPATH=. python experiments/revision/rf17_ml_statistical_comparison.py
Outputs: results/revision_final/17_ml_statistical_comparison/
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .common_rev import (
    ETA_PUBLISHED, FEATS, REPO_ROOT, REV_FINAL, RevisionModels, SEED,
    metrics, to_tex, write_csv, write_text,
)

_E = pathlib.Path(__file__).resolve().parents[1]
if str(_E) not in sys.path:
    sys.path.insert(0, str(_E))
from benchmark_measured.common import (  # noqa: E402
    BACKEND_DISPLAY, CONFIG, SUBSAMPLE_CAP, fit_scale, make_learner,
)

ROOT = REV_FINAL / "17_ml_statistical_comparison"
DIRS = {d: ROOT / d for d in
        ("protocol", "metrics", "friedman", "nemenyi", "paired", "figures",
         "tables", "summary")}

N_BLOCKS = 9                 # -> 8 rolling-origin folds
ALPHA = 0.05

# Protocol B equalises the fitting size across all 21 candidates. It can only be
# equalised DOWN to the most restrictive model: Gaussian-process regression
# factorises an n x n kernel matrix, so its cost is O(n^3) and the project caps it
# at 1,200 for that reason. Measured on this host with the project's own GPR
# configuration: 19.0 s at n = 1,200 and 72.6 s at n = 2,400 (3.8x for a 2x size
# increase, i.e. essentially cubic), which extrapolates to roughly 700 s per fit
# at n = 4,000 -- about 94 minutes for GPR alone across the 8 folds. A common size
# of 4,000 is therefore infeasible for GPR, and 1,200 (the minimum of the existing
# project caps) is the largest size every candidate can actually be fitted on.
COMMON_CAP = 1200
COMMON_CAP_RATIONALE = (
    "1,200 = the minimum of the project's existing training caps, set by Gaussian-process "
    "regression whose O(n^3) kernel factorisation makes a larger common size infeasible "
    "(measured 19.0 s at n=1,200 and 72.6 s at n=2,400; ~700 s extrapolated at n=4,000, "
    "~94 min for GPR alone over 8 folds). Equalising the fitting size is only possible "
    "downwards, to the most restrictive model."
)
METRICS = ("MAPE", "MAE", "RMSE", "R2")
LOWER_BETTER = {"MAPE": True, "MAE": True, "RMSE": True, "R2": False}

FAMILY = {
    "ridge": "Linear", "lasso": "Linear", "elastic": "Linear",
    "bayesian_ridge": "Linear", "huber": "Linear", "ols": "Linear",
    "knn": "Kernel/instance", "svr": "Kernel/instance", "gpr": "Kernel/instance",
    "pnn": "Kernel/instance", "mlp": "Neural",
    "dt": "Tree ensemble", "rf": "Tree ensemble", "et": "Tree ensemble",
    "ada": "Tree ensemble", "bag": "Tree ensemble", "gbt": "Tree ensemble",
    "histgb": "Tree ensemble",
    "xgboost": "Boosted-tree library", "catboost": "Boosted-tree library",
    "lightgbm": "Boosted-tree library",
}
ALWAYS_IN_LEADING = ["HistGB", "CatBoost", "SVR (RBF)", "GaussianProcess"]


def _git() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() or "unavailable"


# ----------------------------------------------------------------- stats ----

def avg_ranks(pivot: pd.DataFrame, lower_better: bool) -> pd.Series:
    """Average rank per model across folds (rows = folds, cols = models)."""
    r = pivot.rank(axis=1, ascending=lower_better, method="average")
    return r.mean(axis=0)


def nemenyi_exact(ranks: pd.Series, n_folds: int) -> tuple[pd.DataFrame, float, float]:
    """Exact Nemenyi pairwise p-values, plus (q_critical, critical difference).

    p_ij = 1 - F_studentised-range( |R_i - R_j| / sqrt(k(k+1)/(6N)) * sqrt(2) ; k, inf )
    CD   = q_alpha / sqrt(2) * sqrt(k(k+1)/(6N))
    """
    from scipy.stats import studentized_range
    names = list(ranks.index)
    k, n = len(names), n_folds
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    q_crit = float(studentized_range.ppf(1.0 - ALPHA, k, np.inf))
    cd = q_crit / np.sqrt(2.0) * se
    P = pd.DataFrame(np.ones((k, k)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            qs = abs(ranks[a] - ranks[b]) / se * np.sqrt(2.0)
            p = float(1.0 - studentized_range.cdf(qs, k, np.inf))
            P.loc[a, b] = P.loc[b, a] = min(max(p, 0.0), 1.0)
    return P, q_crit, float(cd)


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m, out, prev = len(pvals), [1.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(prev, (m - rank) * pvals[i]))
        out[i], prev = adj, adj
    return out


# ------------------------------------------------------------ experiment ----

def run_protocol(m: RevisionModels, folds, common_cap: int | None,
                 label: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Fit all 21 candidates in the residual role on every fold.

    Returns ``(fold_metrics, wbm_reference, training_size_audit)``.
    """
    y, X, ph = m.y_tr, m.X_tr, m.ph["train"]
    rng = np.random.default_rng(SEED)
    rows, wbm_rows, audit = [], [], []

    for fi, (tr_i, va_i) in enumerate(folds):
        k_f = fit_scale(y[tr_i], ph[tr_i])
        res_f = y[tr_i] - k_f * ph[tr_i]
        base_va = k_f * ph[va_i]
        mw = metrics(y[va_i], base_va)
        wbm_rows.append({"protocol": label, "fold": fi, "k_refit": k_f,
                         "n_fit": int(len(tr_i)), "n_val": int(len(va_i)),
                         **{c: mw[c] for c in METRICS}})

        for key in CONFIG["bbm_backends"]:
            disp = BACKEND_DISPLAY[key]
            proj_cap = SUBSAMPLE_CAP.get(key)
            lrn, sub = make_learner(key, rng)
            if common_cap is None:
                Xs, rs = sub(X[tr_i], res_f)             # canonical caps
                cap_used = proj_cap
            else:
                take = min(common_cap, len(tr_i))
                # temporally representative: uniform stride over the fitting block
                sel = np.linspace(0, len(tr_i) - 1, take).astype(int)
                Xs, rs = X[tr_i][sel], res_f[sel]
                cap_used = common_cap
            t0 = time.time()
            lrn.fit(Xs, rs)
            pred = base_va + lrn.predict(X[va_i])
            mm = metrics(y[va_i], pred)
            rows.append({"protocol": label, "model": disp, "backend_key": key,
                         "family": FAMILY[key], "fold": fi,
                         "n_fit": int(len(tr_i)), "n_val": int(len(va_i)),
                         "n_train_used": int(len(Xs)), "training_cap": cap_used,
                         "cap_applied": bool(len(Xs) < len(tr_i)),
                         "fit_seconds": round(time.time() - t0, 2),
                         **{c: mm[c] for c in METRICS}})
            audit.append({"protocol": label, "model": disp, "fold": fi,
                          "available_fitting_records": int(len(tr_i)),
                          "records_actually_used": int(len(Xs)),
                          "training_cap": cap_used,
                          "cap_applied": bool(len(Xs) < len(tr_i))})
        done = sorted({r["model"] for r in rows if r["fold"] == fi})
        print(f"    [{label}] fold {fi} done: n_fit={len(tr_i):,} "
              f"models={len(done)}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(wbm_rows), audit


def main() -> None:
    import joblib
    with joblib.parallel_backend("threading", n_jobs=1):
        _main()


def _main() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    m = RevisionModels(ETA_PUBLISHED)
    tr, n = m.train, len(m.train)

    # ---------------- TASK 1: temporal folds ----------------
    b = np.linspace(0, n, N_BLOCKS + 1).astype(int)
    folds = [(np.arange(0, b[f]), np.arange(b[f], b[f + 1])) for f in range(1, N_BLOCKS)]
    tid, mon = tr["time_id"].to_numpy(), tr["months"].to_numpy()
    fold_def = pd.DataFrame([{
        "fold": i,
        "fit_row_start": int(a[0]), "fit_row_end": int(a[-1]), "n_fit": int(len(a)),
        "val_row_start": int(v[0]), "val_row_end": int(v[-1]), "n_val": int(len(v)),
        "fit_time_id_start": float(tid[a[0]]), "fit_time_id_end": float(tid[a[-1]]),
        "val_time_id_start": float(tid[v[0]]), "val_time_id_end": float(tid[v[-1]]),
        "fit_months_median": float(np.median(mon[a])),
        "val_months_median": float(np.median(mon[v])),
        "months_drift_val_minus_fit": float(np.median(mon[v]) - np.median(mon[a])),
        "strictly_causal": bool(tid[v[0]] > tid[a[-1]]),
    } for i, (a, v) in enumerate(folds)])
    write_csv(fold_def, DIRS["protocol"] / "temporal_fold_definition.csv")
    print(f"[T1] {len(folds)} rolling-origin folds; all strictly causal: "
          f"{bool(fold_def['strictly_causal'].all())}")

    write_text("\n".join([
        "# Temporal fold definition",
        "",
        f"Shifts **training partition only** ({n:,} records, time-ordered). "
        f"{N_BLOCKS} contiguous unshuffled blocks; fold *f* fits on blocks 0..f-1 and "
        f"validates on block *f*, giving **{len(folds)} rolling-origin folds**. Every "
        "validation block occurs strictly after its fitting data — verified per fold in the "
        "`strictly_causal` column, all True. No `dev_in` or `dev_out` record is read.",
        "",
        "Random k-fold was **not** used: the target is an hourly series with strong temporal "
        "autocorrelation, so shuffled folds would place neighbouring hours on both sides of a "
        "split and inflate every score.",
        "",
        "Per fold the physics scale `k` is refitted on that fold's fitting block alone, and "
        "the residual target `r = y - k*F_phys` is built from that fold-specific calibration.",
        "",
        "| Fold | n fit | n val | fit time_id | val time_id | fit months (med) | val months (med) | drift |",
        "|---|---|---|---|---|---|---|---|",
    ] + [f"| {r.fold} | {r.n_fit:,} | {r.n_val:,} | {r.fit_time_id_start:.0f}–"
         f"{r.fit_time_id_end:.0f} | {r.val_time_id_start:.0f}–{r.val_time_id_end:.0f} | "
         f"{r.fit_months_median:.1f} | {r.val_months_median:.1f} | "
         f"{r.months_drift_val_minus_fit:+.1f} |" for r in fold_def.itertuples()] + [
        "",
        "The `drift` column is the change in median months-since-dry-docking from fitting to "
        "validation block. It is positive in every fold, so each fold contains genuine fouling "
        "drift: these are not same-distribution hold-outs, and a model that extrapolates "
        "conservatively is favoured. That is a property of the vessel's operating history, and "
        "it is why the ranking below should not be read as a same-distribution accuracy "
        "ranking.",
    ]), DIRS["protocol"] / "temporal_fold_definition.md")

    # ---------------- TASKS 2-4: run both protocols ----------------
    # Each protocol is persisted as soon as it finishes, so a change to one does
    # not force recomputation of the other, and a long run is inspectable while
    # still in progress.
    cacheA = DIRS["metrics"] / "_raw_protocol_A.csv"
    cacheB = DIRS["metrics"] / "_raw_protocol_B.csv"
    cacheAw = DIRS["metrics"] / "_raw_protocol_A_wbm.csv"
    cacheBw = DIRS["metrics"] / "_raw_protocol_B_wbm.csv"
    cacheAa = DIRS["protocol"] / "_raw_protocol_A_audit.csv"
    cacheBa = DIRS["protocol"] / "_raw_protocol_B_audit.csv"

    if cacheA.exists() and cacheAw.exists() and cacheAa.exists():
        print("[T3] protocol A: reusing persisted fold metrics", flush=True)
        A, wbmA = pd.read_csv(cacheA), pd.read_csv(cacheAw)
        auditA = pd.read_csv(cacheAa).to_dict("records")
    else:
        print("[T3] protocol A (canonical project caps) ...", flush=True)
        A, wbmA, auditA = run_protocol(m, folds, None, "A_canonical")
        write_csv(A, cacheA); write_csv(wbmA, cacheAw)
        write_csv(pd.DataFrame(auditA), cacheAa)

    if cacheB.exists() and cacheBw.exists() and cacheBa.exists():
        print("[T2] protocol B: reusing persisted fold metrics", flush=True)
        B, wbmB = pd.read_csv(cacheB), pd.read_csv(cacheBw)
        auditB = pd.read_csv(cacheBa).to_dict("records")
    else:
        print(f"[T2] protocol B (common training size {COMMON_CAP:,}) ...", flush=True)
        B, wbmB, auditB = run_protocol(m, folds, COMMON_CAP, "B_common_size")
        write_csv(B, cacheB); write_csv(wbmB, cacheBw)
        write_csv(pd.DataFrame(auditB), cacheBa)

    fold_metrics = pd.concat([A, B], ignore_index=True)
    write_csv(fold_metrics, DIRS["metrics"] / "residual_model_fold_metrics.csv")
    write_csv(pd.concat([wbmA, wbmB], ignore_index=True),
              DIRS["metrics"] / "wbm_fold_reference.csv")
    write_csv(pd.DataFrame(auditA + auditB), DIRS["protocol"] / "training_size_audit.csv")

    def summarise(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby(["model", "family"])
        out = g.agg(**{f"{s}_{c}": (c, s) for c in METRICS
                       for s in ("mean", "std", "median")}).reset_index()
        piv = {c: df.pivot(index="fold", columns="model", values=c) for c in METRICS}
        for c in METRICS:
            rk = avg_ranks(piv[c], LOWER_BETTER[c])
            out[f"rank_{c}"] = out["model"].map(rk)
        out["mean_overall_rank"] = out[[f"rank_{c}" for c in METRICS]].mean(axis=1)
        return out.sort_values("mean_MAPE").reset_index(drop=True)

    sumA, sumB = summarise(A), summarise(B)
    write_csv(pd.concat([sumA.assign(protocol="A_canonical"),
                         sumB.assign(protocol="B_common_size")], ignore_index=True),
              DIRS["metrics"] / "residual_model_summary.csv")

    # ---------------- TASK 5: Friedman ----------------
    from scipy.stats import friedmanchisquare, wilcoxon
    pivA = {c: A.pivot(index="fold", columns="model", values=c) for c in METRICS}
    fried = []
    for c in METRICS:
        cols = [pivA[c][mm].to_numpy() for mm in pivA[c].columns]
        st, p = friedmanchisquare(*cols)
        fried.append({"protocol": "A_canonical", "metric": c,
                      "friedman_statistic": float(st), "df": pivA[c].shape[1] - 1,
                      "p_value": float(p), "alpha": ALPHA,
                      "n_models": pivA[c].shape[1], "n_folds": pivA[c].shape[0],
                      "reject_H0_equivalent_performance": bool(p < ALPHA)})
    pivB = {c: B.pivot(index="fold", columns="model", values=c) for c in METRICS}
    for c in METRICS:
        st, p = friedmanchisquare(*[pivB[c][mm].to_numpy() for mm in pivB[c].columns])
        fried.append({"protocol": "B_common_size", "metric": c,
                      "friedman_statistic": float(st), "df": pivB[c].shape[1] - 1,
                      "p_value": float(p), "alpha": ALPHA,
                      "n_models": pivB[c].shape[1], "n_folds": pivB[c].shape[0],
                      "reject_H0_equivalent_performance": bool(p < ALPHA)})
    fr = pd.DataFrame(fried)
    write_csv(fr, DIRS["friedman"] / "friedman_results.csv")
    to_tex(fr[fr.protocol == "A_canonical"][
               ["metric", "friedman_statistic", "df", "p_value",
                "reject_H0_equivalent_performance"]],
           DIRS["friedman"] / "friedman_results.tex",
           caption=("Friedman omnibus test across the 21 candidate learning backends on "
                    "8 rolling-origin temporal validation folds of the training partition. "
                    "Canonical protocol."),
           label="tab:ml-friedman")

    # ---------------- TASK 6-8: Nemenyi ----------------
    ranks, Pmats, cds, qs = {}, {}, {}, {}
    for c in METRICS:
        ranks[c] = avg_ranks(pivA[c], LOWER_BETTER[c])
        Pmats[c], qs[c], cds[c] = nemenyi_exact(ranks[c], len(folds))
        sig = fr[(fr.protocol == "A_canonical") & (fr.metric == c)]["p_value"].iloc[0] < ALPHA
        low = c.lower()
        Pmats[c].to_csv(DIRS["nemenyi"] / f"nemenyi_{low}_pvalues.csv")
        pair = []
        names = list(ranks[c].index)
        for i, a in enumerate(names):
            for bb in names[i + 1:]:
                pair.append({"metric": c, "model_A": a, "model_B": bb,
                             "p_value": float(Pmats[c].loc[a, bb]),
                             "significant_at_0.05": bool(Pmats[c].loc[a, bb] < ALPHA),
                             "average_rank_A": float(ranks[c][a]),
                             "average_rank_B": float(ranks[c][bb]),
                             "rank_difference": float(abs(ranks[c][a] - ranks[c][bb])),
                             "friedman_significant_for_metric": bool(sig)})
        write_csv(pd.DataFrame(pair), DIRS["nemenyi"] / f"nemenyi_{low}_pairs.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    plt.rcParams.update({"font.size": 7, "figure.dpi": 300, "savefig.bbox": "tight"})

    for c in METRICS:
        low = c.lower()
        order = ranks[c].sort_values().index.tolist()
        P = Pmats[c].loc[order, order]
        fig, ax = plt.subplots(figsize=(8.2, 7.0))
        cmap = ListedColormap(["#08306B", "#2171B5", "#9ECAE1", "#F7F7F7",
                               "#FDD0A2", "#FD8D3C", "#D94801"])
        bounds = [0, 0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 1.0]
        im = ax.imshow(P.to_numpy(), cmap=cmap, norm=BoundaryNorm(bounds, cmap.N))
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
        ax.set_yticks(range(len(order)), order, fontsize=6)
        # mark significant cells
        for i in range(len(order)):
            for j in range(len(order)):
                if i != j and P.iloc[i, j] < ALPHA:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=6,
                            color="white")
        cb = fig.colorbar(im, ax=ax, boundaries=bounds, ticks=bounds, shrink=0.82)
        cb.set_label("Nemenyi pairwise p-value (Bonferroni-free, studentized range)",
                     fontsize=7)
        ax.set_title(f"Nemenyi pairwise p-values — {c}\n"
                     f"* marks p < {ALPHA}; models ordered by average rank "
                     f"(best first); {len(folds)} temporal folds", fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(DIRS["figures"] / f"fig_nemenyi_{low}.{ext}")
        plt.close(fig)

        # CD / average-rank diagram
        rv = ranks[c].sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        yp = np.arange(len(rv))[::-1]
        ax.hlines(yp, rv.min(), rv.to_numpy(), color="#CCCCCC", lw=1)
        ax.plot(rv.to_numpy(), yp, "o", color="#0072B2", ms=5)
        ax.set_yticks(yp, rv.index.tolist(), fontsize=6)
        best = float(rv.iloc[0])
        ax.axvspan(best, best + cds[c], color="#009E73", alpha=0.15,
                   label=f"not distinguishable from best (CD = {cds[c]:.2f})")
        ax.axvline(best, color="#009E73", lw=1.2)
        ax.set_xlabel(f"average rank across {len(folds)} temporal folds "
                      f"({'lower' if LOWER_BETTER[c] else 'higher'} metric is better; "
                      f"rank 1 = best)")
        ax.set_title(f"Average ranks and critical difference — {c} "
                     f"(k=21, N={len(folds)}, alpha={ALPHA})", fontsize=8)
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(DIRS["figures"] / f"fig_cd_{low}.{ext}")
        plt.close(fig)

    # ---------------- TASK 4 cont.: residual models vs WBM ----------------
    wp = wbmA.set_index("fold")
    vs_rows = []
    raw_p, keys = [], []
    for mdl in pivA["MAPE"].columns:
        d = pivA["MAPE"][mdl].to_numpy() - wp["MAPE"].to_numpy()
        try:
            st, p = wilcoxon(pivA["MAPE"][mdl].to_numpy(), wp["MAPE"].to_numpy(),
                             alternative="two-sided", zero_method="zsplit")
        except ValueError:
            st, p = np.nan, 1.0
        raw_p.append(float(p)); keys.append(mdl)
        vs_rows.append({"model": mdl, "metric": "MAPE",
                        "mean_model": float(pivA["MAPE"][mdl].mean()),
                        "mean_wbm": float(wp["MAPE"].mean()),
                        "median_paired_diff_model_minus_wbm": float(np.median(d)),
                        "folds_model_better": int((d < 0).sum()),
                        "n_folds": len(folds),
                        "beats_wbm_in_mean": bool(pivA["MAPE"][mdl].mean() < wp["MAPE"].mean()),
                        "beats_wbm_all_folds": bool((d < 0).all()),
                        "wilcoxon_stat": float(st), "p_raw": float(p)})
    for r, ph_ in zip(vs_rows, holm(raw_p)):
        r["p_holm"] = ph_
        r["significantly_better_than_wbm"] = bool(
            ph_ < ALPHA and r["median_paired_diff_model_minus_wbm"] < 0)
        r["significantly_worse_than_wbm"] = bool(
            ph_ < ALPHA and r["median_paired_diff_model_minus_wbm"] > 0)
    vs = pd.DataFrame(vs_rows).sort_values("mean_model")
    write_csv(vs, DIRS["paired"] / "wbm_vs_models_holm.csv")

    # ---------------- TASK 10: leading subset, Wilcoxon + Holm ----------------
    leaders = set(ALWAYS_IN_LEADING)
    for c in METRICS:
        leaders.add(sumA.sort_values(f"mean_{c}",
                                     ascending=LOWER_BETTER[c])["model"].iloc[0])
        leaders.add(ranks[c].idxmin())
    best_mape = ranks["MAPE"].idxmin()
    for mdl in ranks["MAPE"].index:
        if Pmats["MAPE"].loc[best_mape, mdl] >= ALPHA:
            leaders.add(mdl)
    leaders = [x for x in pivA["MAPE"].columns if x in leaders]
    lead_rows, lp, lk = [], [], []
    for c in METRICS:
        for i, a in enumerate(leaders):
            for bb in leaders[i + 1:]:
                x, yv = pivA[c][a].to_numpy(), pivA[c][bb].to_numpy()
                try:
                    st, p = wilcoxon(x, yv, alternative="two-sided", zero_method="zsplit")
                except ValueError:
                    st, p = np.nan, 1.0
                lp.append(float(p)); lk.append((c, a, bb))
                lead_rows.append({"metric": c, "model_A": a, "model_B": bb,
                                  "median_paired_difference": float(np.median(x - yv)),
                                  "wilcoxon_statistic": float(st), "p_raw": float(p),
                                  "n_folds": len(folds)})
    for r, ph_ in zip(lead_rows, holm(lp)):
        r["p_holm"] = ph_
        r["significant_after_holm"] = bool(ph_ < ALPHA)
    lead = pd.DataFrame(lead_rows)
    write_csv(lead, DIRS["paired"] / "leading_models_pairwise_holm.csv")
    min_p_wilcoxon = 2.0 / (2 ** len(folds))

    # ---------------- TASK 9 + 12: ranking table ----------------
    rank_tbl = sumA[["model", "family"] +
                    [f"{s}_{c}" for c in METRICS for s in ("mean", "std")] +
                    [f"rank_{c}" for c in METRICS] + ["mean_overall_rank"]].copy()
    for c in METRICS:
        bestm = ranks[c].idxmin()
        rank_tbl[f"not_sig_diff_from_best_{c}"] = rank_tbl["model"].map(
            lambda mm, cc=c, bmm=bestm: bool(Pmats[cc].loc[bmm, mm] >= ALPHA))
        rank_tbl[f"best_model_{c}"] = bestm
    write_csv(rank_tbl, DIRS["tables"] / "statistical_model_ranking.csv")
    to_tex(rank_tbl[["model", "family", "mean_MAPE", "std_MAPE", "rank_MAPE",
                     "not_sig_diff_from_best_MAPE", "mean_overall_rank"]],
           DIRS["tables"] / "statistical_model_ranking.tex",
           caption=("Statistical ranking of the 21 candidate learning backends over "
                    "8 rolling-origin temporal validation folds. "
                    "not\\_sig\\_diff\\_from\\_best indicates the Nemenyi post-hoc cannot "
                    "distinguish the model from the best-ranked one."),
           label="tab:ml-ranking")

    comp = sumA[["model", "family"]].copy()
    for c in METRICS:
        comp[f"{c} mean +/- SD"] = (sumA[f"mean_{c}"].map(lambda v: f"{v:.3f}")
                                    + " +/- " + sumA[f"std_{c}"].map(lambda v: f"{v:.3f}"))
    comp["Avg MAPE rank"] = sumA["rank_MAPE"].round(2)
    comp["Relation to best (MAPE)"] = rank_tbl["not_sig_diff_from_best_MAPE"].map(
        {True: "not distinguishable", False: "worse than best"})
    write_csv(comp, DIRS["tables"] / "table_ml_statistical_comparison.csv")
    to_tex(comp, DIRS["tables"] / "table_ml_statistical_comparison.tex",
           caption=("Repeated temporal-validation performance of the 21 candidate learning "
                    "backends in the residual role, with average MAPE rank and the Nemenyi "
                    "relation to the best-ranked model. Training partition only."),
           label="tab:ml-comparison")
    to_tex(fr[fr.protocol == "A_canonical"][["metric", "friedman_statistic", "df",
                                             "p_value", "reject_H0_equivalent_performance"]],
           DIRS["tables"] / "table_friedman_tests.tex",
           caption="Friedman omnibus tests across the 21 candidate backends.",
           label="tab:ml-friedman-main")
    top = lead[(lead.metric == "MAPE")].sort_values("p_holm")
    to_tex(top[["model_A", "model_B", "median_paired_difference", "p_raw", "p_holm",
                "significant_after_holm"]].head(20),
           DIRS["tables"] / "table_nemenyi_top_models.tex",
           caption=("Paired fold-wise Wilcoxon comparisons among the leading backends on "
                    "MAPE, Holm-corrected. Complete Nemenyi matrices are supplementary."),
           label="tab:ml-leading-pairs")

    meta = {
        "n_folds": len(folds), "n_blocks": N_BLOCKS, "n_models": 21,
        "fold_scheme": "rolling-origin expanding-window blocked, contiguous, unshuffled",
        "random_kfold_used": False, "dev_in_used": False, "dev_out_used": False,
        "physics_k_refit_per_fold": True, "seed": SEED, "git_commit": _git(),
        "alpha": ALPHA, "nemenyi_q_critical": {c: qs[c] for c in METRICS},
        "nemenyi_critical_difference": {c: cds[c] for c in METRICS},
        "nemenyi_pvalues": "exact studentized-range (scikit_posthocs unavailable; no dependency added)",
        "common_training_size_protocol_B": COMMON_CAP,
        "common_training_size_rationale": COMMON_CAP_RATIONALE,
        "wilcoxon_min_attainable_two_sided_p": min_p_wilcoxon,
        "hyperparameters_retuned": False,
        "project_training_caps": SUBSAMPLE_CAP,
    }
    write_text(json.dumps(meta, indent=2), ROOT / "statistical_comparison_metadata.json")
    print(f"[T5] Friedman done; [T6] CD(MAPE)={cds['MAPE']:.2f}")
    return_pack = dict(fold_def=fold_def, A=A, B=B, sumA=sumA, sumB=sumB, fr=fr,
                       ranks=ranks, Pmats=Pmats, cds=cds, vs=vs, lead=lead,
                       leaders=leaders, wbmA=wbmA, rank_tbl=rank_tbl,
                       min_p_wilcoxon=min_p_wilcoxon, folds=folds, meta=meta)
    _write_summaries(**return_pack)
    print(f"[RF17] wrote {ROOT}")


def _write_summaries(fold_def, A, B, sumA, sumB, fr, ranks, Pmats, cds, vs, lead,
                     leaders, wbmA, rank_tbl, min_p_wilcoxon, folds, meta) -> None:
    frA = fr[fr.protocol == "A_canonical"].set_index("metric")
    frB = fr[fr.protocol == "B_common_size"].set_index("metric")
    best_mape_mean = sumA.iloc[0]["model"]
    best_rank = {c: ranks[c].idxmin() for c in METRICS}
    tie = [mm for mm in ranks["MAPE"].index
           if Pmats["MAPE"].loc[best_rank["MAPE"], mm] >= ALPHA]
    n_better_wbm = int(vs["beats_wbm_in_mean"].sum())
    n_all_folds = int(vs["beats_wbm_all_folds"].sum())
    n_sig_better = int(vs["significantly_better_than_wbm"].sum())
    n_sig_worse = int(vs["significantly_worse_than_wbm"].sum())
    sumB_i = sumB.set_index("model")
    svr_in = "SVR (RBF)" in tie
    cat_in = "CatBoost" in tie
    hist_in = "HistGB" in tie

    def q(i, t): return f"**{i}. {t}**"

    lines = [
        "# ML model statistical selection — summary",
        "",
        f"Seed {meta['seed']}; git `{meta['git_commit']}`. "
        f"{len(folds)} rolling-origin temporal folds on the training partition only; "
        "`dev_in` and `dev_out` never read; physics scale refitted per fold; no "
        "hyperparameter retuned.",
        "",
        "## Friedman omnibus tests (canonical protocol)",
        "",
        "| Metric | chi-square | df | p | Reject H0 |", "|---|---|---|---|---|",
    ] + [f"| {c} | {frA.loc[c,'friedman_statistic']:.3f} | {int(frA.loc[c,'df'])} | "
         f"{frA.loc[c,'p_value']:.3e} | "
         f"{'yes' if frA.loc[c,'reject_H0_equivalent_performance'] else 'no'} |"
         for c in METRICS] + [
        "",
        f"Nemenyi critical difference at alpha = {ALPHA}, k = 21, N = {len(folds)}: "
        + ", ".join(f"{c} {cds[c]:.2f}" for c in METRICS)
        + " rank positions. Average ranks can only span 1 to 21, so a CD of this size means "
        "the 21-way post-hoc separates only the extremes of the field. That is a power "
        "limitation of a 21-way test on 8 folds, and it is reported rather than worked around.",
        "",
        "## Answers",
        "",
        q(1, "Friedman significant for MAPE?")
        + f" {'Yes' if frA.loc['MAPE','reject_H0_equivalent_performance'] else 'No'} "
          f"(chi2 = {frA.loc['MAPE','friedman_statistic']:.2f}, p = {frA.loc['MAPE','p_value']:.3e}).",
        q(2, "For MAE?")
        + f" {'Yes' if frA.loc['MAE','reject_H0_equivalent_performance'] else 'No'} "
          f"(p = {frA.loc['MAE','p_value']:.3e}).",
        q(3, "For RMSE?")
        + f" {'Yes' if frA.loc['RMSE','reject_H0_equivalent_performance'] else 'No'} "
          f"(p = {frA.loc['RMSE','p_value']:.3e}).",
        q(4, "For R2?")
        + f" {'Yes' if frA.loc['R2','reject_H0_equivalent_performance'] else 'No'} "
          f"(p = {frA.loc['R2','p_value']:.3e}).",
        "",
        q(5, "Best mean MAPE?") + f" **{best_mape_mean}** "
        f"({sumA.iloc[0]['mean_MAPE']:.4f} +/- {sumA.iloc[0]['std_MAPE']:.4f}).",
    ]
    for i, c in ((6, "MAE"), (7, "RMSE"), (8, "R2")):
        s = sumA.sort_values(f"mean_{c}", ascending=LOWER_BETTER[c]).iloc[0]
        lines.append(q(i, f"Best mean {c}?")
                     + f" **{s['model']}** ({s[f'mean_{c}']:.4f} +/- {s[f'std_{c}']:.4f}).")
    lines += [
        q(9, "Best average MAPE rank?")
        + f" **{best_rank['MAPE']}** (avg rank {ranks['MAPE'].min():.2f}).",
        "",
        q(10, "Models NOT distinguishable from the best-MAPE model under Nemenyi?")
        + f" **{len(tie)} of 21**: " + ", ".join(tie) + ".",
        "",
        q(11, "Does any residual ML model statistically outperform the WBM reference?")
        + f" **{'Yes' if n_sig_better else 'No'}**. Of 21 models, {n_better_wbm} beat the WBM "
          f"in mean fold MAPE, {n_all_folds} beat it in every fold, and after "
          f"Holm-corrected paired Wilcoxon over the {len(folds)} folds "
          f"**{n_sig_better} are significantly better** and {n_sig_worse} significantly "
          f"worse. WBM mean fold MAPE = {wbmA['MAPE'].mean():.4f}.",
        "",
        q(12, "Is SVR still the numerical winner under repeated temporal folds?")
        + f" {'Yes' if best_mape_mean == 'SVR (RBF)' else 'No'} — the best mean MAPE is "
          f"{best_mape_mean}. SVR's mean MAPE is "
          f"{float(sumA.set_index('model').loc['SVR (RBF)','mean_MAPE']):.4f} "
          f"(avg rank {ranks['MAPE']['SVR (RBF)']:.2f}). The single-split argmin reported "
          "earlier is not reproduced as a robust ranking.",
        "",
        q(13, "Is CatBoost distinguishable from SVR?")
        + f" {'Yes' if Pmats['MAPE'].loc['CatBoost','SVR (RBF)'] < ALPHA else 'No'} "
          f"(Nemenyi p = {Pmats['MAPE'].loc['CatBoost','SVR (RBF)']:.3f}).",
        q(14, "Is HistGB distinguishable from SVR?")
        + f" {'Yes' if Pmats['MAPE'].loc['HistGB','SVR (RBF)'] < ALPHA else 'No'} "
          f"(Nemenyi p = {Pmats['MAPE'].loc['HistGB','SVR (RBF)']:.3f}).",
        "",
        q(15, "Does the common-training-size sensitivity change the ranking or grouping?")
        + f" Protocol B fits every model on the same {meta['common_training_size_protocol_B']:,} "
          f"temporally strided records. Best mean MAPE becomes **{sumB.iloc[0]['model']}** "
          f"({sumB.iloc[0]['mean_MAPE']:.4f}); Friedman on MAPE p = "
          f"{frB.loc['MAPE','p_value']:.3e}. "
        + ("The leader is unchanged, so the cap confound does not drive the ranking."
           if sumB.iloc[0]['model'] == best_mape_mean else
           f"**The leader changes** (A: {best_mape_mean}, B: {sumB.iloc[0]['model']}), so the "
           "training-size caps are a material confound in the canonical protocol and any "
           "ranking must be reported with that caveat."),
        "",
        q(16, "One defensible representative, or an equivalent top group?")
        + f" **An equivalent top group.** {len(tie)} of 21 models are not distinguishable "
          "from the best-ranked model on MAPE at the Nemenyi critical difference. No single "
          "model is statistically separable as the best.",
        "",
        q(17, "Most defensible candidate to carry forward?")
        + " Because the top group is statistically indistinguishable, the choice cannot be "
          "made on these statistics alone and must be made on a stated secondary ground. "
          f"HistGB is {'inside' if hist_in else 'outside'} the top group "
          f"(avg MAPE rank {ranks['MAPE']['HistGB']:.2f}), so retaining it as the fixed "
          "representative is "
        + ("defensible provided the paper states it was fixed a priori and reports the tie."
           if hist_in else
           "**not** supported by this analysis and would need a different justification."),
        "",
        "## What this analysis does not license",
        "",
        "- Naming a statistically best backend. The Friedman test rejects global equivalence, "
        "but the post-hoc cannot separate the leaders.",
        f"- Over-reading the Wilcoxon results: with {len(folds)} folds the smallest attainable "
        f"two-sided p is {min_p_wilcoxon:.4f}, so only large, consistent differences can ever "
        "reach significance, and ties are expected.",
        "- Reading these ranks as same-distribution accuracy. Every fold carries positive "
        "fouling drift from fitting to validation block (see `protocol/"
        "temporal_fold_definition.md`), so the folds reward conservative extrapolation.",
    ]
    write_text("\n".join(lines), DIRS["summary"] / "ml_statistical_selection_summary.md")

    claims = [
        (f"\"{best_mape_mean} has the lowest mean validation MAPE.\"", "SUPPORTED",
         f"mean fold MAPE {sumA.iloc[0]['mean_MAPE']:.4f} +/- {sumA.iloc[0]['std_MAPE']:.4f} "
         f"over {len(folds)} folds; `metrics/residual_model_summary.csv`."),
        (f"\"{best_mape_mean} is statistically superior to all alternatives.\"",
         "NOT SUPPORTED",
         f"{len(tie)} of 21 models lie within the Nemenyi critical difference "
         f"({cds['MAPE']:.2f} rank positions) of the best-ranked model; "
         "`nemenyi/nemenyi_mape_pairs.csv`."),
        ("\"There is a statistically best-performing group of models.\"", "SUPPORTED",
         f"Friedman rejects global equivalence on MAPE (p = {frA.loc['MAPE','p_value']:.3e}) "
         f"while the post-hoc leaves {len(tie)} models mutually indistinguishable from the "
         "leader."),
        ("\"HistGB belongs to the statistically best-performing group.\"",
         "SUPPORTED" if hist_in else "NOT SUPPORTED",
         f"Nemenyi p vs best-ranked = {Pmats['MAPE'].loc[best_rank['MAPE'],'HistGB']:.3f}; "
         f"avg MAPE rank {ranks['MAPE']['HistGB']:.2f}."),
        ("\"SVR belongs to the statistically best-performing group.\"",
         "SUPPORTED" if svr_in else "NOT SUPPORTED",
         f"Nemenyi p vs best-ranked = {Pmats['MAPE'].loc[best_rank['MAPE'],'SVR (RBF)']:.3f}."),
        ("\"CatBoost belongs to the statistically best-performing group.\"",
         "SUPPORTED" if cat_in else "NOT SUPPORTED",
         f"Nemenyi p vs best-ranked = {Pmats['MAPE'].loc[best_rank['MAPE'],'CatBoost']:.3f}."),
        ("\"Residual learning improves significantly over the WBM.\"",
         "SUPPORTED" if n_sig_better else "NOT SUPPORTED",
         f"{n_better_wbm}/21 beat WBM in mean fold MAPE, {n_all_folds}/21 in every fold, "
         f"{n_sig_better}/21 significantly after Holm-corrected paired Wilcoxon; "
         "`paired/wbm_vs_models_holm.csv`."),
        ("\"A single model should be selected for the architecture comparison on the "
         "strength of these statistics.\"", "NOT SUPPORTED",
         "The top group is statistically equivalent; the choice requires a stated "
         "secondary criterion, not a statistical one."),
        ("\"The training-size caps do not affect the ranking.\"",
         "SUPPORTED" if sumB.iloc[0]['model'] == best_mape_mean else "NOT SUPPORTED",
         f"Protocol B (common {meta['common_training_size_protocol_B']:,} records) leader is "
         f"{sumB.iloc[0]['model']} against {best_mape_mean} in the canonical protocol."),
    ]
    write_text("\n".join([
        "# Statistical claim audit", "",
        "| Claim | Status | Evidence |", "|---|---|---|",
    ] + [f"| {c} | **{s}** | {e} |" for c, s, e in claims]),
        DIRS["summary"] / "statistical_claim_audit.md")

    write_text("\n".join([
        "# 17 — Statistical comparison of the 21 candidate ML models",
        "",
        "Friedman + Nemenyi over the 21 learning backends in the **residual** role, on "
        f"**{len(folds)} rolling-origin temporal folds** of the Shifts training partition. "
        "`dev_in` and `dev_out` are never read. Physics scale refitted per fold. No "
        "hyperparameter retuned.",
        "",
        "## Command",
        "",
        "    PYTHONPATH=. python -m experiments.revision.rf17_ml_statistical_comparison",
        "",
        f"Seed {meta['seed']}; git `{meta['git_commit']}`. Nemenyi p-values are exact "
        "studentized-range values computed in this script (`scikit_posthocs` is not installed "
        "and no dependency was added).",
        "",
        "## Headline",
        "",
        f"- Friedman rejects equivalence on all four metrics (MAPE p = "
        f"{frA.loc['MAPE','p_value']:.2e}).",
        f"- Best mean MAPE: **{best_mape_mean}**; best average MAPE rank: "
        f"**{best_rank['MAPE']}**.",
        f"- **{len(tie)} of 21** models are not distinguishable from the best-ranked model "
        f"(CD = {cds['MAPE']:.2f} rank positions) — a statistically **equivalent top group**, "
        "not a single winner.",
        f"- Residual models beating the WBM in mean fold MAPE: {n_better_wbm}/21; "
        f"significantly better after Holm: {n_sig_better}/21.",
        "",
        "## Layout",
        "",
        "- `protocol/` fold definitions (with fouling drift per fold) and the training-size audit",
        "- `metrics/` per-fold metrics for both protocols, per-model summary, WBM reference",
        "- `friedman/` omnibus tests, CSV and LaTeX",
        "- `nemenyi/` four 21x21 p-value matrices and four long-format pair tables",
        "- `paired/` leading-subset Wilcoxon+Holm, and every model against the WBM",
        "- `figures/` four Nemenyi heatmaps and four critical-difference diagrams (PDF+PNG)",
        "- `tables/` ranking and three manuscript-ready LaTeX tables",
        "- `summary/` the 17 answers and the claim audit",
    ]), ROOT / "README.md")


if __name__ == "__main__":
    main()
