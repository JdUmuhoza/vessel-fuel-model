"""Friedman + Nemenyi comparison of the four architectures (WBM, BBM, S-GBM, C-GBM).

Applies to the *architectures* the two-step non-parametric protocol already used for
learner selection in ``ml_statistical_comparison`` (Friedman omnibus test, Nemenyi post-hoc test with a
critical-difference diagram, alpha = 0.05), as in Fousseni, Otis and Ziane (2026),
Machine Learning with Applications 23, 100813, Sections 3.8 and 4.3-4.4.

Two pre-declared analyses, both fixed before any result was seen:

A. Rolling-origin cross-validation on the development data only.
   The chronological development file (train + validation + internal-test blocks of
   the folder-01 60/20/20 split, 530,706 records) is cut into ``N_BLOCKS`` contiguous
   unshuffled blocks, giving ``N_BLOCKS - 1`` folds: fold f trains on blocks 0..f and
   tests on block f+1. Inside each fold the training window is split in time into a
   fitting portion (first ``1 - TAIL_FRACTION``) and a selection tail (last
   ``TAIL_FRACTION``): the physics scale k, the Huber residual learner and the Huber
   BBM are fitted on the fitting portion, and lambda is selected on the tail by MAPE
   over the 101-point grid. dev_in and dev_out are never read.

B. Held-out partitions with the frozen manuscript models.
   The folder-01 models (k on the training block, Huber residual and Huber BBM on the
   training block's fitting sub-block, lambda* = 0.70) are rebuilt and asserted to
   reproduce the folder-01 dev_in/dev_out MAPE. Each held-out partition, ordered by
   time, is cut into ``N_EVAL_BLOCKS`` contiguous blocks that act as repeated measures.

Statistics per analysis and metric (MAPE, MAE, RMSE, R2): Shapiro-Wilk on the
per-fold scores (justifies the non-parametric route), Friedman test, exact Nemenyi
pairwise p-values and critical difference, and two-sided Wilcoxon signed-rank tests of
C-GBM against each other architecture with Holm correction (3 comparisons).

Nothing existing is overwritten; no manuscript file is touched.

Run from the repository root::

    PYTHONPATH=. python -m experiments.revision.s10_architecture_friedman_nemenyi
"""

from __future__ import annotations

import json
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, shapiro, wilcoxon

from .common_rev import (
    ETA_PUBLISHED, FEATS, REPO_ROOT, REV_FINAL, SEED, physics_power_eta,
    write_csv, write_text,
)
from .ml_statistical_comparison import avg_ranks, holm, nemenyi_exact
from .s01_full_data_602020 import ES_FRACTION, fit_learner, metrics7
from benchmark_measured.common import fit_scale, load_split  # noqa: E402

OUT = REV_FINAL / "10_architecture_friedman_nemenyi"
F01 = REV_FINAL / "01_full_data_602020"

# ---- pre-declared protocol constants -------------------------------------
N_BLOCKS = 9                 # analysis A: 9 contiguous blocks -> 8 rolling-origin folds
TAIL_FRACTION = 0.20         # analysis A: lambda-selection tail of each training window
N_EVAL_BLOCKS = 10           # analysis B: contiguous blocks per held-out partition
ALPHA = 0.05
LAMBDA_STAR = 0.70           # folder-01 / manuscript value, used in analysis B only
LAM = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 4)
LEARNER = "huber"            # residual learner and BBM backend (folder-01 definition)
ARCHS = ["WBM", "BBM", "S-GBM", "C-GBM"]
METRICS = ["MAPE", "MAE", "RMSE", "R2"]
LOWER_BETTER = {"MAPE": True, "MAE": True, "RMSE": True, "R2": False}
MAPE_TOL = 1e-4              # analysis B must reproduce folder-01 MAPE to this tolerance


def _git() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------------ data ----

def load_development() -> dict:
    """Load the Shifts train file and the two held-out files with cached physics.

    Returns
    -------
    dict
        ``dev`` (development frame in time order), ``ph_dev`` (uncalibrated physics
        power, same order), the folder-01 block boundaries ``i60``/``i80``, and the
        ``dev_in`` / ``dev_out`` frames with their physics arrays.
    """
    tr_full = load_split("train.csv")
    n = len(tr_full)
    i60, i80 = int(0.60 * n), int(0.80 * n)
    blocks = {"train": tr_full.iloc[:i60], "validation": tr_full.iloc[i60:i80],
              "internal_test": tr_full.iloc[i80:]}
    ph = {k: physics_power_eta(v.reset_index(drop=True), f"602020_{k}", ETA_PUBLISHED)
          for k, v in blocks.items()}
    out = {"dev": tr_full.reset_index(drop=True),
           "ph_dev": np.concatenate([ph["train"], ph["validation"], ph["internal_test"]]),
           "i60": i60, "i80": i80}
    for nm in ("dev_in", "dev_out"):
        d = load_split(f"{nm}.csv").reset_index(drop=True)
        out[nm] = d
        out[f"ph_{nm}"] = physics_power_eta(d, f"602020_{nm}", ETA_PUBLISHED)
    if not np.all(np.diff(out["dev"]["time_id"].to_numpy()) >= 0):
        raise AssertionError("development file is not in time order")
    return out


def predictions(k: float, resid, bbm, lam: float, X: np.ndarray,
                ph: np.ndarray) -> dict[str, np.ndarray]:
    """Predictions of the four architectures for one design matrix."""
    base = k * ph
    r = resid.predict(X)
    return {"WBM": base, "BBM": bbm.predict(X), "S-GBM": base + r,
            "C-GBM": base + lam * r}


# ---------------------------------------------------------- analysis A ----

def analysis_a(D: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin folds on the development data; lambda reselected per fold."""
    dev, ph = D["dev"], D["ph_dev"]
    y, X = dev["power"].to_numpy(), dev[FEATS].to_numpy()
    tid = dev["time_id"].to_numpy()
    b = np.linspace(0, len(dev), N_BLOCKS + 1).astype(int)
    rows, fold_def = [], []
    for f in range(N_BLOCKS - 1):
        win = np.arange(0, b[f + 1])
        test = np.arange(b[f + 1], b[f + 2])
        n_fit = int((1.0 - TAIL_FRACTION) * len(win))
        fit_i, tail_i = win[:n_fit], win[n_fit:]
        t0 = time.time()
        k = fit_scale(y[fit_i], ph[fit_i])
        res = y - k * ph
        # Huber has no early stopping, so the es index is unused; fit on fit_i only.
        resid = fit_learner(LEARNER, X, res, fit_i, np.array([], dtype=int))[0]
        bbm = fit_learner(LEARNER, X, y, fit_i, np.array([], dtype=int))[0]
        r_tail = resid.predict(X[tail_i])
        mape_tail = [metrics7(y[tail_i], k * ph[tail_i] + l * r_tail)["MAPE"] for l in LAM]
        lam_f = float(LAM[int(np.argmin(mape_tail))])
        pr = predictions(k, resid, bbm, lam_f, X[test], ph[test])
        for a in ARCHS:
            m = metrics7(y[test], pr[a])
            rows.append({"analysis": "A_rolling_origin_cv", "block": f, "architecture": a,
                         **{c: m[c] for c in METRICS}})
        fold_def.append({"fold": f, "n_fit": len(fit_i), "n_tail": len(tail_i),
                         "n_test": len(test), "k": k, "lambda_selected": lam_f,
                         "fit_time_id_start": float(tid[fit_i[0]]),
                         "fit_time_id_end": float(tid[fit_i[-1]]),
                         "tail_time_id_end": float(tid[tail_i[-1]]),
                         "test_time_id_start": float(tid[test[0]]),
                         "test_time_id_end": float(tid[test[-1]]),
                         "strictly_causal": bool(tid[test[0]] > tid[tail_i[-1]]),
                         "seconds": round(time.time() - t0, 1)})
        print(f"[A] fold {f}: n_fit={len(fit_i):,} test={len(test):,} "
              f"k={k:.4f} lambda={lam_f:.2f} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(fold_def)


# ---------------------------------------------------------- analysis B ----

def analysis_b(D: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen folder-01 models scored on contiguous blocks of dev_in and dev_out."""
    dev, ph = D["dev"], D["ph_dev"]
    i60 = D["i60"]
    y_tr, X_tr, ph_tr = (dev["power"].to_numpy()[:i60], dev[FEATS].to_numpy()[:i60],
                         ph[:i60])
    k = fit_scale(y_tr, ph_tr)
    n_fit = int((1.0 - ES_FRACTION) * i60)
    fi, ei = np.arange(n_fit), np.arange(n_fit, i60)
    resid = fit_learner(LEARNER, X_tr, y_tr - k * ph_tr, fi, ei)[0]
    bbm = fit_learner(LEARNER, X_tr, y_tr, fi, ei)[0]

    ref = pd.read_csv(F01 / "final_dev_in_dev_out.csv")
    ref["architecture"] = ref["architecture"].str.replace(r"^C-GBM.*", "C-GBM", regex=True)
    rows, ver = [], []
    for nm in ("dev_in", "dev_out"):
        d = D[nm]
        order = np.argsort(d["time_id"].to_numpy(), kind="stable")
        y, X, p_nm = (d["power"].to_numpy()[order], d[FEATS].to_numpy()[order],
                      D[f"ph_{nm}"][order])
        pr = predictions(k, resid, bbm, LAMBDA_STAR, X, p_nm)
        for a in ARCHS:                      # reproduction check on the whole partition
            whole = metrics7(y, pr[a])["MAPE"]
            exp = float(ref[(ref.partition == nm) & (ref.architecture == a)].MAPE.iloc[0])
            ver.append({"partition": nm, "architecture": a, "MAPE_recomputed": whole,
                        "MAPE_folder24": exp, "abs_diff": abs(whole - exp),
                        "reproduces": abs(whole - exp) < MAPE_TOL})
        edges = np.linspace(0, len(y), N_EVAL_BLOCKS + 1).astype(int)
        for bi in range(N_EVAL_BLOCKS):
            s = slice(edges[bi], edges[bi + 1])
            for a in ARCHS:
                m = metrics7(y[s], pr[a][s])
                rows.append({"analysis": f"B_frozen_{nm}", "block": bi, "architecture": a,
                             **{c: m[c] for c in METRICS}})
    vdf = pd.DataFrame(ver)
    if not vdf.reproduces.all():
        raise AssertionError("frozen models do not reproduce folder-01 MAPE:\n"
                             + vdf.to_string())
    print(f"[B] frozen models reproduce folder-01 MAPE on all 8 cells "
          f"(k={k:.6f})", flush=True)
    return pd.DataFrame(rows), vdf


# ------------------------------------------------------------ statistics ----

def statistics(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Shapiro-Wilk, Friedman, Nemenyi and Holm-corrected Wilcoxon per analysis/metric."""
    fr, nem, wil, rk, sw = [], [], [], [], []
    for an, g in scores.groupby("analysis", sort=False):
        for c in METRICS:
            piv = g.pivot(index="block", columns="architecture", values=c)[ARCHS]
            n = len(piv)
            for a in ARCHS:
                p_sw = float(shapiro(piv[a]).pvalue) if n >= 3 else np.nan
                sw.append({"analysis": an, "metric": c, "architecture": a,
                           "shapiro_p": p_sw, "normal_at_alpha": p_sw >= ALPHA})
            stat, p = friedmanchisquare(*[piv[a].to_numpy() for a in ARCHS])
            ranks = avg_ranks(piv, LOWER_BETTER[c])
            P, q, cd = nemenyi_exact(ranks, n)
            fr.append({"analysis": an, "metric": c, "n_blocks": n, "k_models": len(ARCHS),
                       "friedman_chi2": float(stat), "df": len(ARCHS) - 1,
                       "p_value": float(p), "reject_equal_performance": p < ALPHA,
                       "nemenyi_q_alpha": q, "critical_difference": cd,
                       "best_ranked": str(ranks.idxmin())})
            for a in ARCHS:
                rk.append({"analysis": an, "metric": c, "architecture": a,
                           "average_rank": float(ranks[a]),
                           "mean_score": float(piv[a].mean()),
                           "std_score": float(piv[a].std(ddof=1)),
                           "blocks_ranked_first": int((piv.rank(
                               axis=1, ascending=LOWER_BETTER[c])[a] == 1).sum())})
            for i, a in enumerate(ARCHS):
                for b_ in ARCHS[i + 1:]:
                    nem.append({"analysis": an, "metric": c, "model_a": a, "model_b": b_,
                                "rank_a": float(ranks[a]), "rank_b": float(ranks[b_]),
                                "rank_difference": float(abs(ranks[a] - ranks[b_])),
                                "critical_difference": cd,
                                "nemenyi_p": float(P.loc[a, b_]),
                                "significant": float(P.loc[a, b_]) < ALPHA})
            others = [a for a in ARCHS if a != "C-GBM"]
            raw = []
            for o in others:
                d = piv["C-GBM"].to_numpy() - piv[o].to_numpy()
                raw.append(float(wilcoxon(d, alternative="two-sided").pvalue)
                           if np.any(d != 0) else 1.0)
            adj = holm(raw)
            for o, pr_, pa in zip(others, raw, adj):
                d = piv["C-GBM"] - piv[o]
                better = (d < 0) if LOWER_BETTER[c] else (d > 0)
                wil.append({"analysis": an, "metric": c, "comparison": f"C-GBM vs {o}",
                            "median_difference": float(d.median()),
                            "blocks_cgbm_better": int(better.sum()), "n_blocks": n,
                            "wilcoxon_p": pr_, "p_holm": pa,
                            "significant_holm": pa < ALPHA})
    return {"friedman": pd.DataFrame(fr), "nemenyi": pd.DataFrame(nem),
            "wilcoxon": pd.DataFrame(wil), "ranks": pd.DataFrame(rk),
            "shapiro": pd.DataFrame(sw)}


# --------------------------------------------------------------- figures ----

def cd_diagram(ranks: pd.Series, cd: float, title: str, path_stem) -> None:
    """Demsar-style critical-difference diagram (rank 1 = best, on the left)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 2.0))
    _draw_cd(ax, ranks, cd, title)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def _draw_cd(ax, ranks: pd.Series, cd: float, title: str) -> None:
    """Draw one critical-difference diagram on an existing axis."""
    k = len(ranks)
    r = ranks.sort_values()
    ax.set_xlim(-0.75, k + 1.75)              # rank 1 (best) on the left; room for labels
    ax.set_ylim(0, 1.08)
    ax.hlines(0.8, 1, k, color="black", lw=0.8)
    for t in range(1, k + 1):
        ax.vlines(t, 0.78, 0.82, color="black", lw=0.8)
        ax.text(t, 0.86, str(t), ha="center", va="bottom", fontsize=8)
    ax.hlines(0.97, 1, 1 + cd, color="black", lw=1.2)
    ax.text(1 + cd / 2, 0.99, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    names = list(r.index)
    half = (k + 1) // 2
    for i, nm in enumerate(names):
        x = float(r[nm])
        left = i < half
        # left side: best on top; right side: worst on top, so connectors never cross
        # right side offset by half a step so tied ranks never merge into one line
        ylab = 0.62 - 0.14 * i if left else 0.55 - 0.14 * (k - 1 - i)
        xt = 0.85 if left else k + 0.15
        ax.plot([x, x], [0.8, ylab], color="black", lw=0.7)
        ax.plot([x, xt], [ylab, ylab], color="black", lw=0.7)
        ax.text(xt, ylab, f"{nm} ({x:.2f})", ha="right" if left else "left",
                va="center", fontsize=7)
    # Bars join each maximal group of models whose rank spread is below CD
    # (not distinguishable by Nemenyi). Groups are drawn once, left to right.
    yb, vals, last_j = 0.74, r.to_numpy(), -1
    for i in range(k):
        j = max(jj for jj in range(i, k) if vals[jj] - vals[i] < cd)
        if j > i and j > last_j:
            ax.hlines(yb, vals[i] - 0.03, vals[j] + 0.03, color="black", lw=2.2)
            yb -= 0.05
            last_j = j
    ax.set_ylim(0.15, 1.08)                   # trim the empty band under the labels
    ax.axis("off")
    ax.set_title(title, fontsize=7)


# ------------------------------------------------------- manuscript assets ----

PANELS = [("A_rolling_origin_cv", "(a) Rolling-origin cross-validation, 8 folds"),
          ("B_frozen_dev_in", "(b) In-domain set, 10 blocks"),
          ("B_frozen_dev_out", "(c) Shifted set, 10 blocks")]


def paper_figure(st: dict) -> None:
    """Three stacked MAPE critical-difference panels for the manuscript."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.4))
    for ax, (an, lab) in zip(axes, PANELS):
        f = st["friedman"][(st["friedman"].analysis == an)
                           & (st["friedman"].metric == "MAPE")].iloc[0]
        rr = st["ranks"][(st["ranks"].analysis == an)
                         & (st["ranks"].metric == "MAPE")].set_index("architecture")
        _draw_cd(ax, rr["average_rank"], f.critical_difference,
                 f"{lab}; Friedman p = {f.p_value:.2g}")
    fig.tight_layout(h_pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_cd_mape_three_panels.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)


HEATMAP_METRICS = [("R2", "$R^2$"), ("RMSE", "RMSE"), ("MAE", "MAE")]  # as the reference


def heatmap_figure(st: dict, analysis: str, stem: str) -> None:
    """Annotated Nemenyi p-value heatmaps, one panel per metric (R2, RMSE, MAE).

    Same presentation as ``generate_statistical_heatmaps`` in the reference
    ``statistical_analysis.py``: a matrix of pairwise Nemenyi p-values per metric,
    ``coolwarm`` colour map on [0, 1] centred at 0.05 (the seaborn ``center=0.05``
    setting, reproduced with a two-slope norm since seaborn is not installed), the
    diagonal set to 1 (a model against itself), and a colour bar as legend.
    Significant pairs (p < ALPHA) are also printed in bold.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    norm = TwoSlopeNorm(vmin=0.0, vcenter=ALPHA, vmax=1.0)
    cmap = plt.get_cmap("coolwarm")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    im = None
    for pi, (ax, (c, lab)) in enumerate(zip(axes, HEATMAP_METRICS)):
        nem = st["nemenyi"][(st["nemenyi"].analysis == analysis) & (st["nemenyi"].metric == c)]
        P = pd.DataFrame(np.ones((len(ARCHS), len(ARCHS))), index=ARCHS, columns=ARCHS)
        for r in nem.itertuples():
            P.loc[r.model_a, r.model_b] = P.loc[r.model_b, r.model_a] = r.nemenyi_p
        im = ax.imshow(P.to_numpy(), cmap=cmap, norm=norm)
        for i in range(len(ARCHS)):
            for j in range(len(ARCHS)):
                p = float(P.iloc[i, j])
                shade = norm(p)                     # 0 = darkest blue, 1 = darkest red
                ax.text(j, i, "<0.001" if p < 0.001 else f"{p:.3f}", ha="center",
                        va="center", fontsize=7.5,
                        color="white" if (shade < 0.15 or shade > 0.85) else "black",
                        fontweight="bold" if p < ALPHA else "normal")
        ax.set_xticks(range(len(ARCHS)), ARCHS, rotation=35, ha="right")
        ax.set_yticks(range(len(ARCHS)), ARCHS if pi == 0 else [""] * len(ARCHS))
        ax.set_xticks(np.arange(-0.5, len(ARCHS)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ARCHS)), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        f = st["friedman"][(st["friedman"].analysis == analysis)
                           & (st["friedman"].metric == c)].iloc[0]
        ax.set_title(f"({'abc'[pi]}) {lab}, Friedman $p$ = "
                     + ("<0.001" if f.p_value < 0.001 else f"{f.p_value:.3f}"), fontsize=8)
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.945, 0.24, 0.013, 0.62])
    # Two-slope norm: lower half of the bar spans 0..ALPHA, upper half ALPHA..1,
    # so ticks are spread over both halves rather than crowded around ALPHA.
    ticks = [0.0, 0.01, 0.025, ALPHA, 0.25, 0.5, 0.75, 1.0]
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    cb.ax.set_yticklabels(["0", "0.01", "0.025", f"{ALPHA:g}", "0.25", "0.50", "0.75", "1"],
                          fontsize=7)
    cb.set_label(f"Nemenyi $p$-value (< {ALPHA:g}: significant)", fontsize=7.5)
    plt.rcParams["svg.fonttype"] = "none"       # keep SVG text editable, not outlined
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def _p(p: float) -> str:
    """Format a p-value for the table; bold when below ALPHA."""
    s = "$<$0.001" if p < 0.001 else f"{p:.3f}"
    return f"\\textbf{{{s}}}" if p < ALPHA else s


def _against(row: pd.Series, metric: str) -> str:
    """Dagger when the median block difference favours the other architecture."""
    worse = row["median_difference"] > 0 if LOWER_BETTER[metric] else row["median_difference"] < 0
    return "$^{\\dagger}$" if worse else ""


def paper_table(st: dict) -> str:
    """LaTeX table of Friedman tests, average ranks and Holm-corrected Wilcoxon tests."""
    names = {"A_rolling_origin_cv": "Cross-validation",
             "B_frozen_dev_in": "In-domain set", "B_frozen_dev_out": "Shifted set"}
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Friedman tests, average ranks and Holm-corrected Wilcoxon tests of "
         r"the four architectures.}",
         r"\label{tab:friedman-nemenyi-tests}", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.06}",
         r"\begin{tabular}{llrrcccccccc}", r"\toprule", r"\rowcolor{gray!10}",
         r"\textbf{Analysis} & \textbf{Metric} & \textbf{$\chi^2_F$} & "
         r"\textbf{$p_F$} & \textbf{WBM} & \textbf{BBM} & \textbf{S-GBM} & "
         r"\textbf{C-GBM} & \textbf{CD} & \textbf{vs WBM} & \textbf{vs BBM} & "
         r"\textbf{vs S-GBM}\\", r"\midrule"]
    for ai, (an, _) in enumerate(PANELS):
        for mi, c in enumerate(METRICS):
            f = st["friedman"][(st["friedman"].analysis == an)
                               & (st["friedman"].metric == c)].iloc[0]
            r = st["ranks"][(st["ranks"].analysis == an)
                            & (st["ranks"].metric == c)].set_index("architecture")
            w = st["wilcoxon"][(st["wilcoxon"].analysis == an)
                               & (st["wilcoxon"].metric == c)].set_index("comparison")
            best = r["average_rank"].min()
            rk = [f"\\textbf{{{r.loc[a, 'average_rank']:.2f}}}"
                  if r.loc[a, "average_rank"] == best else
                  f"{r.loc[a, 'average_rank']:.2f}" for a in ARCHS]
            first = f"\\multirow{{4}}{{*}}{{{names[an]}}}" if mi == 0 else ""
            L.append(f"{first} & {'$R^2$' if c == 'R2' else c} & "
                     f"{f.friedman_chi2:.2f} & {_p(f.p_value)} & " + " & ".join(rk)
                     + f" & {f.critical_difference:.2f} & "
                     + " & ".join(_p(w.loc[f'C-GBM vs {o}', 'p_holm'])
                                  + _against(w.loc[f'C-GBM vs {o}'], c)
                                  for o in ("WBM", "BBM", "S-GBM")) + r"\\")
        if ai < len(PANELS) - 1:
            L.append(r"\midrule")
    L += [r"\midrule",
          r"\multicolumn{12}{@{}p{16.5cm}@{}}{\footnotesize Columns WBM to C-GBM give "
          r"average Friedman ranks (1 = best, lowest in bold); CD is the Nemenyi critical "
          r"difference at the 5\% significance level; the last three columns give Holm-corrected "
          r"Wilcoxon $p$-values of C-GBM against each architecture, in bold when below "
          r"0.05. $^{\dagger}$ marks a difference in favour of the other architecture.}\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main ----

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    t0 = time.time()
    D = load_development()
    print(f"[S10] development {len(D['dev']):,}; dev_in {len(D['dev_in']):,}; "
          f"dev_out {len(D['dev_out']):,}", flush=True)
    sa, folds = analysis_a(D)
    sb, ver = analysis_b(D)
    scores = pd.concat([sa, sb], ignore_index=True)
    st = statistics(scores)

    write_csv(scores, OUT / "per_block_scores.csv")
    write_csv(folds, OUT / "analysis_A_fold_definition.csv")
    write_csv(ver, OUT / "analysis_B_reproduction_check.csv")
    for name, df in st.items():
        write_csv(df, OUT / f"{name}.csv")

    labels = {"A_rolling_origin_cv": f"{N_BLOCKS - 1} rolling-origin folds (development)",
              "B_frozen_dev_in": f"{N_EVAL_BLOCKS} blocks, in-domain set (frozen models)",
              "B_frozen_dev_out": f"{N_EVAL_BLOCKS} blocks, shifted set (frozen models)"}
    for _, row in st["friedman"].iterrows():
        rr = st["ranks"][(st["ranks"].analysis == row.analysis)
                         & (st["ranks"].metric == row.metric)].set_index("architecture")
        cd_diagram(rr["average_rank"], row.critical_difference,
                   f"{row.metric}: {labels[row.analysis]}; Friedman p = {row.p_value:.3g}",
                   OUT / "figures" / f"cd_{row.analysis}_{row.metric}")

    paper_figure(st)
    heatmap_figure(st, "A_rolling_origin_cv", "fig_nemenyi_heatmaps_cv")       # manuscript
    heatmap_figure(st, "B_frozen_dev_in", "fig_nemenyi_heatmaps_in_domain")    # record
    heatmap_figure(st, "B_frozen_dev_out", "fig_nemenyi_heatmaps_shifted")     # record
    write_text(paper_table(st), OUT / "tab_friedman_nemenyi_tests.tex")

    meta = {"purpose": "Friedman + Nemenyi comparison of WBM, BBM, S-GBM and C-GBM",
            "reference_protocol": "Fousseni, Otis & Ziane (2026) MLWA 23:100813, s3.8, s4.3-4.4",
            "analysis_A": {"n_blocks": N_BLOCKS, "n_folds": N_BLOCKS - 1,
                           "tail_fraction_for_lambda": TAIL_FRACTION,
                           "lambda_grid": "0.00-1.00 step 0.01", "selection_metric": "MAPE",
                           "dev_in_dev_out_used": False},
            "analysis_B": {"n_blocks_per_partition": N_EVAL_BLOCKS,
                           "lambda_star": LAMBDA_STAR, "models": "folder-01 frozen",
                           "reproduces_folder24": bool(ver.reproduces.all())},
            "learner": LEARNER, "metrics": METRICS, "alpha": ALPHA, "seed": SEED,
            "eta_wave_loss": ETA_PUBLISHED, "git_commit": _git(),
            "runtime_s": round(time.time() - t0, 1)}
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_text(report(st, folds, labels), OUT / "README.md")
    print(f"[S10] done in {time.time() - t0:.0f}s -> {OUT}", flush=True)
    return 0


def report(st: dict, folds: pd.DataFrame, labels: dict) -> str:
    """Markdown summary read back from the computed tables."""
    L = ["# 38 - Friedman + Nemenyi comparison of the four architectures", "",
         "Protocol of Fousseni, Otis & Ziane (2026), MLWA 23:100813 (s3.8, s4.3-4.4), "
         "applied to WBM, BBM, S-GBM and C-GBM. Generated by "
         "`experiments/revision/s10_architecture_friedman_nemenyi.py`; every number "
         "below is read from the CSVs in this folder.", "",
         f"- Analysis A: {N_BLOCKS - 1} rolling-origin folds on the development data; "
         f"lambda reselected per fold on the last {int(TAIL_FRACTION * 100)}% of each "
         f"training window (selected values: "
         f"{', '.join(f'{v:.2f}' for v in folds.lambda_selected)}).",
         f"- Analysis B: frozen manuscript models (Huber, lambda* = {LAMBDA_STAR}), "
         f"{N_EVAL_BLOCKS} contiguous blocks per held-out partition.", ""]
    for an, lab in labels.items():
        L += [f"## {lab}", "", "| Metric | Friedman chi2 | p | CD | Avg rank WBM | BBM | "
              "S-GBM | C-GBM | C-GBM vs WBM (Holm p) | vs BBM | vs S-GBM |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in METRICS:
            f = st["friedman"][(st["friedman"].analysis == an) & (st["friedman"].metric == c)].iloc[0]
            r = st["ranks"][(st["ranks"].analysis == an) & (st["ranks"].metric == c)].set_index("architecture")
            w = st["wilcoxon"][(st["wilcoxon"].analysis == an) & (st["wilcoxon"].metric == c)].set_index("comparison")
            cell = lambda o: (f"{w.loc[f'C-GBM vs {o}', 'p_holm']:.3g}"
                              f" ({int(w.loc[f'C-GBM vs {o}', 'blocks_cgbm_better'])}/"
                              f"{int(w.loc[f'C-GBM vs {o}', 'n_blocks'])})")
            L.append(f"| {c} | {f.friedman_chi2:.2f} | {f.p_value:.3g} | "
                     f"{f.critical_difference:.2f} | "
                     + " | ".join(f"{r.loc[a, 'average_rank']:.2f}" for a in ARCHS)
                     + f" | {cell('WBM')} | {cell('BBM')} | {cell('S-GBM')} |")
        L.append("")
        sig = st["nemenyi"][(st["nemenyi"].analysis == an) & st["nemenyi"].significant]
        L.append("Nemenyi pairs significant at alpha = 0.05: "
                 + ("; ".join(f"{s.metric}: {s.model_a}-{s.model_b} (p={s.nemenyi_p:.3g})"
                              for s in sig.itertuples()) or "none") + ".")
        L.append("")
    L += ["Wilcoxon cells give the Holm-corrected p-value and, in brackets, the number "
          "of blocks in which C-GBM scored better. Normality of the per-block scores is "
          "tested in `shapiro.csv`.", ""]
    return "\n".join(L)


def _guarded() -> int:
    import joblib
    with joblib.parallel_backend("threading", n_jobs=1):
        return main()


if __name__ == "__main__":
    raise SystemExit(_guarded())
