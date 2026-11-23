"""Revised manuscript Figures 3 and 4, rebuilt on the folder-01 frozen configuration.

Source of truth: results/revision_final/01_full_data_602020/ (full-data chronological
60/20/20, calibrated WBM with k = 0.393586, Huber residual learner, lambda* = 0.70).
The BBM in this configuration is the Huber learner fitted directly on measured power,
which is the definition folder 01 uses; the independent ExtraTrees BBM of folder 02 is a
later alternative experiment and is deliberately not used here.

Figure 3 - architecture-level predictive performance on dev_in and dev_out.
Figure 4 - measured versus predicted shaft power, WBM and C-GBM, both partitions.

Both are predictive diagnostics only. No physical-consistency, D_phys, speed-exponent or
operating-regime quantity is computed or plotted.

Nothing is refitted, retuned or reselected on dev_in or dev_out. Every metric printed on
a figure is recomputed from the frozen predictions and asserted against the folder-01
reference table before any file is written.
"""

from __future__ import annotations

import json
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from .common_rev import REPO_ROOT, REV_FINAL, SEED, write_csv
from .s01_full_data_602020 import FEATS, build_splits, fit_learner, metrics7, physics

_E = REPO_ROOT / "experiments"
if str(_E) not in sys.path:
    sys.path.insert(0, str(_E))
from benchmark_measured.common import fit_scale  # noqa: E402

F01 = REV_FINAL / "01_full_data_602020"
OUT = REV_FINAL / "04_figures_3_4"
LAMBDA_STAR = 0.70
RESID_KEY = BBM_KEY = "huber"          # folder-01 definition of both roles
K_REFERENCE = 0.393586
TOL_PP = 0.001                          # rounding tolerance on a value quoted to 3 dp
ARCHS = ["WBM", "BBM", "S-GBM", "C-GBM"]
COL = {"WBM": "#4C4C4C", "BBM": "#C44E52", "S-GBM": "#4C72B0", "C-GBM": "#55A868"}
PART_HATCH = {"dev_in": "", "dev_out": "///"}
# Names printed on the figures; the Shifts file names dev_in/dev_out stay as data keys.
PART_LABEL = {"dev_in": "In-domain", "dev_out": "Shifted"}

# Section 5.3 values the figures must reproduce (MAPE, %)
EXPECTED = {
    ("dev_in", "WBM"): 8.848, ("dev_in", "BBM"): 18.820,
    ("dev_in", "S-GBM"): 8.788, ("dev_in", "C-GBM"): 8.238,
    ("dev_out", "WBM"): 9.214, ("dev_out", "BBM"): 21.930,
    ("dev_out", "S-GBM"): 9.545, ("dev_out", "C-GBM"): 8.732,
}
EXPECTED_DELTA = {"WBM": 0.366, "BBM": 3.110, "S-GBM": 0.757, "C-GBM": 0.495}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W2 = 7.16    # elsarticle 5p double-column width, inches


def _git() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


def save(fig, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"    {stem}.{{pdf,svg,png}}", flush=True)


def build_predictions() -> tuple[dict, dict, float]:
    """Rebuild the frozen folder-01 models and return per-partition predictions."""
    parts = build_splits()
    ph = physics(parts)
    tr = parts["train"]
    y_tr, X_tr = tr["power"].to_numpy(), tr[FEATS].to_numpy()
    k = fit_scale(y_tr, ph["train"])
    res_tr = y_tr - k * ph["train"]
    fi, ei = parts["_fit_idx"], parts["_es_idx"]
    bbm = fit_learner(BBM_KEY, X_tr, y_tr, fi, ei)[0]
    resid = fit_learner(RESID_KEY, X_tr, res_tr, fi, ei)[0]
    print(f"[F34] k = {k:.6f}; residual = Huber; BBM = Huber on y "
          f"(folder-01 definition)", flush=True)

    pred, meas = {}, {}
    for nm in ("dev_in", "dev_out"):
        d = parts[nm].reset_index(drop=True)
        X = d[FEATS].to_numpy()
        base = k * ph[nm]
        r = resid.predict(X)
        meas[nm] = d["power"].to_numpy()
        pred[nm] = {"WBM": base, "BBM": bbm.predict(X),
                    "S-GBM": base + r, "C-GBM": base + LAMBDA_STAR * r}
    return meas, pred, float(k)


def verify(meas: dict, pred: dict, k: float) -> pd.DataFrame:
    """Recompute every metric and assert reproduction of the Section 5.3 values."""
    ref = pd.read_csv(F01 / "final_dev_in_dev_out.csv")
    ref["architecture"] = ref["architecture"].str.replace(
        r"^C-GBM.*", "C-GBM", regex=True)
    rows, fails = [], []
    for nm in ("dev_in", "dev_out"):
        for a in ARCHS:
            m = metrics7(meas[nm], pred[nm][a])
            rr = ref[(ref.partition == nm) & (ref.architecture == a)].iloc[0]
            exp = EXPECTED[(nm, a)]
            d_ref = abs(m["MAPE"] - float(rr.MAPE))
            d_exp = abs(round(m["MAPE"], 3) - exp)
            ok = d_ref < 1e-4 and d_exp <= TOL_PP
            if not ok:
                fails.append(f"{nm}/{a}: recomputed {m['MAPE']:.4f}, folder-01 "
                             f"{rr.MAPE:.4f}, Section 5.3 {exp:.3f}")
            slope, intercept = np.polyfit(meas[nm], pred[nm][a], 1)
            rows.append({"partition": nm, "architecture": a, "n": len(meas[nm]),
                         **{f"{key}_recomputed": v for key, v in m.items()},
                         "MAPE_folder24": float(rr.MAPE),
                         "MAPE_section53": exp,
                         "abs_diff_vs_folder24": d_ref,
                         "abs_diff_vs_section53": d_exp,
                         "reproduces": ok,
                         "regression_slope": float(slope),
                         "regression_intercept_kw": float(intercept)})
    ver = pd.DataFrame(rows)

    # delta MAPE check
    for a in ARCHS:
        din = float(ver[(ver.partition == "dev_in") & (ver.architecture == a)]
                    ["MAPE_recomputed"].iloc[0])
        dout = float(ver[(ver.partition == "dev_out") & (ver.architecture == a)]
                     ["MAPE_recomputed"].iloc[0])
        if abs(round(dout - din, 3) - EXPECTED_DELTA[a]) > TOL_PP:
            fails.append(f"delta {a}: recomputed {dout - din:+.3f} pp, "
                         f"Section 5.3 {EXPECTED_DELTA[a]:+.3f} pp")
    if abs(k - K_REFERENCE) > 1e-6:
        fails.append(f"k: recomputed {k:.6f}, frozen {K_REFERENCE:.6f}")

    if fails:
        print("\n[F34] STOP - the frozen predictions do NOT reproduce the "
              "Section 5.3 values:", flush=True)
        for f in fails:
            print(f"       {f}", flush=True)
        raise SystemExit(2)
    print(f"[F34] verified: 8/8 MAPE values, 4/4 delta values and k reproduce "
          f"within {TOL_PP} pp", flush=True)
    return ver


def figure3(ver: pd.DataFrame) -> pd.DataFrame:
    dat = []
    for nm in ("dev_in", "dev_out"):
        for a in ARCHS:
            r = ver[(ver.partition == nm) & (ver.architecture == a)].iloc[0]
            dat.append({"partition": nm, "architecture": a,
                        "MAPE_pct": r.MAPE_recomputed, "sMAPE_pct": r.sMAPE_recomputed,
                        "MAE_kw": r.MAE_recomputed, "RMSE_kw": r.RMSE_recomputed,
                        "R2": r.R2_recomputed, "n": int(r.n)})
    d = pd.DataFrame(dat)

    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.75))
    x, w = np.arange(len(ARCHS)), 0.36
    for j, nm in enumerate(("dev_in", "dev_out")):
        for panel, col in ((ax[0], "MAPE_pct"), (ax[1], "R2")):
            v = [float(d[(d.partition == nm) & (d.architecture == a)][col].iloc[0])
                 for a in ARCHS]
            panel.bar(x + (j - 0.5) * w, v, w,
                      color=[COL[a] for a in ARCHS], edgecolor="black", lw=0.5,
                      hatch=PART_HATCH[nm],
                      label=PART_LABEL[nm] if panel is ax[0] else None)
    # delta MAPE annotation, stated as a description of the two bars it sits above
    for i, a in enumerate(ARCHS):
        din = float(d[(d.partition == "dev_in") & (d.architecture == a)]["MAPE_pct"].iloc[0])
        dout = float(d[(d.partition == "dev_out") & (d.architecture == a)]["MAPE_pct"].iloc[0])
        ax[0].text(i, max(din, dout) + 0.7, f"{dout - din:+.2f}", ha="center",
                   fontsize=7.2, color="black")
    ax[0].set_ylabel("MAPE (%)")
    ax[0].set_ylim(0, 26)
    ax[1].set_ylabel(r"$R^2$")
    ax[1].set_ylim(0.78, 1.0)
    for a_, t in zip(ax, ("(a)", "(b)")):
        a_.set_xticks(x)
        a_.set_xticklabels(ARCHS, fontsize=8)
        a_.grid(axis="y", color="0.9", lw=0.5)
        a_.set_axisbelow(True)
        a_.text(-0.14, 1.05, t, transform=a_.transAxes, fontsize=9,
                fontweight="bold")
    hand = [plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=0.5,
                          hatch=PART_HATCH[nm]) for nm in ("dev_in", "dev_out")]
    ax[0].legend(hand, [PART_LABEL["dev_in"], PART_LABEL["dev_out"]], frameon=False, ncol=1,
                 loc="upper right", handlelength=1.6)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_regime_performance_representative")
    return d


def figure4(meas: dict, pred: dict, ver: pd.DataFrame) -> pd.DataFrame:
    panels = [("dev_in", "WBM"), ("dev_in", "C-GBM"),
              ("dev_out", "WBM"), ("dev_out", "C-GBM")]
    vals = np.concatenate([meas[p] for p, _ in panels]
                          + [pred[p][a] for p, a in panels])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    pad = 0.02 * (hi - lo)
    lim = (lo - pad, hi + pad)
    counts = [np.histogram2d(meas[p], pred[p][a], bins=40,
                             range=[lim, lim])[0].max() for p, a in panels]
    vmax = float(max(counts))

    fig, axes = plt.subplots(2, 2, figsize=(W2, 6.4), sharex=True, sharey=True)
    hb = None
    for axp, (p, a) in zip(axes.ravel(), panels):
        hb = axp.hexbin(meas[p], pred[p][a], gridsize=45, extent=(*lim, *lim),
                        cmap="viridis", norm=LogNorm(vmin=1, vmax=vmax),
                        mincnt=1, linewidths=0.0)
        axp.plot(lim, lim, color="crimson", lw=1.0, ls="--", zorder=4)
        r = ver[(ver.partition == p) & (ver.architecture == a)].iloc[0]
        axp.text(0.035, 0.965,
                 f"{a} · {PART_LABEL[p]}\n"
                 f"MAPE {r.MAPE_recomputed:.3f} %\n"
                 f"$R^2$ {r.R2_recomputed:.4f}\n"
                 f"slope {r.regression_slope:.4f}\n"
                 f"intercept {r.regression_intercept_kw:+.0f} kW",
                 transform=axp.transAxes, va="top", ha="left", fontsize=7.6,
                 bbox=dict(fc="white", ec="0.5", lw=0.5, alpha=0.88, pad=2.6))
        axp.set_xlim(lim)
        axp.set_ylim(lim)
        axp.set_aspect("equal", adjustable="box")
        axp.grid(color="0.92", lw=0.4)
        axp.set_axisbelow(True)
    for axp in axes[1, :]:
        axp.set_xlabel("Measured shaft power (kW)")
    for axp in axes[:, 0]:
        axp.set_ylabel("Predicted shaft power (kW)")
    fig.tight_layout(w_pad=1.4, h_pad=1.4)
    cb = fig.colorbar(hb, ax=axes.ravel().tolist(), fraction=0.032, pad=0.02)
    cb.set_label("Records per hexagon (log scale)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.solids.set_rasterized(False)
    save(fig, "fig_pred_vs_actual_shrinkage")

    return pd.DataFrame({
        "partition": np.concatenate([[p] * len(meas[p]) for p, _ in panels]),
        "architecture": np.concatenate([[a] * len(meas[p]) for p, a in panels]),
        "measured_power_kw": np.concatenate([meas[p] for p, _ in panels]),
        "predicted_power_kw": np.concatenate([pred[p][a] for p, a in panels]),
    })


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meas, pred, k = build_predictions()
    ver = verify(meas, pred, k)
    write_csv(ver, OUT / "verification_recomputed_metrics.csv")

    print("[F34] figures", flush=True)
    d3 = figure3(ver)
    write_csv(d3, OUT / "fig3_plotting_data.csv")
    d4 = figure4(meas, pred, ver)
    write_csv(d4, OUT / "fig4_plotting_data.csv")

    (OUT / "run_metadata.json").write_text(json.dumps({
        "purpose": "revised manuscript Figures 3 and 4, Section 5.3",
        "source_configuration": "results/revision_final/01_full_data_602020/",
        "protocol": "full-data chronological 60/20/20",
        "physics_k_recomputed": k, "physics_k_frozen": K_REFERENCE,
        "residual_learner": "Huber", "lambda_star": LAMBDA_STAR,
        "bbm_definition": "Huber fitted on measured power (folder-01 definition)",
        "bbm_not_used": "ExtraTrees BBM of folder 02 (later alternative experiment)",
        "partitions": {"dev_in": int(len(meas["dev_in"])),
                       "dev_out": int(len(meas["dev_out"]))},
        "refit_on_dev": False, "lambda_reselected_on_dev": False,
        "scope": "predictive diagnostics only; no physical-consistency, D_phys, "
                 "speed-exponent or operating-regime quantity is computed",
        "verification": "8/8 MAPE, 4/4 delta MAPE and k asserted against "
                        "folder 01 and the Section 5.3 values before saving",
        "seed": SEED, "git_commit": _git()}, indent=2), encoding="utf-8")
    print("[F34] done", flush=True)
    return 0


def _guarded() -> int:
    import joblib
    with joblib.parallel_backend("threading", n_jobs=1):
        return main()


if __name__ == "__main__":
    sys.exit(_guarded())
