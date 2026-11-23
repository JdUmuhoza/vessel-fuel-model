"""Single manuscript figure summarising the FuelCast cross-vessel results.

Reads the Part A table produced by s06 (cube-law prior, corrected wind-aware feature
set, wind-quantile shift) and plots it. No model is fitted and no metric is recomputed
from predictions here: every plotted value is read back from the frozen CSVs and
cross-checked against the per-metric file before anything is drawn.

Encoding choices are deliberately architecture-neutral: colour carries the evaluation
partition, never the architecture, so no single architecture is visually privileged.
The selected lambda is printed as a panel note rather than plotted on a second axis.
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common_rev import REV_FINAL, write_csv

SRC = REV_FINAL / "06_fuelcast_wind_and_wbm"
STEM = "fig_fuelcast_cross_vessel_performance"
VESSELS = ["Poseidon", "Triton", "Ceto"]
ARCH_ORDER = ["WBM-lite (cube law)", "BBM (Huber on target)",
              "S-GBM (Huber residual)", "C-GBM (Huber residual)"]
ARCH_LABEL = {"WBM-lite (cube law)": "WBM-lite", "BBM (Huber on target)": "BBM",
              "S-GBM (Huber residual)": "S-GBM", "C-GBM (Huber residual)": "C-GBM"}
# colour = partition, not architecture
C_IN, C_OUT = "#5B7DB1", "#C4813F"
TOL = 1e-9

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W2 = 7.16     # elsarticle double-column width, inches


def load_and_verify() -> pd.DataFrame:
    tab = pd.read_csv(SRC / "partA_manuscript_table_mape.csv")
    full = pd.read_csv(SRC / "partA_all_metrics.csv")
    if len(tab) != len(VESSELS) * len(ARCH_ORDER):
        raise SystemExit(f"STOP: expected {len(VESSELS) * len(ARCH_ORDER)} table rows, "
                         f"found {len(tab)}")
    rows, fails = [], []
    for ves in VESSELS:
        for arch in ARCH_ORDER:
            t = tab[(tab.Vessel == ves) & (tab.Architecture == arch)]
            f = full[(full.vessel_short == ves) & (full.architecture == arch)]
            if len(t) != 1 or len(f) != 1:
                fails.append(f"{ves}/{arch}: {len(t)} table rows, {len(f)} metric rows")
                continue
            t, f = t.iloc[0], f.iloc[0]
            d_in = abs(float(t.MAPE_in_pct) - float(f.MAPE_in))
            d_out = abs(float(t.MAPE_out_pct) - float(f.MAPE_out))
            d_delta = abs(float(t.delta_MAPE_pp)
                          - (float(f.MAPE_out) - float(f.MAPE_in)))
            if max(d_in, d_out, d_delta) > TOL:
                fails.append(f"{ves}/{arch}: table vs metrics differ by "
                             f"{max(d_in, d_out, d_delta):.3e}")
            lam = float(t.lambda_star) if not pd.isna(t.lambda_star) else np.nan
            rows.append({"vessel": ves, "architecture": arch,
                         "architecture_label": ARCH_LABEL[arch],
                         "MAPE_in_pct": float(f.MAPE_in),
                         "MAPE_out_pct": float(f.MAPE_out),
                         "delta_MAPE_pp": float(f.MAPE_out) - float(f.MAPE_in),
                         "lambda_star": lam,
                         "n_dev_in": int(f.n_dev_in), "n_dev_out": int(f.n_dev_out),
                         "source_table": "partA_manuscript_table_mape.csv",
                         "source_metrics": "partA_all_metrics.csv"})
    if fails:
        print("STOP: plotted values do not reproduce the final FuelCast results:",
              flush=True)
        for x in fails:
            print(f"   {x}", flush=True)
        raise SystemExit(2)
    d = pd.DataFrame(rows)
    lam = {v: float(d[(d.vessel == v) & d.lambda_star.notna()].lambda_star.iloc[0])
           for v in VESSELS}
    print(f"[fig] verified {len(d)} MAPE pairs against both source CSVs "
          f"(max deviation < {TOL:g})", flush=True)
    print(f"[fig] lambda* = {lam}", flush=True)
    return d


def figure(d: pd.DataFrame) -> None:
    ymax = float(max(d.MAPE_in_pct.max(), d.MAPE_out_pct.max()))
    ytop = 5.0 * np.ceil((ymax * 1.08) / 5.0)      # common scale, headroom for labels
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.9), sharey=True)
    x, w = np.arange(len(ARCH_ORDER)), 0.36

    for ax, ves, tag in zip(axes, VESSELS, ("(a)", "(b)", "(c)")):
        s = d[d.vessel == ves].set_index("architecture").loc[ARCH_ORDER]
        vin, vout = s.MAPE_in_pct.to_numpy(), s.MAPE_out_pct.to_numpy()
        ax.bar(x - w / 2, vin, w, color=C_IN, edgecolor="black", lw=0.5,
               label="in-domain")
        ax.bar(x + w / 2, vout, w, color=C_OUT, edgecolor="black", lw=0.5,
               hatch="///", label="shifted")
        for xi, v in zip(x - w / 2, vin):
            ax.text(xi, v + ytop * 0.014, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.4, rotation=90)
        for xi, v in zip(x + w / 2, vout):
            ax.text(xi, v + ytop * 0.014, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.4, rotation=90)
        lam = float(s.lambda_star.dropna().iloc[0])
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABEL[a] for a in ARCH_ORDER], fontsize=7.4,
                           rotation=18, ha="right")
        ax.set_ylim(0, ytop)
        ax.grid(axis="y", color="0.9", lw=0.5)
        ax.set_axisbelow(True)
        # lambda* lives in the panel header, so it can never overlap a bar label
        # and is never encoded on an axis
        ax.text(0.0, 1.045, f"{tag} {ves}", transform=ax.transAxes, fontsize=8.6,
                fontweight="bold", ha="left")
        ax.text(1.0, 1.05, rf"C-GBM $\lambda^{{*}}\!=\!{lam:.2f}$",
                transform=ax.transAxes, fontsize=7.6, ha="right")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.90),
                   handlelength=1.5, borderpad=0.2)
    fig.tight_layout(w_pad=1.1)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(SRC / f"{STEM}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"[fig] {STEM}.{{pdf,svg,png}}  common y-axis 0-{ytop:g} %", flush=True)


def main() -> int:
    d = load_and_verify()
    write_csv(d, SRC / f"{STEM}_data.csv")
    figure(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
