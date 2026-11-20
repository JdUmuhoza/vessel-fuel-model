"""Redraw fig2_candidate_602020_fulldata with the endpoint labels placed inside the axes.

The curve, the data and the selected lambda are unchanged: the sweep is read verbatim from
``results/revision_final/24_full_data_602020/lambda_sweep_validation.csv`` and the optimum
is recomputed from it, then asserted against lambda* = 0.70. Only the annotation geometry
changes.

Two placement faults in the original are corrected:

* ``WBM`` was annotated 6 pt *above* the point at lambda = 0, which is the maximum of the
  curve. Matplotlib does not expand the y-limits to accommodate annotation text, so the
  label was drawn outside the axes.
* ``S-GBM (Huber)`` was a two-line label offset 6 pt *below* the point at lambda = 1, which
  put it on top of the rising branch of the curve.

Both are now placed in verified-empty regions inside the axes, with explicit y-limit
headroom and the legend pinned rather than left to ``loc="best"`` so the layout is
deterministic. Written to a new folder so folder 24, whose report documents the original
figure, is left untouched.
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common_rev import REV_FINAL, write_csv

SRC = REV_FINAL / "24_full_data_602020"
OUT = REV_FINAL / "36_fig2_lambda_relabelled"
STEM = "fig2_candidate_602020_fulldata"
PRIMARY = "MAPE"
LAMBDA_STAR = 0.70
BACKEND_DISPLAY = "Huber"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    curve = pd.read_csv(SRC / "lambda_sweep_validation.csv")
    lam = curve["lambda"].to_numpy()
    mu = curve[PRIMARY].to_numpy()

    i_star = int(np.argmin(mu))
    if abs(float(lam[i_star]) - LAMBDA_STAR) > 1e-9:
        raise SystemExit(f"STOP: sweep minimum is at lambda = {lam[i_star]}, "
                         f"not {LAMBDA_STAR}")
    if len(lam) != 101:
        raise SystemExit(f"STOP: expected a 101-point grid, found {len(lam)}")
    print(f"[fig] sweep verified: {len(lam)} points, minimum {mu[i_star]:.4f} % at "
          f"lambda = {lam[i_star]:.2f}; endpoints WBM {mu[0]:.4f} %, "
          f"S-GBM {mu[-1]:.4f} %", flush=True)

    plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.bbox": "tight",
                         "axes.grid": True, "grid.alpha": 0.22,
                         "axes.axisbelow": True})
    fig, ax = plt.subplots(figsize=(3.46, 2.55))
    ax.plot(lam, mu, "-", color="#1f78b4", lw=1.6)

    # explicit headroom, so annotation text placed near the extremes stays inside
    span = float(mu.max() - mu.min())
    ax.set_ylim(mu.min() - 0.10 * span, mu.max() + 0.16 * span)
    ax.set_xlim(-0.02, 1.02)

    # The two endpoints are identified in the legend rather than by floating text, so
    # no label can land outside the axes or on the curve. Colour AND marker shape both
    # differ, so the three points stay distinguishable in greyscale print.
    for x, lbl, marker, colour, msize in (
            (0.0, rf"WBM ($\lambda = 0$)", "o", "#1b9e46", 5.5),
            (1.0, rf"S-GBM, {BACKEND_DISPLAY} ($\lambda = 1$)", "s", "#333333", 5.0)):
        i = int(np.argmin(np.abs(lam - x)))
        ax.plot([x], [mu[i]], marker, color=colour, ms=msize, mec="white", mew=0.7,
                zorder=5, label=lbl, lw=0)

    ax.plot([lam[i_star]], [mu[i_star]], "D", color="#b2182b", ms=6.5, mec="white",
            mew=0.7, zorder=6, lw=0,
            label=rf"selected $\lambda^{{\ast}} = {lam[i_star]:.2f}$")
    ax.set_xticks(np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1))
    ax.set_xlabel(r"Residual-shrinkage coefficient $\lambda$", fontsize=9)
    ax.set_ylabel(f"Validation {PRIMARY} (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    # pinned, not "best": deterministic placement in the empty upper-right region.
    # A light frame keeps the three entries legible over the gridlines.
    leg = ax.legend(fontsize=7.2, loc="upper right", handlelength=1.0,
                    borderaxespad=0.5, labelspacing=0.35, handletextpad=0.5,
                    frameon=True, framealpha=0.92, borderpad=0.45)
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_linewidth(0.6)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", format=ext)
    plt.close(fig)

    write_csv(curve, OUT / f"{STEM}_data.csv")
    print(f"[fig] {STEM}.{{pdf,svg,png}} written to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
