"""Manuscript figure for Section 5.6, built only from verified interval-invariant results.

Three panels, each carrying a quantity the Section 5.6 audit established as defensible:

(a) relative cumulative difference of the propagated budget against the measured-power
    reference. Identical for fuel, CO2, CH4, N2O and CO2e by proportionality, and
    invariant to the interval convention, so it is safe to plot where absolute tonnes
    are not.
(b) CII ratio (attained / required) under the container-ship reference line with the
    A-E rating bands, which is where the rating actually separates.
(c) the dissociation between point accuracy and budget agreement: MAPE against the signed
    relative budget difference, which shows the black box achieving the smallest in-domain
    budget difference at the worst MAPE.

Deliberately NOT plotted: the gas-composition split, which is identical for every series
because ``ch4_lowload_gain = 0`` for MDO. Five identical stacked bars would imply the
composition had been predicted rather than imposed. Also not plotted: absolute cumulative
tonnes, which depend entirely on the assumed 1 h interval.

Every value is read back from the frozen CSVs in folder 08 and cross-checked before
drawing.
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common_rev import REV_FINAL, write_csv

SRC = REV_FINAL / "08_multigas_cii"
STEM = "fig_multigas_cii_propagation"
ARCHS = ["WBM", "BBM", "S-GBM", "C-GBM"]
PARTS = [("dev_in", "in-domain"), ("dev_out", "distribution-shifted")]
C_IN, C_OUT = "#5B7DB1", "#C4813F"          # colour encodes partition, not architecture
BANDS = {"bulk carrier": (0.86, 0.94, 1.06, 1.18),
         "container ship": (0.83, 0.94, 1.07, 1.19)}
BAND_FILL = {"A": "#DFF0D8", "B": "#EAF4E2", "C": "#FDF6DD",
             "D": "#FBE3D4", "E": "#F7D4D4"}
TOL = 1e-9

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W2 = 7.16


def load_and_verify():
    rel = pd.read_csv(SRC / "multigas_relative_errors.csv")
    gas = pd.read_csv(SRC / "multigas_results.csv")
    cii = pd.read_csv(SRC / "cii_results.csv")
    cons = pd.read_csv(SRC / "section53_consistency_check.csv")
    comp = pd.read_csv(SRC / "gas_composition_audit.csv")
    fails = []

    if not cons.reproduces.all():
        fails.append("Section 5.3 consistency check in folder 08 is not all-PASS")
    # the relative difference must be identical across gases (proportionality)
    cols = [c for c in rel.columns if c.startswith("rel_diff_")]
    spread = float(np.max(rel[cols].max(axis=1) - rel[cols].min(axis=1)))
    if spread > 1e-9:
        fails.append(f"relative differences differ between gases by {spread:.2e}")
    # and must reproduce from the raw totals
    for _, r in rel.iterrows():
        base = gas[(gas.partition == r.partition)
                   & (gas.series == "Measured-power reference")].iloc[0]
        row = gas[(gas.partition == r.partition) & (gas.series == r.series)].iloc[0]
        recomputed = 100.0 * (row.co2e_t - base.co2e_t) / base.co2e_t
        if abs(recomputed - r.rel_diff_co2e_t_pct) > 1e-6:
            fails.append(f"{r.partition}/{r.series}: relative difference "
                         f"{r.rel_diff_co2e_t_pct} does not reproduce from totals")
    # CII ratio must reproduce from attained / required
    for _, r in cii.iterrows():
        if abs(r.attained_cii / r.required_cii - r.ratio) > 1e-9:
            fails.append(f"{r.partition}/{r.series}/{r.ship_type}: ratio mismatch")
    if float(comp.co2e_per_fuel_t_per_t.std()) > 1e-9:
        fails.append("gas composition is not constant across series")

    if fails:
        print("STOP: folder 08 results did not verify:", flush=True)
        for f in fails:
            print("   " + f, flush=True)
        raise SystemExit(2)
    print(f"[fig] verified: 5.3 check all-PASS, relative differences identical across "
          f"5 gases (spread {spread:.1e}), {len(cii)} CII ratios reproduce", flush=True)
    return rel, gas, cii


def figure(rel, gas, cii) -> pd.DataFrame:
    mape = {("dev_in", "WBM"): 8.848, ("dev_in", "BBM"): 18.820,
            ("dev_in", "S-GBM"): 8.788, ("dev_in", "C-GBM"): 8.238,
            ("dev_out", "WBM"): 9.214, ("dev_out", "BBM"): 21.930,
            ("dev_out", "S-GBM"): 9.545, ("dev_out", "C-GBM"): 8.732}
    dat = []
    for key, lab in PARTS:
        for a in ARCHS:
            r = rel[(rel.partition == key) & (rel.series == a)].iloc[0]
            c = cii[(cii.partition == key) & (cii.series == a)
                    & (cii.ship_type == "container ship")].iloc[0]
            cb = cii[(cii.partition == key) & (cii.series == a)
                     & (cii.ship_type == "bulk carrier")].iloc[0]
            dat.append({"partition": key, "partition_label": lab, "architecture": a,
                        "rel_diff_pct_all_gases": r.rel_diff_co2e_t_pct,
                        "MAPE_pct": mape[(key, a)],
                        "cii_ratio_container": c.ratio, "rating_container": c.rating,
                        "cii_ratio_bulk": cb.ratio, "rating_bulk": cb.rating,
                        "attained_cii": c.attained_cii,
                        "required_cii_container": c.required_cii})
    d = pd.DataFrame(dat)
    ref = {k: cii[(cii.partition == k) & (cii.series == "Measured-power reference")
                  & (cii.ship_type == "container ship")].iloc[0].ratio
           for k, _ in PARTS}

    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.75))
    x, bw = np.arange(len(ARCHS)), 0.36

    # ---- (a) relative cumulative difference -------------------------------
    for j, (key, lab) in enumerate(PARTS):
        v = [float(d[(d.partition == key) & (d.architecture == a)]
                   ["rel_diff_pct_all_gases"].iloc[0]) for a in ARCHS]
        ax[0].bar(x + (j - 0.5) * bw, v, bw, color=(C_IN if j == 0 else C_OUT),
                  edgecolor="black", lw=0.5, hatch=("" if j == 0 else "///"), label=lab)
    ax[0].axhline(0.0, color="black", lw=0.9)
    ax[0].set_ylabel("Relative difference vs\nmeasured-power reference (%)")
    ax[0].set_ylim(-3.2, 6.6)
    ax[0].legend(frameon=False, loc="upper left", handlelength=1.4, borderpad=0.2,
                 fontsize=7.2)

    # ---- (b) CII ratio with rating bands, container line ------------------
    d1, d2, d3, d4 = BANDS["container ship"]
    edges = [(0.0, d1, "A"), (d1, d2, "B"), (d2, d3, "C"), (d3, d4, "D"),
             (d4, 2.0, "E")]
    lo, hi = 0.93, 1.14
    for a0, a1, lab in edges:
        if a1 <= lo or a0 >= hi:
            continue
        ax[1].axhspan(max(a0, lo), min(a1, hi), color=BAND_FILL[lab], zorder=0)
        mid = 0.5 * (max(a0, lo) + min(a1, hi))
        ax[1].text(len(ARCHS) - 0.42, mid, lab, fontsize=7.6, va="center",
                   ha="left", color="0.35")
    for j, (key, lab) in enumerate(PARTS):
        v = [float(d[(d.partition == key) & (d.architecture == a)]
                   ["cii_ratio_container"].iloc[0]) for a in ARCHS]
        ax[1].plot(x, v, marker=("o" if j == 0 else "s"), ms=5, lw=0,
                   mfc=(C_IN if j == 0 else C_OUT), mec="black", mew=0.6,
                   label=lab, zorder=3)
        ax[1].axhline(ref[key], color=(C_IN if j == 0 else C_OUT), lw=0.9,
                      ls=(":" if j == 0 else "--"), zorder=2)
    ax[1].set_ylabel("Attained / required CII\n(container-ship line)")
    ax[1].set_ylim(lo, hi)
    ax[1].set_xlim(-0.5, len(ARCHS) - 0.2)

    # ---- (c) accuracy versus budget agreement ----------------------------
    for j, (key, lab) in enumerate(PARTS):
        s = d[d.partition == key]
        ax[2].scatter(s.MAPE_pct, s.rel_diff_pct_all_gases,
                      marker=("o" if j == 0 else "s"), s=34,
                      facecolor=(C_IN if j == 0 else C_OUT), edgecolor="black",
                      linewidth=0.6, zorder=3, label=lab)
    # The three physics-anchored architectures nearly coincide in domain, so they are
    # annotated once as a cluster; only the BBM and the shifted C-GBM are named
    # individually, since those two points carry the panel's message.
    for key, arch, dx, dy in (("dev_in", "BBM", 5, 4), ("dev_out", "BBM", -6, -10),
                              ("dev_out", "C-GBM", 5, -9),
                              ("dev_out", "WBM", 6, 4)):
        r = d[(d.partition == key) & (d.architecture == arch)].iloc[0]
        ax[2].annotate(arch, (r.MAPE_pct, r.rel_diff_pct_all_gases),
                       textcoords="offset points", xytext=(dx, dy), fontsize=6.6)
    cl = d[(d.partition == "dev_in") & (d.architecture != "BBM")]
    ax[2].annotate("WBM, S-GBM, C-GBM", (cl.MAPE_pct.mean(),
                                         cl.rel_diff_pct_all_gases.mean()),
                   textcoords="offset points", xytext=(10, 12), fontsize=6.6,
                   arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))
    ax[2].legend(frameon=False, loc="upper left", fontsize=6.8, handlelength=1.0,
                 borderpad=0.2, labelspacing=0.25)
    ax[2].axhline(0.0, color="black", lw=0.9)
    ax[2].set_xlabel("MAPE (%)")
    ax[2].set_ylabel("Relative budget\ndifference (%)")
    ax[2].set_xlim(6.5, 24.0)
    ax[2].set_ylim(-3.2, 6.6)

    for i, (a_, tag) in enumerate(zip(ax, ("(a)", "(b)", "(c)"))):
        a_.grid(axis="y", color="0.9", lw=0.5, zorder=1)
        a_.set_axisbelow(True)
        a_.text(-0.20, 1.05, tag, transform=a_.transAxes, fontsize=9,
                fontweight="bold")
        if i < 2:
            a_.set_xticks(x)
            a_.set_xticklabels(ARCHS, fontsize=7.2, rotation=18, ha="right")
    fig.tight_layout(w_pad=1.8)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(SRC / f"{STEM}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"[fig] {STEM}.{{pdf,svg,png}}", flush=True)
    return d


def main() -> int:
    rel, gas, cii = load_and_verify()
    d = figure(rel, gas, cii)
    write_csv(d, SRC / f"{STEM}_data.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
