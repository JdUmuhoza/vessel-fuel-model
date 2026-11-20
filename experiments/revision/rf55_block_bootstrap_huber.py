"""Paired moving-block bootstrap of MAPE differences for the manuscript's frozen models.

Recomputes table T15 (FINAL_MANUSCRIPT_RESULTS) for the configuration the manuscript
reports: residual learner Huber, lambda* = 0.70, and the matched-backend BBM (Huber on
the measured target, folder-24 definition). T15 itself used an ExtraTrees BBM, so only
its WBM and S-GBM rows apply to the manuscript; this script recomputes all six rows and
asserts that the WBM and S-GBM rows reproduce T15 exactly (same models, same seed, same
resampler), which verifies the method before the new BBM rows are used.

The resampler and Holm correction are imported unchanged from ``rf36_final_core``:
blocks of ``BLOCK`` consecutive records (the held-out files are in time order; records
are not uniformly spaced, so a block is a record count, not a duration), ``B_BOOT``
resamples, two-sided bootstrap p-values, Holm correction across the comparisons of one
partition.

Run from the repository root::

    PYTHONPATH=. python -m experiments.revision.rf55_block_bootstrap_huber
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common_rev import FEATS, REV_FINAL, SEED, write_csv, write_text
from .rf32_full_data_602020 import ES_FRACTION, fit_learner, metrics7
from .rf36_final_core import B_BOOT, BLOCK, block_idx, holm
from .rf54_architecture_friedman_nemenyi import LAMBDA_STAR, LEARNER, load_development
from benchmark_measured.common import fit_scale  # noqa: E402

OUT = REV_FINAL / "39_block_bootstrap_huber"
T15 = REV_FINAL / "FINAL_MANUSCRIPT_RESULTS" / "T15_statistical_tests.csv"
REFS = ("WBM", "BBM", "S-GBM")
TOL = 1e-9                       # T15 reproduction tolerance (identical computation)


def frozen_predictions(D: dict) -> dict:
    """Folder-24 frozen models: k on the training block, Huber on its fitting sub-block."""
    dev, ph, i60 = D["dev"], D["ph_dev"], D["i60"]
    y_tr, X_tr, ph_tr = dev["power"].to_numpy()[:i60], dev[FEATS].to_numpy()[:i60], ph[:i60]
    k = fit_scale(y_tr, ph_tr)
    n_fit = int((1.0 - ES_FRACTION) * i60)
    fi, ei = np.arange(n_fit), np.arange(n_fit, i60)
    resid = fit_learner(LEARNER, X_tr, y_tr - k * ph_tr, fi, ei)[0]
    bbm = fit_learner(LEARNER, X_tr, y_tr, fi, ei)[0]
    out = {}
    for nm in ("dev_in", "dev_out"):
        d = D[nm]
        X, base = d[FEATS].to_numpy(), k * D[f"ph_{nm}"]
        r = resid.predict(X)
        out[nm] = (d["power"].to_numpy(),
                   {"WBM": base, "BBM": bbm.predict(X), "S-GBM": base + r,
                    "C-GBM": base + LAMBDA_STAR * r})
    return out


def bootstrap(pred: dict) -> pd.DataFrame:
    """The T15 loop of rf36, applied to the manuscript's frozen models."""
    rows = []
    for nm in ("dev_in", "dev_out"):
        y, pr = pred[nm]
        n = len(y)
        raw, recs = [], []
        for ref in REFS:
            point = metrics7(y, pr["C-GBM"])["MAPE"] - metrics7(y, pr[ref])["MAPE"]
            rng = np.random.default_rng(SEED)
            dd = np.empty(B_BOOT)
            for b in range(B_BOOT):
                i = block_idx(n, rng)
                dd[b] = metrics7(y[i], pr["C-GBM"][i])["MAPE"] - metrics7(y[i], pr[ref][i])["MAPE"]
            lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
            pv = float(np.clip(2 * min(float(np.mean(dd <= 0)), float(np.mean(dd >= 0))),
                               1 / B_BOOT, 1))
            raw.append(pv)
            recs.append({"partition": nm, "comparison": f"C-GBM minus {ref}", "reference": ref,
                         "MAPE_cgbm": metrics7(y, pr["C-GBM"])["MAPE"],
                         "MAPE_reference": metrics7(y, pr[ref])["MAPE"],
                         "effect_size_pp": point, "ci95_lo": lo, "ci95_hi": hi, "p_raw": pv,
                         "block_length_records": BLOCK, "n_resamples": B_BOOT, "n_records": n})
        for r_, p_ in zip(recs, holm(raw)):
            r_["p_holm"] = p_
            r_["statistically_resolved"] = bool(p_ < 0.05 and (r_["ci95_lo"] > 0 or r_["ci95_hi"] < 0))
            rows.append(r_)
    return pd.DataFrame(rows)


def check_against_t15(res: pd.DataFrame) -> pd.DataFrame:
    """WBM and S-GBM rows must equal T15; the BBM rows are expected to differ."""
    t15 = pd.read_csv(T15)
    chk = []
    for r in res.itertuples():
        t = t15[(t15.partition == r.partition) & (t15.reference == r.reference)].iloc[0]
        same = all(abs(getattr(r, c) - float(t[c])) < TOL
                   for c in ("effect_size_pp", "ci95_lo", "ci95_hi", "p_raw"))
        chk.append({"partition": r.partition, "reference": r.reference,
                    "identical_to_T15": same, "expected_identical": r.reference != "BBM",
                    "T15_effect_pp": float(t.effect_size_pp), "new_effect_pp": r.effect_size_pp})
    c = pd.DataFrame(chk)
    bad = c[c.expected_identical & ~c.identical_to_T15]
    if len(bad):
        raise AssertionError("WBM/S-GBM rows do not reproduce T15:\n" + bad.to_string())
    return c


def latex_table(res: pd.DataFrame) -> str:
    """Manuscript table, numbers read from the bootstrap result."""
    names = {"dev_in": "In-domain", "dev_out": "Shifted"}
    fmt_p = lambda p: ("$<$0.001" if p < 0.001 else f"{p:.3f}")
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Paired moving-block bootstrap of the MAPE difference between C-GBM and each architecture.}",
         r"\label{tab:block-bootstrap-mape}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
         r"\renewcommand{\arraystretch}{1.06}", r"\begin{tabular}{llrcr}", r"\toprule",
         r"\rowcolor{gray!10}",
         r"\textbf{Set} & \textbf{Comparison} & \textbf{$\Delta$ (pp)} & \textbf{95\% CI} & \textbf{$p_{\mathrm{Holm}}$}\\",
         r"\midrule"]
    for i, nm in enumerate(("dev_in", "dev_out")):
        g = res[res.partition == nm]
        for j, r in enumerate(g.itertuples()):
            first = f"\\multirow{{3}}{{*}}{{{names[nm]}}}" if j == 0 else ""
            p = fmt_p(r.p_holm)
            p = f"\\textbf{{{p}}}" if r.statistically_resolved else p
            L.append(f"{first} & C-GBM $-$ {r.reference} & {r.effect_size_pp:+.3f} & "
                     f"[{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] & {p}\\\\")
        if i == 0:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = load_development()
    for nm in ("dev_in", "dev_out"):
        if not np.all(np.diff(D[nm]["time_id"].to_numpy()) >= 0):
            raise AssertionError(f"{nm} is not in time order; consecutive-record blocks invalid")
    res = bootstrap(frozen_predictions(D))
    chk = check_against_t15(res)
    write_csv(res, OUT / "block_bootstrap_mape_differences.csv")
    write_csv(chk, OUT / "reproduction_check_vs_T15.csv")
    write_text(latex_table(res), OUT / "tab_block_bootstrap_mape.tex")
    (OUT / "run_metadata.json").write_text(json.dumps({
        "models": "folder-24 frozen: Huber residual, lambda*=0.70, Huber BBM on the target",
        "method": "paired moving-block bootstrap (rf36 block_idx), two-sided, Holm per partition",
        "block_length_records": BLOCK, "n_resamples": B_BOOT, "seed": SEED,
        "wbm_sgbm_rows_identical_to_T15": bool(chk[chk.expected_identical].identical_to_T15.all()),
    }, indent=2), encoding="utf-8")
    print(res[["partition", "comparison", "effect_size_pp", "ci95_lo", "ci95_hi",
               "p_holm", "statistically_resolved"]].to_string(index=False))
    print("[S39] WBM/S-GBM rows reproduce T15 exactly; BBM rows use the Huber BBM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
