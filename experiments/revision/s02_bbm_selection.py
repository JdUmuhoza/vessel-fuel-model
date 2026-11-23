"""Standalone BBM model selection on the full-data 60/20/20 protocol.

Folder 01 selected a backend in the **residual** role, where the target is
``r = y - k*F_phys``, and then reused that learner as the standalone black-box
model. Huber is an excellent residual learner and a poor direct predictor of
shaft power, so that reuse understates what a black-box model can do.

This stage selects the standalone BBM independently, training every candidate
directly on the measured target ``y``. Everything else is inherited unchanged
from folder 01 by importing its own functions: the chronological 60/20/20
splits, the physics cache, the runtime-exclusion decisions, the no-subsampling
rule and the early-stopping sub-block.

The residual-learner result of folder 01 is not modified, lambda* = 0.70 is not
modified, folder 01 is not written to, and no manuscript file is touched.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .common_rev import REPO_ROOT, REV_FINAL, SEED, write_csv, write_text
from .s01_full_data_602020 import (
    BUDGET_S, EXCLUDED_MEASURED, FEATS, LOWER_BETTER, METRICS, PRIMARY,
    build_splits, feasibility, fit_learner, metrics7, physics,
)

_E = REPO_ROOT / "experiments"
if str(_E) not in sys.path:
    sys.path.insert(0, str(_E))
from benchmark_measured.common import (  # noqa: E402
    BACKEND_DISPLAY, CONFIG, fit_scale,
)

OUT = REV_FINAL / "02_bbm_selection"
CACHE = OUT / "_cache"
F01 = REV_FINAL / "01_full_data_602020"
LAMBDA_STAR = 0.70                 # carried from folder 01, not re-selected


def _git() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


def bbm_selection(parts, excluded) -> tuple[pd.DataFrame, str]:
    """Every candidate trained directly on measured shaft power y."""
    tr, va = parts["train"], parts["validation"]
    y_tr, X_tr = tr["power"].to_numpy(), tr[FEATS].to_numpy()
    y_va, X_va = va["power"].to_numpy(), va[FEATS].to_numpy()
    fit_idx, es_idx = parts["_fit_idx"], parts["_es_idx"]

    drop = set(excluded[~excluded.feasible_at_full_n].backend_key)
    drop |= set(EXCLUDED_MEASURED)
    cand = [b for b in CONFIG["bbm_backends"] if b not in drop]

    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / "bbm_selection.csv"
    rows = pd.read_csv(cf).to_dict("records") if cf.exists() else []
    have = {r["backend_key"] for r in rows}

    for key in cand:
        if key in have:
            continue
        t0 = time.time()
        try:
            lrn, info = fit_learner(key, X_tr, y_tr, fit_idx, es_idx)
            m = metrics7(y_va, lrn.predict(X_va))
        except Exception as exc:
            print(f"    {BACKEND_DISPLAY[key]:16s} FAILED: {exc}", flush=True)
            rows.append({"model": BACKEND_DISPLAY[key], "backend_key": key,
                         "status": f"failed: {type(exc).__name__}",
                         "fit_seconds": time.time() - t0})
            pd.DataFrame(rows).to_csv(cf, index=False)
            continue
        rows.append({"model": BACKEND_DISPLAY[key], "backend_key": key,
                     "status": "ok", "role": "standalone BBM (target = y)",
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
    write_csv(ok, OUT / "bbm_selection_validation_full_data.csv")
    sel = str(ok.iloc[0]["backend_key"])
    print(f"[S02] standalone BBM selected on validation {PRIMARY}: "
          f"{ok.iloc[0]['model']} ({ok.iloc[0][PRIMARY]:.4f})", flush=True)
    return ok, sel


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = build_splits()                    # identical construction to folder 01
    ph = physics(parts)                       # identical cache keys
    tr = parts["train"]
    y_tr, X_tr = tr["power"].to_numpy(), tr[FEATS].to_numpy()
    k = fit_scale(y_tr, ph["train"])
    res_tr = y_tr - k * ph["train"]

    # split identity proof against folder 01
    s24 = pd.read_csv(F01 / "split_metadata.csv").set_index("partition")
    ident = []
    for name in ("train", "validation", "internal_test", "dev_in", "dev_out"):
        d = parts[name]
        ident.append({
            "partition": name, "n_here": int(len(d)),
            "n_folder24": int(s24.loc[name, "n_records"]),
            "time_id_min_here": int(d["time_id"].min()),
            "time_id_min_folder24": int(s24.loc[name, "time_id_min"]),
            "time_id_max_here": int(d["time_id"].max()),
            "time_id_max_folder24": int(s24.loc[name, "time_id_max"]),
            "identical": bool(len(d) == s24.loc[name, "n_records"]
                              and d["time_id"].min() == s24.loc[name, "time_id_min"]
                              and d["time_id"].max() == s24.loc[name, "time_id_max"]),
            "used_in_bbm_selection": name in ("train", "validation"),
            "dev_leakage": False})
    idf = pd.DataFrame(ident)
    write_csv(idf, OUT / "split_identity_check.csv")
    if not idf.identical.all():
        raise AssertionError("splits differ from folder 01")
    k01 = json.loads((F01 / "run_metadata.json").read_text(encoding="utf-8"))["physics_k"]
    assert abs(k - k01) < 1e-12, f"physics k differs from folder 01: {k} vs {k01}"
    print(f"[S02] splits and physics k ({k:.6f}) identical to folder 01", flush=True)

    excl = feasibility(X_tr, res_tr)
    write_csv(excl, OUT / "excluded_models_feasibility.csv")
    sel_tbl, sel = bbm_selection(parts, excl)

    # frozen evaluation: selected BBM, the folder-01 BBM (Huber on y) for a
    # like-for-like comparison, and the unchanged WBM / S-GBM / C-GBM family.
    fit_idx, es_idx = parts["_fit_idx"], parts["_es_idx"]
    bbm_sel, _ = fit_learner(sel, X_tr, y_tr, fit_idx, es_idx)
    bbm_huber, _ = fit_learner("huber", X_tr, y_tr, fit_idx, es_idx)
    resid, _ = fit_learner("huber", X_tr, res_tr, fit_idx, es_idx)

    rows = []
    for name in ("internal_test", "dev_in", "dev_out"):
        d = parts[name]
        y, X = d["power"].to_numpy(), d[FEATS].to_numpy()
        base = k * ph[name]
        r = resid.predict(X)
        preds = {
            f"BBM ({BACKEND_DISPLAY[sel]}, selected on y)": bbm_sel.predict(X),
            "BBM (Huber, folder-01 reuse of the residual learner)":
                bbm_huber.predict(X),
            "WBM": base,
            "S-GBM (Huber residual)": base + r,
            f"C-GBM (Huber residual, lambda*={LAMBDA_STAR:g})":
                base + LAMBDA_STAR * r,
        }
        for arch, p_ in preds.items():
            rows.append({"partition": name, "architecture": arch,
                         "n": int(len(y)), **metrics7(y, p_)})
    frozen = pd.DataFrame(rows)
    write_csv(frozen[frozen.partition == "internal_test"].reset_index(drop=True),
              OUT / "selected_bbm_internal_test.csv")
    write_csv(frozen[frozen.partition != "internal_test"].reset_index(drop=True),
              OUT / "selected_bbm_dev_in_dev_out.csv")
    write_csv(frozen, OUT / "frozen_all_partitions.csv")

    (OUT / "run_metadata.json").write_text(json.dumps({
        "purpose": "standalone BBM selection, target = measured shaft power y",
        "splits_identical_to_folder24": True,
        "physics_k": float(k), "lambda_star_carried": LAMBDA_STAR,
        "residual_learner_unchanged": "Huber",
        "selected_standalone_bbm": BACKEND_DISPLAY[sel],
        "different_from_residual_learner": bool(sel != "huber"),
        "excluded_models": sorted(set(excl[~excl.feasible_at_full_n].model)
                                  | {BACKEND_DISPLAY[k_] for k_ in EXCLUDED_MEASURED}),
        "budget_seconds": BUDGET_S, "subsampling_applied": False,
        "dev_in_used_in_selection": False, "dev_out_used_in_selection": False,
        "primary_metric": PRIMARY, "seed": SEED, "git_commit": _git()},
        indent=2), encoding="utf-8")

    report(sel_tbl, sel, frozen, idf, k)
    return 0


def report(sel_tbl, sel, frozen, idf, k) -> None:
    L: list[str] = []
    A = L.append
    A("# Standalone BBM selection on the full-data 60/20/20 protocol")
    A("")
    A("Folder 01 selected a backend in the **residual** role and reused it as the "
      "standalone BBM. This stage selects the BBM independently, training every "
      "candidate directly on measured shaft power. The residual learner (Huber) and "
      f"lambda* = {LAMBDA_STAR:g} are carried over unchanged.")
    A("")
    A("## Split identity - no drift, no leakage")
    A("")
    A("| partition | n here | n folder 01 | time_id range matches | used in BBM selection |")
    A("|---|---|---|---|---|")
    for _, r in idf.iterrows():
        A(f"| {r['partition']} | {r['n_here']:,} | {r['n_folder24']:,} | "
          f"{'yes' if r['identical'] else 'NO'} | "
          f"{'yes' if r['used_in_bbm_selection'] else 'no'} |")
    A("")
    A(f"Physics scale k = {k:.6f}, identical to folder 01. `dev_in` and `dev_out` "
      f"were used only after the BBM was selected.")
    A("")
    A("## BBM validation ranking (target = measured shaft power)")
    A("")
    A("| model | MAPE | sMAPE | MAE | nMAE | RMSE | NRMSE | R2 | mean rank |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in sel_tbl.iterrows():
        A(f"| {r['model']} | {r['MAPE']:.4f} | {r['sMAPE']:.4f} | {r['MAE']:.1f} | "
          f"{r['nMAE']:.4f} | {r['RMSE']:.1f} | {r['NRMSE']:.4f} | {r['R2']:.4f} | "
          f"{r['mean_rank_all_metrics']:.2f} |")
    A("")
    A(f"**Selected standalone BBM: {BACKEND_DISPLAY[sel]}**")
    A("")
    for part in ("internal_test", "dev_in", "dev_out"):
        A(f"## Frozen evaluation - {part}")
        A("")
        A("| architecture | MAPE | sMAPE | MAE | nMAE | RMSE | NRMSE | R2 |")
        A("|---|---|---|---|---|---|---|---|")
        for _, r in frozen[frozen.partition == part].iterrows():
            A(f"| {r['architecture']} | {r['MAPE']:.4f} | {r['sMAPE']:.4f} | "
              f"{r['MAE']:.1f} | {r['nMAE']:.4f} | {r['RMSE']:.1f} | "
              f"{r['NRMSE']:.4f} | {r['R2']:.4f} |")
        A("")
    write_text("\n".join(L) + "\n", OUT / "bbm_selection_report.md")
    print("[S02] report written", flush=True)


def _guarded_main() -> int:
    import joblib
    with joblib.parallel_backend("threading", n_jobs=1):
        return main()


if __name__ == "__main__":
    sys.exit(_guarded_main())
