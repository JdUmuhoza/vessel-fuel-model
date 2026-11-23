"""Was selecting Huber in Table 6 justified? Score every candidate after selection.

Table 6 selects the residual learner on the validation block only (stage s01, folder 01).
This stage refits every Table 6 candidate with the identical split, physics model and
learner settings, and scores each S-GBM (physics + full residual) on the validation block
(the selection data, reproducing Table 6), the internal test block (never used for
selection) and the held-out in-domain and shifted sets. A selection that only reflects the
validation block would show another learner winning on the later partitions.

Diagnostic only: nothing here changes the selection. Folder 01 is not modified.

Run:  python -m experiments.revision.s12_table6_selection_check
Outputs: results/revision_final/12_table6_selection_check/
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from . import s01_full_data_602020 as s01
from .common_rev import FEATS, REV_FINAL, write_csv

_E = s01.REPO_ROOT / "experiments"
if str(_E) not in sys.path:
    sys.path.insert(0, str(_E))
from benchmark_measured.common import BACKEND_DISPLAY, fit_scale  # noqa: E402

OUT = REV_FINAL / "12_table6_selection_check"
F01 = REV_FINAL / "01_full_data_602020"
PARTS = ("validation", "internal_test", "dev_in", "dev_out")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s01.OUT = OUT                      # build_splits writes split metadata; keep folder 01 intact
    parts = s01.build_splits()
    ph = s01.physics(parts)
    tr = parts["train"]
    y_tr, X_tr = tr["power"].to_numpy(), tr[FEATS].to_numpy()
    k = fit_scale(y_tr, ph["train"])
    res_tr = y_tr - k * ph["train"]
    keys = pd.read_csv(F01 / "ml_selection_validation_full_data.csv")["backend_key"].tolist()
    cache = OUT / "_rows.csv"
    rows = pd.read_csv(cache).to_dict("records") if cache.exists() else []
    done = {r["backend_key"] for r in rows}
    for key in keys:
        if key in done:
            continue
        t0 = time.time()
        lrn, _ = s01.fit_learner(key, X_tr, res_tr, parts["_fit_idx"], parts["_es_idx"])
        r = {"model": BACKEND_DISPLAY[key], "backend_key": key}
        for p in PARTS:
            d = parts[p]
            m = s01.metrics7(d["power"].to_numpy(), k * ph[p] + lrn.predict(d[FEATS].to_numpy()))
            r.update({f"{p}_MAPE": m["MAPE"], f"{p}_RMSE": m["RMSE"], f"{p}_R2": m["R2"]})
        rows.append(r)
        pd.DataFrame(rows).to_csv(cache, index=False)
        print(f"  {r['model']:14s} " + "  ".join(f"{p} {r[f'{p}_MAPE']:6.2f}" for p in PARTS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    tab = pd.DataFrame(rows)
    for p in PARTS:
        tab[f"{p}_rank"] = tab[f"{p}_MAPE"].rank().astype(int)
    tab = tab.sort_values("validation_MAPE").reset_index(drop=True)
    write_csv(tab, OUT / "selection_check_all_partitions.csv")
    ref = pd.read_csv(F01 / "ml_selection_validation_full_data.csv").set_index("backend_key")
    gap = float(np.max(np.abs(tab.set_index("backend_key")["validation_MAPE"] - ref["MAPE"])))
    print(f"\nTable 6 reproduction (validation MAPE): max |diff| = {gap:.2e}")
    print(tab[["model"] + [f"{p}_MAPE" for p in PARTS] + [f"{p}_rank" for p in PARTS]]
          .round(3).to_string(index=False))
    return 0


def _guarded() -> int:
    import joblib
    with joblib.parallel_backend("threading", n_jobs=1):
        return main()


if __name__ == "__main__":
    sys.exit(_guarded())
