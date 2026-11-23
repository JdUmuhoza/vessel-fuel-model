"""Does the MLP rank in Table 6 depend on its hidden-layer architecture?

Table 6 uses one MLP architecture, two hidden layers (64, 32) (paper hyperparameter table),
and the MLP ranks 10th on validation MAPE. This stage refits the MLP residual learner with
other architectures, more layers (3 and 5) and smaller networks (1 layer; 3 and 5 neurons),
on the identical split, physics model, training block and scoring as stage s12, and scores
each S-GBM (physics + full residual) on validation, internal test, dev_in and dev_out.
The (64, 32) control must reproduce Table 6's MLP validation MAPE.

Sensitivity only: the paper's selection is unchanged. Folders 01 and 12 are not modified.

Run:  python -m experiments.revision.s13_mlp_architecture_sensitivity
Outputs: results/revision_final/13_mlp_architecture_sensitivity/
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

from . import s01_full_data_602020 as s01
from .common_rev import FEATS, REV_FINAL, SEED, write_csv

_E = s01.REPO_ROOT / "experiments"
if str(_E) not in sys.path:
    sys.path.insert(0, str(_E))
from benchmark_measured.common import fit_scale  # noqa: E402
from vessel_fuel.research.hybrid import _make_learner  # noqa: E402

OUT = REV_FINAL / "13_mlp_architecture_sensitivity"
F01 = REV_FINAL / "01_full_data_602020"
PARTS = ("validation", "internal_test", "dev_in", "dev_out")
# label -> hidden layer sizes; (64, 32) is the Table 6 control.
ARCHS = {
    "2 layers (64,32) [Table 6]": (64, 32),
    "1 layer (32)": (32,),
    "1 layer (5 neurons)": (5,),
    "1 layer (3 neurons)": (3,),
    "3 layers (64,32,16)": (64, 32, 16),
    "5 layers (64,64,32,32,16)": (64, 64, 32, 32, 16),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s01.OUT = OUT                      # build_splits writes split metadata; keep folder 01 intact
    parts = s01.build_splits()
    ph = s01.physics(parts)
    tr = parts["train"]
    y_tr, X_tr = tr["power"].to_numpy(), tr[FEATS].to_numpy()
    k = fit_scale(y_tr, ph["train"])
    res_tr = y_tr - k * ph["train"]
    fit_idx = parts["_fit_idx"]
    cache = OUT / "_rows.csv"
    rows = pd.read_csv(cache).to_dict("records") if cache.exists() else []
    done = {r["architecture"] for r in rows}
    for label, hidden in ARCHS.items():
        if label in done:
            continue
        t0 = time.time()
        lrn = _make_learner("mlp", hidden=hidden, seed=SEED)
        lrn.fit(X_tr[fit_idx], res_tr[fit_idx])
        r = {"architecture": label, "hidden": str(hidden), "n_hidden_layers": len(hidden),
             "fit_seconds": round(time.time() - t0, 1)}
        for p in PARTS:
            d = parts[p]
            m = s01.metrics7(d["power"].to_numpy(), k * ph[p] + lrn.predict(d[FEATS].to_numpy()))
            r.update({f"{p}_MAPE": m["MAPE"], f"{p}_RMSE": m["RMSE"], f"{p}_R2": m["R2"]})
        rows.append(r)
        pd.DataFrame(rows).to_csv(cache, index=False)
        print(f"  {label:28s} " + "  ".join(f"{p} {r[f'{p}_MAPE']:6.2f}" for p in PARTS)
              + f"  ({r['fit_seconds']:.0f}s)", flush=True)
    tab = pd.DataFrame(rows).sort_values("validation_MAPE").reset_index(drop=True)
    write_csv(tab, OUT / "mlp_architectures_all_partitions.csv")
    ref = pd.read_csv(F01 / "ml_selection_validation_full_data.csv").set_index("backend_key")
    ctl = tab.loc[tab.hidden == str((64, 32)), "validation_MAPE"].iloc[0]
    gap = abs(ctl - float(ref.loc["mlp", "MAPE"]))
    (OUT / "run_metadata.json").write_text(json.dumps({
        "protocol": "s12 S-GBM scoring, MLP hidden layers varied, other settings default",
        "architectures": {k: list(v) for k, v in ARCHS.items()}, "seed": SEED,
        "control_gap_vs_table6": gap}, indent=2), encoding="utf-8")
    print(f"\nControl (64,32) vs Table 6 MLP validation MAPE: |diff| = {gap:.2e}")
    print(f"Table 6 reference: Huber validation MAPE = {ref.loc['huber', 'MAPE']:.3f}")
    print(tab[["architecture"] + [f"{p}_MAPE" for p in PARTS]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
