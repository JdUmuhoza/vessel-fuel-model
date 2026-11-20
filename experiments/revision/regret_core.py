"""Decision-regret machinery shared by Tasks 2 and 3 of the revision round.

Reproduces the published protocol of
``experiments/benchmark_measured/run_decision_regret.py`` exactly -- same
mid-band fit, same condition bins, same 1-kn speed cells, same economics, same
seed -- and adds the two things the published script does not provide:

1. **Validation-tail selection of the constraint parameters inside the regret
   fit.** The published script hard-codes ``lambda = 0.25`` and
   ``alpha = 100``, and carries no bounded-residual variant. Because the regret
   protocol fits on the interquartile speed band rather than the whole training
   split, its selection set is defined here as the final 15 % of that mid band
   in time -- the same 85/15 temporal rule the primary protocol uses, applied
   to the mid band. This is stated rather than silent: it is the one procedural
   addition in this module.
2. **Per-row prediction caching**, so a paired bootstrap can resample the
   underlying records and recompute the whole decision procedure without
   re-invoking any learner.

The economics constants and the two decision tasks are copied verbatim from the
published script; changing them would change the published result rather than
quantify its uncertainty.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common_rev import (
    FEATS, FIT_FRACTION, LAMBDAS, QS, ALPHAS, SEED,
    ETA_PUBLISHED, fit_scale, load_shifts_splits, make_learner, metrics,
    physics_power_eta,
)

# --- economics and binning: verbatim from run_decision_regret.py ---
SFOC_G_KWH = 170.0
FUEL_USD_T = 600.0
CHARTER_USD_DAY = 15_000.0
TARGET_RED = 0.20
MIN_CELL = 5
MIN_CELLS_PER_BIN = 4

ARCH_REGRET = ["WBM", "BBM", "S-GBM", "C-GBM-shrink", "C-GBM-bounded", "C-GBM-ridge"]


def cost_per_nm(p_kw: np.ndarray, v_kn: np.ndarray) -> np.ndarray:
    """Unit transport cost ($/nm) from shaft power and speed."""
    fuel_usd_h = p_kw * SFOC_G_KWH / 1e6 * FUEL_USD_T
    return (fuel_usd_h + CHARTER_USD_DAY / 24.0) / v_kn


class RegretModels:
    """Architectures fit on the training mid speed band.

    Parameters
    ----------
    fit_mode : {"published", "selected"}
        ``"published"`` reproduces ``run_decision_regret.py`` exactly: learners
        fit on the **whole** mid band, ``lambda`` fixed at 0.25 and ``alpha``
        fixed at 100, no bounded variant (its ``q`` is set to 0.1 so the
        architecture list stays comparable, and this is flagged in the output).
        ``"selected"`` mirrors the primary protocol instead: learners fit on the
        first 85 % of the mid band in time, and ``lambda``, ``q`` and ``alpha``
        selected by minimum MAPE on the final 15 %.
    eta_wave_loss : float
        Wave-efficiency penalty coefficient of the physics core.
    """

    PUBLISHED_LAMBDA = 0.25
    PUBLISHED_ALPHA = 100.0
    PUBLISHED_Q = 0.1

    def __init__(self, fit_mode: str = "selected",
                 eta_wave_loss: float = ETA_PUBLISHED,
                 backend: str = "histgb") -> None:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        # Representative learner as a parameter; "histgb" reproduces the published run.
        self.backend = str(backend)
        _rng = np.random.default_rng(SEED)

        if fit_mode not in ("published", "selected"):
            raise ValueError(f"fit_mode must be 'published' or 'selected', got {fit_mode!r}")
        self.fit_mode = fit_mode
        self.eta = float(eta_wave_loss)

        tr, self.dev_in, self.dev_out = load_shifts_splits()
        self.q1, self.q3 = tr["stw"].quantile([0.25, 0.75])
        mid = tr[(tr["stw"] >= self.q1) & (tr["stw"] <= self.q3)].reset_index(drop=True)
        self.mid = mid
        self.n_mid = len(mid)

        y = mid["power"].to_numpy()
        X = mid[FEATS].to_numpy()
        ph = physics_power_eta(mid, "train_midband", self.eta)
        self.k = fit_scale(y, ph)
        res = y - self.k * ph

        if fit_mode == "published":
            fit_i = np.arange(len(mid))          # published: fit on the whole mid band
            val_i = np.arange(0, 0)
        else:
            n_fit = int(FIT_FRACTION * len(mid))
            fit_i, val_i = np.arange(n_fit), np.arange(n_fit, len(mid))
        self.n_fit, self.n_val = len(fit_i), len(val_i)

        self._f_ml, sub_r = make_learner(self.backend, _rng)
        Xr, yr = sub_r(X[fit_i], res[fit_i])
        self._f_ml.fit(Xr, yr)
        self._bbm, sub_b = make_learner(self.backend, _rng)
        Xb, yb = sub_b(X[fit_i], y[fit_i])
        self._bbm.fit(Xb, yb)
        self._sc = StandardScaler().fit(X[fit_i])
        self.val_scores: dict[str, dict[float, float]] = {}

        if fit_mode == "published":
            self.lam_star = self.PUBLISHED_LAMBDA
            self.q_star = self.PUBLISHED_Q
            self.alpha_star = self.PUBLISHED_ALPHA
            self._ridge = Ridge(alpha=self.PUBLISHED_ALPHA).fit(
                self._sc.transform(X[fit_i]), res[fit_i])
            return

        base_val = self.k * ph[val_i]
        r_val = self._f_ml.predict(X[val_i])
        y_val = y[val_i]

        lam = {l: metrics(y_val, base_val + l * r_val)["MAPE"] for l in LAMBDAS}
        self.val_scores["lambda"] = {float(a): float(b) for a, b in lam.items()}
        self.lam_star = float(min(lam, key=lam.get))

        qq = {q: metrics(y_val, base_val
                         + np.clip(r_val, -q * base_val, q * base_val))["MAPE"] for q in QS}
        self.val_scores["q"] = {float(a): float(b) for a, b in qq.items()}
        self.q_star = float(min(qq, key=qq.get))

        best, alp = (np.inf, None, None), {}
        for a in ALPHAS:
            rr = Ridge(alpha=a).fit(self._sc.transform(X[fit_i]), res[fit_i])
            s = metrics(y_val, base_val + rr.predict(self._sc.transform(X[val_i])))["MAPE"]
            alp[float(a)] = float(s)
            if s < best[0]:
                best = (s, float(a), rr)
        self.val_scores["alpha"] = alp
        self.alpha_star, self._ridge = best[1], best[2]

    def predict(self, name: str, X: np.ndarray, phys: np.ndarray) -> np.ndarray:
        base = self.k * phys
        if name == "WBM":
            return base
        if name == "BBM":
            return self._bbm.predict(X)
        r = self._f_ml.predict(X)
        if name == "S-GBM":
            return base + r
        if name == "C-GBM-shrink":
            return base + self.lam_star * r
        if name == "C-GBM-bounded":
            return base + np.clip(r, -self.q_star * base, self.q_star * base)
        if name == "C-GBM-ridge":
            return base + self._ridge.predict(self._sc.transform(X))
        raise ValueError(name)


def prepare_split(models: RegretModels, split: str) -> dict[str, Any]:
    """Bin one evaluation set and precompute per-row predictions.

    Returns
    -------
    dict
        ``bins``: list of usable condition bins, each a dict with the sorted
        speed grid ``vs`` and, per speed cell, the row indices into the frame;
        ``pred``: per-architecture prediction array over the whole frame;
        ``power``: measured power array; ``mape``: per-architecture MAPE.
    """
    dset = {"dev_in": models.dev_in, "dev_out": models.dev_out}[split]
    phys = physics_power_eta(dset, split, models.eta)
    d = dset.reset_index(drop=True).copy()
    d["_phys"] = phys
    d["_cell"] = d["stw"].round().astype(int)
    d["_draft_bin"] = pd.qcut(d["draft"], 3, labels=False, duplicates="drop")
    d["_wind_bin"] = pd.qcut(d["wind_kn"], 3, labels=False, duplicates="drop")

    X = d[FEATS].to_numpy()
    ph = d["_phys"].to_numpy()
    power = d["power"].to_numpy()
    pred = {m: models.predict(m, X, ph) for m in ARCH_REGRET}
    mape = {m: metrics(power, pred[m])["MAPE"] for m in ARCH_REGRET}

    bins: list[dict[str, Any]] = []
    for _, g in d.groupby(["_draft_bin", "_wind_bin"]):
        cells = {c: cg.index.to_numpy() for c, cg in g.groupby("_cell")
                 if len(cg) >= MIN_CELL}
        if len(cells) < MIN_CELLS_PER_BIN:
            continue
        vs = np.array(sorted(cells), dtype=float)
        idx = [cells[int(v)] for v in vs]
        cur = max(cells, key=lambda c: len(cells[c]))
        bins.append({"vs": vs, "idx": idx,
                     "i_cur": int(np.where(vs == cur)[0][0]),
                     "n_records": int(sum(len(i) for i in idx))})
    return {"bins": bins, "pred": pred, "power": power, "mape": mape,
            "n_records": int(sum(b["n_records"] for b in bins))}


def regret_once(prep: dict[str, Any],
                resample: np.random.Generator | None = None
                ) -> tuple[dict[str, float], dict[str, float]]:
    """Run the full decision procedure once, optionally on a resample.

    When ``resample`` is given, row indices inside every speed cell are drawn
    with replacement at the cell's own size, and the **same** resampled indices
    are used for the measured curve and for every architecture, so architecture
    differences are paired.

    Returns
    -------
    tuple of dict
        ``(mean speed regret % per architecture, mean CII shortfall pp per
        architecture)``.
    """
    power, pred = prep["power"], prep["pred"]
    speed_acc = {m: [] for m in ARCH_REGRET}
    cii_acc = {m: [] for m in ARCH_REGRET}

    for b in prep["bins"]:
        vs, i_cur = b["vs"], b["i_cur"]
        if resample is None:
            idx = b["idx"]
        else:
            idx = [resample.choice(i, size=len(i), replace=True) for i in b["idx"]]
        meas_p = np.array([np.median(power[i]) for i in idx])
        meas_J = cost_per_nm(meas_p, vs)
        j_true = meas_J.min()
        meas_e = meas_p / vs
        for m in ARCH_REGRET:
            pp = np.array([np.median(pred[m][i]) for i in idx])
            v_pick = int(np.argmin(cost_per_nm(pp, vs)))
            speed_acc[m].append(100.0 * (meas_J[v_pick] - j_true) / j_true)
            pe = pp / vs
            rel = 1.0 - pe / max(pe[i_cur], 1e-9)
            ok = [i for i in range(len(vs)) if vs[i] <= vs[i_cur] and rel[i] >= TARGET_RED]
            if not ok:
                continue
            i_rec = max(ok, key=lambda i: vs[i])
            ach = 1.0 - meas_e[i_rec] / max(meas_e[i_cur], 1e-9)
            cii_acc[m].append(100.0 * max(TARGET_RED - ach, 0.0))

    speed = {m: float(np.mean(v)) if v else float("nan") for m, v in speed_acc.items()}
    cii = {m: float(np.mean(v)) if v else float("nan") for m, v in cii_acc.items()}
    return speed, cii
