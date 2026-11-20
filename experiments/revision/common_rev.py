"""Shared machinery for the mandatory revision experiments (SciSpace V2 round).

Nothing here changes the modelling methodology. It provides two things the
existing scripts do not expose:

1. **A physics core whose vessel particulars can be overridden**, so the
   wave-efficiency penalty ``eta_wave_loss`` can be set to zero without
   touching ``experiments/benchmark_measured/common.py``. The override gets its
   own cache namespace, so the committed physics caches of the published draft
   are never read or overwritten.
2. **A single architecture builder with the full validation-selection ladder**,
   including selection of the bounded-residual ``q``, which the published
   scripts hard-code rather than select.

Every split, feature set, seed, learner configuration and selection criterion
is copied from the published protocol:

* splits: ``load_shifts_splits()`` (train uniformly temporally subsampled to
  40,000; ``dev_in`` / ``dev_out`` as released by the benchmark authors);
* features: ``FEATS`` (9 columns), unchanged;
* fit block: first 85 % of train in time; selection set: final 15 % ("the
  temporal validation tail");
* selection criterion: **minimum MAPE on that validation tail**, the criterion
  ``run_cgbm_pareto.py`` already applies to lambda and alpha, now applied to q
  as well;
* seed 42 everywhere; HistGB ``max_iter=400``.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

_EXPERIMENTS_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from benchmark_measured.common import (  # noqa: E402
    CONFIG, FEATS, SEED, fit_scale, load_shifts_splits, make_learner, metrics,
    speed_sweep_frame,
)
from vessel_fuel import fuel_components  # noqa: E402

REPO_ROOT = _EXPERIMENTS_DIR.parent
REV_ROOT = REPO_ROOT / "results" / "revision_round"
CACHE = pathlib.Path("data/external/shifts_vessel/cache")

FIT_FRACTION = 0.85
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
QS = [0.1, 0.2, 0.3]
ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]

# Published setting of the contested wave-efficiency penalty, and the revision
# setting in which wave effects enter through STAWAVE-1 added resistance alone.
ETA_PUBLISHED = 0.03
ETA_REVISED = 0.0


def particulars(eta_wave_loss: float) -> dict[str, Any]:
    """Canonical vessel particulars with the wave-efficiency penalty overridden."""
    return dict(CONFIG["particulars"], eta_wave_loss=float(eta_wave_loss))


def physics_power_eta(df: pd.DataFrame, tag: str | None,
                      eta_wave_loss: float) -> np.ndarray:
    """Uncalibrated Holtrop shaft power (kW) at a given ``eta_wave_loss``.

    Identical to ``benchmark_measured.common.physics_power`` except that the
    vessel particulars are taken from :func:`particulars` and the cache key
    carries the eta value, so the published caches are neither read nor
    overwritten.

    Parameters
    ----------
    df : pandas.DataFrame
        Benchmark-schema frame (needs ``stw``, ``draft``, ``trim``, ``wind_kn``,
        ``wind_ang``, ``hs``, ``cur_kn``, ``cur_ang``, ``months``).
    tag : str or None
        Cache tag; ``None`` disables caching.
    eta_wave_loss : float
        Coefficient of the linear wave penalty on propulsive efficiency,
        ``eta = eta_0 * (1 - eta_wave_loss * Hs)``.

    Returns
    -------
    numpy.ndarray
        Uncalibrated shaft power per row (kW).
    """
    key = f"eta{eta_wave_loss:g}".replace(".", "p")
    if tag is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / f"rev_{key}_{tag}_{len(df)}.npz"
        if f.exists():
            return np.load(f)["p"]
    part = particulars(eta_wave_loss)
    out = np.empty(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        env = {"wind_kn": r.wind_kn, "wind_angle_deg": r.wind_ang, "Hs": r.hs,
               "Tp": 8.0, "wave_angle_deg": 0.0, "current_kn": r.cur_kn,
               "current_angle_deg": r.cur_ang, "sst_c": 12.0, "depth_m": 2000.0}
        prm = dict(part, T=float(r.draft), trim_m=float(r.trim),
                   months_since_cleaning=float(r.months))
        out[i] = float(fuel_components(100.0, float(r.stw), env, prm)["shaft_power_kw"])
    if tag is not None:
        np.savez_compressed(CACHE / f"rev_{key}_{tag}_{len(df)}.npz", p=out)
    return out


class RevisionModels:
    """The architecture family, fit on train only, at one ``eta_wave_loss``.

    Adds validation-tail selection of the bounded-residual ``q`` to the ladder
    the published scripts already select ``lambda`` and ``alpha`` on. Selection
    uses the same criterion for all three knobs: minimum MAPE on the final 15 %
    of the training split in time.

    Attributes
    ----------
    k : float
        Physics scale factor, least squares on the whole training split.
    lam_star, q_star, alpha_star : float
        Validation-tail-selected constraint parameters.
    val_scores : dict
        Per-knob validation-tail MAPE for every candidate value, so the
        selection is auditable rather than asserted.
    """

    def __init__(self, eta_wave_loss: float, backend: str = "histgb") -> None:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self.eta = float(eta_wave_loss)
        # The representative learner is a parameter so the architecture family can
        # be rebuilt on a different backend without touching the published run.
        # Default "histgb" reproduces the published draft byte-for-byte.
        self.backend = str(backend)
        _rng = np.random.default_rng(SEED)
        self.train, self.dev_in, self.dev_out = load_shifts_splits()
        self.ph = {
            "train": physics_power_eta(self.train, "train", self.eta),
            "dev_in": physics_power_eta(self.dev_in, "dev_in", self.eta),
            "dev_out": physics_power_eta(self.dev_out, "dev_out", self.eta),
        }
        self.y_tr = self.train["power"].to_numpy()
        self.X_tr = self.train[FEATS].to_numpy()
        self.k = fit_scale(self.y_tr, self.ph["train"])

        n_fit = int(FIT_FRACTION * len(self.train))
        self.fit_idx = np.arange(n_fit)
        self.val_idx = np.arange(n_fit, len(self.train))
        res = self.y_tr - self.k * self.ph["train"]

        # make_learner() is the same factory the 21-backend catalogue uses, so the
        # learner configuration here is identical to the candidate comparison.
        self._f_ml, sub_r = make_learner(self.backend, _rng)
        Xr, yr = sub_r(self.X_tr[self.fit_idx], res[self.fit_idx])
        self._f_ml.fit(Xr, yr)
        self._bbm, sub_b = make_learner(self.backend, _rng)
        Xb, yb = sub_b(self.X_tr[self.fit_idx], self.y_tr[self.fit_idx])
        self._bbm.fit(Xb, yb)
        self.n_fit_residual, self.n_fit_bbm = len(Xr), len(Xb)

        y_val = self.y_tr[self.val_idx]
        ph_val = self.ph["train"][self.val_idx]
        base_val = self.k * ph_val
        r_val = self._f_ml.predict(self.X_tr[self.val_idx])
        self.val_scores: dict[str, dict[float, float]] = {}

        lam = {l: metrics(y_val, base_val + l * r_val)["MAPE"] for l in LAMBDAS}
        self.val_scores["lambda"] = {float(a): float(b) for a, b in lam.items()}
        self.lam_star = float(min(lam, key=lam.get))

        qs = {q: metrics(y_val, base_val
                         + np.clip(r_val, -q * base_val, q * base_val))["MAPE"]
              for q in QS}
        self.val_scores["q"] = {float(a): float(b) for a, b in qs.items()}
        self.q_star = float(min(qs, key=qs.get))

        self._scaler = StandardScaler().fit(self.X_tr[self.fit_idx])
        alp: dict[float, float] = {}
        best = (np.inf, None, None)
        for a in ALPHAS:
            rr = Ridge(alpha=a).fit(self._scaler.transform(self.X_tr[self.fit_idx]),
                                    res[self.fit_idx])
            s = metrics(y_val, base_val
                        + rr.predict(self._scaler.transform(self.X_tr[self.val_idx])))["MAPE"]
            alp[float(a)] = float(s)
            if s < best[0]:
                best = (s, float(a), rr)
        self.val_scores["alpha"] = alp
        self.alpha_star, self._ridge = best[1], best[2]

    # -- prediction --------------------------------------------------------

    def predict(self, name: str, X: np.ndarray, phys: np.ndarray) -> np.ndarray:
        """Predict shaft power (kW) for one architecture name."""
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
            return base + self._ridge.predict(self._scaler.transform(X))
        if name.startswith("C-GBM-lambda-"):
            return base + float(name.rsplit("-", 1)[1]) * r
        if name.startswith("C-GBM-bounded-q"):
            q = float(name.rsplit("q", 1)[1])
            return base + np.clip(r, -q * base, q * base)
        raise ValueError(f"unknown architecture {name!r}")

    def frame(self, split: str) -> pd.DataFrame:
        return {"train": self.train, "dev_in": self.dev_in, "dev_out": self.dev_out}[split]

    def predictions(self, name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
        df = self.frame(split)
        return (df["power"].to_numpy(),
                self.predict(name, df[FEATS].to_numpy(), self.ph[split]))

    # -- physical consistency ---------------------------------------------

    def physics_row(self, name: str, sweep: pd.DataFrame) -> dict[str, float]:
        """Monotonicity, fitted speed-power exponent (+95 % CI) and D_phys."""
        from scipy.stats import t as tdist
        ph_sweep = physics_power_eta(sweep, None, self.eta)
        p = self.predict(name, sweep[FEATS].to_numpy(), ph_sweep)
        v = sweep["stw"].to_numpy()
        viol = float(np.mean(np.diff(p) < 0.0))
        ok = p > 0
        if ok.sum() >= 10:
            x, yv = np.log(v[ok]), np.log(p[ok])
            b, c = np.polyfit(x, yv, 1)
            n = len(x)
            s2 = float(np.sum((yv - (b * x + c)) ** 2) / (n - 2))
            se = float(np.sqrt(s2 / np.sum((x - x.mean()) ** 2)))
            tc = float(tdist.ppf(0.975, n - 2))
            lo, hi = b - tc * se, b + tc * se
        else:
            b = lo = hi = float("nan")
        d = {}
        for split in ("dev_in", "dev_out"):
            _, p_s = self.predictions(name, split)
            wbm = self.k * self.ph[split]
            d[split] = float(np.mean(np.abs(p_s - wbm) / np.maximum(wbm, 1e-9)))
        return {"mono_violation_frac": viol, "propeller_exponent": float(b),
                "exponent_lo95": float(lo), "exponent_hi95": float(hi),
                "D_phys_in": d["dev_in"], "D_phys_out": d["dev_out"]}

    def conformal(self, name: str, alpha: float = 0.10) -> dict[str, dict[str, float]]:
        """Split-conformal coverage and width, radius from the train tail only."""
        y_val = self.y_tr[self.val_idx]
        p_val = self.predict(name, self.X_tr[self.val_idx], self.ph["train"][self.val_idx])
        q = float(np.quantile(np.abs(y_val - p_val), 1.0 - alpha))
        out = {}
        for split in ("dev_in", "dev_out"):
            y, p = self.predictions(name, split)
            lo, hi = np.maximum(p - q, 0.0), p + q
            out[split] = {"coverage": float(np.mean((y >= lo) & (y <= hi))),
                          "width_kw": float(np.mean(hi - lo)), "q_kw": q}
        return out


ARCHITECTURES = ["WBM", "BBM", "S-GBM", "C-GBM-shrink", "C-GBM-bounded", "C-GBM-ridge"]


def sweep(models: RevisionModels, n: int = 60) -> pd.DataFrame:
    return speed_sweep_frame(models.train, n=n)


def write_csv(df: pd.DataFrame, path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_text(text: str, path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def to_tex(df: pd.DataFrame, path: pathlib.Path, caption: str,
           label: str, floatfmt: str = "%.3f", note: str = "") -> pathlib.Path:
    r"""Write a standalone LaTeX table (no manuscript numbering, no \input hooks).

    Parameters
    ----------
    df : pandas.DataFrame
        Table body; float columns are formatted with *floatfmt*.
    path : pathlib.Path
        Destination ``.tex`` file.
    caption, label : str
        Caption text and ``\label`` key.
    floatfmt : str, default="%.3f"
        Format applied to float columns.
    note : str, default=""
        Optional footnote placed under the tabular in a ``\footnotesize`` line.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    body = df.copy()
    for c in body.columns:
        if pd.api.types.is_bool_dtype(body[c]):
            body[c] = body[c].map({True: r"\checkmark", False: ""})
        elif pd.api.types.is_float_dtype(body[c]):
            body[c] = body[c].map(lambda v: "" if pd.isna(v) else floatfmt % v)
    cols = "l" + "r" * (len(body.columns) - 1)

    def esc(s: object) -> str:
        t = str(s)
        if t.startswith("\\"):
            return t
        return (t.replace("\\", r"\textbackslash{}").replace("_", r"\_")
                 .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))

    def esc_prose(t: str) -> str:
        """Escape caption/note prose, leaving intentional LaTeX macros alone."""
        if "\\" in t:            # caller supplied deliberate markup
            return t
        return t.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")

    caption, note = esc_prose(caption), esc_prose(note)
    lines = [
        "% Auto-generated for the revision round -- place manually; no numbering assumed.",
        r"\begin{table}[!tb]", r"  \centering", rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}", r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3.5pt}", rf"  \begin{{tabular}}{{{cols}}}",
        r"    \toprule",
        "    " + " & ".join(esc(c) for c in body.columns) + r" \\",
        r"    \midrule",
    ]
    for _, r in body.iterrows():
        lines.append("    " + " & ".join(esc(v) for v in r) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    if note:
        lines.append(r"  \\[2pt] \parbox{\linewidth}{\footnotesize " + note + "}")
    lines += [r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# VFM_RESULTS_ROOT lets a full rerun write to a fresh folder and leave the saved
# results untouched; unset, the saved location is used.
REV_FINAL = (pathlib.Path(os.environ["VFM_RESULTS_ROOT"]) if os.environ.get("VFM_RESULTS_ROOT")
             else REPO_ROOT / "results" / "revision_final")
