"""Shared machinery for the measured-data benchmark (Shifts + FuelCast).

Primary benchmark of the paper: instrument-measured targets only (Shifts
main-engine shaft power; FuelCast Coriolis fuel). Temporal hold-out; the
physics scale factor and every learner are fit on the training split only.
No MRV-derived quantity is used anywhere in this package (MRV trust policy).

All experiments: seed 42, config-driven, one command per table/figure.
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable

import numpy as np
import pandas as pd

from vessel_fuel import fuel_components
from vessel_fuel.research.hybrid import _make_learner

SEED = 42
REAL = pathlib.Path("data/external/shifts_vessel/v2/power_consumption_upload/real_data")
CACHE = pathlib.Path("data/external/shifts_vessel/cache")
OUT = pathlib.Path("outputs/benchmark_measured")
MIN_PER_MONTH = 60.0 * 24.0 * 30.44

FEATS = ["stw", "draft", "trim", "wind_kn", "wind_ang", "cur_kn", "cur_ang", "hs", "months"]

CONFIG: dict[str, Any] = {
    "seed": SEED,
    "n_train_sub": 40_000,     # uniform temporal subsample of train (physics loop cost)
    "pnn_train_sub": 2_500,    # PNN/GRNN LOO-CV is O(n^2); documented subsample
    "speed_min_kn": 4.0,
    "power_min_kw": 100.0,
    # representative large container-ship particulars (vessel anonymised);
    # a single scale factor k absorbs the unknown absolute hull size.
    "particulars": dict(
        L=300.0, B=40.0, T=14.5, S=15_200.0, CB=0.62, MCR=45_000.0,
        sfoc_at_mcr=170.0, A_front=1_200.0, A_lateral=8_000.0,
        design_draft=14.5, eta_0=0.70, eta_wave_loss=0.03,
    ),
    # 21 BBM backends spanning five learning families: linear
    # (ridge/lasso/elastic/bayesian_ridge/huber/ols), kernel/instance-based
    # (knn/svr/gpr/pnn), tree ensembles (dt/rf/et/ada/bag/gbt/histgb),
    # boosted-tree libraries (xgboost/catboost/lightgbm), and neural (mlp).
    # Used for the primary (Shifts) benchmark, where the backend catalog
    # itself is under comparison.
    "bbm_backends": [
        "ridge", "lasso", "elastic", "bayesian_ridge", "huber", "ols",
        "knn", "svr", "gpr", "mlp", "pnn",
        "dt", "rf", "et", "ada", "bag", "gbt", "histgb",
        "xgboost", "catboost", "lightgbm",
    ],
    # FuelCast (secondary benchmark) keeps the original five-backend set:
    # its role is testing the prior-quality conditional across coupling
    # TYPES on a second, weaker physics prior, not re-running the full
    # backend catalog; Poseidon alone has 42k training rows; sweeping 21
    # backends there would not add evidence beyond what the primary
    # benchmark already establishes about backend diversity.
    "bbm_backends_secondary": ["ridge", "mlp", "pnn", "gbt", "histgb"],
}


# ---------------------------------------------------------------- data ----

def load_split(fname: str) -> pd.DataFrame:
    """Load one Shifts real-data csv into the benchmark schema."""
    d = pd.read_csv(REAL / fname)
    out = pd.DataFrame({
        "time_id": d["time_id"],
        "stw": d["stw"],
        "draft": 0.5 * (d["draft_aft_telegram"] + d["draft_fore_telegram"]),
        "trim": d["draft_aft_telegram"] - d["draft_fore_telegram"],
        "wind_kn": np.hypot(d["awind_ucomp_provider"], d["awind_vcomp_provider"]),
        "wind_ang": np.degrees(np.arctan2(d["awind_vcomp_provider"], d["awind_ucomp_provider"])),
        "cur_kn": np.hypot(d["rcurrent_ucomp"], d["rcurrent_vcomp"]),
        "cur_ang": np.degrees(np.arctan2(d["rcurrent_vcomp"], d["rcurrent_ucomp"])),
        "hs": d["comb_wind_swell_wave_height"],
        "months": d["timeSinceDryDock"] / MIN_PER_MONTH,
        "power": d["power"],
    })
    out = out[(out["stw"] > CONFIG["speed_min_kn"]) & (out["power"] > CONFIG["power_min_kw"])]
    return out.dropna().sort_values("time_id").reset_index(drop=True)


def load_shifts_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, dev_in, dev_out); train uniformly temporally subsampled."""
    tr = load_split("train.csv")
    n = CONFIG["n_train_sub"]
    if len(tr) > n:
        tr = tr.iloc[np.linspace(0, len(tr) - 1, n).astype(int)].reset_index(drop=True)
    return tr, load_split("dev_in.csv"), load_split("dev_out.csv")


# ------------------------------------------------------------ fuelcast ----

FC_DIR = pathlib.Path("data/external/fuelcast")

# candidate benchmark columns -> FuelCast source columns (m/s -> kn where noted)
_FC_MAP = {
    "stw": "Ship_SpeedThroughWater",        # kn after conversion
    "sog": "Ship_SpeedOverGround",
    "draft_aft": "Ship_DraftAft",
    "draft_fore": "Ship_DraftFore",
    "wind10_kn": "Weather_WindSpeed10M",
    "wind10_ang": "Weather_WindDirection10M",
    "hs": "Weather_WaveHeight",
    "tp": "Weather_WavePeriod",
    "swell_hs": "Weather_SwellWaveHeight",
    "cur_kn": "Weather_OceanCurrentVelocity",
    "cur_ang": "Weather_OceanCurrentDirection",
    "depth": "Environment_SeaFloorDepth",
    "temp": "Weather_Temperature2M",
    "press": "Weather_SurfacePressure",
    # heading and wave direction are not ML features; they are needed to form
    # vessel-relative wind and wave angles for the Holtrop-Mennen prior
    "heading": "Ship_Heading",
    "wave_ang": "Weather_WaveDirection",
}
_FC_MS2KN = ("stw", "sog", "wind10_kn", "cur_kn")

# Source columns that are genuinely absent for some vessels, so a missing name
# is legitimate and must degrade to NaN rather than raise. Every other name in
# _FC_MAP must exist in every vessel file; a typo there previously produced a
# silently all-NaN column that the feature filter then dropped without warning
# (wind and air temperature were lost this way).
_FC_OPTIONAL = {"Ship_DraftAft", "Ship_DraftFore", "Ship_SpeedThroughWater",
                "Ship_Heading"}

FC_VESSELS: dict[str, dict[str, Any]] = {
    # min_speed_kn: transit filter; Ceto is the DP negative control (low bar)
    "CPS_Poseidon": {"min_speed_kn": 6.0, "role": "benchmark"},
    "CPS_Triton": {"min_speed_kn": 6.0, "role": "benchmark"},
    "OSS_Ceto": {"min_speed_kn": 2.0, "role": "negative_control"},
}


def load_fuelcast_vessel(name: str) -> tuple[pd.DataFrame, list[str]]:
    """Load one FuelCast vessel into the benchmark schema.

    Returns (frame sorted by the 5-min index, usable feature names).
    Target column ``fuel_kgh`` is the measured Coriolis total fuel (kg/h).
    Speed column ``speed`` prefers STW and falls back to SOG when STW is
    missing or degenerate (Triton's STW sensor is constant; documented).
    """
    d = pd.read_parquet(FC_DIR / f"{name}.parquet")
    missing = [s for s in _FC_MAP.values()
               if s not in d.columns and s not in _FC_OPTIONAL]
    if missing:
        raise KeyError(
            f"{name}.parquet is missing required FuelCast columns {missing}; "
            f"check _FC_MAP against the file schema (names are case sensitive)")
    out = pd.DataFrame({"idx": d["index"].to_numpy(dtype=float)})
    for col, src in _FC_MAP.items():
        v = d[src].to_numpy(dtype=float) if src in d.columns else np.full(len(d), np.nan)
        if col in _FC_MS2KN:
            v = v * 1.943844
        out[col] = v
    stw = out["stw"]
    use_stw = stw.notna().any() and float(np.nanstd(stw)) > 0.1
    out["speed"] = out["stw"] if use_stw else out["sog"]
    out["draft"] = 0.5 * (out["draft_aft"] + out["draft_fore"])
    out["trim"] = out["draft_aft"] - out["draft_fore"]
    out["fuel_kgh"] = d["Consumer_Total_MomentaryFuel"].to_numpy(dtype=float) * 3600.0

    min_kn = FC_VESSELS[name]["min_speed_kn"]
    out = out[(out["speed"] > min_kn) & (out["fuel_kgh"] > 0)]
    # the temporal protocol orders on idx, so a row with no index has no
    # position in time and cannot be assigned to a split (Triton and Ceto
    # carry NaN in the index column; Poseidon does not)
    out = out.dropna(subset=["idx"]).sort_values("idx").reset_index(drop=True)

    candidates = ["speed", "draft", "trim", "wind10_kn", "wind10_ang", "hs", "tp",
                  "swell_hs", "cur_kn", "cur_ang", "depth", "temp", "press"]
    feats = [c for c in candidates
             if out[c].notna().mean() > 0.95 and float(out[c].std()) > 1e-9]
    return out.dropna(subset=feats + ["fuel_kgh"]).reset_index(drop=True), feats


def fuelcast_shift_splits(
    df: pd.DataFrame, shift_var: str = "wind10_kn", q: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Temporal + weather-quantile shift splits mirroring the Shifts protocol.

    train = first 70% (temporal); of the last 30%: dev_in = rows at or below
    the train's q-quantile of ``shift_var``, dev_out = rows above it (rough-
    weather shift). Falls back to ``hs`` if the wind column is unusable or a
    side would be degenerate (<300 rows).
    """
    n_tr = int(0.70 * len(df))
    tr, te = df.iloc[:n_tr], df.iloc[n_tr:]
    for var in (shift_var, "hs"):
        if var not in df.columns or tr[var].isna().all():
            continue
        thr = float(tr[var].quantile(q))
        din, dout = te[te[var] <= thr], te[te[var] > thr]
        if len(din) >= 300 and len(dout) >= 300:
            return (tr.reset_index(drop=True), din.reset_index(drop=True),
                    dout.reset_index(drop=True), var)
    return tr.reset_index(drop=True), te.reset_index(drop=True), te.iloc[0:0], "none"


def fit_cube(y_train: np.ndarray, v_train: np.ndarray) -> float:
    """Admiralty-type cube-law coefficient a (train only): y ~ a * v^3."""
    return float(np.sum(y_train * v_train ** 3) / np.sum(v_train ** 6))


# ------------------------------------------------------------- physics ----

def physics_power(df: pd.DataFrame, tag: str | None = None) -> np.ndarray:
    """Uncalibrated Holtrop shaft power (kW) per row; cached per (tag, n)."""
    if tag is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / f"phys_{tag}_{len(df)}.npz"
        if f.exists():
            return np.load(f)["p"]
    part = CONFIG["particulars"]
    out = np.empty(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        env = {"wind_kn": r.wind_kn, "wind_angle_deg": r.wind_ang, "Hs": r.hs,
               "Tp": 8.0, "wave_angle_deg": 0.0, "current_kn": r.cur_kn,
               "current_angle_deg": r.cur_ang, "sst_c": 12.0, "depth_m": 2000.0}
        prm = dict(part, T=float(r.draft), trim_m=float(r.trim),
                   months_since_cleaning=float(r.months))
        out[i] = float(fuel_components(100.0, float(r.stw), env, prm)["shaft_power_kw"])
    if tag is not None:
        np.savez_compressed(CACHE / f"phys_{tag}_{len(df)}.npz", p=out)
    return out


def fit_scale(y_train: np.ndarray, phys_train: np.ndarray) -> float:
    """Least-squares single scale factor k (train split only)."""
    return float(np.sum(y_train * phys_train) / np.sum(phys_train ** 2))


# ------------------------------------------------------------- metrics ----

def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err / y)) * 100.0),
        "R2": float(1.0 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
    }


# ------------------------------------------------------------- learners ----

class _ChunkedPredict:
    """Wrap a learner so predict() runs in row chunks (PNN memory)."""

    def __init__(self, learner: Any, chunk: int = 2000):
        self._l, self._chunk = learner, chunk

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_ChunkedPredict":
        self._l.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([self._l.predict(x[i:i + self._chunk])
                               for i in range(0, len(x), self._chunk)])


# Documented training-set caps for backends whose fit cost is superlinear in
# n and would be impractical at the 34,000-record training size used here.
# PNN's leave-one-out bandwidth search is O(n^2); SVR's dual solve is
# O(n^2)-O(n^3); Gaussian-process regression factorises an n x n kernel
# matrix, O(n^3), the most restrictive of the three.
SUBSAMPLE_CAP: dict[str, int] = {
    "pnn": CONFIG["pnn_train_sub"],
    "svr": 4_000,
    "gpr": 1_200,
}

# Display names for tables and printouts.
BACKEND_DISPLAY: dict[str, str] = {
    "ridge": "Ridge", "lasso": "Lasso", "elastic": "ElasticNet",
    "bayesian_ridge": "BayesianRidge", "huber": "Huber", "ols": "OLS",
    "mlp": "MLP", "pnn": "PNN (GRNN)", "knn": "KNN", "svr": "SVR (RBF)",
    "gpr": "GaussianProcess", "dt": "DecisionTree", "rf": "RandomForest",
    "et": "ExtraTrees", "ada": "AdaBoost", "bag": "Bagging",
    "gbt": "GBT", "histgb": "HistGB",
    "xgboost": "XGBoost", "catboost": "CatBoost", "lightgbm": "LightGBM",
}


def make_learner(name: str, rng: np.random.Generator) -> tuple[Any, Callable]:
    """Return (learner, subsampler) for a backend name.

    The subsampler maps (X, y) -> (X', y') applied to the TRAIN split only;
    identity except for the backends listed in SUBSAMPLE_CAP.
    """
    if name == "histgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=SEED, max_iter=400), lambda X, y: (X, y)

    cap = SUBSAMPLE_CAP.get(name)
    if cap is not None:
        def sub(X: np.ndarray, y: np.ndarray, n: int = cap) -> tuple[np.ndarray, np.ndarray]:
            if len(X) <= n:
                return X, y
            idx = rng.choice(len(X), size=n, replace=False)
            return X[idx], y[idx]

        if name == "pnn":
            return _ChunkedPredict(_make_learner("pnn")), sub
        return _make_learner(name, seed=SEED, random_state=SEED), sub

    return _make_learner(name, seed=SEED, random_state=SEED), lambda X, y: (X, y)


# ------------------------------------------- physical-inconsistency index ----

def speed_sweep_frame(train: pd.DataFrame, n: int = 60,
                      v_lo: float = 6.0, v_hi: float = 22.0) -> pd.DataFrame:
    """Fixed-condition speed sweep (median train conditions, varying stw)."""
    med = train[FEATS].median()
    grid = pd.DataFrame({c: np.full(n, med[c]) for c in FEATS})
    grid["stw"] = np.linspace(v_lo, v_hi, n)
    return grid


def inconsistency_index(predict: Callable[[pd.DataFrame], np.ndarray],
                        sweep: pd.DataFrame,
                        speed_col: str = "stw") -> dict[str, float]:
    """Monotonicity violations + fitted propeller-law exponent on the sweep.

    Reuses the physical-consistency metric concept from the recent literature
    (cited in the paper); target exponent for displacement hulls is ~2.5-3.5.
    """
    p = np.asarray(predict(sweep), dtype=float)
    v = sweep[speed_col].to_numpy()
    dp = np.diff(p)
    viol = float(np.mean(dp < 0.0))
    ok = p > 0
    if ok.sum() >= 10:
        b = float(np.polyfit(np.log(v[ok]), np.log(p[ok]), 1)[0])
    else:
        b = float("nan")
    return {"mono_violation_frac": viol, "propeller_exponent": b}
