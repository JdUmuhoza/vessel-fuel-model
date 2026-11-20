"""Data pipeline for AIS/metocean/engine fusion workflows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from vessel_fuel.model import fuel_model

REQUIRED_KEYS = ("distance_km", "speed_tw_kn", "env", "vessel_params", "fuel_mt")


def quality_flag_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Attach quality flags for missingness and plausible ranges."""
    out = dict(obs)
    flags = {
        "missing_env": int("env" not in obs or obs.get("env") is None),
        "missing_vessel": int("vessel_params" not in obs or obs.get("vessel_params") is None),
        "invalid_speed": int(float(obs.get("speed_tw_kn", 0.0)) < 0.0 or float(obs.get("speed_tw_kn", 0.0)) > 35.0),
        "invalid_distance": int(float(obs.get("distance_km", 0.0)) <= 0.0),
        "invalid_fuel": int(float(obs.get("fuel_mt", 0.0)) < 0.0),
    }
    out["quality_flags"] = flags
    out["quality_score"] = 1.0 - min(sum(flags.values()) / 5.0, 1.0)
    return out


def clean_observations(observations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Impute missing values and filter anomalies.

    Strategy
    --------
    - Apply quality flags.
    - Remove records with structurally invalid core fields.
    - Median-impute selected environmental values.
    - Clip gross outliers by median absolute deviation.
    """
    flagged = [quality_flag_observation(o) for o in observations]

    kept = [
        o
        for o in flagged
        if o["quality_flags"]["missing_env"] == 0
        and o["quality_flags"]["missing_vessel"] == 0
        and o["quality_flags"]["invalid_speed"] == 0
        and o["quality_flags"]["invalid_distance"] == 0
        and o["quality_flags"]["invalid_fuel"] == 0
    ]

    if len(kept) == 0:
        return []

    env_keys = ["wind_kn", "wind_angle_deg", "Hs", "Tp", "current_kn", "current_angle_deg", "sst_c", "depth_m", "wave_angle_deg"]
    med: dict[str, float] = {}
    for k in env_keys:
        vals = [float(o["env"].get(k, np.nan)) for o in kept]
        arr = np.asarray(vals, dtype=float)
        med[k] = float(np.nanmedian(arr)) if np.isfinite(np.nanmedian(arr)) else 0.0

    cleaned: list[dict[str, Any]] = []
    for o in kept:
        oo = dict(o)
        env = dict(o["env"])
        for k in env_keys:
            v = float(env.get(k, np.nan))
            if not np.isfinite(v):
                env[k] = med[k]
        oo["env"] = env
        cleaned.append(oo)

    fuel = np.asarray([float(o["fuel_mt"]) for o in cleaned], dtype=float)
    med_f = float(np.median(fuel))
    mad = float(np.median(np.abs(fuel - med_f))) + 1e-9
    lo, hi = med_f - 6.0 * mad, med_f + 6.0 * mad
    return [o for o in cleaned if lo <= float(o["fuel_mt"]) <= hi]


def fit_category_maps(observations: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Build ordinal encodings for categorical fields from a training set.

    Call this once on the training split and pass the result to
    :func:`build_feature_matrix` for every split (train, val, test) so that
    the same category always receives the same integer code.

    Parameters
    ----------
    observations : iterable of dict
        Training observations. Only the ``vessel_class``, ``route_id``, and
        ``season`` fields are used.

    Returns
    -------
    dict
        Mapping ``{"vessel_class": {...}, "route_id": {...}, "season": {...}}``.
        Unknown categories encountered at inference time are mapped to ``-1``.
    """
    obs = list(observations)
    vessel_classes = sorted({str(o.get("vessel_class", "UNK")) for o in obs})
    routes         = sorted({str(o.get("route_id",     "UNK")) for o in obs})
    seasons        = sorted({str(o.get("season",       "UNK")) for o in obs})
    return {
        "vessel_class": {c: i for i, c in enumerate(vessel_classes)},
        "route_id":     {r: i for i, r in enumerate(routes)},
        "season":       {s: i for i, s in enumerate(seasons)},
    }


def build_feature_matrix(
    observations: Iterable[dict[str, Any]],
    cat_maps: dict[str, dict[str, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build deterministic tabular features for ML models.

    Parameters
    ----------
    observations : iterable of dict
        Voyage observations.
    cat_maps : dict, optional
        Ordinal encodings returned by :func:`fit_category_maps`.
        When ``None``, encodings are built from *observations* (fine for a
        single-split call, but train/val/test splits may get inconsistent codes
        — always pass training ``cat_maps`` for multi-split workflows).
    """
    obs = list(observations)
    if len(obs) == 0:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float), []

    # Ordinal-encode categorical fields.
    # Use caller-supplied maps (fitted on train) so codes are consistent across splits.
    if cat_maps is not None:
        _vc_map  = cat_maps["vessel_class"]
        _rt_map  = cat_maps["route_id"]
        _sea_map = cat_maps["season"]
    else:
        _vc_map  = {c: i for i, c in enumerate(sorted({str(o.get("vessel_class", "UNK")) for o in obs}))}
        _rt_map  = {r: i for i, r in enumerate(sorted({str(o.get("route_id",     "UNK")) for o in obs}))}
        _sea_map = {s: i for i, s in enumerate(sorted({str(o.get("season",       "UNK")) for o in obs}))}

    rows: list[list[float]] = []
    y: list[float] = []
    for o in obs:
        e = o["env"]
        v = o["vessel_params"]
        speed = float(o["speed_tw_kn"])
        dist = float(o["distance_km"])
        hs = float(e.get("Hs", 0.0))
        wind = float(e.get("wind_kn", 0.0))
        wa = np.deg2rad(float(e.get("wind_angle_deg", 0.0)))
        ca = np.deg2rad(float(e.get("current_angle_deg", 180.0)))
        tp = float(e.get("Tp", 0.0))
        current_kn = float(e.get("current_kn", 0.0))
        depth_m = float(e.get("depth_m", 0.0))
        sst_c = float(e.get("sst_c", 15.0))

        L = float(v.get("L", 100.0))
        B = float(v.get("B", 18.0))
        T = float(v.get("T", 6.0))
        CB = float(v.get("CB", 0.70))
        S = float(v.get("S", 2.2 * (L * B + L * T)))
        Cp = float(v.get("Cp", 0.68))
        Cm = float(v.get("Cm", 0.98))
        Cwp = float(v.get("Cwp", 0.82))
        months = float(v.get("months_since_cleaning", 0.0))
        trim = float(v.get("trim_m", 0.0))
        design_draft = float(v.get("design_draft", T))
        A_front = float(v.get("A_front", B * T))
        A_lateral = float(v.get("A_lateral", L * T))
        Cd_air = float(v.get("Cd_air", 1.0))
        MCR = float(v.get("MCR", 10000.0))
        sfoc_at_mcr = float(v.get("sfoc_at_mcr", 180.0))
        eta_0 = float(v.get("eta_0", 0.72))
        eta_wave_loss = float(v.get("eta_wave_loss", 0.03))
        include_aux = 1.0 if bool(v.get("include_aux", False)) else 0.0
        aux_power_kw = float(v.get("aux_power_kw", 0.0))
        include_boiler = 1.0 if bool(v.get("include_boiler", False)) else 0.0
        boiler_power_kw = float(v.get("boiler_power_kw", 0.0))

        # -1 signals "unseen category" to tree models; ridge treats it as a small negative
        vc_code  = float(_vc_map.get( str(o.get("vessel_class", "UNK")), -1))
        rt_code  = float(_rt_map.get( str(o.get("route_id",     "UNK")), -1))
        sea_code = float(_sea_map.get(str(o.get("season",       "UNK")), -1))

        row = [
            dist,
            speed,
            speed**2,
            speed**3,
            hs,
            tp,
            wind,
            np.cos(wa),
            np.sin(wa),
            current_kn,
            np.cos(ca),
            np.sin(ca),
            depth_m,
            sst_c,
            L,
            B,
            T,
            CB,
            S,
            Cp,
            Cm,
            Cwp,
            months,
            trim,
            design_draft,
            A_front,
            A_lateral,
            Cd_air,
            MCR,
            sfoc_at_mcr,
            eta_0,
            eta_wave_loss,
            include_aux,
            aux_power_kw,
            include_boiler,
            boiler_power_kw,
            speed * hs,
            speed * wind,
            speed * current_kn,
            dist * hs,
            dist * wind,
            trim * (T - design_draft),
            # Operational context — ordinal-encoded for tree models
            vc_code,
            rt_code,
            sea_code,
        ]
        rows.append([float(x) for x in row])
        y.append(float(o["fuel_mt"]))

    names = [
        "distance_km",
        "speed_kn",
        "speed2",
        "speed3",
        "Hs",
        "Tp",
        "wind_kn",
        "cos_wind",
        "sin_wind",
        "current_kn",
        "cos_current",
        "sin_current",
        "depth_m",
        "sst_c",
        "L",
        "B",
        "T",
        "CB",
        "S",
        "Cp",
        "Cm",
        "Cwp",
        "months_since_cleaning",
        "trim_m",
        "design_draft",
        "A_front",
        "A_lateral",
        "Cd_air",
        "MCR",
        "sfoc_at_mcr",
        "eta_0",
        "eta_wave_loss",
        "include_aux",
        "aux_power_kw",
        "include_boiler",
        "boiler_power_kw",
        "speed_x_Hs",
        "speed_x_wind",
        "speed_x_current",
        "distance_x_Hs",
        "distance_x_wind",
        "trim_x_draft_offset",
        "vessel_class_code",
        "route_code",
        "season_code",
    ]
    return np.asarray(rows, dtype=float), np.asarray(y, dtype=float), names


def split_train_val_test(
    observations: Iterable[dict[str, Any]],
    seed: int = 42,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict[str, list[dict[str, Any]]]:
    """Group-aware split to reduce leakage across operational regimes.

    Group key: (vessel_class, route_id, season).
    """
    obs = list(observations)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for o in obs:
        key = (str(o.get("vessel_class", "UNK")), str(o.get("route_id", "UNK")), str(o.get("season", "UNK")))
        buckets[key].append(o)

    keys = list(buckets.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n = len(keys)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    if n == 1:
        n_train, n_val = 1, 0
    elif n == 2:
        n_train, n_val = 1, 0
    else:
        n_train = max(1, int(np.floor(train_frac * n)))
        n_val = max(1, int(np.floor(val_frac * n)))
        if n_train + n_val >= n:
            n_val = 1
            n_train = max(1, n - 2)

    train_k = set(keys[:n_train])
    val_k = set(keys[n_train : n_train + n_val])
    test_k = set(keys[n_train + n_val :])

    def gather(ks: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
        return [x for k in ks for x in buckets[k]]

    return {"train": gather(train_k), "val": gather(val_k), "test": gather(test_k)}


def generate_synthetic_operational_dataset(n_samples: int = 1500, seed: int = 42) -> list[dict[str, Any]]:
    r"""Generate synthetic AIS-like records with latent residual structure.

    The target is created as:
    $y = y_{physics} + f_{residual}(x) + \epsilon$.
    """
    rng = np.random.default_rng(seed)
    vessel_classes = ["tanker", "bulker", "container"]
    routes = ["atlantic", "pacific", "indian"]
    seasons = ["winter", "spring", "summer", "autumn"]

    all_obs: list[dict[str, Any]] = []
    for i in range(n_samples):
        vc = vessel_classes[i % len(vessel_classes)]
        route = routes[rng.integers(0, len(routes))]
        season = seasons[rng.integers(0, len(seasons))]

        L = float(rng.uniform(110.0, 280.0))
        B = float(rng.uniform(18.0, 42.0))
        T = float(rng.uniform(6.0, 14.0))
        cb = float(rng.uniform(0.62, 0.84))
        speed = float(rng.uniform(7.0, 18.0))
        dist = float(rng.uniform(60.0, 260.0))
        hs = float(rng.uniform(0.0, 4.5))
        wind = float(rng.uniform(0.0, 35.0))

        env = {
            "wind_kn": wind,
            "wind_angle_deg": float(rng.uniform(0.0, 180.0)),
            "Hs": hs,
            "Tp": float(rng.uniform(4.0, 13.0)),
            "current_kn": float(rng.uniform(0.0, 3.0)),
            "current_angle_deg": float(rng.uniform(0.0, 180.0)),
            "sst_c": float(rng.uniform(-1.0, 30.0)),
            "depth_m": float(rng.uniform(9.0, 400.0)),
            "wave_angle_deg": float(rng.uniform(0.0, 180.0)),
            "wave_method": "kwon" if rng.uniform() < 0.45 else "stawave1",
        }

        vessel = {
            "L": L,
            "B": B,
            "T": T,
            "design_draft": float(rng.uniform(0.95 * T, 1.05 * T)),
            "trim_m": float(rng.uniform(0.0, 1.2)),
            "S": float(2.2 * (L * B + L * T)),
            "CB": cb,
            "Cp": float(rng.uniform(0.62, 0.76)),
            "Cm": float(rng.uniform(0.95, 0.99)),
            "Cwp": float(rng.uniform(0.78, 0.92)),
            "MCR": float(rng.uniform(7000.0, 30000.0)),
            "sfoc_at_mcr": float(rng.uniform(165.0, 190.0)),
            "eta_0": float(rng.uniform(0.65, 0.78)),
            "eta_wave_loss": float(rng.uniform(0.02, 0.05)),
            "A_front": float(B * T * rng.uniform(0.9, 1.3)),
            "A_lateral": float(L * T * rng.uniform(0.8, 1.3)),
            "Cd_air": float(rng.uniform(0.85, 1.2)),
            "months_since_cleaning": float(rng.uniform(0.0, 36.0)),
            "include_aux": bool(rng.uniform() < 0.7),
            "aux_power_kw": float(rng.uniform(250.0, 1200.0)),
            "aux_sfoc": float(rng.uniform(195.0, 220.0)),
            "include_boiler": bool(rng.uniform() < 0.5),
            "boiler_power_kw": float(rng.uniform(80.0, 500.0)),
            "boiler_sfoc": float(rng.uniform(250.0, 310.0)),
        }

        phys = float(fuel_model(dist, speed, env, vessel))

        # latent non-linear residual pattern (unseen by physics core)
        seasonal = {"winter": 0.12, "spring": 0.03, "summer": -0.04, "autumn": 0.05}[season]
        route_term = {"atlantic": 0.06, "pacific": 0.01, "indian": 0.04}[route]
        class_term = {"tanker": 0.03, "bulker": 0.01, "container": 0.05}[vc]
        residual = phys * (0.03 * np.sin(speed / 2.5) + 0.02 * np.cos(np.deg2rad(env["wind_angle_deg"])) + seasonal + route_term + class_term)
        noise = rng.normal(0.0, 0.04 * max(phys, 1e-6))
        fuel_mt = max(0.0, phys + residual + noise)

        all_obs.append(
            {
                "sample_id": i,
                "vessel_id": f"V{rng.integers(1, 90):03d}",
                "vessel_class": vc,
                "route_id": route,
                "season": season,
                "distance_km": dist,
                "speed_tw_kn": speed,
                "env": env,
                "vessel_params": vessel,
                "fuel_mt": float(fuel_mt),
            }
        )

    return all_obs
