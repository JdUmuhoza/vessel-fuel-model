"""End-to-end benchmark and ablation orchestration."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .data_pipeline import build_feature_matrix, fit_category_maps, split_train_val_test
from .hybrid import (
    CubicLawModel,
    HybridResidualModel,
    LSTMModel,
    PhysicsOnlyModel,
    ProphetModel,
    PureMLModel,
    ReferencePhysicsReducedModel,
    SpeedPowerBaseline,
    SymbolicRegressionModel,
)
from .metrics import paired_bootstrap_pvalue, regression_metrics
from .decision_matrix import generate_decision_matrix_report
from .statistical_analysis import kfold_cross_validate, model_selection_report
from .reporting import generate_confusion_matrices
from .tracking import ExperimentTracker
from .uncertainty import conformal_interval, interval_coverage, permutation_sensitivity


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if len(rows) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _set_nested_feature(obs: dict[str, Any], feature_name: str, value: float) -> None:
    if feature_name in {"distance_km", "speed_tw_kn"}:
        obs[feature_name] = float(value)
        return

    env_map = {
        "wind_kn",
        "wind_angle_deg",
        "Hs",
        "Tp",
        "current_kn",
        "current_angle_deg",
        "sst_c",
        "depth_m",
        "wave_angle_deg",
    }
    if feature_name in env_map:
        env = dict(obs.get("env", {}))
        env[feature_name] = float(value)
        obs["env"] = env
        return

    vp = dict(obs.get("vessel_params", {}))
    vp[feature_name] = float(value)
    obs["vessel_params"] = vp


def _get_nested_feature(obs: dict[str, Any], feature_name: str) -> float:
    if feature_name in {"distance_km", "speed_tw_kn"}:
        return float(obs.get(feature_name, 0.0))

    if feature_name in {
        "wind_kn",
        "wind_angle_deg",
        "Hs",
        "Tp",
        "current_kn",
        "current_angle_deg",
        "sst_c",
        "depth_m",
        "wave_angle_deg",
    }:
        return float(dict(obs.get("env", {})).get(feature_name, 0.0))

    return float(dict(obs.get("vessel_params", {})).get(feature_name, 0.0))


def _profile_raw_features(observations: list[dict[str, Any]], features: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in features:
        vals = np.asarray([_get_nested_feature(o, name) for o in observations], dtype=float)
        out.append(
            {
                "feature": name,
                "min": float(np.min(vals)) if len(vals) else 0.0,
                "max": float(np.max(vals)) if len(vals) else 0.0,
                "mean": float(np.mean(vals)) if len(vals) else 0.0,
                "std": float(np.std(vals)) if len(vals) else 0.0,
            }
        )
    return out


def _total_model_permutation_sensitivity(
    hybrid: HybridResidualModel,
    observations: list[dict[str, Any]],
    y_true: np.ndarray,
    cat_maps: dict[str, dict[str, int]] | None = None,
    n_repeats: int = 10,
    seed: int = 42,
) -> list[tuple[str, float]]:
    """Permutation sensitivity for total hybrid prediction (physics + residual).

    This permutation happens on raw observation fields, then predictions are
    recomputed using both the physics and residual terms.
    """
    if len(observations) == 0:
        return []

    raw_features = [
        "distance_km",
        "speed_tw_kn",
        "wind_kn",
        "wind_angle_deg",
        "Hs",
        "Tp",
        "current_kn",
        "current_angle_deg",
        "sst_c",
        "depth_m",
        "wave_angle_deg",
        "L",
        "B",
        "T",
        "S",
        "CB",
        "months_since_cleaning",
        "trim_m",
        "MCR",
        "sfoc_at_mcr",
    ]

    x_base, _, _ = build_feature_matrix(observations, cat_maps=cat_maps)
    base_pred = hybrid.predict(observations, x_base)
    base_mae = float(regression_metrics(y_true, base_pred)["mae"])

    rng = np.random.default_rng(seed)
    sens: list[tuple[str, float]] = []
    for feat in raw_features:
        source = np.asarray([_get_nested_feature(o, feat) for o in observations], dtype=float)
        if np.std(source) < 1e-12:
            sens.append((feat, 0.0))
            continue

        deltas: list[float] = []
        for _ in range(n_repeats):
            idx = np.arange(len(observations))
            rng.shuffle(idx)

            obs_perm = [deepcopy(o) for o in observations]
            for i, j in enumerate(idx):
                _set_nested_feature(obs_perm[i], feat, float(source[j]))

            x_perm, _, _ = build_feature_matrix(obs_perm, cat_maps=cat_maps)
            pred = hybrid.predict(obs_perm, x_perm)
            mae = float(regression_metrics(y_true, pred)["mae"])
            deltas.append(mae - base_mae)

        sens.append((feat, float(np.mean(deltas))))

    return sorted(sens, key=lambda t: t[1], reverse=True)


def run_benchmark_suite(
    observations: list[dict[str, Any]],
    out_dir: str | Path,
    seed: int = 42,
    alpha: float = 0.1,
    external_baselines: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Run baseline comparison, significance, uncertainty, and ablation.

    Returns
    -------
    dict
        Structured results for publication tables.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(out, run_name="experiment_log")
    tracker.log({"event": "start", "seed": seed, "n_samples": len(observations)})

    splits = split_train_val_test(observations, seed=seed)
    train, val, test = splits["train"], splits["val"], splits["test"]

    # Fit category maps on training data so all splits use consistent ordinal codes
    cat_maps = fit_category_maps(train)

    x_train, y_train, feature_names = build_feature_matrix(train, cat_maps=cat_maps)
    x_val, y_val, _ = build_feature_matrix(val, cat_maps=cat_maps)
    x_test, y_test, _ = build_feature_matrix(test, cat_maps=cat_maps)

    # Baseline 1: speed-power law
    baseline = SpeedPowerBaseline().fit(train)
    y_b = baseline.predict(test)

    # Baseline 1b: fixed cubic law
    cubic = CubicLawModel().fit(train)
    y_cubic = cubic.predict(test)

    # Baseline 2: pure ML (Ridge)
    pure_ml = PureMLModel(learner_kwargs={"lam": 0.05}).fit(x_train, y_train)
    y_ml = pure_ml.predict_from_matrix(x_test)

    # Baseline 3: physics only
    physics = PhysicsOnlyModel().fit(train)
    y_phys = physics.predict(test)

    # Baseline 4: reduced-feature reference physics
    physics_reduced = ReferencePhysicsReducedModel().fit(train)
    y_phys_reduced = physics_reduced.predict(test)

    # Target model: hybrid residual (Ridge)
    hybrid = HybridResidualModel(learner_kwargs={"lam": 0.05}).fit(train, x_train, y_train)
    y_h = hybrid.predict(test, x_test)

    predictions: dict[str, np.ndarray] = {
        "speed_power": y_b,
        "cubic_law": y_cubic,
        "physics_only": y_phys,
        "physics_reduced": y_phys_reduced,
        "pure_ml_ridge": y_ml,
        "hybrid_ridge": y_h,
    }
    model_labels = {
        "speed_power": "Speed-Power Baseline (fitted exponent)",
        "cubic_law": "Cubic Law Baseline (v^3)",
        "physics_only": "Physics-Only",
        "physics_reduced": "Reference Physics (Reduced)",
        "pure_ml_ridge": "Pure ML (Ridge)",
        "hybrid_ridge": "Hybrid (Physics + Ridge)",
    }

    # Additional ML backends — each wrapped so a missing library skips gracefully
    _extra_learners = [
        ("mlp",            {"epochs": 300},                          "MLP"),
        ("pnn",            {"cv_bandwidths": 20, "seed": seed},      "PNN"),
        ("xgboost",        {"n_estimators": 300, "max_depth": 4},    "XGBoost"),
        ("catboost",       {"iterations": 300, "depth": 4},          "CatBoost"),
        ("lightgbm",       {"n_estimators": 300, "max_depth": 4},    "LightGBM"),
        ("rf",             {"n_estimators": 200},                    "Random Forest"),
        ("gbt",            {"n_estimators": 200, "max_depth": 3},    "Gradient Boosting"),
        ("ols",            {},                                        "OLS"),
        ("ridge_sklearn",  {"alpha": 1.0},                           "Ridge (sklearn)"),
        ("bayesian_ridge", {"max_iter": 300},                        "Bayesian Ridge"),
        ("knn",            {"n_neighbors": 5},                       "k-NN"),
        ("svr",            {"C": 10.0, "kernel": "rbf"},             "SVR"),
        ("quantile",       {"quantile": 0.5, "alpha": 0.0},          "Quantile Regression"),
        ("gpr",            {"n_restarts_optimizer": 3,
                            "random_state": seed},                   "Gaussian Process"),
    ]
    for learner_name, lkw, label in _extra_learners:
        try:
            pm = PureMLModel(learner=learner_name, learner_kwargs=lkw).fit(x_train, y_train)
            predictions[f"pure_ml_{learner_name}"] = pm.predict_from_matrix(x_test)
            model_labels[f"pure_ml_{learner_name}"] = f"Pure ML ({label})"

            hm = HybridResidualModel(learner=learner_name, learner_kwargs=lkw).fit(train, x_train, y_train)
            predictions[f"hybrid_{learner_name}"] = hm.predict(test, x_test)
            model_labels[f"hybrid_{learner_name}"] = f"Hybrid (Physics + {label})"
        except Exception:
            pass  # library not installed or training error — skip silently

    # Standalone models requiring optional heavy dependencies
    # Each entry: (learner_key, learner_kwargs, display_label)
    _standalone_configs: list[tuple[str, dict, str]] = [
        ("lstm",     {"seed": seed},                              "LSTM"),
        ("prophet",  {},                                           "Prophet"),
        ("symbolic", {"generations": 10, "random_state": seed},  "Symbolic Regression"),
    ]
    _standalone_instances: dict[str, Any] = {
        "lstm":     LSTMModel(seed=seed),
        "prophet":  ProphetModel(),
        "symbolic": SymbolicRegressionModel(generations=10, random_state=seed),
    }
    for key, lkw, label in _standalone_configs:
        # Pure-ML version
        try:
            _sm = _standalone_instances[key]
            _sm.fit(x_train, y_train)
            predictions[f"pure_ml_{key}"] = np.maximum(
                _sm.predict(x_test), 0.0
            )
            model_labels[f"pure_ml_{key}"] = f"Pure ML ({label})"
        except Exception:
            pass  # package not installed — skip silently

        # Hybrid version: physics baseline + this learner on the residual
        try:
            hm_s = HybridResidualModel(learner=key, learner_kwargs=lkw).fit(
                train, x_train, y_train
            )
            predictions[f"hybrid_{key}"] = hm_s.predict(test, x_test)
            model_labels[f"hybrid_{key}"] = f"Hybrid (Physics + {label})"
        except Exception:
            pass  # package not installed or training error — skip silently

    # Keep backward-compatible keys for significance / uncertainty steps below
    y_ml = predictions.get("pure_ml_ridge", y_ml)
    y_h  = predictions.get("hybrid_ridge",  y_h)

    # Also keep "pure_ml" and "hybrid" as aliases for Ridge so downstream code works
    predictions["pure_ml"] = y_ml
    predictions["hybrid"]   = y_h
    model_labels["pure_ml"] = "Pure ML (Ridge)"
    model_labels["hybrid"]  = "Hybrid (Physics + Ridge)"

    # Optional external/frozen/vendor baselines by segment_id mapping
    if external_baselines:
        test_ids = [str(o.get("segment_id", "")) for o in test]
        for name, sid_to_pred in external_baselines.items():
            if len(test_ids) == 0:
                continue
            if all((sid in sid_to_pred) for sid in test_ids):
                predictions[name] = np.asarray([float(sid_to_pred[sid]) for sid in test_ids], dtype=float)

    # Metrics
    rows = []
    for name, pred in predictions.items():
        m = regression_metrics(y_test, pred)
        rows.append(
            {
                "model": name,
                "model_display": model_labels.get(name, name),
                "is_primary_model": int(name == "hybrid"),
                **{k: float(v) for k, v in m.items()},
            }
        )
    _write_csv(out / "results_table.csv", rows)

    test_ids = [str(o.get("segment_id", f"test_{i}")) for i, o in enumerate(test)]
    pred_rows: list[dict[str, Any]] = []
    for i, sid in enumerate(test_ids):
        row: dict[str, Any] = {
            "segment_id": sid,
            "y_true": float(y_test[i]),
        }
        for model_name, pred in predictions.items():
            row[f"pred_{model_name}"] = float(pred[i])
        pred_rows.append(row)
    _write_csv(out / "predictions_by_segment.csv", pred_rows)

    # Statistical significance: hybrid vs physics-only and pure ML
    sig_phys = paired_bootstrap_pvalue(y_test, y_phys, y_h, metric="mae", seed=seed)
    sig_ml = paired_bootstrap_pvalue(y_test, y_ml, y_h, metric="mae", seed=seed)

    with (out / "significance.json").open("w", encoding="utf-8") as f:
        json.dump({"hybrid_vs_physics": sig_phys, "hybrid_vs_pure_ml": sig_ml}, f, indent=2)

    # Uncertainty with split conformal (calibration split = val)
    y_val_h = hybrid.predict(val, x_val)
    lo, hi = conformal_interval(y_val, y_val_h, y_h, alpha=alpha)
    coverage = interval_coverage(y_test, lo, hi)

    interval_rows = [
        {"y_true": float(t), "y_pred": float(p), "lo": float(l), "hi": float(h)}
        for t, p, l, h in zip(y_test, y_h, lo, hi)
    ]
    _write_csv(out / "prediction_intervals.csv", interval_rows)

    # Sensitivity on hybrid residual term (matrix-only view)
    sens_residual = permutation_sensitivity(lambda xx: hybrid.predict(test, xx), x_test, y_test, feature_names, seed=seed)
    _write_csv(out / "sensitivity_residual_ranking.csv", [{"feature": k, "delta_mae": v} for k, v in sens_residual])

    # Sensitivity on full hybrid prediction (physics + residual)
    sens_total = _total_model_permutation_sensitivity(hybrid, test, y_test, cat_maps=cat_maps, seed=seed)
    _write_csv(out / "sensitivity_ranking.csv", [{"feature": k, "delta_mae": v} for k, v in sens_total])

    raw_profile_features = [
        "distance_km",
        "speed_tw_kn",
        "wind_kn",
        "Hs",
        "Tp",
        "current_kn",
        "sst_c",
        "depth_m",
        "L",
        "B",
        "T",
        "S",
        "CB",
        "months_since_cleaning",
        "trim_m",
    ]
    _write_csv(out / "feature_profile_test_split.csv", _profile_raw_features(test, raw_profile_features))

    # Ablation: remove one feature block at a time from hybrid residual learner
    blocks = {
        "no_wind_features": ["wind_kn", "cos_wind", "sin_wind"],
        "no_wave_features": ["Hs", "Tp"],
        "no_current_features": ["current_kn", "cos_current", "sin_current"],
        "no_loading_features": ["T", "CB", "trim_m", "design_draft"],
        "no_fouling_feature": ["months_since_cleaning"],
    }

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    ablation_rows = []
    for label, feats in blocks.items():
        keep_idx = [i for n, i in name_to_idx.items() if n not in feats]
        xtr = x_train[:, keep_idx]
        xte = x_test[:, keep_idx]

        ab_model = HybridResidualModel(learner_kwargs={"lam": 0.05}).fit(train, xtr, y_train)
        yp = ab_model.predict(test, xte)
        m = regression_metrics(y_test, yp)
        ablation_rows.append({"ablation": label, **m})

    _write_csv(out / "ablation_table.csv", ablation_rows)

    summary = {
        "metrics": rows,
        "significance": {"hybrid_vs_physics": sig_phys, "hybrid_vs_pure_ml": sig_ml},
        "uncertainty": {"alpha": alpha, "empirical_coverage": coverage},
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "model_identity": {
            "primary_model_key": "hybrid",
            "primary_model_display": "Hybrid (Physics + ML Residual)",
        },
        "diagnostics": {
            "y_test_mean": float(np.mean(y_test)) if len(y_test) else 0.0,
            "y_test_std": float(np.std(y_test)) if len(y_test) else 0.0,
            "note": (
                "Negative R2 means a model performs worse than predicting the test-set mean. "
                "Inspect feature_profile_test_split.csv and predictions_by_segment.csv for target variance and outliers."
            ),
        },
    }

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── K-fold cross-validation and statistical model selection ──────────────
    # Build a representative subset of models for CV (skip GPR and Symbolic
    # by default to keep runtime manageable; they are present in predictions).
    cv_models: dict[str, Any] = {
        "speed_power":      SpeedPowerBaseline(),
        "cubic_law":        CubicLawModel(),
        "physics_only":     PhysicsOnlyModel(),
        "physics_reduced":  ReferencePhysicsReducedModel(),
        "pure_ml_ridge":    PureMLModel(learner_kwargs={"lam": 0.05}),
        "hybrid_ridge":     HybridResidualModel(learner_kwargs={"lam": 0.05}),
        "pure_ml_ols":      PureMLModel(learner="ols"),
        "pure_ml_ridge_sk": PureMLModel(learner="ridge_sklearn", learner_kwargs={"alpha": 1.0}),
        "pure_ml_br":       PureMLModel(learner="bayesian_ridge"),
        "pure_ml_knn":      PureMLModel(learner="knn", learner_kwargs={"n_neighbors": 5}),
        "pure_ml_svr":      PureMLModel(learner="svr", learner_kwargs={"C": 10.0}),
        "pure_ml_quantile": PureMLModel(learner="quantile"),
        "pure_ml_mlp":      PureMLModel(learner="mlp", learner_kwargs={"epochs": 150}),
        "pure_ml_pnn":      PureMLModel(learner="pnn"),
    }

    # Add tree-based backends if they were successfully used above
    for _tree in ("rf", "gbt", "xgboost", "catboost", "lightgbm"):
        if f"pure_ml_{_tree}" in predictions:
            cv_models[f"pure_ml_{_tree}"] = PureMLModel(
                learner=_tree,
                learner_kwargs={
                    "rf":        {"n_estimators": 100},
                    "gbt":       {"n_estimators": 100, "max_depth": 3},
                    "xgboost":   {"n_estimators": 100, "max_depth": 4},
                    "catboost":  {"iterations": 100, "depth": 4},
                    "lightgbm":  {"n_estimators": 100, "max_depth": 4},
                }.get(_tree, {}),
            )

    # Add LSTM and Prophet for CV with reduced epochs to keep runtime manageable.
    # Symbolic Regression is skipped in CV due to its high training cost.
    for _sa_key, _sa_lkw in (
        ("lstm",    {"seed": seed, "epochs": 50}),
        ("prophet", {}),
    ):
        if f"pure_ml_{_sa_key}" in predictions:
            cv_models[f"pure_ml_{_sa_key}"] = PureMLModel(
                learner=_sa_key, learner_kwargs=_sa_lkw
            )
        if f"hybrid_{_sa_key}" in predictions:
            cv_models[f"hybrid_{_sa_key}"] = HybridResidualModel(
                learner=_sa_key, learner_kwargs=_sa_lkw
            )

    try:
        cv_results = kfold_cross_validate(cv_models, observations, k=5, seed=seed)
        stat_summary = model_selection_report(
            cv_results,
            out_dir=out,
            metric="mae",
            lower_is_better=True,
            alpha=0.05,
        )
        summary["statistical_analysis"] = stat_summary
    except Exception:
        pass  # statistical analysis is best-effort; never block the main benchmark

    # ── Decision matrix ───────────────────────────────────────────────────────
    try:
        # Build metrics dict from the rows already computed
        dm_metrics = {r["model"]: {k: r[k] for k in ("mae", "rmse", "mape", "r2", "bias")}
                      for r in rows if all(k in r for k in ("mae", "rmse", "r2"))}
        dm_result = generate_decision_matrix_report(
            dm_metrics,
            out_dir=out,
            cv_results=cv_results if "cv_results" in dir() else None,
        )
        summary["decision_matrix"] = {
            "pareto_efficient": dm_result["pareto_efficient"],
            "top_model_balanced":       dm_result["rankings"]["balanced"][0][0]
                                        if dm_result["rankings"].get("balanced") else None,
            "top_model_research":       dm_result["rankings"]["research"][0][0]
                                        if dm_result["rankings"].get("research") else None,
            "top_model_operational":    dm_result["rankings"]["operational"][0][0]
                                        if dm_result["rankings"].get("operational") else None,
            "top_model_interpretability": dm_result["rankings"]["interpretability"][0][0]
                                          if dm_result["rankings"].get("interpretability") else None,
        }
    except Exception:
        pass  # decision matrix is best-effort

    # ── Regression confusion matrices ─────────────────────────────────────────
    try:
        generate_confusion_matrices(out)
    except Exception:
        pass  # best-effort; never block the main benchmark

    # Re-write summary with statistical analysis included
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CLI-facing summary (kept lean for scripted consumers)
    cli_summary: dict[str, Any] = {
        "best_model": summary.get("statistical_analysis", {}).get("best_model"),
        "model_ranking": summary.get("statistical_analysis", {}).get("model_ranking"),
        "friedman_p_value": summary.get("statistical_analysis", {})
            .get("friedman", {}).get("p_value"),
    }
    with (out / "cli_summary.json").open("w", encoding="utf-8") as f:
        json.dump(cli_summary, f, indent=2)

    tracker.log({"event": "end", **{k: v for k, v in summary.items() if k != "statistical_analysis"}})
    return summary
