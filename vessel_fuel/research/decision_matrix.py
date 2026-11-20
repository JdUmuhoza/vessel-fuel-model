"""Multi-criteria decision matrix for model selection in vessel fuel prediction.

Combines quantitative performance metrics (MAE, R², CV stability) with
qualitative engineering criteria (interpretability, physics compliance,
uncertainty quantification, computational cost, data efficiency, calibration
independence) into a weighted score for each model under four use-case scenarios.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

# ── Qualitative scores (1 = worst, 5 = best) ─────────────────────────────────
# Keys correspond to model names used in benchmark results_table.csv.

_QUALITATIVE_SCORES: dict[str, dict[str, int]] = {
    # Physics-based
    "physics_only":         dict(interp=5, physics=5, uq_native=2, compute=5, data_eff=5, calib_free=5),
    "physics_reduced":      dict(interp=5, physics=5, uq_native=2, compute=5, data_eff=5, calib_free=5),
    # Baselines
    "speed_power":          dict(interp=5, physics=3, uq_native=2, compute=5, data_eff=4, calib_free=3),
    "cubic_law":            dict(interp=5, physics=3, uq_native=2, compute=5, data_eff=4, calib_free=3),
    # Hybrid — physics core preserved
    "hybrid_ridge":         dict(interp=4, physics=5, uq_native=2, compute=5, data_eff=4, calib_free=3),
    "hybrid":               dict(interp=4, physics=5, uq_native=2, compute=5, data_eff=4, calib_free=3),
    "hybrid_ols":           dict(interp=4, physics=5, uq_native=2, compute=5, data_eff=4, calib_free=3),
    "hybrid_ridge_sklearn": dict(interp=4, physics=5, uq_native=2, compute=5, data_eff=4, calib_free=3),
    "hybrid_bayesian_ridge":dict(interp=4, physics=5, uq_native=3, compute=5, data_eff=4, calib_free=3),
    "hybrid_quantile":      dict(interp=4, physics=5, uq_native=4, compute=5, data_eff=4, calib_free=3),
    "hybrid_svr":           dict(interp=2, physics=5, uq_native=2, compute=3, data_eff=3, calib_free=3),
    "hybrid_knn":           dict(interp=3, physics=5, uq_native=2, compute=4, data_eff=2, calib_free=3),
    "hybrid_mlp":           dict(interp=2, physics=5, uq_native=2, compute=4, data_eff=3, calib_free=3),
    "hybrid_pnn":           dict(interp=2, physics=5, uq_native=2, compute=2, data_eff=2, calib_free=3),
    "hybrid_rf":            dict(interp=2, physics=5, uq_native=3, compute=3, data_eff=3, calib_free=3),
    "hybrid_gbt":           dict(interp=2, physics=5, uq_native=2, compute=4, data_eff=3, calib_free=3),
    "hybrid_xgboost":       dict(interp=2, physics=5, uq_native=2, compute=4, data_eff=3, calib_free=3),
    "hybrid_catboost":      dict(interp=2, physics=5, uq_native=2, compute=3, data_eff=3, calib_free=3),
    "hybrid_lightgbm":      dict(interp=2, physics=5, uq_native=2, compute=4, data_eff=3, calib_free=3),
    "hybrid_gpr":           dict(interp=3, physics=5, uq_native=5, compute=1, data_eff=4, calib_free=3),
    # Pure ML — no physics prior
    "pure_ml_ridge":        dict(interp=4, physics=1, uq_native=2, compute=5, data_eff=4, calib_free=5),
    "pure_ml":              dict(interp=4, physics=1, uq_native=2, compute=5, data_eff=4, calib_free=5),
    "pure_ml_ols":          dict(interp=5, physics=1, uq_native=2, compute=5, data_eff=4, calib_free=5),
    "pure_ml_ridge_sklearn":dict(interp=4, physics=1, uq_native=2, compute=5, data_eff=4, calib_free=5),
    "pure_ml_bayesian_ridge":dict(interp=4, physics=1, uq_native=4, compute=5, data_eff=5, calib_free=5),
    "pure_ml_quantile":     dict(interp=4, physics=1, uq_native=4, compute=5, data_eff=4, calib_free=5),
    "pure_ml_svr":          dict(interp=2, physics=1, uq_native=2, compute=3, data_eff=3, calib_free=5),
    "pure_ml_knn":          dict(interp=3, physics=1, uq_native=2, compute=4, data_eff=1, calib_free=5),
    "pure_ml_mlp":          dict(interp=1, physics=1, uq_native=2, compute=3, data_eff=2, calib_free=5),
    "pure_ml_pnn":          dict(interp=2, physics=1, uq_native=2, compute=2, data_eff=1, calib_free=5),
    "pure_ml_rf":           dict(interp=2, physics=1, uq_native=3, compute=3, data_eff=3, calib_free=5),
    "pure_ml_gbt":          dict(interp=2, physics=1, uq_native=2, compute=4, data_eff=3, calib_free=5),
    "pure_ml_xgboost":      dict(interp=2, physics=1, uq_native=2, compute=4, data_eff=3, calib_free=5),
    "pure_ml_catboost":     dict(interp=2, physics=1, uq_native=2, compute=3, data_eff=3, calib_free=5),
    "pure_ml_lightgbm":     dict(interp=2, physics=1, uq_native=2, compute=4, data_eff=3, calib_free=5),
    "pure_ml_gpr":          dict(interp=3, physics=1, uq_native=5, compute=1, data_eff=4, calib_free=5),
}

# ── Weight profiles ────────────────────────────────────────────────────────────
# Each profile represents a use-case scenario. Weights must sum to 1.0.

WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "balanced": {
        "mae_score":      0.20,
        "r2_score":       0.15,
        "cv_stability":   0.10,
        "interpretability": 0.10,
        "physics_compliance": 0.10,
        "uq_native":      0.08,
        "compute_cost":   0.07,
        "data_efficiency":0.10,
        "calib_free":     0.10,
    },
    "research": {
        # Prioritises accuracy, interpretability, physics traceability, uncertainty
        "mae_score":      0.25,
        "r2_score":       0.15,
        "cv_stability":   0.10,
        "interpretability": 0.15,
        "physics_compliance": 0.15,
        "uq_native":      0.10,
        "compute_cost":   0.02,
        "data_efficiency":0.05,
        "calib_free":     0.03,
    },
    "operational": {
        # Prioritises accuracy + stability + speed (real-time voyage planning)
        "mae_score":      0.30,
        "r2_score":       0.20,
        "cv_stability":   0.15,
        "interpretability": 0.05,
        "physics_compliance": 0.05,
        "uq_native":      0.05,
        "compute_cost":   0.10,
        "data_efficiency":0.05,
        "calib_free":     0.05,
    },
    "interpretability": {
        # Prioritises explainability and physics compliance (regulatory / audit)
        "mae_score":      0.15,
        "r2_score":       0.10,
        "cv_stability":   0.05,
        "interpretability": 0.20,
        "physics_compliance": 0.20,
        "uq_native":      0.10,
        "compute_cost":   0.05,
        "data_efficiency":0.05,
        "calib_free":     0.10,
    },
}

CRITERION_LABELS: dict[str, str] = {
    "mae_score":          "Accuracy (MAE)",
    "r2_score":           "Accuracy (R²)",
    "cv_stability":       "CV Stability",
    "interpretability":   "Interpretability",
    "physics_compliance": "Physics Compliance",
    "uq_native":          "Uncertainty Quant.",
    "compute_cost":       "Computational Cost",
    "data_efficiency":    "Data Efficiency",
    "calib_free":         "Calibration-Free",
}

SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "balanced":          "Equal emphasis on accuracy, interpretability, and operational factors",
    "research":          "Prioritises accuracy, physics traceability, interpretability, and uncertainty",
    "operational":       "Prioritises predictive accuracy, CV stability, and computational speed",
    "interpretability":  "Prioritises explainability and physics compliance for regulatory contexts",
}


# ── Core computation ───────────────────────────────────────────────────────────

def _normalize_col(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Min-max normalise a column to [0, 1]."""
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if hi - lo < 1e-12:
        return np.ones_like(values, dtype=float)
    norm = (values - lo) / (hi - lo)
    return norm if higher_is_better else (1.0 - norm)


def compute_decision_matrix(
    metrics: dict[str, dict[str, float]],
    cv_results: dict[str, list[dict[str, float]]] | None = None,
    weight_profiles: dict[str, dict[str, float]] | None = None,
    qualitative_scores: dict[str, dict[str, int]] | None = None,
    primary_metric: str = "mae",
) -> dict[str, Any]:
    """Compute a weighted multi-criteria decision matrix for all models.

    Parameters
    ----------
    metrics : dict
        ``{model_name: {mae, rmse, mape, r2, bias}}`` from
        ``results_table.csv`` or ``regression_metrics()``.
    cv_results : dict, optional
        Output from ``kfold_cross_validate``. Used to compute CV stability
        (coefficient of variation of fold MAE). If absent, stability is set
        to 0.5 for all models.
    weight_profiles : dict, optional
        Custom weight profiles. Defaults to ``WEIGHT_PROFILES``.
    qualitative_scores : dict, optional
        Custom qualitative scores. Defaults to ``_QUALITATIVE_SCORES``.
    primary_metric : str
        Metric used for accuracy scoring (``"mae"`` or ``"rmse"``).

    Returns
    -------
    dict
        Keys:

        - ``criteria_table`` — ``{model: {criterion: normalised_score}}``
        - ``scores`` — ``{scenario: {model: weighted_score}}``
        - ``rankings`` — ``{scenario: [(model, score), ...]}`` sorted best-first
        - ``pareto_efficient`` — list of model names on the Pareto frontier
          (accuracy vs interpretability)
        - ``sensitivity`` — weight sensitivity: how much rank changes when
          each criterion weight is perturbed by ±10 pp
        - ``weight_profiles`` — the profiles used
        - ``criteria_labels`` — human-readable criterion names
        - ``scenario_descriptions`` — human-readable scenario descriptions
    """
    profiles = weight_profiles or WEIGHT_PROFILES
    qual = qualitative_scores or _QUALITATIVE_SCORES

    model_names = list(metrics.keys())
    n = len(model_names)

    # ── 1. Build raw criterion matrix ─────────────────────────────────────────

    # Quantitative: MAE and R² from metrics
    mae_raw  = np.array([float(metrics[m].get(primary_metric, np.nan)) for m in model_names])
    r2_raw   = np.array([float(metrics[m].get("r2", np.nan))           for m in model_names])

    # CV stability: 1 - CV_of_fold_MAE  (higher = more stable)
    stab_raw = np.full(n, 0.5)
    if cv_results:
        for i, m in enumerate(model_names):
            folds = cv_results.get(m, [])
            fold_maes = np.array([f.get("mae", np.nan) for f in folds], dtype=float)
            valid = fold_maes[~np.isnan(fold_maes)]
            if len(valid) >= 2:
                cv = float(np.std(valid) / (np.mean(valid) + 1e-12))
                stab_raw[i] = max(0.0, 1.0 - cv)

    # Qualitative criteria (1-5 → treat as continuous)
    qual_keys = ["interp", "physics", "uq_native", "compute", "data_eff", "calib_free"]
    qual_mat = np.full((n, len(qual_keys)), 3.0)   # default = neutral
    for i, m in enumerate(model_names):
        scores = qual.get(m, {})
        for j, k in enumerate(qual_keys):
            if k in scores:
                qual_mat[i, j] = float(scores[k])

    # ── 2. Normalise each criterion to [0, 1] ─────────────────────────────────

    mae_norm   = _normalize_col(mae_raw,  higher_is_better=False)   # lower MAE = better
    r2_norm    = _normalize_col(r2_raw,   higher_is_better=True)
    stab_norm  = _normalize_col(stab_raw, higher_is_better=True)
    qual_norms = np.column_stack([
        _normalize_col(qual_mat[:, j], higher_is_better=True)
        for j in range(len(qual_keys))
    ])

    # Assemble criterion matrix: columns match CRITERION_LABELS key order
    criterion_cols = [
        "mae_score", "r2_score", "cv_stability",
        "interpretability", "physics_compliance",
        "uq_native", "compute_cost", "data_efficiency", "calib_free",
    ]
    crit_matrix = np.column_stack([
        mae_norm, r2_norm, stab_norm,
        qual_norms[:, 0],   # interpretability
        qual_norms[:, 1],   # physics
        qual_norms[:, 2],   # uq_native
        qual_norms[:, 3],   # compute
        qual_norms[:, 4],   # data_eff
        qual_norms[:, 5],   # calib_free
    ])

    criteria_table: dict[str, dict[str, float]] = {
        model_names[i]: {
            criterion_cols[j]: float(crit_matrix[i, j])
            for j in range(len(criterion_cols))
        }
        for i in range(n)
    }

    # ── 3. Weighted scores per scenario ───────────────────────────────────────

    scores: dict[str, dict[str, float]] = {}
    rankings: dict[str, list[tuple[str, float]]] = {}

    for scenario, weights in profiles.items():
        w = np.array([weights.get(c, 0.0) for c in criterion_cols])
        w = w / (w.sum() + 1e-12)
        raw_scores = crit_matrix @ w
        scenario_scores = {model_names[i]: float(raw_scores[i]) for i in range(n)}
        scores[scenario] = scenario_scores
        rankings[scenario] = sorted(
            scenario_scores.items(), key=lambda t: t[1], reverse=True
        )

    # ── 4. Pareto efficiency (accuracy vs interpretability) ───────────────────
    # A model is Pareto-efficient if no other model dominates it on both criteria.
    acc_scores  = mae_norm            # higher = better accuracy
    interp_scores = qual_norms[:, 0]  # higher = more interpretable

    pareto_efficient: list[str] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if acc_scores[j] >= acc_scores[i] and interp_scores[j] >= interp_scores[i]:
                if acc_scores[j] > acc_scores[i] or interp_scores[j] > interp_scores[i]:
                    dominated = True
                    break
        if not dominated:
            pareto_efficient.append(model_names[i])

    # ── 5. Sensitivity analysis ───────────────────────────────────────────────
    # Perturb each criterion weight by +0.10 (clamped) for the "balanced" scenario
    # and record how the average rank of the top-3 models changes.
    sensitivity: dict[str, dict[str, Any]] = {}
    base_profile = profiles.get("balanced", next(iter(profiles.values())))
    base_w = np.array([base_profile.get(c, 0.0) for c in criterion_cols])
    base_w = base_w / (base_w.sum() + 1e-12)
    base_ranks = {m: r + 1 for r, (m, _) in enumerate(
        sorted({model_names[i]: float((crit_matrix @ base_w)[i]) for i in range(n)}.items(),
               key=lambda t: t[1], reverse=True)
    )}

    for ci, cname in enumerate(criterion_cols):
        perturbed = base_w.copy()
        perturbed[ci] = min(perturbed[ci] + 0.10, 1.0)
        perturbed = perturbed / (perturbed.sum() + 1e-12)
        pert_scores = crit_matrix @ perturbed
        pert_ranks = {
            model_names[i]: r + 1
            for r, (i, _) in enumerate(
                sorted(enumerate(pert_scores), key=lambda t: t[1], reverse=True)
            )
        }
        rank_changes = {m: base_ranks[m] - pert_ranks[m] for m in model_names}
        sensitivity[cname] = {
            "top_3_unchanged": all(
                pert_ranks[m] <= 3
                for m in [t[0] for t in sorted(base_ranks.items(), key=lambda x: x[1])[:3]]
            ),
            "largest_gainer": max(rank_changes, key=lambda m: rank_changes[m]),
            "largest_loser":  min(rank_changes, key=lambda m: rank_changes[m]),
        }

    return {
        "criteria_table":        criteria_table,
        "scores":                scores,
        "rankings":              rankings,
        "pareto_efficient":      pareto_efficient,
        "sensitivity":           sensitivity,
        "weight_profiles":       {k: dict(v) for k, v in profiles.items()},
        "criteria_labels":       dict(CRITERION_LABELS),
        "scenario_descriptions": dict(SCENARIO_DESCRIPTIONS),
    }


def generate_decision_matrix_report(
    metrics: dict[str, dict[str, float]],
    out_dir: str | Path,
    cv_results: dict[str, list[dict[str, float]]] | None = None,
    weight_profiles: dict[str, dict[str, float]] | None = None,
    qualitative_scores: dict[str, dict[str, int]] | None = None,
    primary_metric: str = "mae",
) -> dict[str, Any]:
    """Compute and write the full decision-matrix report to *out_dir*.

    Writes
    ------
    - ``decision_matrix.csv``        — normalised criterion scores per model
    - ``decision_matrix_scores.csv`` — weighted total per model per scenario
    - ``decision_matrix.json``       — full result dict
    - ``pareto_efficient.json``      — Pareto-efficient model names

    Parameters
    ----------
    metrics : dict
        ``{model_name: {mae, rmse, r2, …}}`` from ``results_table.csv``.
    out_dir : str or Path
        Directory for output files (created if absent).
    cv_results : dict, optional
        K-fold CV results for stability scoring.
    weight_profiles : dict, optional
        Custom weight profiles; defaults to ``WEIGHT_PROFILES``.
    qualitative_scores : dict, optional
        Custom qualitative scores; defaults to ``_QUALITATIVE_SCORES``.
    primary_metric : str
        ``"mae"`` or ``"rmse"``.

    Returns
    -------
    dict
        Full result from :func:`compute_decision_matrix`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = compute_decision_matrix(
        metrics,
        cv_results=cv_results,
        weight_profiles=weight_profiles,
        qualitative_scores=qualitative_scores,
        primary_metric=primary_metric,
    )

    # decision_matrix.csv — one row per model, one column per criterion
    crit_rows: list[dict[str, Any]] = []
    for model, crits in result["criteria_table"].items():
        row: dict[str, Any] = {"model": model}
        row.update({CRITERION_LABELS.get(k, k): round(v, 4) for k, v in crits.items()})
        crit_rows.append(row)
    _write_csv(out / "decision_matrix.csv", crit_rows)

    # decision_matrix_scores.csv — one row per model, one column per scenario
    score_rows: list[dict[str, Any]] = []
    all_models = list(result["criteria_table"].keys())
    for model in all_models:
        row = {"model": model}
        for scenario in result["scores"]:
            row[f"score_{scenario}"] = round(result["scores"][scenario].get(model, 0.0), 4)
            row[f"rank_{scenario}"] = next(
                (r + 1 for r, (m, _) in enumerate(result["rankings"][scenario]) if m == model),
                len(all_models),
            )
        score_rows.append(row)
    # Sort by balanced score descending
    score_rows.sort(key=lambda r: r.get("score_balanced", 0.0), reverse=True)
    _write_csv(out / "decision_matrix_scores.csv", score_rows)

    # JSON dumps
    with (out / "decision_matrix.json").open("w", encoding="utf-8") as f:
        # rankings contains tuples → convert to lists
        serialisable = {
            k: v for k, v in result.items()
            if k not in ("rankings",)
        }
        serialisable["rankings"] = {
            scenario: [[m, s] for m, s in pairs]
            for scenario, pairs in result["rankings"].items()
        }
        json.dump(serialisable, f, indent=2)

    with (out / "pareto_efficient.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pareto_efficient_models": result["pareto_efficient"],
                "description": (
                    "Models that are not dominated on both "
                    "accuracy (normalised MAE) and interpretability."
                ),
            },
            f, indent=2,
        )

    return result


# ── CSV helper ────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
