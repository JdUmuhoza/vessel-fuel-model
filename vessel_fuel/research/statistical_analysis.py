"""Statistical model comparison: k-fold CV, Friedman, Nemenyi, Wilcoxon, heatmaps.

Methodology follows the reference notebooks in Reference Statistical_Analysis/:
  - statistics_test.ipynb  (Friedman + Wilcoxon all-pairs + Nemenyi matrix)
  - re-trainingV8.ipynb     (k-fold CV with LSTM/Prophet/RF/XGBoost/LightGBM)
  - 4 algo_cross_validation.ipynb (per-metric analysis)

Tests are run on R², RMSE, and MAE separately, then combined into a global ranking.
The pairwise comparison matrix (Nemenyi p-values as a heatmap) is generated for
the top-N models on each metric, matching the seaborn heatmaps in the reference.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

from .data_pipeline import build_feature_matrix, fit_category_maps
from .hybrid import (
    CubicLawModel,
    HybridResidualModel,
    PhysicsOnlyModel,
    PureMLModel,
    ReferencePhysicsReducedModel,
    SpeedPowerBaseline,
)
from .metrics import regression_metrics


# ── Interface detection ───────────────────────────────────────────────────────

def _is_observations_model(model: Any) -> bool:
    return isinstance(
        model,
        (PhysicsOnlyModel, ReferencePhysicsReducedModel, SpeedPowerBaseline, CubicLawModel),
    )


def _is_hybrid_model(model: Any) -> bool:
    return isinstance(model, HybridResidualModel)


def _is_pure_ml_model(model: Any) -> bool:
    return isinstance(model, PureMLModel)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cv_to_dataframe(
    cv_results: dict[str, list[dict[str, float]]],
    metric: str,
) -> "Any":  # returns pandas DataFrame
    import pandas as pd  # type: ignore[import]
    model_names = list(cv_results.keys())
    n_folds = min(len(cv_results[m]) for m in model_names)
    data: dict[str, list[float]] = {}
    for m in model_names:
        vals = [cv_results[m][i].get(metric, float("nan")) for i in range(n_folds)]
        if not any(np.isnan(v) for v in vals):
            data[m] = vals
    return pd.DataFrame(data)


# ── Core statistical functions ────────────────────────────────────────────────

def kfold_cross_validate(
    models: dict[str, Any],
    observations: list[dict[str, Any]],
    k: int = 5,
    seed: int = 42,
    metric_keys: Sequence[str] = ("mae", "rmse", "mape", "r2"),
) -> dict[str, list[dict[str, float]]]:
    """K-fold cross-validation for a set of heterogeneous models.

    Each model is deep-copied per fold to avoid state leakage. Supports
    three model interfaces:

    - ``fit(observations)`` / ``predict(observations)`` — physics / baseline
    - ``fit(observations, x, y)`` / ``predict(observations, x)`` — hybrid
    - ``fit(x, y)`` / ``predict(x)`` — pure-ML, LSTM, Prophet, Symbolic

    Parameters
    ----------
    models : dict of str to model
        Named model instances. Each is deep-copied before fitting each fold.
    observations : list of dict
        Full dataset (all splits combined).
    k : int
        Number of folds.
    seed : int
        Shuffle seed for fold assignment.
    metric_keys : sequence of str
        Subset of ``regression_metrics`` keys to record per fold.

    Returns
    -------
    dict
        ``{model_name: [fold_0_metrics, ..., fold_{k-1}_metrics]}``.
        Folds that raise an exception contain ``NaN`` for all metrics.

    References
    ----------
    Stone, M. (1974). Cross-validatory choice and assessment of statistical
    predictions. Journal of the Royal Statistical Society B, 36(2), 111–147.
    """
    rng = np.random.default_rng(seed)
    n = len(observations)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds: list[list[int]] = [arr.tolist() for arr in np.array_split(idx, k)]

    results: dict[str, list[dict[str, float]]] = {name: [] for name in models}
    nan_fold = {mk: float("nan") for mk in metric_keys}

    for fold_i in range(k):
        val_idx = folds[fold_i]
        train_idx = [j for fi in range(k) if fi != fold_i for j in folds[fi]]

        train = [observations[i] for i in train_idx]
        val_obs = [observations[i] for i in val_idx]

        cat_maps = fit_category_maps(train)
        x_train, y_train, _ = build_feature_matrix(train, cat_maps=cat_maps)
        x_val, y_val, _ = build_feature_matrix(val_obs, cat_maps=cat_maps)

        for name, model in models.items():
            m_clone = deepcopy(model)
            try:
                if _is_observations_model(m_clone):
                    m_clone.fit(train)
                    y_pred = m_clone.predict(val_obs)
                elif _is_hybrid_model(m_clone):
                    m_clone.fit(train, x_train, y_train)
                    y_pred = m_clone.predict(val_obs, x_val)
                elif _is_pure_ml_model(m_clone):
                    m_clone.fit(x_train, y_train)
                    y_pred = m_clone.predict_from_matrix(x_val)
                else:
                    # LSTMModel, ProphetModel, SymbolicRegressionModel
                    # — any object with fit(x, y) / predict(x) interface
                    m_clone.fit(x_train, y_train)
                    y_pred = np.maximum(
                        np.asarray(m_clone.predict(x_val), dtype=float), 0.0
                    )
                all_m = regression_metrics(y_val, y_pred)
                results[name].append({mk: all_m[mk] for mk in metric_keys if mk in all_m})
            except Exception:
                results[name].append(dict(nan_fold))

    return results


def rank_models(
    cv_results: dict[str, list[dict[str, float]]],
    metric: str = "mae",
    lower_is_better: bool = True,
) -> dict[str, float]:
    """Compute average Friedman rank of each model across CV folds.

    Parameters
    ----------
    cv_results : dict
        Output from :func:`kfold_cross_validate`.
    metric : str
        Metric key to rank on.
    lower_is_better : bool
        If ``True``, smaller values receive rank 1.

    Returns
    -------
    dict
        ``{model_name: average_rank}`` sorted best (rank 1) first.

    References
    ----------
    Demšar, J. (2006). Statistical comparisons of classifiers over
    multiple data sets. Journal of Machine Learning Research, 7, 1–30.
    """
    model_names = list(cv_results.keys())
    n_models = len(model_names)
    n_folds = max(len(v) for v in cv_results.values())

    matrix = np.full((n_folds, n_models), np.nan)
    for j, name in enumerate(model_names):
        for i, fold_m in enumerate(cv_results[name]):
            if i < n_folds:
                matrix[i, j] = fold_m.get(metric, np.nan)

    ranks = np.full_like(matrix, np.nan)
    for i in range(n_folds):
        row = matrix[i]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        sign = 1.0 if lower_is_better else -1.0
        ranks[i, valid] = stats.rankdata(sign * row[valid])

    avg_ranks = np.nanmean(ranks, axis=0)
    result = {name: float(avg_ranks[j]) for j, name in enumerate(model_names)}
    return dict(sorted(result.items(), key=lambda t: t[1]))


def combined_ranking(
    cv_results: dict[str, list[dict[str, float]]],
) -> dict[str, float]:
    """Combined ranking averaging Friedman ranks on R², RMSE, and MAE.

    Replicates the *classement final global* from the reference notebook
    ``statistics_test.ipynb``. R² uses ``lower_is_better=False`` (higher is
    better); RMSE and MAE use ``lower_is_better=True``.

    Returns
    -------
    dict
        ``{model_name: combined_avg_rank}`` sorted best first.
    """
    ranks_r2   = rank_models(cv_results, metric="r2",   lower_is_better=False)
    ranks_rmse = rank_models(cv_results, metric="rmse",  lower_is_better=True)
    ranks_mae  = rank_models(cv_results, metric="mae",   lower_is_better=True)

    all_models = set(ranks_r2) | set(ranks_rmse) | set(ranks_mae)
    worst = len(all_models) + 1.0
    combined: dict[str, float] = {}
    for m in all_models:
        combined[m] = (
            ranks_r2.get(m, worst) +
            ranks_rmse.get(m, worst) +
            ranks_mae.get(m, worst)
        ) / 3.0

    return dict(sorted(combined.items(), key=lambda t: t[1]))


def friedman_test(
    cv_results: dict[str, list[dict[str, float]]],
    metric: str = "mae",
) -> dict[str, Any]:
    """Friedman chi-square test for significant differences across models.

    Parameters
    ----------
    cv_results : dict
        Output from :func:`kfold_cross_validate`.
    metric : str
        Metric key to test on.

    Returns
    -------
    dict
        Keys: ``statistic``, ``p_value``, ``df``, ``n_models``,
        ``n_folds_used``, ``significant_at_0_05``.

    References
    ----------
    Friedman, M. (1937). The use of ranks to avoid the assumption of
    normality implicit in the analysis of variance.
    Journal of the American Statistical Association, 32(200), 675–701.
    """
    model_names = list(cv_results.keys())
    k = len(model_names)
    n_folds = min(len(cv_results[name]) for name in model_names)

    matrix = np.full((n_folds, k), np.nan)
    for j, name in enumerate(model_names):
        for i, fold_m in enumerate(cv_results[name][:n_folds]):
            matrix[i, j] = fold_m.get(metric, np.nan)

    valid_rows = ~np.any(np.isnan(matrix), axis=1)
    mat = matrix[valid_rows]
    n_valid = int(mat.shape[0])

    if n_valid < 3 or k < 3:
        return {
            "statistic": float("nan"), "p_value": float("nan"),
            "df": k - 1, "n_models": k, "n_folds_used": n_valid,
            "significant_at_0_05": False,
            "note": "Insufficient data — need >= 3 models and >= 3 complete folds.",
        }

    stat, p_value = stats.friedmanchisquare(*[mat[:, j] for j in range(k)])
    return {
        "statistic": float(stat), "p_value": float(p_value),
        "df": k - 1, "n_models": k, "n_folds_used": n_valid,
        "significant_at_0_05": bool(p_value < 0.05),
    }


def friedman_test_all_metrics(
    cv_results: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    """Run :func:`friedman_test` on R², RMSE, and MAE simultaneously.

    Replicates the three ``perform_stat_tests`` calls in the reference notebook.

    Returns
    -------
    dict
        ``{"r2": {...}, "rmse": {...}, "mae": {...}}``.
    """
    return {
        "r2":   friedman_test(cv_results, metric="r2"),
        "rmse": friedman_test(cv_results, metric="rmse"),
        "mae":  friedman_test(cv_results, metric="mae"),
    }


def nemenyi_posthoc(
    cv_results: dict[str, list[dict[str, float]]],
    metric: str = "mae",
    lower_is_better: bool = True,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Nemenyi post-hoc pairwise test (Demšar 2006, Bonferroni-corrected).

    Returns
    -------
    dict
        Keys: ``model_names``, ``avg_ranks``, ``p_values``,
        ``significant_pairs``, ``critical_difference``, ``alpha``.
    """
    model_names = list(cv_results.keys())
    k = len(model_names)
    n_folds = min(len(cv_results[name]) for name in model_names)

    avg_ranks_dict = rank_models(cv_results, metric=metric, lower_is_better=lower_is_better)
    avg_ranks = np.array([avg_ranks_dict[name] for name in model_names])

    se = np.sqrt(k * (k + 1) / (6.0 * n_folds))
    n_pairs = k * (k - 1) // 2

    p_matrix = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            z = float(abs(avg_ranks[i] - avg_ranks[j]) / se)
            p_raw = float(2.0 * (1.0 - stats.norm.cdf(z)))
            p_corrected = min(p_raw * n_pairs, 1.0)
            p_matrix[i, j] = p_corrected
            p_matrix[j, i] = p_corrected

    _q005 = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 12: 3.268,
        15: 3.394, 20: 3.578,
    }
    q = _q005.get(k, 1.960 + 0.5 * float(np.log(k)))
    cd = q * float(np.sqrt(k * (k + 1) / (6.0 * n_folds)))

    return {
        "model_names": model_names,
        "avg_ranks": avg_ranks_dict,
        "p_values": {
            f"{model_names[i]} vs {model_names[j]}": float(p_matrix[i, j])
            for i in range(k) for j in range(i + 1, k)
        },
        "significant_pairs": [
            f"{model_names[i]} vs {model_names[j]}"
            for i in range(k) for j in range(i + 1, k)
            if p_matrix[i, j] < alpha
        ],
        "critical_difference": float(cd),
        "alpha": float(alpha),
    }


def nemenyi_matrix_posthoc(
    cv_results: dict[str, list[dict[str, float]]],
    metric: str = "mae",
) -> dict[str, Any]:
    """Full N×N Nemenyi p-value matrix via ``scikit_posthocs``.

    Uses ``scikit_posthocs.posthoc_nemenyi_friedman`` — the same function used
    in the reference notebook ``statistics_test.ipynb``.

    Returns
    -------
    dict
        ``{"model_names": [...], "matrix": {model: {model: p_value}}}``.
        Returns empty dict if ``scikit_posthocs`` is unavailable.

    References
    ----------
    Demšar, J. (2006). Statistical comparisons of classifiers over
    multiple data sets. Journal of Machine Learning Research, 7, 1–30.
    """
    try:
        import scikit_posthocs as sp  # type: ignore[import]
    except ImportError:
        return {}

    df = _cv_to_dataframe(cv_results, metric)
    if df.empty or df.shape[1] < 3:
        return {}

    matrix = sp.posthoc_nemenyi_friedman(df)
    return {
        "model_names": list(df.columns),
        "matrix": {
            str(row): {str(col): float(matrix.loc[row, col]) for col in matrix.columns}
            for row in matrix.index
        },
    }


def wilcoxon_pairwise(
    cv_results: dict[str, list[dict[str, float]]],
    baseline_key: str,
    metric: str = "mae",
    lower_is_better: bool = True,
) -> dict[str, dict[str, float]]:
    """Wilcoxon signed-rank test: each model vs a reference baseline.

    Parameters
    ----------
    cv_results : dict
    baseline_key : str
        Reference model key.
    metric : str
    lower_is_better : bool

    Returns
    -------
    dict
        ``{model_name: {"statistic": float, "p_value": float}}``.

    References
    ----------
    Wilcoxon, F. (1945). Individual comparisons by ranking methods.
    Biometrics Bulletin, 1(6), 80–83.
    """
    if baseline_key not in cv_results:
        raise KeyError(f"baseline_key {baseline_key!r} not found in cv_results.")

    baseline_scores = np.array(
        [fold.get(metric, np.nan) for fold in cv_results[baseline_key]]
    )
    result: dict[str, dict[str, float]] = {}

    for name, folds in cv_results.items():
        if name == baseline_key:
            continue
        scores = np.array([fold.get(metric, np.nan) for fold in folds])
        valid = ~(np.isnan(scores) | np.isnan(baseline_scores))
        if valid.sum() < 3:
            result[name] = {"statistic": float("nan"), "p_value": float("nan")}
            continue
        try:
            alt = "greater" if lower_is_better else "less"
            stat, p = stats.wilcoxon(baseline_scores[valid], scores[valid], alternative=alt)
            result[name] = {"statistic": float(stat), "p_value": float(p)}
        except Exception:
            result[name] = {"statistic": float("nan"), "p_value": float("nan")}

    return result


def wilcoxon_all_pairs(
    cv_results: dict[str, list[dict[str, float]]],
    metric: str = "mae",
    lower_is_better: bool = True,
) -> dict[str, dict[str, float]]:
    """Wilcoxon signed-rank test for every pair of models.

    Replicates the all-pairs Wilcoxon analysis in the reference notebook
    ``statistics_test.ipynb``.

    Returns
    -------
    dict
        ``{"ModelA vs ModelB": {"statistic": float, "p_value": float}}``.
    """
    model_names = list(cv_results.keys())
    result: dict[str, dict[str, float]] = {}

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            s1 = np.array([f.get(metric, np.nan) for f in cv_results[m1]])
            s2 = np.array([f.get(metric, np.nan) for f in cv_results[m2]])
            valid = ~(np.isnan(s1) | np.isnan(s2))
            pair_key = f"{m1} vs {m2}"
            if valid.sum() < 3:
                result[pair_key] = {"statistic": float("nan"), "p_value": float("nan")}
                continue
            try:
                stat, p = stats.wilcoxon(s1[valid], s2[valid])
                result[pair_key] = {"statistic": float(stat), "p_value": float(p)}
            except Exception:
                result[pair_key] = {"statistic": float("nan"), "p_value": float("nan")}

    return result


def generate_statistical_heatmaps(
    cv_results: dict[str, list[dict[str, float]]],
    out_dir: str | Path,
    top_n: int = 10,
    physics_only_key: str = "physics_only",
) -> list[Path]:
    """Generate Nemenyi p-value heatmaps for the top-N models.

    Produces one heatmap per metric (R², RMSE, MAE) using seaborn, matching
    the visual style of the reference notebook ``statistics_test.ipynb``.
    An additional combined heatmap shows the top-N models by combined rank.

    Parameters
    ----------
    cv_results : dict
        Output from :func:`kfold_cross_validate`.
    out_dir : str or Path
        Directory where PNG files are written.
    top_n : int
        Number of top models to include (default 10).
    physics_only_key : str
        If present in top_n, it is always kept regardless of rank position to
        ensure physics-only is always included in comparisons.

    Returns
    -------
    list of Path
        Paths to generated PNG files.

    References
    ----------
    Demšar, J. (2006). Statistical comparisons of classifiers over
    multiple data sets. Journal of Machine Learning Research, 7, 1–30.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import scikit_posthocs as sp  # type: ignore[import]
    except ImportError:
        return []

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Top-N by combined ranking; always include physics_only if present
    combined = combined_ranking(cv_results)
    top_models = list(combined.keys())[:top_n]
    if physics_only_key in cv_results and physics_only_key not in top_models:
        top_models[-1] = physics_only_key

    top_cv = {m: cv_results[m] for m in top_models if m in cv_results}

    paths: list[Path] = []
    metrics_cfg = [
        ("r2",   "R²",   False),
        ("rmse", "RMSE", True),
        ("mae",  "MAE",  True),
    ]

    for metric, label, _lower in metrics_cfg:
        df = _cv_to_dataframe(top_cv, metric)
        if df.empty or df.shape[1] < 3:
            continue

        matrix = sp.posthoc_nemenyi_friedman(df)
        short_names = {m: m.replace("pure_ml_", "ML:").replace("hybrid_", "Hyb:") for m in df.columns}
        matrix.index   = [short_names[m] for m in matrix.index]
        matrix.columns = [short_names[m] for m in matrix.columns]

        size = max(8, len(df.columns))
        plt.rcParams.update({"font.family": "serif", "font.size": 10})
        fig, ax = plt.subplots(figsize=(size, size * 0.75))

        sns.heatmap(
            matrix, annot=True, cmap="coolwarm", fmt=".4f", ax=ax,
            vmin=0.0, vmax=1.0, center=0.05,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 9},
        )
        ax.set_title(
            f"Nemenyi Post-Hoc Test — {label}  (p-values)\n"
            f"Values < 0.05 indicate significant difference between model pair",
            fontsize=11, pad=12,
        )
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("Model", fontsize=10)
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        fig.tight_layout()

        p = out / f"nemenyi_heatmap_{metric}.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # Combined ranking bar chart
    try:
        fig, ax = plt.subplots(figsize=(max(10, len(top_models) * 0.9), 5))
        models_sorted = list(combined.keys())[:top_n]
        scores = [combined[m] for m in models_sorted]
        short = [m.replace("pure_ml_", "ML:").replace("hybrid_", "Hyb:") for m in models_sorted]
        colors = ["#2980B9" if "hyb" not in m.lower() and "physics" not in m.lower()
                  else ("#27AE60" if "physics" in m.lower() else "#F5A623")
                  for m in models_sorted]
        ax.bar(short, scores, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Combined avg rank (lower = better)", fontsize=10)
        ax.set_title(
            f"Top {top_n} Models — Combined Rank (R² + RMSE + MAE)\n"
            "Blue = Pure ML  |  Orange = Hybrid  |  Green = Physics",
            fontsize=11,
        )
        plt.xticks(rotation=35, ha="right", fontsize=9)
        fig.tight_layout()
        p = out / "combined_ranking_chart.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    except Exception:
        pass

    return paths


# ── Full report ───────────────────────────────────────────────────────────────

def model_selection_report(
    cv_results: dict[str, list[dict[str, float]]],
    out_dir: str | Path,
    metric: str = "mae",
    baseline_key: str | None = None,
    lower_is_better: bool = True,
    alpha: float = 0.05,
    top_n: int = 10,
) -> dict[str, Any]:
    """Compute and write a full statistical model-selection report.

    Follows the methodology of the reference notebook ``statistics_test.ipynb``:
    Friedman test on R², RMSE, and MAE separately; Wilcoxon all-pairs; Nemenyi
    full p-value matrices; combined ranking; heatmap figures for top-N models.

    Writes to ``out_dir``:

    - ``kfold_cv_results.csv``
    - ``model_ranking.csv``
    - ``combined_ranking.csv``         ← new: avg rank across R²+RMSE+MAE
    - ``friedman_test.json``           ← primary metric only (legacy)
    - ``friedman_all_metrics.json``    ← R², RMSE, MAE separately
    - ``nemenyi_posthoc.json``         ← legacy compact format
    - ``nemenyi_matrix_r2.json``       ← full N×N matrix on R²
    - ``nemenyi_matrix_rmse.json``     ← full N×N matrix on RMSE
    - ``nemenyi_matrix_mae.json``      ← full N×N matrix on MAE
    - ``wilcoxon_pairwise.json``       ← each model vs baseline
    - ``wilcoxon_all_pairs.json``      ← every pair (reference notebook style)
    - ``statistical_summary.json``
    - ``nemenyi_heatmap_r2.png``       ← seaborn heatmap (top-N models)
    - ``nemenyi_heatmap_rmse.png``
    - ``nemenyi_heatmap_mae.png``
    - ``combined_ranking_chart.png``

    Returns
    -------
    dict
        Combined summary dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Per-fold CSV ──────────────────────────────────────────────────────────
    cv_rows: list[dict[str, Any]] = []
    for name, folds in cv_results.items():
        for fold_i, m in enumerate(folds):
            cv_rows.append({"model": name, "fold": fold_i, **m})
    _write_csv(out / "kfold_cv_results.csv", cv_rows)

    # ── Rankings ──────────────────────────────────────────────────────────────
    ranks = rank_models(cv_results, metric=metric, lower_is_better=lower_is_better)
    _write_csv(out / "model_ranking.csv",
               [{"model": n, "avg_rank": r} for n, r in ranks.items()])

    comb = combined_ranking(cv_results)
    _write_csv(out / "combined_ranking.csv",
               [{"model": n, "combined_avg_rank": round(r, 3),
                 "rank": i + 1} for i, (n, r) in enumerate(comb.items())])

    # ── Friedman tests ────────────────────────────────────────────────────────
    friedman_primary = friedman_test(cv_results, metric=metric)
    with (out / "friedman_test.json").open("w", encoding="utf-8") as f:
        json.dump(friedman_primary, f, indent=2)

    friedman_all = friedman_test_all_metrics(cv_results)
    with (out / "friedman_all_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(friedman_all, f, indent=2)

    # ── Nemenyi ───────────────────────────────────────────────────────────────
    nemenyi = nemenyi_posthoc(
        cv_results, metric=metric, lower_is_better=lower_is_better, alpha=alpha
    )
    with (out / "nemenyi_posthoc.json").open("w", encoding="utf-8") as f:
        json.dump(nemenyi, f, indent=2)

    for met in ("r2", "rmse", "mae"):
        nm = nemenyi_matrix_posthoc(cv_results, metric=met)
        if nm:
            with (out / f"nemenyi_matrix_{met}.json").open("w", encoding="utf-8") as f:
                json.dump(nm, f, indent=2)

    # ── Wilcoxon ──────────────────────────────────────────────────────────────
    if baseline_key is None:
        baseline_key = list(ranks.keys())[-1]

    wilcoxon = wilcoxon_pairwise(
        cv_results, baseline_key=baseline_key, metric=metric,
        lower_is_better=lower_is_better,
    )
    with (out / "wilcoxon_pairwise.json").open("w", encoding="utf-8") as f:
        json.dump(wilcoxon, f, indent=2)

    wilcoxon_ap = wilcoxon_all_pairs(cv_results, metric=metric,
                                     lower_is_better=lower_is_better)
    with (out / "wilcoxon_all_pairs.json").open("w", encoding="utf-8") as f:
        json.dump(wilcoxon_ap, f, indent=2)

    # ── Heatmap figures ───────────────────────────────────────────────────────
    try:
        generate_statistical_heatmaps(cv_results, out_dir=out, top_n=top_n)
    except Exception:
        pass

    # ── Summary ───────────────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "primary_metric": metric,
        "model_ranking": ranks,
        "combined_ranking": comb,
        "best_model": list(ranks.keys())[0] if ranks else None,
        "best_model_combined": list(comb.keys())[0] if comb else None,
        "friedman": friedman_primary,
        "friedman_all_metrics": friedman_all,
        "nemenyi": {
            "critical_difference": nemenyi.get("critical_difference"),
            "significant_pairs": nemenyi.get("significant_pairs", []),
        },
        "wilcoxon_vs_baseline": {
            "baseline": baseline_key,
            "results": {n: {"p_value": v["p_value"], "statistic": v["statistic"]}
                        for n, v in wilcoxon.items()},
        },
        "wilcoxon_all_pairs_significant": {
            k: v for k, v in wilcoxon_ap.items()
            if not np.isnan(v["p_value"]) and v["p_value"] < alpha
        },
    }
    with (out / "statistical_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
