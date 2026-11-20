"""Reporting helpers: Excel workbook export and figure generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def regression_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 5,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Build a confusion matrix for regression by discretizing into quantile bins.

    Actual values are binned into *bins* equal-quantile classes using thresholds
    derived from *y_true*.  Predicted values are mapped to those same thresholds.
    The resulting matrix has rows = actual bin, columns = predicted bin — identical
    layout to a classification confusion matrix.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth continuous values.
    y_pred : numpy.ndarray
        Model predicted continuous values.
    bins : int
        Number of discrete classes (default 5).
    labels : list of str, optional
        Class labels.  Defaults to ["Very Low", "Low", "Medium", "High", "Very High"]
        when *bins* == 5, otherwise "Bin 1" … "Bin N".

    Returns
    -------
    dict
        matrix (list[list[int]]), labels, thresholds, bin_accuracy,
        weighted_f1, per_bin {label: {precision, recall, f1, support}}.

    References
    ----------
    Inspired by Hernandez-Orallo, J. et al. (2012). A unified view of performance
    metrics for regression problems.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    if labels is None:
        if bins == 5:
            labels = ["Very Low", "Low", "Medium", "High", "Very High"]
        else:
            labels = [f"Bin {i + 1}" for i in range(bins)]

    # Quantile thresholds from actuals (edges at 0, 1/b, 2/b, … 1)
    q = np.linspace(0.0, 100.0, bins + 1)
    thresholds = np.percentile(yt, q).tolist()

    def _bin(vals: np.ndarray) -> np.ndarray:
        # np.searchsorted with clipping so every value lands in [0, bins-1]
        idx = np.searchsorted(thresholds[1:-1], vals, side="right")
        return np.clip(idx, 0, bins - 1)

    actual_bins = _bin(yt)
    pred_bins = _bin(yp)

    matrix = [[0] * bins for _ in range(bins)]
    for a, p in zip(actual_bins, pred_bins):
        matrix[int(a)][int(p)] += 1

    # Per-bin precision / recall / F1
    per_bin: dict[str, dict[str, float]] = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        col_sum = sum(matrix[r][i] for r in range(bins))
        precision = tp / col_sum if col_sum > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_bin[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    total = len(yt)
    diagonal = sum(matrix[i][i] for i in range(bins))
    bin_accuracy = round(diagonal / total, 4) if total > 0 else 0.0

    total_support = sum(v["support"] for v in per_bin.values())
    weighted_f1 = round(
        sum(v["f1"] * v["support"] for v in per_bin.values()) / total_support, 4
    ) if total_support > 0 else 0.0

    return {
        "matrix": matrix,
        "labels": labels,
        "thresholds": [round(t, 4) for t in thresholds],
        "bin_accuracy": bin_accuracy,
        "weighted_f1": weighted_f1,
        "per_bin": per_bin,
        "n_samples": total,
    }


def generate_confusion_matrices(out_dir: str | Path) -> dict[str, Any]:
    """Compute regression confusion matrices for all models in predictions_by_segment.csv.

    Writes:
    - ``confusion_matrix_summary.csv`` — one row per model with bin_accuracy and weighted_f1
    - ``confusion_matrices/<model>.csv`` — the 5×5 count matrix for each model

    Parameters
    ----------
    out_dir : str or Path
        Benchmark output directory containing predictions_by_segment.csv.

    Returns
    -------
    dict
        Mapping model_name → regression_confusion_matrix result dict.
    """
    out = Path(out_dir)
    pred_path = out / "predictions_by_segment.csv"
    if not pred_path.exists():
        return {}

    with pred_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {}

    y_true = np.asarray([float(r["y_true"]) for r in rows], dtype=float)
    pred_cols = [k for k in rows[0].keys() if k.startswith("pred_")]

    cm_dir = out / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    for col in pred_cols:
        model_name = col[len("pred_"):]
        y_pred = np.asarray([float(r[col]) for r in rows], dtype=float)
        cm = regression_confusion_matrix(y_true, y_pred, bins=5)
        results[model_name] = cm

        # Write individual matrix CSV
        mat_path = cm_dir / f"{model_name}.csv"
        with mat_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["actual \\ predicted"] + labels)
            for i, label in enumerate(labels):
                writer.writerow([label] + cm["matrix"][i])

        # Summary row
        summary_rows.append({
            "model": model_name,
            "bin_accuracy": cm["bin_accuracy"],
            "weighted_f1": cm["weighted_f1"],
            **{f"f1_{lbl.lower().replace(' ', '_')}": cm["per_bin"][lbl]["f1"] for lbl in labels},
            "n_samples": cm["n_samples"],
        })

    # Sort summary by bin_accuracy descending
    summary_rows.sort(key=lambda r: float(r["bin_accuracy"]), reverse=True)
    if summary_rows:
        with (out / "confusion_matrix_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    return results


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def export_results_to_excel(out_dir: str | Path, filename: str = "benchmark_report.xlsx") -> Path | None:
    """Export benchmark outputs to a single multi-sheet Excel workbook.

    Returns workbook path, or None when optional dependency is unavailable.
    """
    try:
        from openpyxl import Workbook
    except Exception:
        return None

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    wb.remove(wb.active)

    # CSV sheets
    csv_files = [
        "results_table.csv",
        "predictions_by_segment.csv",
        "prediction_intervals.csv",
        "sensitivity_ranking.csv",
        "sensitivity_residual_ranking.csv",
        "feature_profile_test_split.csv",
        "ablation_table.csv",
    ]

    for name in csv_files:
        rows = _read_csv(out / name)
        if not rows:
            continue
        ws = wb.create_sheet(title=name.replace(".csv", "")[:31])
        headers = list(rows[0].keys())
        ws.append(headers)
        for r in rows:
            ws.append([r.get(h, "") for h in headers])

    # JSON sheets
    json_files = ["summary.json", "significance.json", "cli_summary.json"]
    for name in json_files:
        p = out / name
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        ws = wb.create_sheet(title=name.replace(".json", "")[:31])
        ws.append(["key", "value"])
        ws.append(["__raw_json__", json.dumps(payload, ensure_ascii=False)])

    ws_meta = wb.create_sheet(title="model_legend")
    ws_meta.append(["model", "description", "is_primary_model"])
    ws_meta.append(["speed_power", "Speed-Power Baseline (fitted exponent)", 0])
    ws_meta.append(["cubic_law", "Cubic Law Baseline (v^3)", 0])
    ws_meta.append(["pure_ml", "Pure ML (Ridge)", 0])
    ws_meta.append(["physics_only", "Physics-Only", 0])
    ws_meta.append(["physics_reduced", "Reference Physics (Reduced)", 0])
    ws_meta.append(["hybrid", "Hybrid (Physics + ML Residual)", 1])

    out_path = out / filename
    wb.save(out_path)
    return out_path


def generate_result_figures(out_dir: str | Path) -> list[Path]:
    """Generate basic benchmark figures.

    Returns list of created figure paths. Returns empty list if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    out = Path(out_dir)
    figs: list[Path] = []

    # 1) MAE/RMSE bar chart
    results = _read_csv(out / "results_table.csv")
    if results:
        models = [r.get("model", "") for r in results]
        mae = [float(r.get("mae", 0.0)) for r in results]
        rmse = [float(r.get("rmse", 0.0)) for r in results]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = list(range(len(models)))
        ax.bar([i - 0.2 for i in x], mae, width=0.4, label="MAE")
        ax.bar([i + 0.2 for i in x], rmse, width=0.4, label="RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylabel("Fuel error (MT)")
        ax.set_title("Model Error Comparison")
        ax.legend()
        fig.tight_layout()
        p = out / "figure_model_error_comparison.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append(p)

    # 2) Observed vs predicted scatter for all models
    pred_rows = _read_csv(out / "predictions_by_segment.csv")
    if pred_rows:
        y_true = [float(r.get("y_true", 0.0)) for r in pred_rows]
        pred_cols = [k for k in pred_rows[0].keys() if k.startswith("pred_")]
        n = len(pred_cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        y_min = min(y_true) if y_true else 0.0
        y_max = max(y_true) if y_true else 1.0
        for idx, col in enumerate(pred_cols):
            r = idx // ncols
            c = idx % ncols
            ax = axes[r][c]
            yp = [float(row.get(col, 0.0)) for row in pred_rows]
            ax.scatter(y_true, yp, alpha=0.55, s=20)
            ax.plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1)
            ax.set_title(col.replace("pred_", ""))
            ax.set_xlabel("Observed fuel (MT)")
            ax.set_ylabel("Predicted fuel (MT)")

        for idx in range(n, nrows * ncols):
            r = idx // ncols
            c = idx % ncols
            axes[r][c].axis("off")

        fig.suptitle("Observed vs Predicted by Model", y=1.02)
        fig.tight_layout()
        p = out / "figure_observed_vs_predicted.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append(p)

    # 3) Top sensitivity features
    sens_rows = _read_csv(out / "sensitivity_ranking.csv")
    if sens_rows:
        top = sens_rows[:15]
        names = [r.get("feature", "") for r in top]
        vals = [float(r.get("delta_mae", 0.0)) for r in top]

        fig, ax = plt.subplots(figsize=(9, 6))
        y = list(range(len(names)))[::-1]
        ax.barh(y, vals[::-1])
        ax.set_yticks(y)
        ax.set_yticklabels(names[::-1])
        ax.set_xlabel("ΔMAE after permutation")
        ax.set_title("Top Sensitivity Drivers (Total Hybrid Model)")
        fig.tight_layout()
        p = out / "figure_sensitivity_top15.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append(p)

    # 4) Prediction intervals (ordered)
    interval_rows = _read_csv(out / "prediction_intervals.csv")
    if interval_rows:
        y_true = [float(r.get("y_true", 0.0)) for r in interval_rows]
        y_pred = [float(r.get("y_pred", 0.0)) for r in interval_rows]
        lo = [float(r.get("lo", 0.0)) for r in interval_rows]
        hi = [float(r.get("hi", 0.0)) for r in interval_rows]
        x = list(range(len(interval_rows)))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(x, y_true, label="Observed", linewidth=1.3)
        ax.plot(x, y_pred, label="Predicted", linewidth=1.3)
        ax.fill_between(x, lo, hi, alpha=0.2, label="Conformal interval")
        ax.set_xlabel("Test segment index")
        ax.set_ylabel("Fuel (MT)")
        ax.set_title("Prediction Intervals on Test Segments")
        ax.legend()
        fig.tight_layout()
        p = out / "figure_prediction_intervals.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append(p)

    return figs
