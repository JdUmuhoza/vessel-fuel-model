"""Utilities for loading external comparator baseline predictions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _pick(row: dict[str, str], names: list[str], default: float = 0.0) -> float:
    for n in names:
        if n in row and row[n] not in {"", None}:
            try:
                return float(row[n])
            except (TypeError, ValueError):
                continue
    return default


def load_vendor_baseline_csv(path: str | Path) -> dict[str, float]:
    """Load generic vendor/public baseline predictions keyed by ``segment_id``."""
    rows = _read_csv(path)
    out: dict[str, float] = {}
    for r in rows:
        sid = str(r.get("segment_id") or r.get("Segment") or "").strip()
        if sid == "":
            continue
        out[sid] = _pick(r, ["fuel_mt", "pred_fuel_mt", "predicted_fuel_mt", "AIS_Fuel_tons", "Opt_Fuel_tons"], 0.0)
    return out


def load_frozen_results_by_order(
    root_dir: str | Path,
    ais_segments: list[dict[str, Any]],
    tags: list[str],
) -> dict[str, dict[str, float]]:
    """Load historical baseline fuel outputs (V5/V7/V8...) by vessel segment order.

    Expected folder pattern per tag and vessel:
    ``results{tag}_{VESSEL_ID}/fuel_by_segment.csv``
    where CSV contains ``AIS_Fuel_tons``.
    """
    root = Path(root_dir)
    grouped: dict[str, list[str]] = {}
    for seg in ais_segments:
        vid = str(seg.get("vessel_id", "")).upper()
        sid = str(seg.get("segment_id", ""))
        if vid == "" or sid == "":
            continue
        grouped.setdefault(vid, []).append(sid)

    results: dict[str, dict[str, float]] = {}
    for tag in tags:
        model_name = f"frozen_{tag.lower()}"
        results[model_name] = {}
        for vessel_id, seg_ids in grouped.items():
            fuel_file = root / f"results{tag}_{vessel_id}" / "fuel_by_segment.csv"
            if not fuel_file.exists():
                continue
            rows = _read_csv(fuel_file)
            n = min(len(seg_ids), len(rows))
            for i in range(n):
                sid = seg_ids[i]
                results[model_name][sid] = _pick(rows[i], ["AIS_Fuel_tons", "Opt_Fuel_tons", "fuel_mt"], 0.0)

    return {k: v for k, v in results.items() if len(v) > 0}
