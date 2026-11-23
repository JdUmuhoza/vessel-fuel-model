"""Reproduce the paper's tables: a quick check from shipped results, or a full rerun.

``--quick`` compares every mapped table cell (``table_map.json``) of the shipped result files
with the value printed in the paper (``paper_values.json``) at the paper's precision.

``--full --yes`` first copies ``results/`` to ``results_shipped/``, then deletes the files the
stages read back to skip work already done (every ``_cache/`` folder and ``_rows.csv`` under
``results/``, and the physics cache ``data/external/shifts_vessel/cache``), so that every
learner is refitted and every physics prediction recomputed, then runs every stage of
``stages.json`` in order (each as ``python -m experiments.revision.<module>`` with
``VFM_RESULTS_ROOT`` pointing at ``results/revision_final``), then runs the quick check on
the new results. The benchmark data must be present (``python scripts/download_benchmarks.py``).

Usage:
    python -m reproduce.reproduce --quick [--results results]
    python -m reproduce.reproduce --full --yes [--results results]
Exit codes: 0 every mapped cell matches; 1 a cell differs or a stage failed; 2 refused
(missing result files, missing --yes, or an existing results_shipped/ that would be
overwritten).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Callable

from .compare import compare

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
STAGE_TIMEOUT_S = 4 * 3600
PHYSICS_CACHE = REPO / "data" / "external" / "shifts_vessel" / "cache"
CACHE_NAMES = ("_cache", "_rows.csv")


def clear_caches(results: pathlib.Path, physics_cache: pathlib.Path) -> list[pathlib.Path]:
    """Delete the read-back caches so a full rerun recomputes every learner and prediction.

    Stages s01, s02, s12 and s13 skip any learner already listed in their cache file,
    and the physics model is read back from ``.npz`` files when present. Without this step
    a full rerun over the shipped ``results/`` would replay the shipped numbers.

    Parameters
    ----------
    results : pathlib.Path
        Results folder; every ``_cache`` folder and ``_rows.csv`` file below it is removed.
    physics_cache : pathlib.Path
        Folder of the cached physics predictions; removed with its contents.

    Returns
    -------
    list of pathlib.Path
        The paths that were removed.
    """
    removed = sorted(p for p in results.rglob("*") if p.name in CACHE_NAMES)
    for p in removed:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    if physics_cache.exists():
        shutil.rmtree(physics_cache)
        removed.append(physics_cache)
    return removed


def quick(results: pathlib.Path, paper: dict, table_map: dict) -> int:
    """Compare the mapped paper cells with the result files under ``results``.

    Parameters
    ----------
    results : pathlib.Path
        Folder the ``csv`` paths of ``table_map`` are relative to.
    paper, table_map : dict
        Contents of ``paper_values.json`` and ``table_map.json``.

    Returns
    -------
    int
        0 all cells match, 1 a differing cell or unreadable binding, 2 a mapped file is missing.
    """
    missing = sorted({b["csv"] for bs in table_map.values() for b in bs
                      if not (results / b["csv"]).is_file()})
    if missing:
        print(f"Missing result files under {results}:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 2
    res = compare(paper, table_map, results)
    for r in res:
        ok = not r.differences and not r.errors
        print(f"{'PASS' if ok else 'FAIL'}  {r.label:32s} {r.n_checked:4d} cells checked")
        for cid, want, got in r.differences:
            print(f"      cell {cid}: paper {want}, result {got:.6g}")
        for e in r.errors:
            print(f"      error {e}")
    failed = [r.label for r in res if r.differences or r.errors]
    if failed:
        print(f"\n{len(failed)} table(s) differ from the paper.")
        return 1
    print("\nEvery mapped table cell matches the paper.")
    return 0


def _run_stage(module: str, results: pathlib.Path) -> int:
    """Run one experiment stage in its own process; return its exit code."""
    env = {**os.environ, "VFM_RESULTS_ROOT": str((results / "revision_final").resolve()),
           "PYTHONHASHSEED": "0"}
    t0 = time.time()
    rc = subprocess.run([sys.executable, "-m", f"experiments.revision.{module}"], cwd=REPO,
                        env=env, timeout=STAGE_TIMEOUT_S).returncode
    print(f"  {module:40s} exit {rc}  {time.time() - t0:.0f}s", flush=True)
    return rc


def full(results: pathlib.Path, stages: list[dict], paper: dict, table_map: dict, yes: bool,
         run_stage: Callable[[str, pathlib.Path], int] = _run_stage,
         backup: pathlib.Path | None = None,
         physics_cache: pathlib.Path = PHYSICS_CACHE) -> int:
    """Rerun every stage over ``results`` after backing up the shipped results, then check.

    Parameters
    ----------
    results : pathlib.Path
        Results folder the stages write into and the check reads.
    stages : list of dict
        ``stages.json`` entries; ``module`` and optional ``known_exit``.
    paper, table_map : dict
        As for :func:`quick`.
    yes : bool
        Explicit confirmation; without it nothing is run.
    run_stage : callable
        ``(module, results) -> exit code``; replaced in tests.
    backup : pathlib.Path, optional
        Where the shipped results are copied first (default ``results_shipped`` beside
        ``results``); refused if it already exists.
    physics_cache : pathlib.Path
        Physics prediction cache deleted before the first stage (see :func:`clear_caches`).

    Returns
    -------
    int
        Exit code of the final quick check, 1 if a stage failed, 2 if refused.
    """
    if not yes:
        print("--full overwrites results/; pass --yes to proceed (the shipped results are "
              "copied to results_shipped/ first).", file=sys.stderr)
        return 2
    backup = backup or results.parent / "results_shipped"
    if backup.exists():
        print(f"{backup} already exists; move it away before a new full rerun.", file=sys.stderr)
        return 2
    if results.exists():
        shutil.copytree(results, backup)
    removed = clear_caches(results, physics_cache)
    print(f"Removed {len(removed)} cache path(s) so every stage recomputes its results.")
    for st in stages:
        rc = run_stage(st["module"], results)
        if rc != 0 and rc != st.get("known_exit"):
            print(f"Stage {st['module']} failed with exit code {rc}.", file=sys.stderr)
            return 1
    return quick(results, paper, table_map)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--results", type=pathlib.Path, default=REPO / "results")
    a = ap.parse_args(argv)
    paper = json.loads((HERE / "paper_values.json").read_text(encoding="utf-8"))
    tmap = json.loads((HERE / "table_map.json").read_text(encoding="utf-8"))
    if a.quick:
        return quick(a.results, paper, tmap)
    stages = json.loads((HERE / "stages.json").read_text(encoding="utf-8"))["stages"]
    return full(a.results, stages, paper, tmap, a.yes)


if __name__ == "__main__":
    sys.exit(main())
