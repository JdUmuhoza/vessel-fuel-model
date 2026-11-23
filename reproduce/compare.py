"""Compare every mapped paper table cell with the result file that produced it.

``paper_values.json`` holds the paper's tables as raw cell strings, keyed by LaTeX label.
``table_map.json`` binds each checked cell to one value in one result CSV. A cell matches
when the CSV value, rounded to the number of decimals printed in the paper, equals the
printed value.

Usage:
    python -m reproduce.compare [--results results] [--json]
Exit codes: 0 every mapped cell matches, 1 a differing cell or an unreadable binding.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import NamedTuple

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_RANGE = re.compile(r"\d\s*(?:--|–|—)\s*\$?-?\d")


class TableResult(NamedTuple):
    """Outcome of checking one paper table."""

    label: str
    n_checked: int
    differences: list[tuple[str, float, float]]
    errors: list[str]


def parse_numbers(cell: str) -> list[tuple[float, int]]:
    """Return (value, printed decimals) of every number in a LaTeX table cell, in order.

    Parameters
    ----------
    cell : str
        Raw cell text, e.g. ``"[-0.926, -0.297]"`` or ``"(patience 50; 31 selected)"``.

    Returns
    -------
    list of (float, int)
        Empty when the cell holds no number, or a range such as ``"0.2 -- 26.4"``.
    """
    text = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}", "", cell)
    text = text.replace("$-$", "-").replace("$", "").replace("\\%", "")
    text = re.sub(r"\\[A-Za-z]+\{", "", text).replace("{", "").replace("}", "")
    if _RANGE.search(text):
        return []
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(0).replace(",", "")
        out.append((float(raw), len(raw.split(".")[1]) if "." in raw else 0))
    return out


def parse_number(cell: str) -> tuple[float, int] | None:
    """Return (value, printed decimals) of the first number in a LaTeX table cell.

    Parameters
    ----------
    cell : str
        Raw cell text, e.g. ``"\\textbf{8.44} (1)"``, ``"$-$0.4"``, ``"20,073"``.

    Returns
    -------
    tuple of (float, int) or None
        None when the cell holds no number, or a range such as ``"0.2 -- 26.4"``.
    """
    nums = parse_numbers(cell)
    return nums[0] if nums else None


def _csv_value(results_root: pathlib.Path, b: dict, cache: dict) -> float:
    """Read the value a binding points to; raise ValueError when a row is not unique.

    A binding with ``terms`` returns the weighted sum of its terms, each a binding of its
    own that inherits ``csv`` from the parent and carries an optional ``weight`` (default 1).
    """
    if "terms" in b:
        return sum(float(t.get("weight", 1.0)) * _csv_value(results_root, {"csv": b["csv"], **t}, cache)
                   for t in b["terms"])
    path = results_root / b["csv"]
    if path not in cache:
        if not path.is_file():
            raise ValueError(f"result file not found: {b['csv']}")
        cache[path] = pd.read_csv(path)
    df = cache[path]
    for col, val in b.get("where", {}).items():
        df = df[df[col].astype(str) == str(val)]
    if len(df) != 1:
        raise ValueError(f"{b['csv']} {b.get('where')}: {len(df)} rows match, expected 1")
    return float(df[b["column"]].iloc[0]) * float(b.get("scale", 1.0))


def compare(paper: dict, table_map: dict, results_root: pathlib.Path) -> list[TableResult]:
    """Check every bound cell of every mapped table.

    Parameters
    ----------
    paper : dict
        ``{label: [[cell, ...], ...]}`` as stored in ``paper_values.json``.
    table_map : dict
        ``{label: [binding, ...]}``; a binding has ``paper`` ([row, col]), ``csv`` (path under
        ``results_root``), optional ``where`` ({column: value}), ``column`` and optional
        ``scale``.
    results_root : pathlib.Path
        Folder the ``csv`` paths are relative to.

    Returns
    -------
    list of TableResult
        One per mapped table, in map order.
    """
    out, cache = [], {}
    for label, bindings in table_map.items():
        n, bad, errs = 0, [], []
        for b in bindings:
            r, c = b["paper"]
            cid = json.dumps([r, c])
            idx = int(b.get("index", 0))
            if idx:
                cid = json.dumps([r, c, idx])
            try:
                nums = parse_numbers(paper[label][r][c])
            except (KeyError, IndexError):
                errs.append(f"{cid}: cell absent from the paper table")
                continue
            if idx >= len(nums):
                errs.append(f"{cid}: paper cell has no number at index {idx}")
                continue
            parsed = nums[idx]
            try:
                got = _csv_value(results_root, b, cache)
            except (ValueError, KeyError) as exc:
                errs.append(f"{cid}: {exc}")
                continue
            n += 1
            value, dec = parsed
            # within half a unit of the last printed digit (ties accepted either way)
            if abs(got - value) > 0.5 * 10 ** -dec + 1e-9:
                bad.append((cid, value, got))
        out.append(TableResult(label, n, bad, errs))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    paper = json.loads((HERE / "paper_values.json").read_text(encoding="utf-8"))
    tmap = json.loads((HERE / "table_map.json").read_text(encoding="utf-8"))
    res = compare(paper, tmap, a.results)
    if a.json:
        print(json.dumps([r._asdict() for r in res], indent=2))
    for r in res:
        ok = not r.differences and not r.errors
        print(f"{'PASS' if ok else 'FAIL'}  {r.label:32s} {r.n_checked:4d} cells checked")
        for cid, want, got in r.differences:
            print(f"      differs {cid}: paper {want}  result {got}")
        for e in r.errors:
            print(f"      error {e}")
    return 0 if all(not r.differences and not r.errors for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
