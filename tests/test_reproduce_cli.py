"""Tests for the reproduction entry point (reproduce/reproduce.py); stages are mocked."""

from __future__ import annotations

import pytest

from reproduce import reproduce as rp

PAPER = {"tab:x": [["Huber", "8.00"]]}
TMAP = {"tab:x": [{"paper": [0, 1], "csv": "revision_final/a/t.csv",
                   "where": {"model": "Huber"}, "column": "MAPE"}]}


def _results(root, value):
    d = root / "revision_final" / "a"
    d.mkdir(parents=True)
    (d / "t.csv").write_text(f"model,MAPE\nHuber,{value}\n", encoding="utf-8")
    return root


def test_quick_passes_on_fixture_results(tmp_path):
    assert rp.quick(_results(tmp_path / "results", 8.001), PAPER, TMAP) == 0


def test_quick_exit_1_on_difference(tmp_path):
    assert rp.quick(_results(tmp_path / "results", 8.4), PAPER, TMAP) == 1


def test_quick_exit_2_when_a_result_file_is_missing(tmp_path):
    (tmp_path / "results").mkdir()
    assert rp.quick(tmp_path / "results", PAPER, TMAP) == 2


def test_full_refuses_without_yes(tmp_path):
    results = _results(tmp_path / "results", 8.001)
    called = []
    rc = rp.full(results, [{"module": "s1"}], PAPER, TMAP, yes=False,
                 run_stage=lambda m, root: called.append(m) or 0)
    assert rc == 2 and called == []


def test_full_backs_up_shipped_results_before_running(tmp_path):
    results = _results(tmp_path / "results", 8.001)
    backup = tmp_path / "results_shipped"
    seen = []

    def run_stage(module, root):
        seen.append((module, (backup / "revision_final" / "a" / "t.csv").is_file()))
        return 0

    rc = rp.full(results, [{"module": "s1"}, {"module": "s2"}], PAPER, TMAP, yes=True,
                 run_stage=run_stage, backup=backup)
    assert rc == 0 and seen == [("s1", True), ("s2", True)]


def test_full_refuses_to_overwrite_an_existing_backup(tmp_path):
    results = _results(tmp_path / "results", 8.001)
    backup = tmp_path / "results_shipped"
    backup.mkdir()
    rc = rp.full(results, [{"module": "s1"}], PAPER, TMAP, yes=True,
                 run_stage=lambda m, root: 0, backup=backup)
    assert rc == 2


def test_full_accepts_a_declared_exit_code_and_stops_on_any_other(tmp_path):
    results = _results(tmp_path / "results", 8.001)
    stages = [{"module": "s1", "known_exit": 2}, {"module": "s2"}, {"module": "s3"}]
    codes = {"s1": 2, "s2": 1, "s3": 0}
    ran = []

    def run_stage(module, root):
        ran.append(module)
        return codes[module]

    rc = rp.full(results, stages, PAPER, TMAP, yes=True, run_stage=run_stage,
                 backup=tmp_path / "bk")
    assert rc == 1 and ran == ["s1", "s2"]


def test_cli_requires_one_mode():
    with pytest.raises(SystemExit):
        rp.main([])


def test_full_clears_stage_and_physics_caches_before_the_first_stage(tmp_path):
    results = _results(tmp_path / "results", 8.001)
    rf = results / "revision_final"
    (rf / "24" / "_cache").mkdir(parents=True)
    (rf / "24" / "_cache" / "ml_selection.csv").write_text("x\n", encoding="utf-8")
    (rf / "41").mkdir()
    (rf / "41" / "_rows.csv").write_text("x\n", encoding="utf-8")
    phys = tmp_path / "data" / "cache"
    phys.mkdir(parents=True)
    (phys / "rev_train_10.npz").write_bytes(b"0")
    backup = tmp_path / "results_shipped"
    seen = []

    def run_stage(module, root):
        seen.append(sorted(p.name for p in rf.rglob("*") if p.name in ("_cache", "_rows.csv"))
                    + sorted(p.name for p in phys.glob("*")))
        return 0

    rc = rp.full(results, [{"module": "s1"}], PAPER, TMAP, yes=True, run_stage=run_stage,
                 backup=backup, physics_cache=phys)
    assert rc == 0 and seen == [[]]
    assert (backup / "revision_final" / "24" / "_cache" / "ml_selection.csv").is_file()
    assert (backup / "revision_final" / "41" / "_rows.csv").is_file()
    assert (rf / "a" / "t.csv").is_file()


def test_full_without_yes_leaves_the_caches_in_place(tmp_path):
    results = _results(tmp_path / "results", 8.001)
    cache = results / "revision_final" / "41" / "_rows.csv"
    cache.parent.mkdir()
    cache.write_text("x\n", encoding="utf-8")
    rc = rp.full(results, [{"module": "s1"}], PAPER, TMAP, yes=False,
                 run_stage=lambda m, root: 0, physics_cache=tmp_path / "none")
    assert rc == 2 and cache.is_file()
