"""Tests for the paper-versus-results comparison (reproduce/compare.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from reproduce.compare import compare, parse_number


@pytest.mark.parametrize("cell, expected", [
    ("8.00", (8.0, 2)),
    ("20,073", (20073.0, 0)),
    ("$-$0.4", (-0.4, 1)),
    (r"\textbf{8.44}", (8.44, 2)),
    ("8.44 (1)", (8.44, 2)),
    (r"8.0\%", (8.0, 1)),
    ("0.9444", (0.9444, 4)),
])
def test_parse_number_reads_value_and_printed_decimals(cell, expected):
    assert parse_number(cell) == expected


@pytest.mark.parametrize("cell", ["Huber", "", r"\multirow{6}{*}{Linear}", "0.2 -- 26.4"])
def test_parse_number_returns_none_without_a_single_number(cell):
    assert parse_number(cell) is None


def _setup(tmp_path, value):
    (tmp_path / "t.csv").write_text(f"model,MAPE\nHuber,{value}\nOLS,8.885\n", encoding="utf-8")
    paper = {"tab:x": [["Huber", "8.00"], ["OLS", "8.89"]]}
    tmap = {"tab:x": [
        {"paper": [0, 1], "csv": "t.csv", "where": {"model": "Huber"}, "column": "MAPE"},
        {"paper": [1, 1], "csv": "t.csv", "where": {"model": "OLS"}, "column": "MAPE"},
    ]}
    return paper, tmap


def test_value_matches_at_printed_precision(tmp_path):
    paper, tmap = _setup(tmp_path, 8.001039)
    (res,) = compare(paper, tmap, tmp_path)
    assert res.label == "tab:x" and res.n_checked == 2 and res.differences == [] and res.errors == []


def test_one_decimal_off_differs(tmp_path):
    paper, tmap = _setup(tmp_path, 8.011)
    (res,) = compare(paper, tmap, tmp_path)
    assert res.differences == [("[0, 1]", 8.0, pytest.approx(8.011))]


def test_missing_csv_is_reported_not_skipped(tmp_path):
    paper, tmap = _setup(tmp_path, 8.0)
    tmap["tab:x"][0]["csv"] = "absent.csv"
    (res,) = compare(paper, tmap, tmp_path)
    assert res.errors and "absent.csv" in res.errors[0]


def test_ambiguous_row_selection_is_an_error(tmp_path):
    (tmp_path / "t.csv").write_text("model,MAPE\nHuber,8.0\nHuber,9.0\n", encoding="utf-8")
    paper = {"tab:x": [["Huber", "8.00"]]}
    tmap = {"tab:x": [{"paper": [0, 1], "csv": "t.csv", "where": {"model": "Huber"},
                       "column": "MAPE"}]}
    (res,) = compare(paper, tmap, tmp_path)
    assert res.errors and "2 rows" in res.errors[0]


def test_scale_applied_before_rounding(tmp_path):
    (tmp_path / "t.csv").write_text("k,v\na,0.08001\n", encoding="utf-8")
    paper = {"tab:x": [["a", "8.00"]]}
    tmap = {"tab:x": [{"paper": [0, 1], "csv": "t.csv", "where": {"k": "a"}, "column": "v",
                       "scale": 100}]}
    (res,) = compare(paper, tmap, tmp_path)
    assert res.differences == [] and res.errors == []


def test_paper_cell_without_number_is_an_error(tmp_path):
    paper, tmap = _setup(tmp_path, 8.0)
    tmap["tab:x"][0]["paper"] = [0, 0]
    (res,) = compare(paper, tmap, tmp_path)
    assert res.errors and "no number" in res.errors[0]


def test_index_selects_a_later_number_of_the_cell(tmp_path):
    (tmp_path / "t.csv").write_text("model,lo,hi\nC,-0.926,-0.297\n", encoding="utf-8")
    paper = {"tab:x": [["C", "[-0.926, -0.297]"]]}
    tmap = {"tab:x": [
        {"paper": [0, 1], "csv": "t.csv", "where": {"model": "C"}, "column": "lo"},
        {"paper": [0, 1], "index": 1, "csv": "t.csv", "where": {"model": "C"}, "column": "hi"},
    ]}
    (res,) = compare(paper, tmap, tmp_path)
    assert res.n_checked == 2 and res.differences == [] and res.errors == []


def test_index_beyond_the_numbers_of_the_cell_is_an_error(tmp_path):
    (tmp_path / "t.csv").write_text("model,v\nC,1.0\n", encoding="utf-8")
    paper = {"tab:x": [["C", "1.0"]]}
    tmap = {"tab:x": [{"paper": [0, 1], "index": 2, "csv": "t.csv",
                       "where": {"model": "C"}, "column": "v"}]}
    (res,) = compare(paper, tmap, tmp_path)
    assert res.n_checked == 0 and res.errors


def test_terms_form_a_weighted_sum_of_csv_values(tmp_path):
    (tmp_path / "t.csv").write_text("p,MAPE\nin,8.848\nout,9.214\n", encoding="utf-8")
    paper = {"tab:x": [["WBM", "0.366"]]}
    tmap = {"tab:x": [{"paper": [0, 1], "csv": "t.csv", "terms": [
        {"where": {"p": "out"}, "column": "MAPE", "weight": 1},
        {"where": {"p": "in"}, "column": "MAPE", "weight": -1}]}]}
    (res,) = compare(paper, tmap, tmp_path)
    assert res.n_checked == 1 and res.differences == [] and res.errors == []


def test_a_term_with_no_matching_row_is_an_error_not_zero(tmp_path):
    (tmp_path / "t.csv").write_text("p,n\na,318423\nb,106141\n", encoding="utf-8")
    paper = {"tab:x": [["S", "424,564"]]}
    tmap = {"tab:x": [{"paper": [0, 1], "csv": "t.csv", "terms": [
        {"where": {"p": "a"}, "column": "n"}, {"where": {"p": "missing"}, "column": "n"}]}]}
    (res,) = compare(paper, tmap, tmp_path)
    assert res.n_checked == 0 and res.errors
