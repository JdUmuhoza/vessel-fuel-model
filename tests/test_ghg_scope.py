"""Contract tests for the regulatory gas-basket mapping."""

from __future__ import annotations

from vessel_fuel.ghg_scope import INSTRUMENTS, gases_for


def test_cii_consumes_co2_only():
    assert gases_for("CII") == ("CO2",)


def test_fueleu_and_gfs_are_well_to_wake_multi_gas():
    for instrument in ("FuelEU", "IMO-GFS"):
        entry = INSTRUMENTS[instrument]
        assert entry["boundary"] == "WtW"
        assert set(entry["gases"]) >= {"CO2", "CH4", "N2O"}


def test_every_instrument_declares_boundary_metric_and_reference():
    for name, entry in INSTRUMENTS.items():
        assert entry["boundary"] in {"TtW", "WtW"}, name
        assert entry["metric"] and entry["reference"], name
        assert set(entry["gases"]) <= {"CO2", "CH4", "N2O"}, name


def test_unknown_instrument_raises():
    try:
        gases_for("NOT-A-REGULATION")
    except KeyError as exc:
        assert "NOT-A-REGULATION" in str(exc)
    else:
        raise AssertionError("unknown instrument must raise KeyError")
