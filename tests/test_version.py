"""__version__ must exist and match pyproject.toml."""
from __future__ import annotations

from pathlib import Path

import pytest

import vessel_fuel


def test_version_attribute_exists():
    assert isinstance(vessel_fuel.__version__, str)
    assert vessel_fuel.__version__.count(".") == 2


def test_version_matches_pyproject():
    tomllib = pytest.importorskip("tomllib")  # stdlib on 3.11+
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        expected = tomllib.load(f)["project"]["version"]
    assert vessel_fuel.__version__ == expected
