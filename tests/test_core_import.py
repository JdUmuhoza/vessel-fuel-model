"""Core-import guarantees for a minimal install (physics core only)."""
from __future__ import annotations

import subprocess
import sys


def test_import_vessel_fuel_does_not_load_research():
    code = (
        "import sys\n"
        "import vessel_fuel\n"
        "assert callable(vessel_fuel.fuel_components)\n"
        "sys.exit(1 if 'vessel_fuel.research' in sys.modules else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        "importing vessel_fuel eagerly loaded vessel_fuel.research:\n" + result.stderr
    )


def test_research_names_still_reachable_lazily():
    code = (
        "from vessel_fuel import HybridResidualModel, run_benchmark_suite\n"
        "import vessel_fuel\n"
        "assert vessel_fuel.PhysicsOnlyModel is not None\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_unknown_attribute_raises():
    code = (
        "import vessel_fuel\n"
        "try:\n"
        "    vessel_fuel.does_not_exist\n"
        "except AttributeError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
