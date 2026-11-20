"""Public API of the released package: importable without optional backends, no deployment code."""

from __future__ import annotations

import subprocess
import sys

import numpy as np


def test_import_without_optional_backends():
    # An import hook raising ModuleNotFoundError reproduces a machine without the backends;
    # sys.modules[name] = None would not, as SciPy then reads sys.modules['torch'].Tensor.
    code = (
        "import sys\n"
        "BLOCKED = ('torch', 'prophet', 'pysr', 'gplearn')\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCKED:\n"
        "            raise ModuleNotFoundError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import vessel_fuel\n"
        "assert callable(vessel_fuel.fuel_model)\n"
        "import vessel_fuel.research\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_inference_symbols():
    import vessel_fuel
    for name in ("predict_state", "list_models"):
        assert not hasattr(vessel_fuel, name)
        assert name not in vessel_fuel.__all__


def _params():
    return {
        "L": 120.0, "B": 20.0, "T": 7.0, "design_draft": 6.5,
        "S": 4200.0, "CB": 0.72, "MCR": 10000.0, "sfoc_at_mcr": 180.0,
        "eta_0": 0.72, "eta_wave_loss": 0.03,
        "A_front": 120.0, "A_lateral": 700.0,
        "months_since_cleaning": 10.0, "trim_m": 0.3,
    }


def _env():
    return {"wind_kn": 12.0, "wind_angle_deg": 20.0, "Hs": 1.5, "Tp": 8.0,
            "wave_angle_deg": 0.0, "sst_c": 15.0, "depth_m": 50.0}


def test_fuel_model_scalar_and_array():
    from vessel_fuel import fuel_model
    speeds = np.array([8.0, 10.0, 12.0])
    arr = fuel_model(100.0, speeds, _env(), _params())
    loop = np.array([fuel_model(100.0, float(s), _env(), _params()) for s in speeds])
    np.testing.assert_allclose(arr, loop, rtol=1e-6)
    assert np.all(np.diff(arr) > 0)
