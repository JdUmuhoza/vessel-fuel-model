import numpy as np

from vessel_fuel.engine import auxiliary_fuel, boiler_fuel, sfoc_curve


def test_sfoc_normal_and_zones():
    sf = sfoc_curve(np.array([0.2, 0.4, 0.6, 0.8, 0.95]), sfoc_at_mcr=180.0)
    assert sf.shape == (5,)
    assert np.all(sf > 0.0)


def test_aux_and_boiler_zero_time():
    assert auxiliary_fuel(500.0, 205.0, 0.0) == 0.0
    assert boiler_fuel(300.0, 280.0, 0.0) == 0.0


def test_array_matches_scalar_loop():
    loads = np.array([0.2, 0.5, 0.8])
    arr = sfoc_curve(loads, 180.0)
    loop = np.array([sfoc_curve(float(l), 180.0) for l in loads])
    assert np.allclose(arr, loop)


def test_sfoc_equals_input_at_mcr():
    # IMO-consistent: SFOC at full load (MCR) must equal the rated sfoc_at_mcr.
    assert np.isclose(sfoc_curve(1.0, 180.0), 180.0)


def test_sfoc_factor_scales_mcr_value():
    assert np.isclose(sfoc_curve(1.0, 180.0, sfoc_factor=1.1), 198.0)


def test_sfoc_parabolic_min_below_mcr():
    loads = np.linspace(0.1, 1.0, 91)
    sf = sfoc_curve(loads, 180.0)
    imin = int(np.argmin(sf))
    assert 0.70 <= loads[imin] <= 0.85          # most-efficient point below MCR
    assert sf.min() < sfoc_curve(1.0, 180.0)    # MCR is not the most efficient point
    assert sfoc_curve(0.15, 180.0) > sf.min()   # SFOC rises at low load


# ---------------------------------------------------------------------------
# auxiliary_fuel / boiler_fuel: scalar-power + array-time_h shape defect.
#
# Both functions decided scalar-vs-array output from the *power* argument
# alone. A multi-segment voyage call — scalar nameplate power, per-segment
# array time_h — is a legitimate call shape under the declared signatures
# and raised TypeError instead of returning an array. See
# `.claude/rules/code-style.md` "Scalar/array duality": a function is
# scalar-returning only when *every* numeric input is scalar.
# ---------------------------------------------------------------------------


def test_auxiliary_fuel_scalar_power_array_time_returns_array():
    time_h = np.array([1.0, 2.0, 3.0])
    fuel = auxiliary_fuel(500.0, 205.0, time_h)
    assert isinstance(fuel, np.ndarray)
    assert fuel.shape == time_h.shape


def test_boiler_fuel_scalar_power_array_time_returns_array():
    time_h = np.array([1.0, 2.0, 3.0])
    fuel = boiler_fuel(300.0, 280.0, time_h)
    assert isinstance(fuel, np.ndarray)
    assert fuel.shape == time_h.shape


def test_auxiliary_fuel_array_power_scalar_time_returns_array():
    power = np.array([400.0, 500.0, 600.0])
    fuel = auxiliary_fuel(power, 205.0, 2.0)
    assert isinstance(fuel, np.ndarray)
    assert fuel.shape == power.shape


def test_boiler_fuel_array_power_scalar_time_returns_array():
    power = np.array([200.0, 300.0, 400.0])
    fuel = boiler_fuel(power, 280.0, 2.0)
    assert isinstance(fuel, np.ndarray)
    assert fuel.shape == power.shape


def test_auxiliary_fuel_all_scalar_returns_float():
    fuel = auxiliary_fuel(500.0, 205.0, 2.0)
    assert isinstance(fuel, float)


def test_boiler_fuel_all_scalar_returns_float():
    fuel = boiler_fuel(300.0, 280.0, 2.0)
    assert isinstance(fuel, float)


def test_auxiliary_fuel_array_matches_scalar_loop():
    time_h = np.array([1.0, 2.5, 4.0])
    arr = auxiliary_fuel(500.0, 205.0, time_h)
    loop = np.array([auxiliary_fuel(500.0, 205.0, float(t)) for t in time_h])
    np.testing.assert_allclose(arr, loop, rtol=1e-6)


def test_boiler_fuel_array_matches_scalar_loop():
    time_h = np.array([1.0, 2.5, 4.0])
    arr = boiler_fuel(300.0, 280.0, time_h)
    loop = np.array([boiler_fuel(300.0, 280.0, float(t)) for t in time_h])
    np.testing.assert_allclose(arr, loop, rtol=1e-6)


def test_auxiliary_fuel_zero_power_and_zero_time():
    assert auxiliary_fuel(0.0, 205.0, 0.0) == 0.0
    assert auxiliary_fuel(500.0, 205.0, 0.0) == 0.0
    assert auxiliary_fuel(0.0, 205.0, 2.0) == 0.0


def test_boiler_fuel_zero_power_and_zero_time():
    assert boiler_fuel(0.0, 280.0, 0.0) == 0.0
    assert boiler_fuel(300.0, 280.0, 0.0) == 0.0
    assert boiler_fuel(0.0, 280.0, 2.0) == 0.0
