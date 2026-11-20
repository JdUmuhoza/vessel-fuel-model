import numpy as np

from vessel_fuel.emissions import co2e_interval, emission_factors, fuel_to_co2e


def test_mdo_co2e_matches_imo_composite_factor():
    # IMO 4th GHG factors, AR6 GWP100: 3.206 + 0.010e-3*29.8 + 0.160e-3*273 = 3.250 t/t
    out = fuel_to_co2e(100.0, load_fraction=0.75, fuel_type="MDO", gwp="AR6")
    assert np.isclose(out["co2_mt"], 320.6, rtol=1e-6)
    assert np.isclose(out["co2e_mt"], 325.0, rtol=1e-3)


def test_zero_fuel_gives_zero_emissions():
    out = fuel_to_co2e(0.0, load_fraction=0.5)
    assert out["co2_mt"] == 0.0
    assert out["co2e_mt"] == 0.0


def test_array_matches_scalar_loop():
    fuels = np.array([1.0, 5.0, 20.0])
    arr = fuel_to_co2e(fuels, load_fraction=0.6)["co2e_mt"]
    loop = np.array([fuel_to_co2e(float(f), load_fraction=0.6)["co2e_mt"] for f in fuels])
    np.testing.assert_allclose(arr, loop, rtol=1e-9)


def test_hfo_co2_factor_below_mdo():
    ef_hfo = emission_factors("HFO", load_fraction=0.75)
    ef_mdo = emission_factors("MDO", load_fraction=0.75)
    assert ef_hfo["co2_t_per_t"] < ef_mdo["co2_t_per_t"]


def test_lng_methane_slip_amplified_at_low_load():
    lo = emission_factors("LNG", load_fraction=0.2)["ch4_kg_per_t"]
    hi = emission_factors("LNG", load_fraction=0.9)["ch4_kg_per_t"]
    assert lo > hi                       # low-load slip amplification
    mdo = emission_factors("MDO", load_fraction=0.2)["ch4_kg_per_t"]
    assert mdo < lo                      # MDO slip negligible vs LNG


def test_interval_brackets_point_and_orders():
    lo, hi = co2e_interval(9.0, 11.0, load_fraction=0.7)
    point = fuel_to_co2e(10.0, load_fraction=0.7)["co2e_mt"]
    assert lo < point < hi
    lo2, hi2 = co2e_interval(10.0, 10.0, load_fraction=0.7, ef_uncertainty=False)
    assert np.isclose(lo2, hi2)


def test_lfo_is_available_and_carries_the_mrv_default_co2_factor():
    factors = emission_factors("LFO", load_fraction=0.7)
    assert np.isclose(factors["co2_t_per_t"], 3.151)


def test_lfo_co2e_sits_between_hfo_and_mdo():
    lfo = fuel_to_co2e(100.0, load_fraction=0.7, fuel_type="LFO")["co2e_mt"]
    hfo = fuel_to_co2e(100.0, load_fraction=0.7, fuel_type="HFO")["co2e_mt"]
    mdo = fuel_to_co2e(100.0, load_fraction=0.7, fuel_type="MDO")["co2e_mt"]
    assert hfo < lfo < mdo


def test_lfo_vectorises_over_load():
    loads = np.array([0.1, 0.4, 0.9])
    arr = emission_factors("LFO", load_fraction=loads)["n2o_kg_per_t"]
    assert np.asarray(arr).shape == loads.shape
