"""Multi-gas emission head: fuel and engine load to CO2, CH4, N2O, CO2e."""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLikeFloat = Union[float, np.ndarray]


def _as_array(x: ArrayLikeFloat) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=float)
    return arr, arr.ndim == 0


def _maybe_scalar(x: np.ndarray, is_scalar: bool) -> ArrayLikeFloat:
    if is_scalar:
        return float(np.asarray(x))
    return x


# Fuel-based emission factors (per metric ton of fuel), IMO Fourth GHG Study.
# CO2 is stoichiometric (load-invariant); CH4 is engine slip (strongly
# load-dependent for LNG-Otto, negligible for distillates); N2O is a
# combustion factor (flat default, load interface kept for forward use).
# ``ch4_lowload_gain`` amplifies slip below 50 % MCR (Fourth IMO GHG Study
# low-load adjustment concept). ``unc`` are approximate relative uncertainty
# half-ranges from the study's factor ranges.
_FUELS: dict[str, dict[str, float]] = {
    "MDO": {"co2_t_per_t": 3.206, "ch4_kg_per_t": 0.010, "n2o_kg_per_t": 0.160,
            "ch4_lowload_gain": 0.0,
            "unc_co2": 0.02, "unc_ch4": 0.5, "unc_n2o": 0.4},
    "HFO": {"co2_t_per_t": 3.114, "ch4_kg_per_t": 0.012, "n2o_kg_per_t": 0.160,
            "ch4_lowload_gain": 0.0,
            "unc_co2": 0.02, "unc_ch4": 0.5, "unc_n2o": 0.4},
    # LFO ("Light Fuel Oil", the EU MRV/IMO distillate category between HFO and MDO).
    # co2_t_per_t = 3.151 is the EU MRV Regulation 2015/757 Annex I default for Light Fuel
    # Oil, also the IMO default (eu2015mrv) -- a primary legal source, not an estimate.
    # ch4_kg_per_t and n2o_kg_per_t are NOT independently sourced for LFO: the IMO Fourth
    # GHG Study PDF that would carry them is unretrievable in this environment (see
    # docs/paper/refs/_failed.md). These two factors are adopted from the residual-fuel
    # (HFO) entry pending the IMO Fourth GHG Study table -- a bounded assumption, since LFO
    # and HFO are both residual-family fuels and the two liquid entries already share
    # n2o = 0.160, not fabrication. Replace both with LFO-specific factors from that study
    # once it is retrievable. unc_ch4/unc_n2o are widened relative to the HFO entry (0.5,
    # 0.4 -> 0.75, 0.6) to reflect that these two are adopted, not sourced.
    "LFO": {"co2_t_per_t": 3.151, "ch4_kg_per_t": 0.012, "n2o_kg_per_t": 0.160,
            "ch4_lowload_gain": 0.0,
            "unc_co2": 0.02, "unc_ch4": 0.75, "unc_n2o": 0.6},
    # LNG (Otto-cycle interface, forward-compatible): large, uncertain slip.
    "LNG": {"co2_t_per_t": 2.750, "ch4_kg_per_t": 20.0, "n2o_kg_per_t": 0.110,
            "ch4_lowload_gain": 1.5,
            "unc_co2": 0.02, "unc_ch4": 0.6, "unc_n2o": 0.4},
}

# GWP100 values: AR6 (fossil CH4 29.8, N2O 273), AR5 (28, 265).
_GWP: dict[str, dict[str, float]] = {
    "AR6": {"ch4": 29.8, "n2o": 273.0},
    "AR5": {"ch4": 28.0, "n2o": 265.0},
}


def emission_factors(fuel_type: str = "MDO", load_fraction: ArrayLikeFloat = 1.0) -> dict[str, ArrayLikeFloat]:
    r"""Return load-dependent fuel-based emission factors per gas.

    Parameters
    ----------
    fuel_type : str, default="MDO"
        One of ``"MDO"``, ``"HFO"``, ``"LFO"``, ``"LNG"``.
    load_fraction : float or numpy.ndarray, default=1.0
        Engine load fraction :math:`\ell = P/\mathrm{MCR}`.

    Returns
    -------
    dict
        ``co2_t_per_t`` (t CO2 per t fuel, load-invariant), ``ch4_kg_per_t``
        and ``n2o_kg_per_t`` (kg per t fuel), each matching the input type.

    References
    ----------
    Faber, J. et al. (2020). Fourth IMO GHG Study 2020. International
    Maritime Organization.
    European Parliament and Council of the European Union (2015). Regulation
    (EU) 2015/757, Annex I (CO2 default factors, incl. Light Fuel Oil 3.151).
    """
    f = _FUELS[fuel_type.upper()]
    lf, is_scalar = _as_array(load_fraction)
    x = np.clip(lf, 0.0, 1.0)
    # low-load slip amplification: linear gain below 50 % MCR
    slip = f["ch4_kg_per_t"] * (1.0 + f["ch4_lowload_gain"] * np.maximum(0.0, 0.5 - x) / 0.5)
    return {
        "co2_t_per_t": _maybe_scalar(np.full_like(x, f["co2_t_per_t"]), is_scalar),
        "ch4_kg_per_t": _maybe_scalar(np.asarray(slip, dtype=float), is_scalar),
        "n2o_kg_per_t": _maybe_scalar(np.full_like(x, f["n2o_kg_per_t"]), is_scalar),
    }


def fuel_to_co2e(
    fuel_mt: ArrayLikeFloat,
    load_fraction: ArrayLikeFloat = 1.0,
    fuel_type: str = "MDO",
    gwp: str = "AR6",
) -> dict[str, ArrayLikeFloat]:
    r"""Convert fuel mass to per-gas emissions and CO2-equivalent.

    Parameters
    ----------
    fuel_mt : float or numpy.ndarray
        Fuel consumed (metric tons).
    load_fraction : float or numpy.ndarray, default=1.0
        Engine load fraction :math:`\ell = P/\mathrm{MCR}` (drives CH4 slip).
    fuel_type : str, default="MDO"
        One of ``"MDO"``, ``"HFO"``, ``"LFO"``, ``"LNG"``.
    gwp : str, default="AR6"
        GWP100 set: ``"AR6"`` (CH4 29.8, N2O 273) or ``"AR5"`` (28, 265).

    Returns
    -------
    dict
        ``co2_mt`` (t), ``ch4_kg`` (kg), ``n2o_kg`` (kg), ``co2e_mt`` (t),
        each matching the input type of *fuel_mt*.

    References
    ----------
    Faber, J. et al. (2020). Fourth IMO GHG Study 2020. IMO.
    IPCC (2021). AR6 WG1, Chapter 7 (GWP100 values).
    European Parliament and Council of the European Union (2015). Regulation
    (EU) 2015/757, Annex I (CO2 default factors, incl. Light Fuel Oil 3.151).
    """
    fm, is_scalar = _as_array(fuel_mt)
    fm = np.maximum(fm, 0.0)
    ef = emission_factors(fuel_type, load_fraction)
    g = _GWP[gwp.upper()]
    co2 = fm * np.asarray(ef["co2_t_per_t"], dtype=float)
    ch4 = fm * np.asarray(ef["ch4_kg_per_t"], dtype=float)
    n2o = fm * np.asarray(ef["n2o_kg_per_t"], dtype=float)
    co2e = co2 + (ch4 * g["ch4"] + n2o * g["n2o"]) / 1e3
    return {
        "co2_mt": _maybe_scalar(np.asarray(co2, dtype=float), is_scalar),
        "ch4_kg": _maybe_scalar(np.asarray(ch4, dtype=float), is_scalar),
        "n2o_kg": _maybe_scalar(np.asarray(n2o, dtype=float), is_scalar),
        "co2e_mt": _maybe_scalar(np.asarray(co2e, dtype=float), is_scalar),
    }


def co2e_interval(
    fuel_lo_mt: ArrayLikeFloat,
    fuel_hi_mt: ArrayLikeFloat,
    load_fraction: ArrayLikeFloat = 1.0,
    fuel_type: str = "MDO",
    gwp: str = "AR6",
    ef_uncertainty: bool = True,
) -> tuple[ArrayLikeFloat, ArrayLikeFloat]:
    r"""Propagate a fuel interval to a conservative CO2e interval.

    CO2e is monotone increasing in fuel and in every emission factor, so the
    outer interval is obtained by evaluating the lower fuel bound with the
    lower-range factors and the upper fuel bound with the upper-range factors.

    Parameters
    ----------
    fuel_lo_mt, fuel_hi_mt : float or numpy.ndarray
        Lower / upper fuel bounds (metric tons), e.g. a conformal interval.
    load_fraction : float or numpy.ndarray, default=1.0
        Engine load fraction.
    fuel_type : str, default="MDO"
        One of ``"MDO"``, ``"HFO"``, ``"LFO"``, ``"LNG"``.
    gwp : str, default="AR6"
        GWP100 set.
    ef_uncertainty : bool, default=True
        Include the IMO factor uncertainty half-ranges; if False only the
        fuel interval is propagated.

    Returns
    -------
    tuple of float or numpy.ndarray
        ``(co2e_lo_mt, co2e_hi_mt)``.

    References
    ----------
    Faber, J. et al. (2020). Fourth IMO GHG Study 2020. IMO.
    European Parliament and Council of the European Union (2015). Regulation
    (EU) 2015/757, Annex I (CO2 default factors, incl. Light Fuel Oil 3.151).
    """
    lo_f, s1 = _as_array(fuel_lo_mt)
    hi_f, s2 = _as_array(fuel_hi_mt)
    is_scalar = s1 and s2
    f = _FUELS[fuel_type.upper()]
    g = _GWP[gwp.upper()]
    ef = emission_factors(fuel_type, load_fraction)
    ch4 = np.asarray(ef["ch4_kg_per_t"], dtype=float)
    u_co2 = f["unc_co2"] if ef_uncertainty else 0.0
    u_ch4 = f["unc_ch4"] if ef_uncertainty else 0.0
    u_n2o = f["unc_n2o"] if ef_uncertainty else 0.0

    def _co2e(fuel: np.ndarray, sign: float) -> np.ndarray:
        co2 = fuel * f["co2_t_per_t"] * (1.0 + sign * u_co2)
        ch4_kg = fuel * ch4 * (1.0 + sign * u_ch4)
        n2o_kg = fuel * f["n2o_kg_per_t"] * (1.0 + sign * u_n2o)
        return co2 + (ch4_kg * g["ch4"] + n2o_kg * g["n2o"]) / 1e3

    lo = _co2e(np.maximum(lo_f, 0.0), -1.0)
    hi = _co2e(np.maximum(hi_f, 0.0), +1.0)
    return (_maybe_scalar(np.asarray(lo, dtype=float), is_scalar),
            _maybe_scalar(np.asarray(hi, dtype=float), is_scalar))
