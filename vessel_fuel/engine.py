"""Engine and fuel-consumption utilities."""

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


def sfoc_curve(load_fraction: ArrayLikeFloat, sfoc_at_mcr: float, sfoc_factor: float = 1.0) -> ArrayLikeFloat:
    r"""Compute load-dependent SFOC via the IMO parabolic load correction.

    SFOC follows :math:`\mathrm{SFOC}(\ell)=\mathrm{sfoc\_at\_mcr}\cdot
    \mathrm{CFL}(\ell)/\mathrm{CFL}(1)`, with load-correction factor
    :math:`\mathrm{CFL}(\ell)=0.455\,\ell^2-0.710\,\ell+1.280` (minimum near
    78\% MCR). Normalisation makes ``SFOC(1.0) == sfoc_at_mcr``.

    Parameters
    ----------
    load_fraction : float or numpy.ndarray
        Engine load fraction, :math:`P / MCR`.
    sfoc_at_mcr : float
        Specific fuel oil consumption at MCR (g/kWh).
    sfoc_factor : float, default=1.0
        Calibration multiplier.

    Returns
    -------
    float or numpy.ndarray
        SFOC (g/kWh).

    References
    ----------
    Jalkanen, J.-P., Johansson, L., Kukkonen, J., Brink, A., Kalli, J.,
    Stipa, T. (2012). Extension of an assessment model of ship traffic
    exhaust emissions for particulate matter and carbon monoxide.
    Atmospheric Chemistry and Physics, 12, 2641-2659. (Origin of the
    quadratic relative-SFOC regression on Wartsila engine data.)
    Third and Fourth IMO GHG Study (2014, 2020) --- adoption of the load
    correction in the bottom-up fuel model. The normalisation to the MCR
    anchor (division by CFL(1) = 1.025) is specific to this implementation.
    """
    lf, is_scalar = _as_array(load_fraction)
    x = np.clip(lf, 0.0, 1.0)

    # Parabolic load-correction factor (Third/Fourth IMO GHG Study), minimum
    # near 78% MCR, normalised by its value at full load so SFOC(1.0) == sfoc_at_mcr.
    cfl = 0.455 * x**2 - 0.710 * x + 1.280
    cfl_at_mcr = 0.455 - 0.710 + 1.280
    sfoc = np.maximum(sfoc_at_mcr, 1e-9) * (cfl / cfl_at_mcr) * np.maximum(sfoc_factor, 1e-9)
    return _maybe_scalar(np.asarray(sfoc, dtype=float), is_scalar)


def auxiliary_fuel(aux_power_kw: ArrayLikeFloat, aux_sfoc: ArrayLikeFloat, time_h: ArrayLikeFloat) -> ArrayLikeFloat:
    r"""Compute auxiliary engine fuel consumption.

    Parameters
    ----------
    aux_power_kw : float or numpy.ndarray
        Auxiliary electrical/mechanical power (kW).
    aux_sfoc : float or numpy.ndarray
        Auxiliary engine SFOC (g/kWh).
    time_h : float or numpy.ndarray
        Operating duration (h).

    Returns
    -------
    float or numpy.ndarray
        Auxiliary fuel in metric tons, broadcast to the common shape of
        *aux_power_kw*, *aux_sfoc*, and *time_h*. Scalar only if every one
        of the three inputs is scalar.

    References
    ----------
    Standard fuel mass relation :math:`m = P\,t\,\mathrm{SFOC}`.
    """
    p, p_scalar = _as_array(aux_power_kw)
    s, s_scalar = _as_array(aux_sfoc)
    t, t_scalar = _as_array(time_h)
    is_scalar = p_scalar and s_scalar and t_scalar
    p_b, s_b, t_b = np.broadcast_arrays(p, s, t)
    fuel_mt = np.maximum(p_b, 0.0) * np.maximum(t_b, 0.0) * np.maximum(s_b, 0.0) / 1e6
    return _maybe_scalar(np.asarray(fuel_mt, dtype=float), is_scalar)


def boiler_fuel(boiler_power_kw: ArrayLikeFloat, boiler_sfoc: ArrayLikeFloat, time_h: ArrayLikeFloat) -> ArrayLikeFloat:
    r"""Compute boiler fuel consumption.

    Parameters
    ----------
    boiler_power_kw : float or numpy.ndarray
        Boiler thermal equivalent power (kW).
    boiler_sfoc : float or numpy.ndarray
        Boiler specific fuel consumption (g/kWh equivalent).
    time_h : float or numpy.ndarray
        Operating duration (h).

    Returns
    -------
    float or numpy.ndarray
        Boiler fuel in metric tons, broadcast to the common shape of
        *boiler_power_kw*, *boiler_sfoc*, and *time_h*. Scalar only if
        every one of the three inputs is scalar.

    References
    ----------
    Standard fuel mass relation :math:`m = P\,t\,\mathrm{SFOC}`.
    """
    p, p_scalar = _as_array(boiler_power_kw)
    s, s_scalar = _as_array(boiler_sfoc)
    t, t_scalar = _as_array(time_h)
    is_scalar = p_scalar and s_scalar and t_scalar
    p_b, s_b, t_b = np.broadcast_arrays(p, s, t)
    fuel_mt = np.maximum(p_b, 0.0) * np.maximum(t_b, 0.0) * np.maximum(s_b, 0.0) / 1e6
    return _maybe_scalar(np.asarray(fuel_mt, dtype=float), is_scalar)
