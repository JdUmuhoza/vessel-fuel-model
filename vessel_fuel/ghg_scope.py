"""Regulatory gas-basket and system-boundary mapping.

Each greenhouse-gas reporting instrument a maritime decision-support system
must answer to consumes a different subset of the model's ``{CO2, CH4, N2O}``
emission-head output, and one of two system boundaries: tank-to-wake
("TtW", combustion only -- what leaves the funnel) or well-to-wake ("WtW",
fuel production and transport plus combustion). The emission head
(:mod:`vessel_fuel.emissions`) computes tank-to-wake factors only, so a
caller feeding its output straight to a well-to-wake instrument would be
silently short by the upstream (production/transport) share of the fuel's
GHG footprint. This module makes both the gas-basket and boundary mismatch
explicit and queryable instead of an assumption a caller could violate
unnoticed.
"""

from __future__ import annotations

from typing import Any

# Per-instrument gas basket, system boundary, regulated metric, and legal /
# IMO source. Sourcing notes:
#
# - CII: the IMO Carbon Intensity Indicator rates *operational* carbon
#   intensity -- CO2 only, tank-to-wake, under the current IMO scope
#   (imo2023mepc377, the 2023 IMO GHG strategy resolution that reaffirms
#   and tightens the CII mechanism).
# - EU-ETS: the maritime extension of the EU Emissions Trading System. CO2
#   has been EU-ETS-liable since 2024; CH4 and N2O join the ETS itself from
#   2026 (the year the 40/70/100 % phase-in reaches full liability). All
#   three gases are monitored tank-to-wake under the MRV Regulation (EU)
#   2015/757 (eu2015mrv) from 2024 -- monitoring precedes ETS liability for
#   CH4/N2O by two years. The 2026 ETS-inclusion date (as opposed to the
#   2024 MRV *monitoring* date) was verified 2026-08-02 via a web lookup of
#   the EUR-Lex text of Directive (EU) 2023/959, recital 20: "Those
#   emissions should first be included in Regulation (EU) 2015/757 from
#   2024, and they should be included in the EU ETS from 2026." Directive
#   2023/959 is not itself a references.bib entry, so eu2015mrv (the MRV
#   Regulation, the actual monitoring/reporting basis for all three gases)
#   is the ``reference`` recorded below; the web lookup confirms the date,
#   it is not the citation.
# - FuelEU: FuelEU Maritime, Regulation (EU) 2023/1805, in force from
#   1 January 2025. Well-to-wake GHG intensity of the energy used on board,
#   covering CO2 + CH4 + N2O -- confirmed 2026-08-02 via a web lookup of the
#   regulation's EUR-Lex text and European Commission guidance. Not yet a
#   references.bib entry, so the legal instrument is cited directly rather
#   than through a citation key.
# - IMO-GFS: the IMO GHG Fuel Standard prices the well-to-wake GHG intensity
#   of the fuel itself (wanaka2026gfs).
INSTRUMENTS: dict[str, dict[str, Any]] = {
    "CII": {
        "gases": ("CO2",),
        "boundary": "TtW",
        "metric": "annual attained vs. required operational carbon intensity (A-E rating)",
        "reference": "imo2023mepc377",
    },
    "EU-ETS": {
        "gases": ("CO2", "CH4", "N2O"),
        "boundary": "TtW",
        "metric": "EU allowances (EUAs) surrendered per tonne of MRV-reported CO2e",
        "reference": "eu2015mrv",
    },
    "FuelEU": {
        "gases": ("CO2", "CH4", "N2O"),
        "boundary": "WtW",
        "metric": "annual average GHG intensity of energy used on board (gCO2e/MJ) against a declining limit",
        "reference": "Regulation (EU) 2023/1805 (FuelEU Maritime)",
    },
    "IMO-GFS": {
        "gases": ("CO2", "CH4", "N2O"),
        "boundary": "WtW",
        "metric": "well-to-wake GHG fuel intensity (gCO2e/MJ) against global fuel-intensity limits",
        "reference": "wanaka2026gfs",
    },
}


def gases_for(instrument: str) -> tuple[str, ...]:
    """Return the greenhouse-gas basket a regulatory instrument consumes.

    Parameters
    ----------
    instrument : str
        One of ``"CII"``, ``"EU-ETS"``, ``"FuelEU"``, ``"IMO-GFS"``.

    Returns
    -------
    tuple of str
        Gas symbols the instrument regulates, a subset of
        ``("CO2", "CH4", "N2O")``.

    Raises
    ------
    KeyError
        If *instrument* is not a key of :data:`INSTRUMENTS`; the error
        message names the unrecognized instrument.

    References
    ----------
    International Maritime Organization (2023). Resolution MEPC.377(80),
    2023 IMO Strategy on Reduction of GHG Emissions from Ships (CII scope).
    European Parliament and Council of the European Union (2015). Regulation
    (EU) 2015/757 on monitoring, reporting and verification of CO2 emissions
    from maritime transport (EU-ETS monitoring basis; CH4/N2O ETS inclusion
    from 2026 per Directive (EU) 2023/959, recital 20).
    European Parliament and Council of the European Union (2023). Regulation
    (EU) 2023/1805 (FuelEU Maritime).
    Wanaka, S., Hiekata, K., Takeuchi, T., and Taniguchi, M. (2026).
    Model-based assessment of marine fuel GHG intensity regulation with
    flexible compliance mechanism.
    """
    if instrument not in INSTRUMENTS:
        raise KeyError(
            f"Unknown regulatory instrument {instrument!r}; expected one of "
            f"{tuple(INSTRUMENTS)}"
        )
    return INSTRUMENTS[instrument]["gases"]
