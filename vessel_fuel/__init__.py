"""vessel-fuel-model public API."""

from __future__ import annotations

from importlib import import_module

from .current import (
    current_induced_resistance,
    current_vector_component,
    fuel_savings_from_current,
    speed_over_ground,
    transit_time_hours,
)
from .emissions import co2e_interval, emission_factors, fuel_to_co2e
from .engine import auxiliary_fuel, boiler_fuel, sfoc_curve
from .environment import sw_density, sw_viscosity
from .fouling import fouling_delta_cf
from .ghg_scope import INSTRUMENTS, gases_for
from .model import calibrate_model, calibration_report, fuel_components, fuel_model, validate_model
from .resistance import (
    blendermann_wind_resistance,
    holtrop_mennen_resistance,
    kwon_resistance,
    stawave1_resistance,
)

# Keep in sync with [project] version in pyproject.toml (tests/test_version.py
# enforces the match).
__version__ = "0.3.0"

# Research names resolve lazily (PEP 562) so that `import vessel_fuel` needs
# only numpy + scipy. A minimal install uses the physics core alone; an eager
# research import would let any future research-side dependency break
# `from vessel_fuel import fuel_components` in that environment.
_RESEARCH_NAMES = frozenset(
    {
        "CubicLawModel",
        "HybridResidualModel",
        "LSTMModel",
        "PhysicsOnlyModel",
        "ProphetModel",
        "PureMLModel",
        "ReferencePhysicsReducedModel",
        "SpeedPowerBaseline",
        "SymbolicRegressionModel",
        "WEIGHT_PROFILES",
        "align_and_merge_temporal_data",
        "build_feature_matrix",
        "clean_observations",
        "combined_ranking",
        "compute_decision_matrix",
        "conformal_interval",
        "download_copernicus_era5",
        "download_gebco_bathymetry",
        "download_noaa_ais_archive",
        "fit_category_maps",
        "friedman_test",
        "friedman_test_all_metrics",
        "fuse_operational_data",
        "generate_confusion_matrices",
        "generate_decision_matrix_report",
        "generate_statistical_heatmaps",
        "generate_synthetic_operational_dataset",
        "kfold_cross_validate",
        "load_ais_segments",
        "load_engine_noon",
        "load_frozen_results_by_order",
        "load_metocean",
        "load_vendor_baseline_csv",
        "load_vessel_particulars",
        "model_selection_report",
        "nemenyi_matrix_posthoc",
        "nemenyi_posthoc",
        "permutation_sensitivity",
        "rank_models",
        "regression_confusion_matrix",
        "run_benchmark_suite",
        "split_train_val_test",
        "validate_data_schema",
        "wilcoxon_all_pairs",
        "wilcoxon_pairwise",
    }
)


def __getattr__(name: str):
    if name in _RESEARCH_NAMES:
        return getattr(import_module("vessel_fuel.research"), name)
    raise AttributeError(f"module 'vessel_fuel' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _RESEARCH_NAMES)


__all__ = [
    "co2e_interval",
    "emission_factors",
    "fuel_to_co2e",
    "auxiliary_fuel",
    "boiler_fuel",
    "sfoc_curve",
    "sw_density",
    "sw_viscosity",
    "fouling_delta_cf",
    "INSTRUMENTS",
    "gases_for",
    "fuel_model",
    "fuel_components",
    "calibrate_model",
    "calibration_report",
    "validate_model",
    "holtrop_mennen_resistance",
    "blendermann_wind_resistance",
    "stawave1_resistance",
    "kwon_resistance",
    "current_vector_component",
    "speed_over_ground",
    "transit_time_hours",
    "current_induced_resistance",
    "fuel_savings_from_current",
    "run_benchmark_suite",
    "build_feature_matrix",
    "clean_observations",
    "fit_category_maps",
    "generate_synthetic_operational_dataset",
    "load_ais_segments",
    "load_metocean",
    "load_vessel_particulars",
    "load_engine_noon",
    "fuse_operational_data",
    "download_noaa_ais_archive",
    "download_copernicus_era5",
    "download_gebco_bathymetry",
    "align_and_merge_temporal_data",
    "validate_data_schema",
    "split_train_val_test",
    "HybridResidualModel",
    "PhysicsOnlyModel",
    "PureMLModel",
    "SpeedPowerBaseline",
    "CubicLawModel",
    "ReferencePhysicsReducedModel",
    "LSTMModel",
    "ProphetModel",
    "SymbolicRegressionModel",
    "load_vendor_baseline_csv",
    "load_frozen_results_by_order",
    "conformal_interval",
    "permutation_sensitivity",
    "kfold_cross_validate",
    "rank_models",
    "combined_ranking",
    "friedman_test",
    "friedman_test_all_metrics",
    "nemenyi_posthoc",
    "nemenyi_matrix_posthoc",
    "wilcoxon_pairwise",
    "wilcoxon_all_pairs",
    "generate_statistical_heatmaps",
    "model_selection_report",
    "regression_confusion_matrix",
    "generate_confusion_matrices",
]
