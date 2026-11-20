"""Research framework for hybrid vessel fuel modeling."""

from .benchmark import run_benchmark_suite
from .data_pipeline import (
    build_feature_matrix,
    clean_observations,
    fit_category_maps,
    generate_synthetic_operational_dataset,
    split_train_val_test,
)
from .ingest import (
    align_and_merge_temporal_data,
    download_copernicus_era5,
    download_gebco_bathymetry,
    download_noaa_ais_archive,
    fuse_operational_data,
    load_ais_segments,
    load_engine_noon,
    load_metocean,
    load_vessel_particulars,
    validate_data_schema,
)
from .external_baselines import load_frozen_results_by_order, load_vendor_baseline_csv
from .hybrid import (
    CubicLawModel,
    HybridResidualModel,
    LSTMModel,
    PhysicsOnlyModel,
    ProphetModel,
    PureMLModel,
    ReferencePhysicsReducedModel,
    SpeedPowerBaseline,
    SymbolicRegressionModel,
)
from .decision_matrix import (
    compute_decision_matrix,
    generate_decision_matrix_report,
    WEIGHT_PROFILES,
)
from .statistical_analysis import (
    combined_ranking,
    friedman_test,
    friedman_test_all_metrics,
    generate_statistical_heatmaps,
    kfold_cross_validate,
    model_selection_report,
    nemenyi_matrix_posthoc,
    nemenyi_posthoc,
    rank_models,
    wilcoxon_all_pairs,
    wilcoxon_pairwise,
)
from .tracking import ExperimentTracker
from .uncertainty import conformal_interval, permutation_sensitivity
from .reporting import export_results_to_excel, generate_confusion_matrices, generate_result_figures, regression_confusion_matrix

__all__ = [
    "run_benchmark_suite",
    "build_feature_matrix",
    "clean_observations",
    "generate_synthetic_operational_dataset",
    "split_train_val_test",
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
    # model classes
    "HybridResidualModel",
    "PhysicsOnlyModel",
    "PureMLModel",
    "SpeedPowerBaseline",
    "CubicLawModel",
    "ReferencePhysicsReducedModel",
    "LSTMModel",
    "ProphetModel",
    "SymbolicRegressionModel",
    # data and baselines
    "load_vendor_baseline_csv",
    "load_frozen_results_by_order",
    "ExperimentTracker",
    # uncertainty
    "conformal_interval",
    "permutation_sensitivity",
    # reporting
    "export_results_to_excel",
    "generate_result_figures",
    "regression_confusion_matrix",
    "generate_confusion_matrices",
    # decision matrix
    "compute_decision_matrix",
    "generate_decision_matrix_report",
    "WEIGHT_PROFILES",
    # statistical analysis
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
]
