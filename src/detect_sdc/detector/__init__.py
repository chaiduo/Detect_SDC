"""Significant-SDC detector training and evaluation."""

from .xgboost import (
    XGBoostConfig,
    add_significant_sdc_target,
    binary_metrics,
    calibrate_threshold_at_fpr,
    calibrate_threshold_max_f1,
    get_feature_columns,
    prepare_features,
    run_detector_job,
    run_calibrated_xgboost,
    significant_sdc_negative_mask,
)

__all__ = [
    "XGBoostConfig",
    "add_significant_sdc_target",
    "binary_metrics",
    "calibrate_threshold_at_fpr",
    "calibrate_threshold_max_f1",
    "get_feature_columns",
    "prepare_features",
    "run_detector_job",
    "run_calibrated_xgboost",
    "significant_sdc_negative_mask",
]
