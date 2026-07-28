"""Significant-SDC detector training and evaluation."""

from .xgboost import (
    XGBoostConfig,
    add_significant_sdc_target,
    binary_metrics,
    get_feature_columns,
    prepare_features,
    run_detector_job,
    run_xgboost,
)

__all__ = [
    "XGBoostConfig",
    "add_significant_sdc_target",
    "binary_metrics",
    "get_feature_columns",
    "prepare_features",
    "run_detector_job",
    "run_xgboost",
]
