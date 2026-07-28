"""Compatibility layer for the shared significant-SDC detector."""

from pathlib import Path
import sys


SHARED_SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SHARED_SOURCE))

from detect_sdc.detector import xgboost as _shared  # noqa: E402


RANDOM_STATE = 42
LEARNING_RATE = 0.01
CLASS_LABELS = list(_shared.CLASS_LABELS)
CLASS_NAMES = _shared.CLASS_NAMES
DROP_ALL_FEATURE_NAN = False

target_counts = _shared.target_counts
label_significance_counts = _shared.label_significance_counts
add_significant_sdc_target = _shared.add_significant_sdc_target
get_feature_columns = _shared.get_feature_columns
prepare_features = _shared.prepare_features
all_feature_nan_mask = _shared.all_feature_nan_mask
class_weight_vector = _shared.class_weight_vector
sample_weights = _shared.sample_weights
binary_metrics = _shared.binary_metrics


def split_train_test(frame, group_col):
    split = _shared.split_by_group(
        frame,
        group_column=group_col,
        holdout_ratio=0.15,
        random_state=RANDOM_STATE,
    )
    return split.train, split.holdout


def build_xgb_binary_classifier():
    return _shared.build_classifier(
        _shared.XGBoostConfig(learning_rate=LEARNING_RATE)
    )


def train_binary_model(x_train, y_train, x_test, y_test):
    return _shared.train_binary_model(
        x_train,
        y_train,
        x_test,
        y_test,
        config=_shared.XGBoostConfig(learning_rate=LEARNING_RATE),
    )


def run_ternary_xgboost(train_csv, valid_csv, group_col="orig_id", **kwargs):
    return _shared.run_legacy_xgboost(
        train_csv,
        valid_csv,
        group_col=group_col,
        learning_rate=LEARNING_RATE,
        **kwargs,
    )


def run_ternary_xgboost_compare_nan_modes(
    train_csv,
    valid_csv,
    group_col="orig_id",
):
    return _shared.run_legacy_compare(
        train_csv,
        valid_csv,
        group_col=group_col,
        learning_rate=LEARNING_RATE,
    )
