import numpy as np
import pandas as pd

from src.data_cleaning import BILL_COLUMNS, PAYMENT_COLUMNS, REPAYMENT_COLUMNS, TARGET
from src.modeling import MODEL_FEATURES, prepare_model_frame, review_capacity_table, threshold_table


def _modeling_frame() -> pd.DataFrame:
    data = {
        "LIMIT_BAL": [100_000, 150_000, 80_000, 200_000],
        "AGE": [30, 35, 42, 28],
        TARGET: [0, 1, 0, 1],
    }
    for column, values in zip(
        REPAYMENT_COLUMNS,
        [
            [0, 2, 0, 1],
            [0, 2, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ):
        data[column] = values
    for column in BILL_COLUMNS:
        data[column] = [20_000, 90_000, 10_000, 80_000]
    for column in PAYMENT_COLUMNS:
        data[column] = [5_000, 1_000, 4_000, 1_000]
    return pd.DataFrame(data)


def test_prepare_model_frame_excludes_target_and_keeps_feature_order():
    features, target = prepare_model_frame(_modeling_frame())

    assert list(features.columns) == MODEL_FEATURES
    assert TARGET not in features.columns
    assert target.tolist() == [0, 1, 0, 1]


def test_review_capacity_table_captures_highest_scored_defaults_first():
    target = pd.Series([1, 0, 1, 0, 1])
    probabilities = np.array([0.95, 0.8, 0.7, 0.2, 0.1])

    result = review_capacity_table(target, probabilities, review_shares=(0.4,))

    assert result.loc[0, "reviewed_customers"] == 2
    assert result.loc[0, "defaults_found"] == 1
    assert result.loc[0, "default_capture_rate"] == 1 / 3


def test_threshold_table_reports_confusion_matrix_counts():
    target = pd.Series([1, 0, 1, 0])
    probabilities = np.array([0.8, 0.7, 0.3, 0.1])

    result = threshold_table(target, probabilities, thresholds=(0.5,))

    row = result.iloc[0]
    assert row["true_positive"] == 1
    assert row["false_positive"] == 1
    assert row["false_negative"] == 1
    assert row["true_negative"] == 1
