import numpy as np
import pandas as pd

from src.data_cleaning import BILL_COLUMNS, PAYMENT_COLUMNS, REPAYMENT_COLUMNS
from src.feature_engineering import add_risk_features


def _sample_frame() -> pd.DataFrame:
    data = {
        "LIMIT_BAL": [100.0, 0.0, 200.0],
        "AGE": [30, 40, 50],
        "default_next_month": [0, 1, 1],
    }
    for column, values in zip(REPAYMENT_COLUMNS, [[0, 1, 2], [0, 2, 2], [1, 0, 2], [2, 0, 2], [0, 0, 2], [0, 0, 0]]):
        data[column] = values
    for column in BILL_COLUMNS:
        data[column] = [50.0, 0.0, -10.0]
    for column in PAYMENT_COLUMNS:
        data[column] = [10.0, 5.0, 1.0]
    return pd.DataFrame(data)


def test_feature_engineering_does_not_modify_input():
    source = _sample_frame()
    original = source.copy(deep=True)
    result = add_risk_features(source)
    pd.testing.assert_frame_equal(source, original)
    assert "overdue_months" in result


def test_ratios_handle_zero_and_negative_denominators():
    result = add_risk_features(_sample_frame())
    assert result.loc[0, "credit_utilization"] == 0.5
    assert np.isnan(result.loc[1, "credit_utilization"])
    assert np.isnan(result.loc[1, "repayment_ratio"])
    assert np.isnan(result.loc[2, "repayment_ratio"])


def test_overdue_counts_streaks_and_segments():
    result = add_risk_features(_sample_frame())
    assert result["overdue_months"].tolist() == [2, 2, 5]
    assert result["longest_overdue_streak"].tolist() == [2, 2, 5]
    assert result["risk_segment"].astype(str).tolist() == ["中风险", "中风险", "高风险"]
    assert result["recent_overdue_months"].tolist() == [0, 2, 2]
    assert result["recent_overdue_pattern"].tolist() == ["Neither month", "Both months", "Both months"]

