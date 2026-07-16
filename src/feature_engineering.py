"""构造可解释的信用风险行为指标。"""

import numpy as np
import pandas as pd

from src.data_cleaning import BILL_COLUMNS, PAYMENT_COLUMNS, REPAYMENT_COLUMNS, validate_schema


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = denominator > 0
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def _longest_overdue_streak(row: pd.Series) -> int:
    longest = current = 0
    for value in row:
        if value > 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def add_risk_features(frame: pd.DataFrame) -> pd.DataFrame:
    """返回包含派生指标的新 DataFrame，不修改输入对象。"""
    validate_schema(frame)
    result = frame.copy()
    result["avg_bill_amount"] = result[BILL_COLUMNS].mean(axis=1)
    result["avg_payment_amount"] = result[PAYMENT_COLUMNS].mean(axis=1)
    result["credit_utilization"] = _safe_ratio(result["avg_bill_amount"], result["LIMIT_BAL"])
    result["repayment_ratio"] = _safe_ratio(
        result[PAYMENT_COLUMNS].sum(axis=1), result[BILL_COLUMNS].clip(lower=0).sum(axis=1)
    )
    result["overdue_months"] = (result[REPAYMENT_COLUMNS] > 0).sum(axis=1)
    result["max_overdue_status"] = result[REPAYMENT_COLUMNS].max(axis=1)
    result["longest_overdue_streak"] = result[REPAYMENT_COLUMNS].apply(
        _longest_overdue_streak, axis=1
    )
    result["risk_segment"] = pd.cut(
        result["overdue_months"], bins=[-1, 0, 2, 6], labels=["低风险", "中风险", "高风险"]
    )
    return result
