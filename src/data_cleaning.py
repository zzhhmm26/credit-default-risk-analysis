"""字段标准化与基础质量检查。"""

import pandas as pd

TARGET = "default_next_month"
REPAYMENT_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
REQUIRED_COLUMNS = ["LIMIT_BAL", "AGE", *REPAYMENT_COLUMNS, *BILL_COLUMNS, *PAYMENT_COLUMNS, TARGET]


def validate_schema(frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")
    labels = set(frame[TARGET].dropna().unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"违约标签应为 0/1，实际为: {sorted(labels)}")


def quality_summary(frame: pd.DataFrame) -> dict[str, object]:
    validate_schema(frame)
    repayment_values = pd.unique(frame[REPAYMENT_COLUMNS].to_numpy().ravel())
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "target_values": sorted(frame[TARGET].dropna().unique().astype(int).tolist()),
        "repayment_codes": sorted(pd.Series(repayment_values).dropna().astype(int).tolist()),
    }
