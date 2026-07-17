"""从 UCI 官方接口获取并保存信用卡违约数据。"""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_default.csv"
UCI_COLUMN_NAMES = {
    "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE", "X5": "AGE",
    "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3", "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
    "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3", "X15": "BILL_AMT4",
    "X16": "BILL_AMT5", "X17": "BILL_AMT6", "X18": "PAY_AMT1", "X19": "PAY_AMT2",
    "X20": "PAY_AMT3", "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
}


def fetch_credit_default_data(output_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """获取 UCI 数据集 350，合并特征与目标后保存为 CSV。"""
    dataset = fetch_ucirepo(id=350)
    features = dataset.data.features.copy().rename(columns=UCI_COLUMN_NAMES)
    targets = dataset.data.targets.copy()
    target_name = targets.columns[0]
    targets = targets.rename(columns={target_name: "default_next_month"})
    frame = pd.concat([features, targets], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


if __name__ == "__main__":
    data = fetch_credit_default_data()
    print(f"已保存 {len(data):,} 行、{data.shape[1]} 列到 {RAW_DATA_PATH}")
