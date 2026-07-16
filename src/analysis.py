"""运行第一阶段探索性分析并导出图表与摘要。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_cleaning import TARGET, quality_summary
from src.feature_engineering import add_risk_features

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "credit_default.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "credit_default_features.csv"
FIGURE_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "analysis_summary.json"


def _rate_table(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    return (
        frame.groupby(group, observed=True)[TARGET]
        .agg(customers="size", default_rate="mean")
        .reset_index()
    )


def run_analysis(raw_path: Path = RAW_PATH) -> dict[str, object]:
    frame = pd.read_csv(raw_path)
    quality = quality_summary(frame)
    data = add_risk_features(frame)

    data["credit_limit_band"] = pd.cut(
        data["LIMIT_BAL"], bins=[0, 100_000, 200_000, 500_000, float("inf")],
        labels=["<=100k", "100k-200k", "200k-500k", ">500k"], include_lowest=True
    )
    data["age_band"] = pd.cut(
        data["AGE"], bins=[20, 29, 39, 49, 59, float("inf")],
        labels=["21-29", "30-39", "40-49", "50-59", "60+"], include_lowest=True
    )

    tables = {
        "credit_limit": _rate_table(data, "credit_limit_band"),
        "age": _rate_table(data, "age_band"),
        "overdue_months": _rate_table(data, "overdue_months"),
        "risk_segment": _rate_table(data, "risk_segment"),
    }

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for name, table in tables.items():
        plot_table = table.copy()
        if name == "risk_segment":
            group_column = plot_table.columns[0]
            plot_table[group_column] = plot_table[group_column].astype("object").map(
                {"低风险": "Low", "中风险": "Medium", "高风险": "High"}
            )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=plot_table, x=plot_table.columns[0], y="default_rate", color="#4472C4", ax=ax)
        ax.set(title=f"Default rate by {name.replace('_', ' ')}", xlabel="", ylabel="Default rate")
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"default_rate_by_{name}.png", dpi=160)
        plt.close(fig)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False)
    summary = {
        "quality": quality,
        "overall_default_rate": float(data[TARGET].mean()),
        "zero_or_negative_average_bill_customers": int((data["avg_bill_amount"] <= 0).sum()),
        "undefined_repayment_ratio_customers": int(data["repayment_ratio"].isna().sum()),
        "tables": {
            name: table.assign(**{table.columns[0]: table.iloc[:, 0].astype(str)}).to_dict("records")
            for name, table in tables.items()
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_analysis()
    print(f"总体违约率: {result['overall_default_rate']:.2%}")
    print(f"分析摘要: {SUMMARY_PATH}")
