import pandas as pd

from src.analysis import recent_warning_table, segment_performance


def _insight_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "default_next_month": [0, 0, 1, 0, 1, 1],
            "risk_segment": pd.Categorical(
                ["低风险", "低风险", "中风险", "中风险", "高风险", "高风险"],
                categories=["低风险", "中风险", "高风险"],
                ordered=True,
            ),
            "recent_overdue_pattern": [
                "Neither month",
                "Neither month",
                "Previous month only",
                "Latest month only",
                "Both months",
                "Both months",
            ],
        }
    )


def test_segment_performance_shares_and_lift():
    result = segment_performance(_insight_frame())
    assert result["customer_share"].sum() == 1
    assert result["default_share"].sum() == 1
    high_lift = result.loc[result["risk_segment"] == "高风险", "lift_vs_overall"].iloc[0]
    assert high_lift == 2


def test_recent_warning_table_has_business_order():
    result = recent_warning_table(_insight_frame())
    assert result["recent_overdue_pattern"].astype(str).tolist() == [
        "Neither month",
        "Previous month only",
        "Latest month only",
        "Both months",
    ]
