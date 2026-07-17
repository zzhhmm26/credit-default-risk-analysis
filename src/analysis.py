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


def segment_performance(frame: pd.DataFrame) -> pd.DataFrame:
    """衡量风险分层覆盖了多少客户、捕获了多少违约事件。"""
    result = (
        frame.groupby("risk_segment", observed=True)[TARGET]
        .agg(customers="size", defaults="sum", default_rate="mean")
        .reset_index()
    )
    result["customer_share"] = result["customers"] / len(frame)
    result["default_share"] = result["defaults"] / frame[TARGET].sum()
    result["lift_vs_overall"] = result["default_rate"] / frame[TARGET].mean()
    return result


def recent_warning_table(frame: pd.DataFrame) -> pd.DataFrame:
    """汇总最近两期逾期模式，观察风险是否随信号恶化而上升。"""
    order = ["Neither month", "Previous month only", "Latest month only", "Both months"]
    result = _rate_table(frame, "recent_overdue_pattern")
    result["recent_overdue_pattern"] = pd.Categorical(
        result["recent_overdue_pattern"], categories=order, ordered=True
    )
    return result.sort_values("recent_overdue_pattern").reset_index(drop=True)


def risk_profile_table(frame: pd.DataFrame) -> pd.DataFrame:
    """比较各风险层的额度、使用率、还款和逾期行为。"""
    return (
        frame.groupby("risk_segment", observed=True)
        .agg(
            customers=(TARGET, "size"),
            default_rate=(TARGET, "mean"),
            avg_credit_limit=("LIMIT_BAL", "mean"),
            median_utilization=("credit_utilization", "median"),
            median_repayment_ratio=("repayment_ratio", "median"),
            avg_overdue_months=("overdue_months", "mean"),
            avg_bill_amount=("avg_bill_amount", "mean"),
            avg_payment_amount=("avg_payment_amount", "mean"),
        )
        .reset_index()
    )


def _draw_rate_bars(ax: plt.Axes, table: pd.DataFrame, title: str) -> None:
    """绘制带数值标签的违约率柱状图。"""
    rates = table["default_rate"]
    colors = sns.color_palette("Blues", n_colors=len(table) + 3)[3:]
    labels = table.iloc[:, 0].astype(str).tolist()
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=1)
    ax.set(title=title, xlabel="", ylabel="Default rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_ylim(0, min(1, max(0.1, float(rates.max()) * 1.2)))
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in rates], padding=3, fontsize=9)
    sns.despine(ax=ax)


def _save_overview_figures(data: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    """生成适合 GitHub 首页展示的总体构成图和综合仪表板。"""
    counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.pie(
        counts, labels=["No default", "Default"], colors=["#2563EB", "#EF4444"],
        autopct="%1.1f%%", startangle=90, counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    ax.text(0, 0.06, f"{len(data):,}", ha="center", va="center", fontsize=24, weight="bold")
    ax.text(0, -0.10, "customers", ha="center", va="center", fontsize=10, color="#64748B")
    ax.set_title("Customer default composition", fontsize=16, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "default_composition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].pie(
        counts, labels=["No default", "Default"], colors=["#2563EB", "#EF4444"],
        autopct="%1.1f%%", startangle=90, counterclock=False,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2},
    )
    axes[0, 0].set_title("Portfolio composition", weight="bold")

    overdue = tables["overdue_months"]
    axes[0, 1].plot(
        overdue.iloc[:, 0].astype(str), overdue["default_rate"],
        color="#DC2626", marker="o", linewidth=2.5, markersize=7,
    )
    for x, value in enumerate(overdue["default_rate"]):
        axes[0, 1].annotate(f"{value:.1%}", (x, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    axes[0, 1].set(title="Default rate rises with overdue months", xlabel="Overdue months", ylabel="Default rate")
    axes[0, 1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[0, 1].set_ylim(0, 0.8)
    sns.despine(ax=axes[0, 1])

    _draw_rate_bars(axes[1, 0], tables["credit_utilization"], "Credit utilization")
    _draw_rate_bars(axes[1, 1], tables["repayment_ratio"], "Repayment ratio")
    fig.suptitle("Credit Default Risk Dashboard", fontsize=22, weight="bold", y=1.02)
    fig.text(0.5, 0.005, "Source: UCI Default of Credit Card Clients · 30,000 observations", ha="center", color="#64748B")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "risk_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_business_figures(
    segment_table: pd.DataFrame,
    warning_table: pd.DataFrame,
    utilization_limit_matrix: pd.DataFrame,
    utilization_limit_counts: pd.DataFrame,
) -> None:
    """生成回答分层效果、提前预警和额度策略问题的业务图表。"""
    segment_plot = segment_table.copy()
    segment_plot["risk_segment"] = segment_plot["risk_segment"].astype("object").map(
        {"低风险": "Low", "中风险": "Medium", "高风险": "High"}
    )
    x = list(range(len(segment_plot)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        [value - width / 2 for value in x], segment_plot["customer_share"], width,
        label="Customer share", color="#60A5FA",
    )
    ax.bar(
        [value + width / 2 for value in x], segment_plot["default_share"], width,
        label="Share of all defaults", color="#F87171",
    )
    ax.set_xticks(x, segment_plot["risk_segment"])
    ax.set_ylabel("Share")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("Risk tiers: portfolio size vs defaults captured", weight="bold")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{value:.1%}" for value in container.datavalues],
            padding=3,
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "risk_tier_capture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = ["Neither", "Previous only", "Latest only", "Both months"]
    bars = ax.bar(
        labels,
        warning_table["default_rate"],
        color=["#93C5FD", "#FCD34D", "#FB923C", "#EF4444"],
    )
    ax.set(
        title="Recent overdue signals and next-month default",
        xlabel="Overdue in the latest two months",
        ylabel="Default rate",
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.bar_label(
        bars,
        labels=[
            f"{rate:.1%}\n(n={count:,})"
            for rate, count in zip(warning_table["default_rate"], warning_table["customers"])
        ],
        padding=4,
    )
    ax.set_ylim(0, min(1, warning_table["default_rate"].max() * 1.25))
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "recent_overdue_warning.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    annotations = utilization_limit_matrix.copy().astype(object)
    for row in annotations.index:
        for column in annotations.columns:
            rate = utilization_limit_matrix.loc[row, column]
            count = utilization_limit_counts.loc[row, column]
            annotations.loc[row, column] = f"{rate:.1%}\nn={count:,.0f}"
    small_sample_mask = utilization_limit_counts < 30
    fig, ax = plt.subplots(figsize=(11, 6.2))
    sns.heatmap(
        utilization_limit_matrix,
        annot=annotations,
        fmt="",
        cmap="YlOrRd",
        vmin=0.1,
        vmax=0.4,
        linewidths=1,
        mask=small_sample_mask,
        cbar_kws={"label": "Default rate"},
        ax=ax,
    )
    ax.set(
        title="Default rate by credit limit and utilization (cells with n < 30 hidden)",
        xlabel="Credit utilization",
        ylabel="Credit limit",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "limit_utilization_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    data["utilization_band"] = pd.cut(
        data["credit_utilization"],
        bins=[float("-inf"), 0, 0.25, 0.5, 0.75, 1, float("inf")],
        labels=["<=0", "0-25%", "25-50%", "50-75%", "75-100%", ">100%"],
    )
    data["repayment_ratio_band"] = pd.cut(
        data["repayment_ratio"],
        bins=[-0.001, 0.1, 0.3, 0.5, 1, float("inf")],
        labels=["0-10%", "10-30%", "30-50%", "50-100%", ">100%"],
        include_lowest=True,
    )
    data["avg_bill_quartile"] = pd.qcut(
        data["avg_bill_amount"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    )
    data["avg_payment_quartile"] = pd.qcut(
        data["avg_payment_amount"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    )
    data["two_month_overdue_pattern"] = pd.Series(pd.NA, index=data.index, dtype="object")
    two_overdue = data["overdue_months"] == 2
    data.loc[two_overdue, "two_month_overdue_pattern"] = "Scattered"
    data.loc[
        two_overdue & (data["longest_overdue_streak"] >= 2),
        "two_month_overdue_pattern",
    ] = "Consecutive"

    tables = {
        "credit_limit": _rate_table(data, "credit_limit_band"),
        "age": _rate_table(data, "age_band"),
        "overdue_months": _rate_table(data, "overdue_months"),
        "longest_overdue_streak": _rate_table(data, "longest_overdue_streak"),
        "two_month_overdue_pattern": _rate_table(data, "two_month_overdue_pattern"),
        "credit_utilization": _rate_table(data, "utilization_band"),
        "repayment_ratio": _rate_table(data, "repayment_ratio_band"),
        "average_bill": _rate_table(data, "avg_bill_quartile"),
        "average_payment": _rate_table(data, "avg_payment_quartile"),
        "risk_segment": _rate_table(data, "risk_segment"),
    }
    segment_table = segment_performance(data)
    warning_table = recent_warning_table(data)
    profile_table = risk_profile_table(data)
    utilization_limit_matrix = data.pivot_table(
        index="credit_limit_band",
        columns="utilization_band",
        values=TARGET,
        aggfunc="mean",
        observed=True,
    )
    utilization_limit_counts = data.pivot_table(
        index="credit_limit_band",
        columns="utilization_band",
        values=TARGET,
        aggfunc="size",
        observed=True,
    )

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
        _draw_rate_bars(ax, plot_table, f"Default rate by {name.replace('_', ' ')}")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"default_rate_by_{name}.png", dpi=160)
        plt.close(fig)

    _save_overview_figures(data, tables)
    _save_business_figures(
        segment_table,
        warning_table,
        utilization_limit_matrix,
        utilization_limit_counts,
    )

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
        "business_insights": {
            "risk_segment_performance": segment_table.assign(
                risk_segment=segment_table["risk_segment"].astype(str)
            ).to_dict("records"),
            "recent_overdue_warning": warning_table.assign(
                recent_overdue_pattern=warning_table["recent_overdue_pattern"].astype(str)
            ).to_dict("records"),
            "risk_profiles": profile_table.assign(
                risk_segment=profile_table["risk_segment"].astype(str)
            ).to_dict("records"),
            "limit_utilization_default_rate": {
                str(index): {str(column): value for column, value in row.items()}
                for index, row in utilization_limit_matrix.to_dict("index").items()
            },
            "limit_utilization_customers": {
                str(index): {str(column): value for column, value in row.items()}
                for index, row in utilization_limit_counts.to_dict("index").items()
            },
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_analysis()
    print(f"总体违约率: {result['overall_default_rate']:.2%}")
    print(f"分析摘要: {SUMMARY_PATH}")
