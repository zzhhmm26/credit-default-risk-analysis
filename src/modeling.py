"""第二阶段：信用卡违约风险预测建模。

这个模块刻意保持“入门但完整”：
- 使用训练集 / 测试集切分，避免只在原数据上自我验证。
- 比较 Dummy、Logistic Regression 和 Random Forest。
- 输出 ROC-AUC、PR-AUC、Recall、Precision、F1 等分类指标。
- 额外模拟“只人工审核风险最高 20% 客户”时能覆盖多少违约客户。

注意：这是学习与作品展示用的探索性模型，不是实际授信或风控生产模型。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_cleaning import BILL_COLUMNS, PAYMENT_COLUMNS, REPAYMENT_COLUMNS, TARGET
from src.feature_engineering import add_risk_features

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "credit_default.csv"
FIGURE_DIR = ROOT / "reports" / "figures"
MODELING_SUMMARY_PATH = ROOT / "reports" / "modeling_summary.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# 第一版模型先不使用 SEX、EDUCATION、MARRIAGE 等人口属性，降低敏感属性误用风险。
MODEL_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    *REPAYMENT_COLUMNS,
    *BILL_COLUMNS,
    *PAYMENT_COLUMNS,
    "avg_bill_amount",
    "avg_payment_amount",
    "credit_utilization",
    "repayment_ratio",
    "overdue_months",
    "max_overdue_status",
    "longest_overdue_streak",
    "recent_overdue_months",
]


def prepare_model_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """构造建模用 X/y，并确保不会把目标变量混入特征。"""
    data = add_risk_features(frame)
    missing = sorted(set(MODEL_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"缺少建模字段: {missing}")
    if TARGET in MODEL_FEATURES:
        raise ValueError("目标变量不能作为模型特征。")
    return data[MODEL_FEATURES].copy(), data[TARGET].astype(int).copy()


def split_model_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """按标签比例分层切分，保证训练集和测试集违约率接近。"""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )


def build_models() -> dict[str, Pipeline]:
    """返回本阶段要比较的模型。"""
    numeric_preprocess = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), MODEL_FEATURES),
        ],
        remainder="drop",
    )
    scaled_preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                MODEL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return {
        "dummy_baseline": Pipeline(
            steps=[
                ("preprocess", numeric_preprocess),
                (
                    "model",
                    DummyClassifier(strategy="prior", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", scaled_preprocess),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", numeric_preprocess),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=25,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _positive_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    return probabilities[:, 1]


def classification_metrics(
    target: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """在指定阈值下计算常用分类指标。"""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def review_capacity_table(
    target: pd.Series,
    probabilities: np.ndarray,
    review_shares: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> pd.DataFrame:
    """模拟只审核预测风险最高的一部分客户时的效果。"""
    ranked = pd.DataFrame({"target": target.to_numpy(), "probability": probabilities})
    ranked = ranked.sort_values("probability", ascending=False).reset_index(drop=True)
    total_defaults = ranked["target"].sum()
    rows = []
    for share in review_shares:
        reviewed = ranked.head(max(1, int(round(len(ranked) * share))))
        defaults_found = reviewed["target"].sum()
        rows.append(
            {
                "review_share": share,
                "reviewed_customers": int(len(reviewed)),
                "defaults_found": int(defaults_found),
                "default_capture_rate": float(defaults_found / total_defaults),
                "review_group_default_rate": float(reviewed["target"].mean()),
                "lift_vs_overall": float(reviewed["target"].mean() / ranked["target"].mean()),
            }
        )
    return pd.DataFrame(rows)


def threshold_table(
    target: pd.Series,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6),
) -> pd.DataFrame:
    """展示不同阈值下“多抓违约”和“少误报”之间的取舍。"""
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "predicted_positive_share": float(predictions.mean()),
                "precision": float(precision_score(target, predictions, zero_division=0)),
                "recall": float(recall_score(target, predictions, zero_division=0)),
                "f1": float(f1_score(target, predictions, zero_division=0)),
                "true_positive": int(tp),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_negative": int(tn),
            }
        )
    return pd.DataFrame(rows)


def feature_importance_table(model: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """提取随机森林的特征重要性。"""
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        raise TypeError("当前模型不提供 feature_importances_。")
    return (
        pd.DataFrame(
            {
                "feature": MODEL_FEATURES,
                "importance": estimator.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _save_modeling_figures(
    fitted_models: dict[str, Pipeline],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    best_model_name: str,
    best_probabilities: np.ndarray,
    threshold_results: pd.DataFrame,
    importances: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in fitted_models.items():
        probabilities = _positive_probabilities(model, x_test)
        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            name=name.replace("_", " ").title(),
            ax=ax,
        )
    ax.set_title("ROC curve: model comparison", weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "modeling_roc_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in fitted_models.items():
        probabilities = _positive_probabilities(model, x_test)
        PrecisionRecallDisplay.from_predictions(
            y_test,
            probabilities,
            name=name.replace("_", " ").title(),
            ax=ax,
        )
    ax.set_title("Precision-Recall curve: imbalanced default prediction", weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "modeling_precision_recall_curve.png", dpi=180)
    plt.close(fig)

    predictions = (best_probabilities >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["No default", "Default"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion matrix: {best_model_name.replace('_', ' ').title()}", weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "modeling_confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    plot_importances = importances.sort_values("importance")
    ax.barh(plot_importances["feature"], plot_importances["importance"], color="#2563EB")
    ax.set(title="Top feature importance: Random Forest", xlabel="Importance", ylabel="")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "modeling_feature_importance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(
        threshold_results["threshold"],
        threshold_results["precision"],
        marker="o",
        label="Precision",
        color="#2563EB",
    )
    ax.plot(
        threshold_results["threshold"],
        threshold_results["recall"],
        marker="o",
        label="Recall",
        color="#DC2626",
    )
    ax.plot(
        threshold_results["threshold"],
        threshold_results["predicted_positive_share"],
        marker="o",
        label="Flagged customer share",
        color="#16A34A",
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set(title="Threshold trade-off", xlabel="Decision threshold", ylabel="Rate")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "modeling_threshold_tradeoff.png", dpi=180)
    plt.close(fig)


def run_modeling(raw_path: Path = RAW_PATH) -> dict[str, object]:
    """训练模型、生成图表并保存建模摘要。"""
    frame = pd.read_csv(raw_path)
    features, target = prepare_model_frame(frame)
    x_train, x_test, y_train, y_test = split_model_data(features, target)

    fitted_models = {}
    metrics = {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        probabilities = _positive_probabilities(model, x_test)
        metrics[name] = classification_metrics(y_test, probabilities)

    best_model_name = max(metrics, key=lambda name: metrics[name]["pr_auc"])
    best_model = fitted_models[best_model_name]
    best_probabilities = _positive_probabilities(best_model, x_test)
    thresholds = threshold_table(y_test, best_probabilities)
    review_table = review_capacity_table(y_test, best_probabilities)
    importances = feature_importance_table(fitted_models["random_forest"])

    _save_modeling_figures(
        fitted_models,
        x_test,
        y_test,
        best_model_name,
        best_probabilities,
        thresholds,
        importances,
    )

    summary = {
        "data": {
            "rows": int(len(frame)),
            "features": MODEL_FEATURES,
            "feature_count": len(MODEL_FEATURES),
            "target": TARGET,
            "overall_default_rate": float(target.mean()),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "train_default_rate": float(y_train.mean()),
            "test_default_rate": float(y_test.mean()),
            "random_state": RANDOM_STATE,
        },
        "models": metrics,
        "best_model_by_pr_auc": best_model_name,
        "threshold_tradeoff": thresholds.to_dict("records"),
        "review_capacity": review_table.to_dict("records"),
        "random_forest_feature_importance": importances.to_dict("records"),
        "figures": [
            "reports/figures/modeling_roc_curve.png",
            "reports/figures/modeling_precision_recall_curve.png",
            "reports/figures/modeling_confusion_matrix.png",
            "reports/figures/modeling_feature_importance.png",
            "reports/figures/modeling_threshold_tradeoff.png",
        ],
        "limitations": [
            "模型只用于学习与作品展示，不用于真实授信决策。",
            "当前版本未做时间外验证，不能证明对未来时期同样稳定。",
            "第一版刻意不使用 SEX、EDUCATION、MARRIAGE 等人口属性，以降低敏感属性误用风险。",
            "模型输出的是相关性预测分数，不代表因果关系。",
        ],
    }
    MODELING_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    result = run_modeling()
    best_model = result["best_model_by_pr_auc"]
    best_metrics = result["models"][best_model]
    print(f"最佳模型: {best_model}")
    print(f"ROC-AUC: {best_metrics['roc_auc']:.3f}")
    print(f"PR-AUC: {best_metrics['pr_auc']:.3f}")
    print(f"Recall@0.5: {best_metrics['recall']:.3f}")
    print(f"建模摘要: {MODELING_SUMMARY_PATH}")
