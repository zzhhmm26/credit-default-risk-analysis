# 信用卡客户违约风险分析与预测

本项目基于 UCI `Default of Credit Card Clients` 数据集，分析客户历史账单、还款和逾期行为与下一期违约之间的关系。第一阶段聚焦数据理解、质量检查、探索性分析、风险指标构造和规则式客户风险分层，暂不训练机器学习模型。

## 项目目的

- 用可复现的数据流程回答信用风险业务问题。
- 展示 Pandas、NumPy、Matplotlib 和统计分析能力。
- 为后续机器学习、阈值与误判成本分析建立可靠基础。

## 项目结构

```text
data/                 # 本地数据（原始和处理后数据不提交）
notebooks/            # 数据理解与风险分析
reports/figures/      # 可复现的分析图表
src/                  # 数据获取、检查、特征工程和分析代码
tests/                # 关键指标与边界情况测试
```

## 快速开始

```powershell
D:\python\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.fetch_data
python -m src.analysis
python -m pytest
```

完整分析结论会在数据流程运行验证后补充，所有数字均来自实际输出。

## 数据来源与许可

数据来自 UCI Machine Learning Repository：[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)，DOI：[10.24432/C55S3H](https://doi.org/10.24432/C55S3H)，采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。仓库不分发原始数据，运行获取脚本可从官方来源取得。

## 使用限制

数据反映特定地区和历史时期，结论不能直接外推到当前中国大陆信用市场。性别、婚姻状况等字段仅用于描述性检查，不应被简单解释为授信依据。分析只能说明数据中的相关关系，不代表因果关系。本项目仅用于学习和作品展示，不用于真实个人的自动化信贷决策。
