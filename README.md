# ah-premium-lab

`ah-premium-lab` 是一个面向研究/作品集的 A/H 溢价实验项目，目标是提供：
- Streamlit 仪表盘（Overview + Pair Detail）
- 统计检验（ADF、Engle-Granger 协整、半衰期）
- 成本敏感性回测与 HTML 报告

项目边界：
- 不接券商 API，不做实盘下单
- 不提供收益承诺，仅用于研究分析

## Dashboard 截图

以下截图来自本地样例数据运行的 `streamlit_app.py`，用于展示页面结构与交互要点。

### 1) Overview：全市场概览与核心指标

![Dashboard Overview](docs/screenshots/dashboard_overview.png)

### 2) Overview（筛选）：按关键词/阈值筛选股票对

![Dashboard Overview Filtered](docs/screenshots/dashboard_overview_filtered.png)

### 3) Pair Detail：单对详情、统计检验与时序图

![Dashboard Pair Detail](docs/screenshots/dashboard_pair_detail.png)

## 项目结构

```text
ah-premium-lab/
├── .env.example
├── config/
│   └── default.yaml
├── data/
│   ├── cache/
│   └── pairs.csv
├── docs/
│   └── screenshots/
├── src/
│   └── ah_premium_lab/
│       ├── app/            # Streamlit dashboard
│       ├── backtest/       # 策略与成本敏感性
│       ├── core/           # premium/spread 计算
│       ├── data/           # 数据模型、provider、完整性检查
│       ├── report/         # HTML 报告生成
│       └── stats/          # 统计检验
└── tests/
```

## 数据源说明

默认数据源：
- A/H 价格：`yfinance`（Yahoo Finance）
- FX：`HKDCNY` 或 `CNYHKD`（同样由 `yfinance` 获取）

实现细节：
- 数据访问抽象：`PriceProvider` / `YahooFinanceProvider`
- 本地缓存：`data/cache/*.parquet`（按 ticker + 起止日期哈希）
- 数据完整性检查：
  - 缺失交易日（business day）
  - 异常跳点（收益率 zscore 绝对值 > 8）
- universe 管理：
  - `data/pairs_master.csv`：更大股票池（50+ 对）
  - `data/pairs.csv`：回退小样本
- ticker 映射修正：
  - 拉取失败会写入 `data/mapping_overrides.csv`
  - dashboard 会提示“需要人工修正”
- Data Quality 指标：
  - `coverage_pct`
  - `max_gap_days`
  - `outlier_count`
  - `data_quality_score`（缺失率超阈值会降分并标红）
- 协整滚动稳定性：
  - `rolling_engle_granger(logA, logHfx, window=252, step=21)`
  - 指标：`p_value_pass_rate`、`beta_variance`、`resid_std_drift`
  - Pair Detail 增加 rolling p-value / rolling beta 图
  - report 增加“协整稳定性排名 Top/Bottom 10”

样例股票对：
- `data/pairs.csv` 预置 10 对常见 AH 公司（研究样本，不保证覆盖全市场）
- `data/pairs_master.csv` 预置 50+ 对更大 universe（用于批量扫描）

## 可复现步骤

### 1) 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2) 代码质量与测试

```bash
pre-commit install
ruff check .
pytest -q
```

### 2.1) 研究回归测试（golden metrics）

项目包含固定基线回归测试：
- fixture universe: `tests/fixtures/research_regression/pairs.csv`（5 对）
- 固定参数: `tests/fixtures/research_regression/params.yaml`
- golden 指标: `tests/fixtures/research_regression/golden_metrics.json`

执行方式：

```bash
pytest -q tests/test_research_regression.py
```

### 3) 启动 dashboard

```bash
PYTHONPATH=src streamlit run src/ah_premium_lab/app/streamlit_app.py
```

### 4) 生成研究报告（single）

```bash
PYTHONPATH=src python -m ah_premium_lab.report.generate_report \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --pairs-csv data/pairs.csv \
  --cache-dir data/cache \
  --window 252 \
  --entry 2.0 \
  --exit 0.5 \
  --commission-min 0 --commission-max 10 --commission-step 2 \
  --slippage-min 0 --slippage-max 20 --slippage-step 5 \
  --stamp-min 0 --stamp-max 15 --stamp-step 5 \
  --output outputs/ah_research_report.html \
  --mode single
```

### 5) 生成研究报告（multi）

```bash
PYTHONPATH=src python -m ah_premium_lab.report.generate_report \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --pairs 601318.SS-2318.HK,600036.SS-3968.HK \
  --output outputs/ah_research_report_multi \
  --mode multi
```

输出：
- `single`：`outputs/ah_research_report.html`
- `multi`：`outputs/ah_research_report_multi/index.html` + `outputs/ah_research_report_multi/pairs/*.html`

## 参数解释

### Dashboard 关键参数

- `start_date`, `end_date`：样本区间
- `window`：rolling 窗口（用于 zscore、percentile）
- `entry`, `exit`：策略阈值（zscore 入场/出场）
- `fx_pair`：汇率口径（`HKDCNY` 或 `CNYHKD`）
- `score_threshold`：研究评分下限筛选
- `premium_percentile_threshold`：premium 分位数下限筛选
- `keyword`：按 `name/notes` 做关键词筛选

### 报告 CLI 关键参数

- `--start`, `--end`：回溯区间
- `--pairs`：指定股票对（逗号分隔）；为空时使用 `pairs.csv` 全集
- `--window`, `--entry`, `--exit`：信号参数
- `--commission-*`, `--slippage-*`, `--stamp-*`：成本网格扫描参数
- `--mode`：`single` 或 `multi`
- `--output`：输出文件或目录

## 真实性与口径

### 为什么 AH 溢价可能长期存在

即使是同一发行人的 A/H 股票，价格也可能长期偏离，常见原因包括：
- 市场分割与资本流动约束：两地资金并非完全自由流动
- 投资者结构差异：风险偏好、估值体系和交易行为不同
- 流动性与卖空约束：可交易性、借券与做空机制存在差异
- 交易/结算制度差异：交易日历、结算节奏、规则约束影响套利闭环

### 本项目口径

- `premium_pct = A / (H * fx_hkd_to_cny * share_ratio) - 1`
- `log_spread = log(A) - log(H * fx_hkd_to_cny * share_ratio)`

其中 A 端按 CNY，H 端按 HKD，经 FX 折算后对齐比较。

### 参考文档（引用链接）

- 恒生沪深港通 AH 股溢价指数 Factsheet（HSAHP）：[fs_hsahp.pdf](https://www.hsi.com.hk/static/uploads/contents/en/dl_centre/factsheets/fs_hsahp.pdf)
- 上交所英文站 H50066 方法论文档：[H50066_h50066hbooken_EN.pdf](https://english.sse.com.cn/indices/indices/list/indexmethods/c/H50066_h50066hbooken_EN.pdf)

说明：本仓库仅基于公开口径做研究实现，不声称与任何官方指数的成分、权重、调样规则完全一致。

## 统计检验（研究用途）

`stats/diagnostics.py` 提供：
- `adf_test(series) -> p_value, test_stat, used_lags`
- `engle_granger_test(log_A, log_H_fx) -> p_value, beta, resid_series`
- `half_life_ar1(series) -> half_life_days`
- `summary_score(...) -> 0-100`

解释与局限：
- ADF/EG 反映样本内平稳性与协整证据强弱，不代表未来必然均值回归
- 半衰期基于 AR(1) 线性近似，遇到结构断点可能失真
- `summary_score` 是启发式研究评分，不是交易信号

## 局限性 / 不做什么

- 不接实盘：不包含下单、账户、券商 API
- 不做收益承诺：回测/敏感性结果不构成投资建议
- 不覆盖微观执行：未建模撮合细节、冲击成本、借券可得性、融资成本
- 不保证与官方指数完全一致：本项目是研究复现与方法实验，不是指数复制产品

## pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

默认钩子：
- `black`
- `ruff` / `ruff-format`
- `isort`
