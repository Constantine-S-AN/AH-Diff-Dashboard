# CLI Usage

本文档汇总仓库内可直接调用的命令行入口、参数含义与示例。

## 1. 主 CLI：`python -m ah_premium_lab`

入口文件：`src/ah_premium_lab/cli.py`

查看帮助：

```bash
PYTHONPATH=src python -m ah_premium_lab --help
```

### 1.1 `hello`（冒烟检查）

```bash
PYTHONPATH=src python -m ah_premium_lab hello --message "ah-premium-lab"
```

参数：
- `--message`: 自定义输出文本，默认 `ah-premium-lab`。

### 1.2 `stats`（单对统计检验）

```bash
PYTHONPATH=src python -m ah_premium_lab stats \
  --pair 600036.SS-3968.HK \
  --config config/default.yaml
```

参数：
- `--pair`（必填）: Pair ID，格式 `A_TICKER-H_TICKER`。
- `--config`（可选）: YAML/TOML 配置文件，默认 `config/default.yaml`。

输出：标准输出 JSON，包含 ADF/EG/half-life/summary score 等指标。

### 1.3 `cost-report`（单对成本敏感性 HTML）

```bash
PYTHONPATH=src python -m ah_premium_lab cost-report \
  --pair 600036.SS-3968.HK \
  --config config/default.yaml \
  --output outputs/cost_sensitivity_600036-3968.html
```

参数：
- `--pair`（必填）: Pair ID。
- `--config`（可选）: 配置文件。
- `--output`（可选）: 输出路径；不传则写到 config 里的 report 目录。

## 2. 报告生成 CLI：`ah_premium_lab.report.generate_report`

入口文件：`src/ah_premium_lab/report/generate_report.py`

### 2.1 完整参数

#### 样本与选择
- `--start`（必填）: 开始日期，`YYYY-MM-DD`
- `--end`（必填）: 结束日期，`YYYY-MM-DD`
- `--pairs`: 逗号分隔 Pair ID 列表；留空表示使用 `pairs.csv` 全部
- `--pairs-csv`: Pair 清单路径，默认 `data/pairs.csv`
- `--cache-dir`: 缓存目录，默认 `data/cache`

#### 模型参数
- `--window`: rolling window，默认 `252`
- `--entry`: 入场 z 阈值，默认 `2.0`
- `--exit`: 出场 z 阈值，默认 `0.5`
- `--fx-pair`: `HKDCNY` 或 `CNYHKD`，默认 `HKDCNY`

#### 成本网格（单位 bps）
- `--commission-min / --commission-max / --commission-step`
- `--slippage-min / --slippage-max / --slippage-step`
- `--stamp-min / --stamp-max / --stamp-step`

#### 输出
- `--output`（必填）: `single` 模式下是 HTML 文件；`multi` 模式下是目录
- `--mode`: `single` 或 `multi`，默认 `single`

### 2.2 Single HTML 示例

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

### 2.3 Multi-page 示例

```bash
PYTHONPATH=src python -m ah_premium_lab.report.generate_report \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --pairs 601318.SS-2318.HK,600036.SS-3968.HK \
  --output outputs/ah_research_report_multi \
  --mode multi
```

## 3. 离线 Demo 构建：`scripts/build_offline_demo.py`

```bash
PYTHONPATH=src python scripts/build_offline_demo.py \
  --output-dir docs \
  --cache-dir data/cache \
  --pairs-csv data/pairs.csv \
  --fixture-pairs-csv tests/fixtures/research_regression/pairs.csv \
  --max-pairs 3
```

参数：
- `--output-dir`: 输出目录（默认 `docs`）
- `--cache-dir`: 本地缓存目录（默认 `data/cache`）
- `--pairs-csv`: 主 pair 清单（默认 `data/pairs.csv`）
- `--fixture-pairs-csv`: 离线回退清单（默认 `tests/fixtures/research_regression/pairs.csv`）
- `--max-pairs`: demo 中最多处理 pair 数量（默认 `3`）

## 4. Case Study 构建：`scripts/build_case_study.py`

```bash
PYTHONPATH=src python scripts/build_case_study.py \
  --output docs/case_study.md \
  --reports-dir docs/reports \
  --cache-dir data/cache \
  --pairs "600036.SS-3968.HK,601318.SS-2318.HK" \
  --start 2000-01-01 --end 2100-01-01 \
  --window 252 --entry 2.0 --exit 0.5 \
  --fx-pair HKDCNY
```

参数说明：
- `--pairs` 需要且仅需要 2 对（脚本会校验）。
- 其余参数与报告生成逻辑一致，用于保证复现实验范围稳定。

## 5. 链接推导脚本：`scripts/print_links.py`

```bash
python scripts/print_links.py
```

输出键：
- `pages_url`
- `demo_index`
- `demo_report`

用途：从 `git remote origin` 自动推导 README/Pages 链接。
