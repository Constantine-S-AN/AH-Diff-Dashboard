# Data Quality

本文档定义 Universe 质量字段 `coverage_pct / max_gap_days / outlier_count` 的计算口径、默认阈值与评分逻辑。

实现模块：
- `src/ah_premium_lab/universe/quality.py`
- `src/ah_premium_lab/data/integrity.py`

## 1. 指标定义

### 1.1 `coverage_pct`

定义：样本区间内，观测到有效数据的业务日占比。

- 期望交易日：`pd.bdate_range(start_date, end_date)`
- 观测交易日：序列清洗后（数值化、去空、归一化日期）与期望交易日的交集
- 公式：`coverage_pct = observed_days / expected_days * 100`

### 1.2 `max_gap_days`

定义：样本区间内最大连续缺失业务日长度。

计算方式：
- 以期望业务日序列为基准，统计前导缺口、后导缺口和内部缺口。
- 取三类缺口中的最大值作为 `max_gap_days`。

### 1.3 `outlier_count`

定义：收益率异常点数量。

计算方式：
- 先算 `returns = pct_change(close)`。
- 计算 z-score：`z = (returns - mean) / std`。
- 默认阈值：`|z| > 8.0` 记为异常点。

### 1.4 `missing_rate`

定义：缺失率。
- 公式：`missing_rate = 1 - observed_days / expected_days`

## 2. Pair 聚合规则

三腿（A/H/FX）单序列质量先独立计算，再聚合为 pair 级字段：

- `coverage_pct = min(a.coverage_pct, h.coverage_pct, fx.coverage_pct)`
- `max_gap_days = max(a.max_gap_days, h.max_gap_days, fx.max_gap_days)`
- `outlier_count = a.outlier_count + h.outlier_count + fx.outlier_count`
- `max_missing_rate = max(a.missing_rate, h.missing_rate, fx.missing_rate)`

## 3. 评分与阈值

默认缺失阈值：`missing_threshold = 0.2`（20%）。

惩罚项：
- `missing_penalty = max_missing_rate * 70`
- `gap_penalty = min(20, max_gap_days * 1.0)`
- `outlier_penalty = min(10, outlier_count * 0.5)`

基础分：
- `base_score = max(0, 100 - missing_penalty - gap_penalty - outlier_penalty)`

分档逻辑：
- 若 `max_missing_rate > 0.2`：
  - `quality_score = base_score * 0.45`
  - `quality_flag = poor`
  - `missing_threshold_breached = true`
- 否则若 `base_score < 70`：
  - `quality_flag = warning`
- 其余：
  - `quality_flag = good`

## 4. Dashboard 呈现口径

在 Streamlit Overview 中：
- `quality_flag == poor` 或 `missing_threshold_breached == true` 会触发行高亮（红色）与降分提示。
- 目的不是“删样本”，而是显式展示数据风险，避免静默失真。

## 5. 阈值解释建议（研究用途）

- `coverage_pct < 80%`：统计结论通常不稳定，应优先补数或降权。
- `max_gap_days` 较大：rolling 指标与断点检测更容易受缺口影响。
- `outlier_count` 偏多：需排查复权、停牌复牌、映射错误或极端行情。

这些阈值是工程默认值，不是市场统一标准；更严格口径可在研究分支中调整并记录参数。
