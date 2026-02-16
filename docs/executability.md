# Executability

本文档说明研究回测中的可执行性约束：T+1、做空限制、missed_trades 与执行评分公式。

实现模块：
- `src/ah_premium_lab/backtest/executability.py`
- `src/ah_premium_lab/backtest/pairs_strategy.py`

## 1. 约束输入与输出

输入：
- `target_a_weight`：策略期望 A 腿目标权重
- `target_h_weight`：策略期望 H 腿目标权重
- `ExecutabilityConfig`：
  - `enforce_a_share_t1`（默认 `True`）
  - `allow_short_a`（默认 `False`）
  - `allow_short_h`（默认 `True`）

输出：
- 约束后权重：`a_weight / h_weight`
- 诊断字段：`missed_trade_flag / constraint_violations`
- 汇总指标：`missed_trades / constraint_violation_count / effective_turnover / executability_score`

## 2. 约束规则（逐日应用）

### 2.1 做空限制

- 若 `allow_short_a = False` 且 `target_a_weight < 0`，则 A 腿截断到 0。
- 若 `allow_short_h = False` 且 `target_h_weight < 0`，则 H 腿截断到 0。

每次触发都记一次 `constraint_violation`。

### 2.2 A 股 T+1

当 `enforce_a_share_t1 = True` 时：
- 若前一日已有 `A > 0` 多头，且当天想降低该多头（`adjusted_a < prev_a`），
- 且距离最近一次 A 买入不足 1 个完整 bar（`days_since_last_a_buy < 1`），
- 则禁止减仓，保持 `adjusted_a = prev_a`。

这等价于研究层面对 A 股当日回转受限的近似表达。

## 3. 指标口径

### 3.1 `missed_trades`

若约束后权重与目标权重任一腿不相等，则该日 `missed_trade_flag = True`。
`missed_trades` 为全样本累计次数。

### 3.2 `constraint_violation_count`

全样本所有约束触发次数总和（可同一交易日多次）。

### 3.3 `effective_turnover`

使用约束后权重计算的两腿换手总量：
- 单日换手：`|a_t - a_{t-1}| + |h_t - h_{t-1}|`
- `effective_turnover`：全样本单日换手求和

## 4. 执行评分公式（0-100）

设：
- `miss_rate = missed_trades / max(signal_days, 1)`
- `violation_rate = constraint_violation_count / max(signal_days, 1)`
- `turnover_preservation = effective_turnover / raw_turnover`（若 `raw_turnover=0` 取 1）

评分：

`score = 100 * (0.50 * (1 - miss_rate) + 0.35 * (1 - violation_rate) + 0.15 * turnover_preservation)`

其中：
- `signal_days` 为无约束目标换手大于 0 的天数。
- 最终分数会 clip 到 `[0, 100]`。

## 5. 解读建议

- `missed_trades` 高：策略信号与交易制度冲突较多。
- `violation_count` 高：参数或信号方向与市场约束不匹配。
- `turnover_preservation` 低：约束后策略行为与原始设计偏离明显。
- `executability_score` 低：研究结论不能直接外推为“可实施”。

## 6. 复现方式

可通过 case study 一次性生成执行性结果与成本阈值报告：

```bash
PYTHONPATH=src python scripts/build_case_study.py \
  --pairs "600036.SS-3968.HK,601318.SS-2318.HK"
```

输出包含 `missed_trades / constraint_violations / executability_score` 与对应成本报告。
