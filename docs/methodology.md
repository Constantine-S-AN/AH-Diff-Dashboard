# Methodology

本文档说明本项目在 A/H 溢价研究中的统一口径、统计流程、执行约束与边界条件。所有条目均对应仓库内已实现模块。

## 1. 研究口径（A/H premium 与 log spread）

实现模块：`src/ah_premium_lab/core/premium.py`

核心定义（`compute_premium_metrics`）：
- `h_price_cny_equiv = h_price_hkd * fx_hkd_to_cny * share_ratio`
- `premium_pct = a_price_cny / h_price_cny_equiv - 1`
- `log_spread = log(a_price_cny) - log(h_price_cny_equiv)`

补充指标：
- `rolling_zscore(log_spread, window)`
- `rolling_percentile(premium_pct, window)`

## 2. 数据对齐与样本边界

实现模块：
- `src/ah_premium_lab/data/providers.py`
- `src/ah_premium_lab/data/models.py`

对齐规则：
- A/H/FX 时间索引统一为无时区 `DatetimeIndex`，先去重、排序。
- 三腿数据采用 `inner join` 后再计算溢价，避免非同步日期造成伪信号。
- `fx_pair` 支持 `HKDCNY` 与 `CNYHKD`，后者按倒数转换。
- 离线模式通过 `CacheOnlyPriceProvider` 仅读取 `data/cache/*.parquet`。

## 3. 统计检验（Diagnostics）

实现模块：`src/ah_premium_lab/stats/diagnostics.py`

检验项：
- ADF：`adf_test(series)`
- Engle-Granger：`engle_granger_test(log_A, log_H_fx)`
- 半衰期：`half_life_ar1(series)`
- 综合评分：`summary_score(...)`

研究解释口径：
- `ADF p < 0.05`：更倾向平稳。
- `EG p < 0.05`：更倾向协整。
- 半衰期越短，均值回复速度通常越快，但不直接等价于可交易。

## 4. Rolling 稳定性（Stability）

实现模块：`src/ah_premium_lab/stats/rolling.py`

流程：
- 在滚动窗口内重复 Engle-Granger 检验。
- 输出每窗 `p_value / beta / resid_std`。
- 汇总稳定性指标：
  - `p_value_pass_rate`
  - `beta_variance`
  - `resid_std_drift`
  - `stability_score`

该部分用于识别“只在局部样本有效”的配对关系，减少静态样本内过拟合。

## 5. 结构突变（Breaks）

实现模块：`src/ah_premium_lab/stats/breaks.py`

检测框架：
- rolling 均值差（mean shift）
- Welch t-test
- shift z-score
- CUSUM 统计量

输出字段：
- `break_date`
- `confidence`
- `p_value`
- `shift_zscore`
- `mean_shift`
- `cusum_stat / cusum_p_value`

用途：识别协整关系可能失效的时间段，避免将已突变区间与稳定区间混用。

## 6. 成本敏感性（Cost Sensitivity）

实现模块：
- `src/ah_premium_lab/backtest/costs.py`
- `src/ah_premium_lab/backtest/sensitivity.py`

成本网格支持：
- A/H 双腿 `commission`
- A/H 双腿 `stamp duty`
- `slippage`
- `borrow_bps`

关键输出：
- `net_cagr`
- `net_sharpe`
- `max_dd`
- `breakeven_total_cost`
- `breakeven_slippage`
- `worst_case_net_dd`

## 7. 可执行性约束（Executability）

实现模块：
- `src/ah_premium_lab/backtest/executability.py`
- `src/ah_premium_lab/backtest/pairs_strategy.py`

内置约束：
- `enforce_a_share_t1`: A 股 T+1 卖出约束
- `allow_short_a`: 是否允许做空 A 股
- `allow_short_h`: 是否允许做空 H 股

核心指标：
- `missed_trades`
- `constraint_violation_count`
- `effective_turnover`
- `executability_score`

## 8. 局限与误用风险

局限：
- 当前为日频 close 级别研究，不含盘口冲击、撮合排队、交易容量约束。
- 统计检验与结构突变对窗口、样本区间、缺失处理较敏感。
- 成本阈值结论依赖网格粒度与参数设定，并非唯一真实成本点。

常见误用：
- 将研究评分直接当作实盘信号。
- 忽略 T+1/做空限制后仍声称“可执行”。
- 在结构突变后继续沿用旧参数与旧结论。

## 9. 复现实验入口

```bash
PYTHONPATH=src python scripts/build_case_study.py \
  --pairs "600036.SS-3968.HK,601318.SS-2318.HK"
```

生成文档：`docs/case_study.md`，并输出对应图表与成本报告到 `docs/reports/`。
