# Case Study: AH 双对完整诊断（可复现）

本文档由仓库脚本自动生成，所有表格/阈值结论均来自仓库现有模块计算结果。

## 复现方式

```bash
PYTHONPATH=src python scripts/build_case_study.py --pairs "600036.SS-3968.HK,601318.SS-2318.HK" --start 2000-01-01 --end 2100-01-01 --window 252 --entry 2.00 --exit 0.50 --fx-pair HKDCNY
```

输入数据：
- 仅使用离线缓存：`data/cache`
- pair 来源：`data/pairs_master.csv`（若不存在则自动回退 `data/pairs.csv`）

输出文件：
- 本文档：`docs/case_study.md`
- 报告与表格目录：`docs/reports`

参数：
- 日期区间：`2000-01-01` ~ `2100-01-01`
- rolling window：`252`
- 策略阈值：`entry=2.00`，`exit=0.50`
- FX quote：`HKDCNY`

---

## 600036.SS-3968.HK - China Merchants Bank

### 1) 样本与口径结果
| Metric | Value |
|---|---|
| Pair | `600036.SS-3968.HK` |
| Name | China Merchants Bank |
| A/H | `600036.SS` / `3968.HK` |
| Share Ratio | 1.00 |
| Sample | 2024-02-19 ~ 2026-02-16 |
| Aligned Obs | 521 |
| Latest Premium % | 110.318 |
| Latest Rolling Z | -1.354 |
| Latest Premium Percentile | 0.194 |

### 2) 统计检验
| Metric | Value |
|---|---|
| ADF p-value | 0.591961 |
| ADF stat | -1.3797 |
| ADF lags | 0 |
| EG p-value | 0.526475 |
| EG stat | -1.5142 |
| EG lags | 3 |
| EG beta | -0.1137 |
| Half-life (days) | 96.89 |
| Summary Score (0-100) | 41.72 |

### 3) Rolling 稳定性
| Metric | Value |
|---|---|
| Rolling windows | 14 |
| p-value pass rate (<0.05) | 0.0000 |
| beta variance | 0.250348 |
| resid std drift | 0.2280 |
| stability score (0-100) | 39.28 |

### 4) 结构突变
| Metric | Value |
|---|---|
| Detected breakpoints | 16 |
| Max breakpoint confidence | 1.0000 |
| CUSUM stat | 5.7002 |
| CUSUM p-value | 0.000000 |

### 5) 可执行性约束（研究版回测）
| Metric | Value |
|---|---|
| Gross return | -0.1684 |
| Net return | -0.1684 |
| Max DD gross | -0.2881 |
| Max DD net | -0.2881 |
| Missed trades | 50 |
| Constraint violations | 50 |
| Effective turnover | 4.0000 |
| Executability score (0-100) | 10.00 |

### 6) 成本敏感性阈值
| Metric | Value |
|---|---|
| Best total cost level (bps) | 0.00 |
| Best net CAGR | -0.0605 |
| Best net Sharpe | -0.2203 |
| Breakeven total cost (bps) | 0.00 |
| Breakeven slippage (bps) | 0.00 |
| Worst-case net drawdown | -0.2929 |

结论：
- 统计检验偏弱，rolling 稳定性一般，存在结构突变风险；在成本阈值上，breakeven total cost 约 0.00 bps，breakeven slippage 约 0.00 bps。

图/表产物（全部由仓库模块生成）：
- [Rolling coint table](reports/case_study_600036_SS_3968_HK_rolling_coint.csv)
- [Breakpoints table](reports/case_study_600036_SS_3968_HK_breaks.csv)
- [Cost grid table](reports/case_study_600036_SS_3968_HK_cost_grid.csv)
- [Cost sensitivity report (含热力图/雷达图)](reports/case_study_600036_SS_3968_HK_cost_report.html)


---

## 601318.SS-2318.HK - Ping An Insurance

### 1) 样本与口径结果
| Metric | Value |
|---|---|
| Pair | `601318.SS-2318.HK` |
| Name | Ping An Insurance |
| A/H | `601318.SS` / `2318.HK` |
| Share Ratio | 1.00 |
| Sample | 2024-02-19 ~ 2026-02-16 |
| Aligned Obs | 521 |
| Latest Premium % | 610.842 |
| Latest Rolling Z | 1.147 |
| Latest Premium Percentile | 0.806 |

### 2) 统计检验
| Metric | Value |
|---|---|
| ADF p-value | 0.626836 |
| ADF stat | -1.3051 |
| ADF lags | 0 |
| EG p-value | 0.817316 |
| EG stat | -0.8061 |
| EG lags | 0 |
| EG beta | -0.0730 |
| Half-life (days) | 106.36 |
| Summary Score (0-100) | 31.06 |

### 3) Rolling 稳定性
| Metric | Value |
|---|---|
| Rolling windows | 14 |
| p-value pass rate (<0.05) | 0.0000 |
| beta variance | 0.577870 |
| resid std drift | 0.1756 |
| stability score (0-100) | 33.61 |

### 4) 结构突变
| Metric | Value |
|---|---|
| Detected breakpoints | 16 |
| Max breakpoint confidence | 1.0000 |
| CUSUM stat | 8.4895 |
| CUSUM p-value | 0.000000 |

### 5) 可执行性约束（研究版回测）
| Metric | Value |
|---|---|
| Gross return | 0.1305 |
| Net return | 0.1305 |
| Max DD gross | -0.0161 |
| Max DD net | -0.0161 |
| Missed trades | 12 |
| Constraint violations | 12 |
| Effective turnover | 1.0000 |
| Executability score (0-100) | 7.50 |

### 6) 成本敏感性阈值
| Metric | Value |
|---|---|
| Best total cost level (bps) | 0.00 |
| Best net CAGR | 0.0442 |
| Best net Sharpe | 0.7902 |
| Breakeven total cost (bps) | N/A |
| Breakeven slippage (bps) | N/A |
| Worst-case net drawdown | -0.0359 |

结论：
- 统计检验偏弱，rolling 稳定性一般，存在结构突变风险；在成本阈值上，在当前成本网格内未观察到净收益穿越 0 的阈值点。

图/表产物（全部由仓库模块生成）：
- [Rolling coint table](reports/case_study_601318_SS_2318_HK_rolling_coint.csv)
- [Breakpoints table](reports/case_study_601318_SS_2318_HK_breaks.csv)
- [Cost grid table](reports/case_study_601318_SS_2318_HK_cost_grid.csv)
- [Cost sensitivity report (含热力图/雷达图)](reports/case_study_601318_SS_2318_HK_cost_report.html)

