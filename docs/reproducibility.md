# Reproducibility

本文档说明如何用仓库现有机制稳定复跑研究结果：缓存、fixtures、golden metrics 与离线产物。

## 1. 固定资产与目录

缓存与输入：
- `data/cache/`：A/H/FX 本地缓存（parquet）
- `data/pairs.csv`、`data/pairs_master.csv`：研究 pair 清单

回归夹具：
- `tests/fixtures/research_regression/pairs.csv`
- `tests/fixtures/research_regression/params.yaml`
- `tests/fixtures/research_regression/golden_metrics.json`

离线文档产物：
- `docs/index.html`
- `docs/reports/*`

## 2. 固定参数与 Golden Metrics

`tests/test_research_regression.py` 会：
- 读取固定参数（`params.yaml`）与固定 pair 集合。
- 使用确定性合成序列计算指标。
- 与 `golden_metrics.json` 做近似匹配校验（绝对/相对容忍阈值）。

目的：防止研究核心指标在重构后发生无感漂移。

## 3. 最小复跑命令

### 3.1 研究回归测试

```bash
PYTHONPATH=src pytest -q tests/test_research_regression.py
```

### 3.2 生成离线 demo 文档

```bash
PYTHONPATH=src python scripts/build_offline_demo.py
```

## 4. 完整复跑建议

```bash
PYTHONPATH=src pytest -q
OFFLINE=1 PYTHONPATH=src streamlit run src/ah_premium_lab/app/streamlit_app.py
```

说明：
- `OFFLINE=1` 会强制只读缓存，不调用 `yfinance`。
- 若缓存不足，可先在联网模式补齐一次缓存，再切回离线模式验证。

## 5. 结果变更治理

当研究逻辑确实发生预期变化时：
- 先保留原参数与旧 golden 文件对比差异。
- 在 PR 中记录“为什么指标变了”。
- 再更新 `golden_metrics.json`，避免无说明漂移。
