# Demo

本文档说明如何离线生成并查看项目 demo 页面与报告。

## 1. 目标产物

`build_offline_demo.py` 生成以下文件：
- `docs/index.html`
- `docs/reports/cost_sensitivity_demo.html`
- `docs/screenshots/dashboard_*_demo.svg`

这些文件可直接本地打开，无需联网。

## 2. 生成命令

```bash
PYTHONPATH=src python scripts/build_offline_demo.py
```

可选参数（完整）：

```bash
PYTHONPATH=src python scripts/build_offline_demo.py \
  --output-dir docs \
  --cache-dir data/cache \
  --pairs-csv data/pairs.csv \
  --fixture-pairs-csv tests/fixtures/research_regression/pairs.csv \
  --max-pairs 3
```

## 3. 数据来源优先级

脚本按以下顺序构建 demo：
1. `data/cache`（优先）
2. `tests/fixtures/research_regression`（缓存不足时回退）

因此在无网络环境下，只要缓存或 fixtures 可用，就能稳定产出 demo 页面。

## 4. 本地查看

- 直接在浏览器打开 `docs/index.html`
- 页面内点击报告链接进入 `docs/reports/cost_sensitivity_demo.html`

## 5. 配套文档

- 方法说明：`docs/methodology.md`
- 可复现流程：`docs/reproducibility.md`
- 两对样例诊断：`docs/case_study.md`
