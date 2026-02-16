# AH-Diff-Dashboard

[![CI](https://img.shields.io/github/actions/workflow/status/Constantine-S-AN/AH-Diff-Dashboard/ci-pages.yml?branch=main&label=CI)](https://github.com/Constantine-S-AN/AH-Diff-Dashboard/actions/workflows/ci-pages.yml) [![GitHub Pages](https://img.shields.io/website?url=https%3A%2F%2Fconstantine-s-an.github.io%2FAH-Diff-Dashboard%2F&label=GitHub%20Pages)](https://constantine-s-an.github.io/AH-Diff-Dashboard/) [![License](https://img.shields.io/github/license/Constantine-S-AN/AH-Diff-Dashboard)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![Ruff](https://img.shields.io/badge/Lint-Ruff-46A758?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)

这是一个面向研究与作品集展示的 AH 溢价分析平台，聚焦统计有效性与工程可复现性。

**Quick links（四个链接）**
- Demo（GitHub Pages）：[Open](https://constantine-s-an.github.io/AH-Diff-Dashboard/)
- Demo Report（HTML）：[Open](https://constantine-s-an.github.io/AH-Diff-Dashboard/reports/cost_sensitivity_demo.html)
- Quickstart：[Jump](#quickstart)
- Methodology：[docs/methodology.md](docs/methodology.md)

_Links are derived from `git remote origin` via `scripts/print_links.py`._

## What it does
该项目用于回答一个研究问题：AH 溢价是否具备稳定统计特征，以及该特征是否会被交易成本与执行约束吞噬。

- 统一口径计算 `AH premium / log spread`，并完成 FX 对齐与样本对齐。
- 提供 ADF、Engle-Granger、half-life、rolling 稳定性与结构突变诊断。
- 支持缓存优先与离线模式，减少数据源波动对研究结论的影响。
- 输出数据质量指标（coverage / max_gap / outliers）并纳入评分流程。
- 建模可执行性约束（T+1、做空限制、missed trades）并量化执行评分。
- 通过 golden metrics 回归测试与离线 HTML 报告保证可复现交付。

## Quickstart
### Docker
```bash
# make sure Docker Desktop/daemon is running
docker compose up --build
# open:
http://localhost:8501
```

### Local
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]  # package: ah-premium-lab
ruff check . && PYTHONPATH=src pytest -q
PYTHONPATH=src python -c "import ah_premium_lab.app.streamlit_app as app; print('streamlit app import check passed')"
# optional: start dashboard
PYTHONPATH=src streamlit run src/ah_premium_lab/app/streamlit_app.py
```

## Reproducibility
- 缓存目录：`data/cache/`；关键参数由脚本固定，产物可回溯。
- 研究回归测试（golden metrics）：
```bash
PYTHONPATH=src pytest -q tests/test_research_regression.py
```
- 生成离线 demo 与报告：
```bash
PYTHONPATH=src python scripts/build_offline_demo.py
PYTHONPATH=src python scripts/build_case_study.py --pairs "600036.SS-3968.HK,601318.SS-2318.HK"
```

## Docs
- Methodology: [docs/methodology.md](docs/methodology.md)
- CLI: [docs/cli.md](docs/cli.md)
- Data quality: [docs/data_quality.md](docs/data_quality.md)
- Executability: [docs/executability.md](docs/executability.md)
- Reproducibility: [docs/reproducibility.md](docs/reproducibility.md)
- Demo guide: [docs/demo.md](docs/demo.md)

## Disclaimer
- 仅用于研究与工程展示，不用于实盘交易。
- 不对接券商 API，不提供下单或执行能力。
- 不构成投资建议，历史结果不代表未来表现。

## License & Citation
- License: [MIT](LICENSE)
- Citation metadata: [CITATION.cff](CITATION.cff)
- One-pager: [docs/one_pager.md](docs/one_pager.md)
