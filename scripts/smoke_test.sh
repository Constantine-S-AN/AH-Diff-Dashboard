#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${SMOKE_VENV_DIR:-.venv-smoke}"

echo "[smoke] root=${ROOT_DIR}"
echo "[smoke] python=${PYTHON_BIN}"
echo "[smoke] venv=${VENV_DIR}"

rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
pip install -e ".[dev]"

ruff check .
PYTHONPATH=src pytest -q

# Import-only check for Streamlit app module; does not start server.
PYTHONPATH=src python -c "import ah_premium_lab.app.streamlit_app as app; print('streamlit app import check passed')"

echo "[smoke] success"
