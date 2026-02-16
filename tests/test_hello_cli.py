"""Tests for minimal hello CLI command."""

from __future__ import annotations

import subprocess
import sys


def test_python_module_hello_cli_runs() -> None:
    """`python -m ah_premium_lab hello` should run successfully."""
    completed = subprocess.run(
        [sys.executable, "-m", "ah_premium_lab", "hello", "--message", "ok"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "hello cli: ok" in completed.stdout
