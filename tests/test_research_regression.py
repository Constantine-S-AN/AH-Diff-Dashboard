"""Research regression tests against fixed golden metrics."""

from __future__ import annotations

import json
import math

from tests.research_regression_utils import (
    GOLDEN_PATH,
    compute_fixture_metrics,
    load_regression_pairs,
    load_regression_params,
)

ABS_TOL = 1e-6
REL_TOL = 1e-3


def test_research_regression_metrics_match_golden() -> None:
    """Deterministic fixture metrics should stay close to committed golden values."""

    params = load_regression_params()
    pairs = load_regression_pairs()
    assert len(pairs) == 5

    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = compute_fixture_metrics(pairs, params)

    assert set(actual.keys()) == set(expected.keys())

    for pair_id, expected_metrics in expected.items():
        actual_metrics = actual[pair_id]
        assert set(actual_metrics.keys()) == set(expected_metrics.keys())

        for metric_name, expected_value in expected_metrics.items():
            actual_value = actual_metrics[metric_name]
            _assert_close(
                pair_id=pair_id,
                metric_name=metric_name,
                expected_value=float(expected_value),
                actual_value=float(actual_value),
            )


def _assert_close(
    *,
    pair_id: str,
    metric_name: str,
    expected_value: float,
    actual_value: float,
) -> None:
    """Assert scalar closeness by absolute or relative tolerance."""

    assert math.isfinite(actual_value), f"{pair_id}:{metric_name} is not finite"

    diff = abs(actual_value - expected_value)
    rel = diff / (abs(expected_value) + 1e-12)

    assert (
        diff <= ABS_TOL or rel <= REL_TOL
    ), (
        f"{pair_id}:{metric_name} expected={expected_value}, "
        f"actual={actual_value}, diff={diff}, rel={rel}"
    )
