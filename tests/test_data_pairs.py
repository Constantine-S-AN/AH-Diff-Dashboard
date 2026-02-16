"""Tests for A/H pair CSV parsing."""

from __future__ import annotations

from pathlib import Path

from ah_premium_lab.data import load_ah_pairs


def test_load_ah_pairs_from_sample_file() -> None:
    """`data/pairs.csv` sample should parse into non-empty AhPair list."""

    pairs = load_ah_pairs(Path("data/pairs.csv"))

    assert len(pairs) >= 2
    assert pairs[0].name
    assert pairs[0].a_ticker
    assert pairs[0].h_ticker


def test_load_ah_pairs_defaults_share_ratio(tmp_path: Path) -> None:
    """Missing `share_ratio` in CSV should default to 1.0."""

    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("name,a_ticker,h_ticker\nDemo,AAA,BBB\n", encoding="utf-8")

    pairs = load_ah_pairs(csv_path)

    assert len(pairs) == 1
    assert pairs[0].share_ratio == 1.0
