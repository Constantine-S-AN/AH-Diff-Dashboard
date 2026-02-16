"""Generate demo dashboard image placeholders for README/docs usage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoAssetSpec:
    """Specification for one generated SVG placeholder."""

    file_name: str
    title: str
    subtitle: str
    bullets: tuple[str, ...]


ASSETS: tuple[DemoAssetSpec, ...] = (
    DemoAssetSpec(
        file_name="dashboard_overview_demo.svg",
        title="AH Premium Lab - Overview",
        subtitle="Universe-level table and filters",
        bullets=(
            "latest premium%",
            "rolling zscore",
            "ADF / EG p-values",
            "executability score",
        ),
    ),
    DemoAssetSpec(
        file_name="dashboard_overview_filtered_demo.svg",
        title="AH Premium Lab - Overview (Filtered)",
        subtitle="Score / percentile / keyword filters",
        bullets=(
            "score > threshold",
            "premium percentile filter",
            "data quality & executability",
            "mapping warning panel",
        ),
    ),
    DemoAssetSpec(
        file_name="dashboard_pair_detail_demo.svg",
        title="AH Premium Lab - Pair Detail",
        subtitle="Time series + diagnostics + sensitivity",
        bullets=(
            "premium / log_spread / zscore",
            "rolling cointegration",
            "structural breaks",
            "cost sensitivity cache",
        ),
    ),
)


def generate_demo_assets(output_dir: Path) -> list[Path]:
    """Generate deterministic SVG demo placeholders under `output_dir`."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for item in ASSETS:
        content = _render_svg(item)
        path = output_dir / item.file_name
        path.write_text(content, encoding="utf-8")
        output_paths.append(path)
    return output_paths


def _render_svg(spec: DemoAssetSpec) -> str:
    """Render a simple UI-like SVG card for README previews."""

    bullet_lines = "".join(
        [
            f'<text x="86" y="{220 + i * 38}" font-size="22" fill="#334155">'
            f"• {text}</text>"
            for i, text in enumerate(spec.bullets)
        ]
    )

    return f"""<svg
  xmlns="http://www.w3.org/2000/svg"
  width="1280"
  height="760"
  viewBox="0 0 1280 760"
>
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#f8fafc"/>
      <stop offset="100%" stop-color="#e2e8f0"/>
    </linearGradient>
  </defs>
  <rect width="1280" height="760" fill="url(#bg)"/>
  <rect x="36" y="36" width="1208" height="688" rx="18" fill="#ffffff" stroke="#cbd5e1"/>
  <rect x="36" y="36" width="1208" height="92" rx="18" fill="#0f172a"/>
  <text
    x="74"
    y="92"
    font-size="34"
    font-family="Arial, sans-serif"
    fill="#f8fafc"
  >{spec.title}</text>
  <text
    x="74"
    y="126"
    font-size="20"
    font-family="Arial, sans-serif"
    fill="#cbd5e1"
  >{spec.subtitle}</text>

  <rect x="64" y="164" width="420" height="520" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="86" y="202" font-size="26" fill="#0f172a">Key Panels</text>
  {bullet_lines}

  <rect x="520" y="164" width="690" height="240" rx="12" fill="#eef2ff" stroke="#c7d2fe"/>
  <text x="546" y="204" font-size="24" fill="#1e293b">Time Series Zone</text>
  <polyline
    points="550,352 625,296 700,318 775,262 850,300 925,230 1000,264 1075,226 1150,246"
    fill="none"
    stroke="#2563eb"
    stroke-width="5"
  />

  <rect x="520" y="426" width="332" height="258" rx="12" fill="#ecfeff" stroke="#bae6fd"/>
  <text x="546" y="466" font-size="24" fill="#0f172a">Diagnostics</text>
  <text x="546" y="510" font-size="20" fill="#334155">ADF / EG / Half-life</text>
  <text x="546" y="542" font-size="20" fill="#334155">Breakpoints / CUSUM</text>
  <text x="546" y="574" font-size="20" fill="#334155">Executability metrics</text>

  <rect x="878" y="426" width="332" height="258" rx="12" fill="#fef9c3" stroke="#fde68a"/>
  <text x="904" y="466" font-size="24" fill="#0f172a">Sensitivity</text>
  <text x="904" y="510" font-size="20" fill="#334155">Cost heatmap</text>
  <text x="904" y="542" font-size="20" fill="#334155">Tolerance radar</text>
  <text x="904" y="574" font-size="20" fill="#334155">Breakeven stats</text>
</svg>
"""


def main() -> None:
    """CLI entry for generating placeholder demo assets."""

    output_dir = Path("docs/screenshots")
    paths = generate_demo_assets(output_dir)
    for item in paths:
        print(item.as_posix())


if __name__ == "__main__":
    main()
