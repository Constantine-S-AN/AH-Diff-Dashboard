"""Generate demo dashboard image placeholders for README/docs usage."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class DemoAssetSpec:
    """Specification for one generated SVG placeholder."""

    file_name: str
    title: str
    subtitle: str
    bullets: tuple[str, ...]
    accent_start: str
    accent_end: str


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
        accent_start="#3B82F6",
        accent_end="#10B981",
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
        accent_start="#06B6D4",
        accent_end="#22C55E",
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
        accent_start="#2563EB",
        accent_end="#7C3AED",
    ),
)


def generate_demo_assets(output_dir: Path) -> list[Path]:
    """Generate deterministic SVG demo placeholders under `output_dir`."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    showcase_path = output_dir / "dashboard_showcase_demo.svg"
    showcase_path.write_text(_render_showcase_svg(), encoding="utf-8")
    output_paths.append(showcase_path)

    for item in ASSETS:
        content = _render_svg(item)
        path = output_dir / item.file_name
        path.write_text(content, encoding="utf-8")
        output_paths.append(path)
    return output_paths


def _render_svg(spec: DemoAssetSpec) -> str:
    """Render a simple UI-like SVG card for README previews."""

    title = escape(spec.title)
    subtitle = escape(spec.subtitle)
    bullet_lines = "".join(
        [
            f'<text x="86" y="{228 + i * 40}" font-size="21" fill="#CBD5E1">'
            f"- {escape(text)}</text>"
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
      <stop offset="0%" stop-color="#020617"/>
      <stop offset="100%" stop-color="#0B1220"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0%" stop-color="{spec.accent_start}"/>
      <stop offset="100%" stop-color="{spec.accent_end}"/>
    </linearGradient>
  </defs>
  <rect width="1280" height="760" fill="url(#bg)"/>
  <circle cx="190" cy="124" r="220" fill="#1D4ED8" opacity="0.2"/>
  <circle cx="1120" cy="636" r="280" fill="#14B8A6" opacity="0.15"/>
  <rect x="36" y="36" width="1208" height="688" rx="20" fill="#0B1220" stroke="#1E293B"/>
  <rect x="36" y="36" width="1208" height="104" rx="20" fill="#0F172A"/>
  <rect x="36" y="124" width="1208" height="4" fill="url(#accent)"/>
  <text
    x="74"
    y="90"
    font-size="38"
    font-weight="700"
    font-family="Helvetica, Arial, sans-serif"
    fill="#F8FAFC"
  >{title}</text>
  <text
    x="74"
    y="120"
    font-size="20"
    font-family="Helvetica, Arial, sans-serif"
    fill="#94A3B8"
  >{subtitle}</text>

  <rect x="64" y="164" width="420" height="520" rx="14" fill="#111827" stroke="#1E293B"/>
  <text x="86" y="202" font-size="27" font-weight="700" fill="#E2E8F0">Key Panels</text>
  {bullet_lines}

  <rect x="520" y="164" width="690" height="240" rx="14" fill="#0F172A" stroke="#334155"/>
  <rect x="544" y="186" width="130" height="34" rx="17" fill="#1E293B"/>
  <text x="562" y="209" font-size="16" fill="#E2E8F0">Premium%</text>
  <rect x="684" y="186" width="124" height="34" rx="17" fill="#1E293B"/>
  <text x="704" y="209" font-size="16" fill="#E2E8F0">Z-score</text>
  <rect x="818" y="186" width="128" height="34" rx="17" fill="#1E293B"/>
  <text x="836" y="209" font-size="16" fill="#E2E8F0">Half-life</text>
  <text x="546" y="254" font-size="24" font-weight="700" fill="#E2E8F0">Time Series Zone</text>
  <polyline
    points="550,352 625,296 700,318 775,262 850,300 925,230 1000,264 1075,226 1150,246"
    fill="none"
    stroke="url(#accent)"
    stroke-width="6"
  />

  <rect x="520" y="426" width="332" height="258" rx="14" fill="#0F172A" stroke="#334155"/>
  <text x="546" y="466" font-size="24" font-weight="700" fill="#E2E8F0">Diagnostics</text>
  <text x="546" y="510" font-size="20" fill="#94A3B8">ADF / EG / Half-life</text>
  <text x="546" y="542" font-size="20" fill="#94A3B8">Breakpoints / CUSUM</text>
  <text x="546" y="574" font-size="20" fill="#94A3B8">Executability metrics</text>

  <rect x="878" y="426" width="332" height="258" rx="14" fill="#0F172A" stroke="#334155"/>
  <text x="904" y="466" font-size="24" font-weight="700" fill="#E2E8F0">Sensitivity</text>
  <text x="904" y="510" font-size="20" fill="#94A3B8">Cost heatmap</text>
  <text x="904" y="542" font-size="20" fill="#94A3B8">Tolerance radar</text>
  <text x="904" y="574" font-size="20" fill="#94A3B8">Breakeven stats</text>
</svg>
"""


def _render_showcase_svg() -> str:
    """Render a wide README hero preview for quick visual showcase."""

    return """<svg
  xmlns="http://www.w3.org/2000/svg"
  width="1600"
  height="900"
  viewBox="0 0 1600 900"
>
  <defs>
    <linearGradient id="heroBg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#020617"/>
      <stop offset="100%" stop-color="#111827"/>
    </linearGradient>
    <linearGradient id="heroLine" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0%" stop-color="#2563EB"/>
      <stop offset="50%" stop-color="#06B6D4"/>
      <stop offset="100%" stop-color="#22C55E"/>
    </linearGradient>
  </defs>
  <rect width="1600" height="900" fill="url(#heroBg)"/>
  <circle cx="210" cy="180" r="220" fill="#1D4ED8" opacity="0.22"/>
  <circle cx="1430" cy="740" r="320" fill="#14B8A6" opacity="0.16"/>

  <text
    x="96"
    y="108"
    font-size="56"
    font-weight="700"
    font-family="Helvetica, Arial, sans-serif"
    fill="#F8FAFC"
  >
    AH Premium Lab Showcase
  </text>
  <text x="96" y="148" font-size="26" font-family="Helvetica, Arial, sans-serif" fill="#94A3B8">
    Dashboard + Diagnostics + Executability + Cost Sensitivity
  </text>
  <rect x="96" y="174" width="508" height="8" rx="4" fill="url(#heroLine)"/>

  <rect x="96" y="238" width="452" height="290" rx="18" fill="#0F172A" stroke="#1E293B"/>
  <rect x="96" y="238" width="452" height="56" rx="18" fill="#111827"/>
  <text x="126" y="275" font-size="24" font-weight="700" fill="#E2E8F0">Overview</text>
  <polyline
    points="128,488 178,430 228,446 278,400 328,430 378,378 428,404 478,366 518,384"
    fill="none"
    stroke="#3B82F6"
    stroke-width="5"
  />
  <text x="126" y="330" font-size="18" fill="#94A3B8">- Premium monitor</text>
  <text x="126" y="360" font-size="18" fill="#94A3B8">- ADF / EG / Half-life</text>
  <text x="126" y="390" font-size="18" fill="#94A3B8">- Data quality + executability</text>

  <rect x="574" y="238" width="452" height="290" rx="18" fill="#0F172A" stroke="#1E293B"/>
  <rect x="574" y="238" width="452" height="56" rx="18" fill="#111827"/>
  <text x="604" y="275" font-size="24" font-weight="700" fill="#E2E8F0">Overview Filtered</text>
  <polyline
    points="606,488 656,438 706,458 756,404 806,434 856,382 906,412 956,372 996,390"
    fill="none"
    stroke="#06B6D4"
    stroke-width="5"
  />
  <text x="604" y="330" font-size="18" fill="#94A3B8">- Score and percentile filters</text>
  <text x="604" y="360" font-size="18" fill="#94A3B8">- Keyword screening</text>
  <text x="604" y="390" font-size="18" fill="#94A3B8">- High-quality candidate set</text>

  <rect x="1052" y="238" width="452" height="290" rx="18" fill="#0F172A" stroke="#1E293B"/>
  <rect x="1052" y="238" width="452" height="56" rx="18" fill="#111827"/>
  <text x="1082" y="275" font-size="24" font-weight="700" fill="#E2E8F0">Pair Detail</text>
  <polyline
    points="1084,488 1134,430 1184,446 1234,400 1284,430 1334,378 1384,404 1434,366 1474,384"
    fill="none"
    stroke="#8B5CF6"
    stroke-width="5"
  />
  <text x="1082" y="330" font-size="18" fill="#94A3B8">- Spread and z-score views</text>
  <text x="1082" y="360" font-size="18" fill="#94A3B8">- Rolling cointegration</text>
  <text x="1082" y="390" font-size="18" fill="#94A3B8">- Breakpoints + sensitivity cache</text>

  <rect x="96" y="588" width="220" height="74" rx="37" fill="#1E293B"/>
  <text x="140" y="634" font-size="24" font-weight="700" fill="#BFDBFE">ADF / EG</text>
  <rect x="336" y="588" width="258" height="74" rx="37" fill="#1E293B"/>
  <text x="378" y="634" font-size="24" font-weight="700" fill="#99F6E4">Break Detection</text>
  <rect x="616" y="588" width="250" height="74" rx="37" fill="#1E293B"/>
  <text x="654" y="634" font-size="24" font-weight="700" fill="#FDE68A">Executability</text>
  <rect x="886" y="588" width="314" height="74" rx="37" fill="#1E293B"/>
  <text x="930" y="634" font-size="24" font-weight="700" fill="#DDD6FE">Cost Sensitivity</text>
  <rect x="1222" y="588" width="282" height="74" rx="37" fill="#1E293B"/>
  <text x="1270" y="634" font-size="24" font-weight="700" fill="#86EFAC">Research Report</text>
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
