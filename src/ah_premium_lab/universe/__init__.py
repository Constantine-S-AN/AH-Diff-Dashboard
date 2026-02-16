"""Universe management for large A/H pair lists and data quality."""

from .loader import (
    DEFAULT_PAIRS_FALLBACK_PATH,
    DEFAULT_PAIRS_MASTER_PATH,
    load_universe_frame,
    load_universe_pairs,
    resolve_universe_path,
)
from .mapping import (
    DEFAULT_MAPPING_OVERRIDES_PATH,
    MappingRequiredError,
    load_mapping_overrides,
    record_mapping_issue,
)
from .quality import (
    PairQuality,
    SeriesQuality,
    combine_pair_quality,
    compute_series_quality,
)

__all__ = [
    "DEFAULT_MAPPING_OVERRIDES_PATH",
    "DEFAULT_PAIRS_FALLBACK_PATH",
    "DEFAULT_PAIRS_MASTER_PATH",
    "MappingRequiredError",
    "PairQuality",
    "SeriesQuality",
    "combine_pair_quality",
    "compute_series_quality",
    "load_mapping_overrides",
    "load_universe_frame",
    "load_universe_pairs",
    "record_mapping_issue",
    "resolve_universe_path",
]
