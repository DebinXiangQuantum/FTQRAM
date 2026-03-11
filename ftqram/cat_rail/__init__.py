"""Phase-protected cat-rail QRAM."""

from .code import (
    CatQubitParameters,
    classify_cat_rail_faults,
    interval_probabilities,
    phase_dual_rail_codewords,
    phase_dual_rail_logical_ops,
)
from .noise import (
    CatRailNoiseAnalyzer,
    NoisePoint,
    cat_rail_quadratic_theory,
    fit_log_slope,
    native_linear_theory,
)
from .qram import (
    BucketBrigadeReferenceQram,
    CatRailPhaseProtectedQram,
    QueryFrame,
    build_memory_cases,
    distributions_close,
)

__all__ = [
    "BucketBrigadeReferenceQram",
    "CatQubitParameters",
    "CatRailNoiseAnalyzer",
    "CatRailPhaseProtectedQram",
    "NoisePoint",
    "QueryFrame",
    "build_memory_cases",
    "cat_rail_quadratic_theory",
    "classify_cat_rail_faults",
    "distributions_close",
    "fit_log_slope",
    "interval_probabilities",
    "native_linear_theory",
    "phase_dual_rail_codewords",
    "phase_dual_rail_logical_ops",
]
