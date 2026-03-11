"""Phase-encoded dual-rail QRAM for biased noise (Cat Qubits)."""

from .qubits import (
    PhaseDualRailPair,
    make_phase_dual_rail_register,
    split_phase_dual_rail_register,
    logical_h,
    logical_x,
    logical_z,
    prepare_logical_zero,
    prepare_logical_one,
)
from .ops import (
    swap_dual_rail,
    cswap_dual_rail,
    measure_x_parity_syndrome,
    measure_conservation_syndrome,
)
from .router import PhaseDualRailRouterNode, ft_router, ft_reverse_router
from .qram import PhaseDualRailQram

__all__ = [
    "PhaseDualRailPair",
    "make_phase_dual_rail_register",
    "split_phase_dual_rail_register",
    "logical_h",
    "logical_x",
    "logical_z",
    "prepare_logical_zero",
    "prepare_logical_one",
    "swap_dual_rail",
    "cswap_dual_rail",
    "measure_x_parity_syndrome",
    "measure_conservation_syndrome",
    "PhaseDualRailRouterNode",
    "ft_router",
    "ft_reverse_router",
    "PhaseDualRailQram",
]
