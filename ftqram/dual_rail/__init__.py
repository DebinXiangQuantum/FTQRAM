"""Dual-rail FT-QRAM implementation."""

from .qram import DualRailQram
from .qubits import (
    DualRailPair,
    logical_h,
    logical_x,
    logical_z,
    prepare_logical_zero,
    prepare_logical_one,
    split_dual_rail_register,
)

__all__ = [
    "DualRailQram",
    "DualRailPair",
    "logical_h",
    "logical_x",
    "logical_z",
    "prepare_logical_zero",
    "prepare_logical_one",
    "split_dual_rail_register",
]
