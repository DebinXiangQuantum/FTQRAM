"""Dual-rail logical qubit helpers.

Dual-rail encoding:
  Logical |0_L> = |01> (rail1 = 1)
  Logical |1_L> = |10> (rail0 = 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import UnitaryGate


@dataclass(frozen=True)
class DualRailPair:
    """A lightweight view of a dual-rail logical qubit.

    rail0 and rail1 are physical qubits. rail0 corresponds to logical |1_L>.
    """

    rail0: Qubit
    rail1: Qubit
    name: str = ""


def make_dual_rail_register(name: str) -> QuantumRegister:
    """Create a 2-qubit register for one dual-rail logical qubit."""

    return QuantumRegister(2, name)


def pair_from_register(qreg: QuantumRegister, index: int, name: str | None = None) -> DualRailPair:
    """Build a DualRailPair from a flat register with pairs [2*i, 2*i+1]."""

    base = 2 * index
    if base + 1 >= len(qreg):
        raise ValueError("Register too small for dual-rail pair index")
    return DualRailPair(qreg[base], qreg[base + 1], name or f"{qreg.name}_{index}")


def split_dual_rail_register(qreg: QuantumRegister) -> List[DualRailPair]:
    """Split a flat register into dual-rail pairs."""

    if len(qreg) % 2 != 0:
        raise ValueError("Dual-rail register length must be even")
    pairs = []
    for i in range(len(qreg) // 2):
        pairs.append(pair_from_register(qreg, i))
    return pairs


# Logical Hadamard acting on the single-excitation subspace.
# Basis order: |00>, |01>, |10>, |11>
_LOGICAL_H_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0],
        [0.0, 1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=complex,
)
_LOGICAL_H_GATE = UnitaryGate(_LOGICAL_H_MATRIX, label="H_L")


def logical_h(circuit: QuantumCircuit, pair: DualRailPair) -> None:
    """Apply a logical Hadamard in the dual-rail code space."""

    circuit.append(_LOGICAL_H_GATE, [pair.rail0, pair.rail1])


def logical_x(circuit: QuantumCircuit, pair: DualRailPair) -> None:
    """Logical X swaps the rails."""

    circuit.swap(pair.rail0, pair.rail1)


def logical_z(circuit: QuantumCircuit, pair: DualRailPair) -> None:
    """Logical Z applies Z to rail0 (|1_L> component)."""

    circuit.z(pair.rail0)


def prepare_logical_zero(circuit: QuantumCircuit, pair: DualRailPair) -> None:
    """Prepare |0_L> = |01>. Assumes both rails start in |0>."""

    circuit.x(pair.rail1)


def prepare_logical_one(circuit: QuantumCircuit, pair: DualRailPair) -> None:
    """Prepare |1_L> = |10>. Assumes both rails start in |0>."""

    circuit.x(pair.rail0)

