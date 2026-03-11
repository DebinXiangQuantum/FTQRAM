"""Phase-encoded dual-rail logical qubit helpers.

Phase-Encoded Dual-rail encoding:
  Logical |0_L> = |+-> (rail1 = |->, rail0 = |+>)
  Logical |1_L> = |-+> (rail0 = |->, rail1 = |+>)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import UnitaryGate

# Standard logical H matrix for reference
_LOGICAL_H_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0],
        [0.0, 1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=complex,
)
_LOGICAL_H_GATE = UnitaryGate(_LOGICAL_H_MATRIX, label="H_L_std")


@dataclass(frozen=True)
class PhaseDualRailPair:
    """A lightweight view of a phase-encoded dual-rail logical qubit.

    rail0 and rail1 are physical qubits.
    """

    rail0: Qubit
    rail1: Qubit
    name: str = ""


def make_phase_dual_rail_register(name: str) -> QuantumRegister:
    """Create a 2-qubit register for one phase dual-rail logical qubit."""

    return QuantumRegister(2, name)


def pair_from_register(qreg: QuantumRegister, index: int, name: str | None = None) -> PhaseDualRailPair:
    """Build a PhaseDualRailPair from a flat register with pairs [2*i, 2*i+1]."""

    base = 2 * index
    if base + 1 >= len(qreg):
        raise ValueError("Register too small for dual-rail pair index")
    return PhaseDualRailPair(qreg[base], qreg[base + 1], name or f"{qreg.name}_{index}")


def split_phase_dual_rail_register(qreg: QuantumRegister) -> List[PhaseDualRailPair]:
    """Split a flat register into phase dual-rail pairs."""

    if len(qreg) % 2 != 0:
        raise ValueError("Dual-rail register length must be even")
    pairs = []
    for i in range(len(qreg) // 2):
        pairs.append(pair_from_register(qreg, i))
    return pairs


def logical_h(circuit: QuantumCircuit, pair: PhaseDualRailPair) -> None:
    """Apply a logical Hadamard in the phase dual-rail code space.
    
    Effectively conjugates the standard H_L by physical H gates.
    """
    circuit.h(pair.rail0)
    circuit.h(pair.rail1)
    circuit.append(_LOGICAL_H_GATE, [pair.rail0, pair.rail1])
    circuit.h(pair.rail0)
    circuit.h(pair.rail1)


def logical_x(circuit: QuantumCircuit, pair: PhaseDualRailPair) -> None:
    """Logical X swaps the rails. SWAP(|+->) = |-+>."""

    circuit.swap(pair.rail0, pair.rail1)


def logical_z(circuit: QuantumCircuit, pair: PhaseDualRailPair) -> None:
    """Logical Z applies X to rail0. X|+> = |+>, X|-> = -|->."""

    circuit.x(pair.rail0)


def prepare_logical_zero(circuit: QuantumCircuit, pair: PhaseDualRailPair) -> None:
    """Prepare |0_L> = |+->. Assumes both rails start in |0>."""

    circuit.h(pair.rail0)
    circuit.x(pair.rail1)
    circuit.h(pair.rail1)


def prepare_logical_one(circuit: QuantumCircuit, pair: PhaseDualRailPair) -> None:
    """Prepare |1_L> = |-+>. Assumes both rails start in |0>."""

    circuit.x(pair.rail0)
    circuit.h(pair.rail0)
    circuit.h(pair.rail1)
