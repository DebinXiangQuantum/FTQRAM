"""Low-level dual-rail operations (swap, cswap, parity checks)."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from .qubits import DualRailPair


def swap_dual_rail(circuit: QuantumCircuit, a: DualRailPair, b: DualRailPair) -> None:
    """Swap two dual-rail logical qubits (rail-wise swap)."""

    circuit.swap(a.rail0, b.rail0)
    circuit.swap(a.rail1, b.rail1)


def cswap_dual_rail(circuit: QuantumCircuit, control: Qubit, a: DualRailPair, b: DualRailPair) -> None:
    """Controlled swap for a dual-rail logical qubit (rail-wise CSWAP)."""

    circuit.cswap(control, a.rail0, b.rail0)
    circuit.cswap(control, a.rail1, b.rail1)


def measure_parity(
    circuit: QuantumCircuit,
    pair: DualRailPair,
    ancilla: Qubit,
    cbit,
) -> None:
    """Measure parity (odd/even) of a dual-rail pair into classical bit.

    Odd parity (01 or 10) corresponds to a valid dual-rail state.
    """

    circuit.reset(ancilla)
    circuit.cx(pair.rail0, ancilla)
    circuit.cx(pair.rail1, ancilla)
    circuit.measure(ancilla, cbit)


def measure_conservation(
    circuit: QuantumCircuit,
    left: DualRailPair,
    right: DualRailPair,
    ancilla: Qubit,
    cbit_left,
    cbit_right,
) -> None:
    """Measure parity of left/right output rails for conservation checks."""

    measure_parity(circuit, left, ancilla, cbit_left)
    measure_parity(circuit, right, ancilla, cbit_right)

