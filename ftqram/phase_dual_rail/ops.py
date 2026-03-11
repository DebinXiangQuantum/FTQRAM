"""Low-level phase-encoded dual-rail operations."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from .qubits import PhaseDualRailPair


def swap_dual_rail(circuit: QuantumCircuit, a: PhaseDualRailPair, b: PhaseDualRailPair) -> None:
    """Swap two phase dual-rail logical qubits."""
    circuit.swap(a.rail0, b.rail0)
    circuit.swap(a.rail1, b.rail1)


def cswap_dual_rail(circuit: QuantumCircuit, control: Qubit, a: PhaseDualRailPair, b: PhaseDualRailPair) -> None:
    """Controlled swap for a phase dual-rail logical qubit."""
    circuit.cswap(control, a.rail0, b.rail0)
    circuit.cswap(control, a.rail1, b.rail1)


def measure_x_parity_syndrome(
    circuit: QuantumCircuit,
    pair: PhaseDualRailPair,
    ancilla: Qubit,
    cbit,
) -> None:
    """Measure X-parity syndrome of a phase dual-rail pair.
    
    0 = valid (odd X-parity, eigenvalue -1), 1 = error (even X-parity).
    """
    circuit.reset(ancilla)
    # Initialize ancilla to |1> so that if parity is odd (-1), 
    # it flips to |0> (syndrome 0 = valid).
    # Wait, CX with control in X basis:
    # If we apply H to data, then CX data->ancilla, then H to data.
    # Data is in |+> or |->. H maps |+> to |0>, |-> to |1>.
    # So if data is |+->, H maps to |01>.
    # CX to ancilla (starts at |0>) gives |1>.
    # So odd parity gives |1> on ancilla.
    # To make syndrome 0 = valid, we can start ancilla at |1>.
    # Then odd parity flips it to |0>.
    circuit.x(ancilla)
    circuit.h(pair.rail0)
    circuit.h(pair.rail1)
    circuit.cx(pair.rail0, ancilla)
    circuit.cx(pair.rail1, ancilla)
    circuit.h(pair.rail0)
    circuit.h(pair.rail1)
    circuit.measure(ancilla, cbit)


def measure_conservation_syndrome(
    circuit: QuantumCircuit,
    left: PhaseDualRailPair,
    right: PhaseDualRailPair,
    ancilla: Qubit,
    cbit_left,
    cbit_right,
) -> None:
    """Conservation check with syndrome convention: 0 = valid, 1 = error."""
    measure_x_parity_syndrome(circuit, left, ancilla, cbit_left)
    measure_x_parity_syndrome(circuit, right, ancilla, cbit_right)
