"""Dual-rail router node and fault-tolerant routing primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qiskit import QuantumCircuit, QuantumRegister

from .ops import (
    cswap_dual_rail,
    measure_conservation,
    measure_conservation_syndrome,
    measure_parity,
    measure_parity_syndrome,
)
from .qubits import DualRailPair, make_dual_rail_register


@dataclass
class DualRailRouterNode:
    """Router node in a binary tree for dual-rail QRAM."""

    index: int
    level: int
    direction: str
    parent: Optional["DualRailRouterNode"]

    left: Optional["DualRailRouterNode"] = None
    right: Optional["DualRailRouterNode"] = None

    def __post_init__(self) -> None:
        # Dual-rail address storage and bus buffer.
        self.addr_reg = make_dual_rail_register(self.reg_name("addr"))
        self.bus_reg = make_dual_rail_register(self.reg_name("bus"))
        # Local ancillas for parity/flag checks.
        self.flag_reg = QuantumRegister(1, self.reg_name("flag"))
        self.parity_reg = QuantumRegister(1, self.reg_name("par"))

    @property
    def address(self) -> str:
        if self.parent is None:
            return self.direction
        return self.parent.address + self.direction

    def reg_name(self, suffix: str) -> str:
        if self.address:
            return f"router_{self.level}_{self.address}_{suffix}"
        return f"router_{self.level}_{suffix}"

    @property
    def addr(self) -> DualRailPair:
        return DualRailPair(self.addr_reg[0], self.addr_reg[1], self.reg_name("addr"))

    @property
    def bus(self) -> DualRailPair:
        return DualRailPair(self.bus_reg[0], self.bus_reg[1], self.reg_name("bus"))

    @property
    def flag(self):
        return self.flag_reg[0]

    @property
    def parity(self):
        return self.parity_reg[0]

    def add_registers(self, circuit: QuantumCircuit, *, skip_ancilla: bool = False) -> None:
        """Attach this node's registers to a circuit (if missing).

        When *skip_ancilla* is True only addr/bus registers are added
        (flag and parity are managed externally as shared qubits).
        """

        regs = [self.addr_reg, self.bus_reg]
        if not skip_ancilla:
            regs += [self.flag_reg, self.parity_reg]
        for reg in regs:
            if reg not in circuit.qregs:
                circuit.add_register(reg)


def ft_router(
    circuit: QuantumCircuit,
    addr: DualRailPair,
    bus: DualRailPair,
    left_bus: DualRailPair,
    right_bus: DualRailPair,
    flag_qubit,
    parity_qubit,
    syndrome,
) -> None:
    """Fault-tolerant dual-rail router with parity + flag checks.

    syndrome is a tracker that yields classical bits via syndrome.next().
    """

    # 1) Pre-check: address must be one-hot (odd parity)
    # Uses syndrome convention: 0 = valid, 1 = error
    measure_parity_syndrome(circuit, addr, parity_qubit, syndrome.next())

    # 2) Flagged routing to RIGHT (addr rail0)
    circuit.reset(flag_qubit)
    circuit.cx(addr.rail0, flag_qubit)
    cswap_dual_rail(circuit, flag_qubit, bus, right_bus)
    circuit.cx(addr.rail0, flag_qubit)
    circuit.measure(flag_qubit, syndrome.next())

    # 3) Flagged routing to LEFT (addr rail1)
    circuit.reset(flag_qubit)
    circuit.cx(addr.rail1, flag_qubit)
    cswap_dual_rail(circuit, flag_qubit, bus, left_bus)
    circuit.cx(addr.rail1, flag_qubit)
    circuit.measure(flag_qubit, syndrome.next())

    # 4) Post-check: conservation between outputs
    # Uses syndrome convention: 0 = valid, 1 = error
    measure_conservation_syndrome(
        circuit,
        left_bus,
        right_bus,
        parity_qubit,
        syndrome.next(),
        syndrome.next(),
    )



def ft_reverse_router(
    circuit: QuantumCircuit,
    addr: DualRailPair,
    bus: DualRailPair,
    left_bus: DualRailPair,
    right_bus: DualRailPair,
    flag_qubit,
    parity_qubit,
    syndrome,
) -> None:
    """Reverse of ft_router with the same syndrome checks.

    Undoes the routing performed by ft_router while recording syndrome data
    for fault-tolerance.
    """

    # 1) Pre-check: conservation of left/right outputs
    # Uses syndrome convention: 0 = valid, 1 = error
    measure_conservation_syndrome(
        circuit,
        left_bus,
        right_bus,
        parity_qubit,
        syndrome.next(),
        syndrome.next(),
    )

    # 2) Reverse flagged routing LEFT (addr rail1) — undo step 3 of ft_router
    circuit.reset(flag_qubit)
    circuit.cx(addr.rail1, flag_qubit)
    cswap_dual_rail(circuit, flag_qubit, bus, left_bus)
    circuit.cx(addr.rail1, flag_qubit)
    circuit.measure(flag_qubit, syndrome.next())

    # 3) Reverse flagged routing RIGHT (addr rail0) — undo step 2 of ft_router
    circuit.reset(flag_qubit)
    circuit.cx(addr.rail0, flag_qubit)
    cswap_dual_rail(circuit, flag_qubit, bus, right_bus)
    circuit.cx(addr.rail0, flag_qubit)
    circuit.measure(flag_qubit, syndrome.next())

    # 4) Post-check: address parity
    # Uses syndrome convention: 0 = valid, 1 = error
    measure_parity_syndrome(circuit, addr, parity_qubit, syndrome.next())

