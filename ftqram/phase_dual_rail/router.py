"""Phase-encoded dual-rail router node and fault-tolerant routing primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qiskit import QuantumCircuit, QuantumRegister

from .ops import (
    cswap_dual_rail,
    measure_conservation_syndrome,
    measure_x_parity_syndrome,
)
from .qubits import PhaseDualRailPair, make_phase_dual_rail_register


@dataclass
class PhaseDualRailRouterNode:
    """Router node in a binary tree for phase dual-rail QRAM."""

    index: int
    level: int
    direction: str
    parent: Optional["PhaseDualRailRouterNode"]

    left: Optional["PhaseDualRailRouterNode"] = None
    right: Optional["PhaseDualRailRouterNode"] = None

    def __post_init__(self) -> None:
        self.addr_reg = make_phase_dual_rail_register(self.reg_name("addr"))
        self.bus_reg = make_phase_dual_rail_register(self.reg_name("bus"))
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
    def addr(self) -> PhaseDualRailPair:
        return PhaseDualRailPair(self.addr_reg[0], self.addr_reg[1], self.reg_name("addr"))

    @property
    def bus(self) -> PhaseDualRailPair:
        return PhaseDualRailPair(self.bus_reg[0], self.bus_reg[1], self.reg_name("bus"))

    @property
    def flag(self):
        return self.flag_reg[0]

    @property
    def parity(self):
        return self.parity_reg[0]

    def add_registers(self, circuit: QuantumCircuit, *, skip_ancilla: bool = False) -> None:
        regs = [self.addr_reg, self.bus_reg]
        if not skip_ancilla:
            regs += [self.flag_reg, self.parity_reg]
        for reg in regs:
            if reg not in circuit.qregs:
                circuit.add_register(reg)


def ft_router(
    circuit: QuantumCircuit,
    addr: PhaseDualRailPair,
    bus: PhaseDualRailPair,
    left_bus: PhaseDualRailPair,
    right_bus: PhaseDualRailPair,
    flag_qubit,
    parity_qubit,
    syndrome,
) -> None:
    """Fault-tolerant phase-encoded dual-rail router."""
    # 1) Pre-check: address must be valid (odd X-parity)
    measure_x_parity_syndrome(circuit, addr, parity_qubit, syndrome.next())

    # 2) Flagged routing to RIGHT (conditional on addr.rail0 being |->)
    circuit.reset(flag_qubit)
    circuit.h(addr.rail0)
    circuit.cx(addr.rail0, flag_qubit)
    circuit.h(addr.rail0)
    # If addr.rail0 was |->, flag is now |1>
    cswap_dual_rail(circuit, flag_qubit, bus, right_bus)
    # Uncompute flag
    circuit.h(addr.rail0)
    circuit.cx(addr.rail0, flag_qubit)
    circuit.h(addr.rail0)
    circuit.measure(flag_qubit, syndrome.next())

    # 3) Flagged routing to LEFT (conditional on addr.rail1 being |->)
    circuit.reset(flag_qubit)
    circuit.h(addr.rail1)
    circuit.cx(addr.rail1, flag_qubit)
    circuit.h(addr.rail1)
    cswap_dual_rail(circuit, flag_qubit, bus, left_bus)
    circuit.h(addr.rail1)
    circuit.cx(addr.rail1, flag_qubit)
    circuit.h(addr.rail1)
    circuit.measure(flag_qubit, syndrome.next())

    # 4) Post-check: conservation between outputs
    measure_conservation_syndrome(
        circuit, left_bus, right_bus, parity_qubit, syndrome.next(), syndrome.next()
    )


def ft_reverse_router(
    circuit: QuantumCircuit,
    addr: PhaseDualRailPair,
    bus: PhaseDualRailPair,
    left_bus: PhaseDualRailPair,
    right_bus: PhaseDualRailPair,
    flag_qubit,
    parity_qubit,
    syndrome,
) -> None:
    """Reverse of ft_router with the same syndrome checks."""

    measure_conservation_syndrome(
        circuit, left_bus, right_bus, parity_qubit, syndrome.next(), syndrome.next()
    )

    circuit.reset(flag_qubit)
    circuit.h(addr.rail1)
    circuit.cx(addr.rail1, flag_qubit)
    circuit.h(addr.rail1)
    cswap_dual_rail(circuit, flag_qubit, bus, left_bus)
    circuit.h(addr.rail1)
    circuit.cx(addr.rail1, flag_qubit)
    circuit.h(addr.rail1)
    circuit.measure(flag_qubit, syndrome.next())

    circuit.reset(flag_qubit)
    circuit.h(addr.rail0)
    circuit.cx(addr.rail0, flag_qubit)
    circuit.h(addr.rail0)
    cswap_dual_rail(circuit, flag_qubit, bus, right_bus)
    circuit.h(addr.rail0)
    circuit.cx(addr.rail0, flag_qubit)
    circuit.h(addr.rail0)
    circuit.measure(flag_qubit, syndrome.next())

    measure_x_parity_syndrome(circuit, addr, parity_qubit, syndrome.next())
