"""Dual-rail fault-tolerant QRAM builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from .qubits import DualRailPair, logical_h, prepare_logical_zero, split_dual_rail_register
from .ops import swap_dual_rail
from .router import DualRailRouterNode, ft_router


class SyndromeTracker:
    """Sequential allocator for syndrome classical bits."""

    def __init__(self, creg: ClassicalRegister) -> None:
        self.creg = creg
        self.index = 0

    def next(self):
        if self.index >= len(self.creg):
            raise RuntimeError("Syndrome register exhausted")
        bit = self.creg[self.index]
        self.index += 1
        return bit


class ReuseSyndromeTracker:
    """Always returns the same classical bit (overwrite mode)."""

    def __init__(self, creg: ClassicalRegister) -> None:
        self.creg = creg

    def next(self):
        return self.creg[0]


def _router_calls_for_depth(depth: int) -> int:
    """Number of router calls up to (but not including) a given depth."""

    if depth <= 0:
        return 0
    return (1 << depth) - 1


class DualRailQram:
    """Dual-rail, fault-tolerant QRAM circuit builder.

    This follows the bucket-brigade logic of bucktele.py but replaces all
    logical qubits with dual-rail pairs and inserts parity/flag checks.
    """

    def __init__(
        self,
        address,
        data: List[int],
        *regs,
        bandwidth: int = 1,
        record_syndrome: bool = True,
        prepare_bus: bool = True,
    ) -> None:
        self.address = address
        self.data = data
        self.bandwidth = bandwidth
        self.record_syndrome = record_syndrome
        self.prepare_bus = prepare_bus

        self.depth = self._infer_depth(address)
        if len(data) != (1 << self.depth):
            raise ValueError("data length must be 2^address_bits")

        self.routers: List[List[DualRailRouterNode]] = [
            [] for _ in range(self.depth)
        ]
        self.root = self._build_tree(level=0, direction="", parent=None)

        self.circuit = QuantumCircuit(*regs) if regs else None
        self.syndrome_tracker: Optional[SyndromeTracker] = None

    def _infer_depth(self, address) -> int:
        if isinstance(address, int):
            return address
        if isinstance(address, list) and address:
            return len(address[0])
        raise ValueError("address must be an int or list of binary strings")

    def _build_tree(
        self, level: int, direction: str, parent: Optional[DualRailRouterNode]
    ) -> DualRailRouterNode:
        node = DualRailRouterNode(
            index=len(self.routers[level]),
            level=level,
            direction=direction,
            parent=parent,
        )
        self.routers[level].append(node)
        if level < self.depth - 1:
            node.left = self._build_tree(level + 1, "0", node)
            node.right = self._build_tree(level + 1, "1", node)
        return node

    def add_router_tree(self, circuit: QuantumCircuit) -> None:
        """Attach all router registers to the circuit."""

        def walk(node: DualRailRouterNode) -> None:
            node.add_registers(circuit)
            if node.left is not None:
                walk(node.left)
            if node.right is not None:
                walk(node.right)

        walk(self.root)

    def __call__(self, *args) -> QuantumCircuit:
        """Build the QRAM circuit.

        Accepts either (circuit, address_reg, bus_reg) or (address_reg, bus_reg).
        """

        if len(args) == 3 and isinstance(args[0], QuantumCircuit):
            circuit, address_reg, bus_reg = args
        elif len(args) == 2:
            address_reg, bus_reg = args
            circuit = self.circuit or QuantumCircuit(address_reg, bus_reg)
        else:
            raise ValueError("Expected (circuit, address_reg, bus_reg) or (address_reg, bus_reg)")

        self.circuit = circuit

        # Attach router resources
        self.add_router_tree(circuit)

        # Syndrome register
        if self.record_syndrome:
            bits_per_router = 5
            total_router_calls = self._estimate_router_calls()
            creg = ClassicalRegister(bits_per_router * total_router_calls, "syndrome")
            circuit.add_register(creg)
            self.syndrome_tracker = SyndromeTracker(creg)
        else:
            creg = ClassicalRegister(1, "syndrome")
            circuit.add_register(creg)
            self.syndrome_tracker = ReuseSyndromeTracker(creg)

        # Store external references
        self.address_reg = address_reg
        self.bus_reg = bus_reg

        # Build the algorithm
        self.decompose_circuit()
        return circuit

    def _estimate_router_calls(self) -> int:
        # Store address bits
        total = 0
        for i in range(self.depth):
            total += 2 * _router_calls_for_depth(i)
        # Route bus down + up
        total += 2 * _router_calls_for_depth(self.depth - 1)
        # Restore address bits
        for i in range(self.depth):
            total += 2 * _router_calls_for_depth(i)
        # Ensure at least 1 to avoid zero-sized creg
        return max(total, 1)

    def _router(self, node: DualRailRouterNode) -> None:
        if node.left is None or node.right is None:
            return
        ft_router(
            self.circuit,
            node.addr,
            node.bus,
            node.left.bus,
            node.right.bus,
            node.flag,
            node.parity,
            self.syndrome_tracker,
        )

    def _route_down(self, node: DualRailRouterNode, target_depth: int) -> None:
        if node.level >= target_depth:
            return
        self._router(node)
        if node.left is not None:
            self._route_down(node.left, target_depth)
        if node.right is not None:
            self._route_down(node.right, target_depth)

    def _route_up(self, node: DualRailRouterNode, target_depth: int) -> None:
        if node.level >= target_depth:
            return
        if node.left is not None:
            self._route_up(node.left, target_depth)
        if node.right is not None:
            self._route_up(node.right, target_depth)
        self._router(node)

    def _address_pairs(self) -> List[DualRailPair]:
        return split_dual_rail_register(self.address_reg)

    def _bus_pairs(self) -> List[DualRailPair]:
        return split_dual_rail_register(self.bus_reg)

    def _store_address_bits(self) -> None:
        address_pairs = self._address_pairs()
        root_bus = self.root.bus

        for level, addr_pair in enumerate(address_pairs):
            # Move address bit into root bus buffer
            swap_dual_rail(self.circuit, addr_pair, root_bus)
            # Route down to the target depth
            self._route_down(self.root, target_depth=level)
            # Store the address bit into routers at this level
            for node in self.routers[level]:
                swap_dual_rail(self.circuit, node.bus, node.addr)
            # Route back up, leaving root bus empty
            self._route_up(self.root, target_depth=level)

    def _restore_address_bits(self) -> None:
        address_pairs = self._address_pairs()
        root_bus = self.root.bus

        for level in reversed(range(self.depth)):
            # Route empty root bus down to reach stored address bits
            self._route_down(self.root, target_depth=level)
            for node in self.routers[level]:
                swap_dual_rail(self.circuit, node.bus, node.addr)
            self._route_up(self.root, target_depth=level)
            # Restore to external address register
            swap_dual_rail(self.circuit, address_pairs[level], root_bus)

    def _memory_interaction(self) -> None:
        """Phase-oracle style memory interaction at leaf routers."""

        if self.depth == 0:
            return

        for node in self.routers[-1]:
            prefix = node.address
            left_addr = prefix + "0"
            right_addr = prefix + "1"
            left_idx = int(left_addr, 2)
            right_idx = int(right_addr, 2)

            # Encode classical memory bits as phase flips on the bus logical |1_L> rail.
            if self.data[left_idx] == 1:
                # Address last bit = 0 -> addr.rail1 is active
                self.circuit.cz(node.addr.rail1, node.bus.rail0)
            if self.data[right_idx] == 1:
                # Address last bit = 1 -> addr.rail0 is active
                self.circuit.cz(node.addr.rail0, node.bus.rail0)

    def _route_bus_query(self) -> None:
        root_bus = self.root.bus

        for bus_pair in self._bus_pairs():
            # Prepare bus if requested
            if self.prepare_bus:
                prepare_logical_zero(self.circuit, bus_pair)
                logical_h(self.circuit, bus_pair)

            swap_dual_rail(self.circuit, bus_pair, root_bus)
            self._route_down(self.root, target_depth=self.depth - 1)
            self._memory_interaction()
            self._route_up(self.root, target_depth=self.depth - 1)
            swap_dual_rail(self.circuit, bus_pair, root_bus)

            if self.prepare_bus:
                logical_h(self.circuit, bus_pair)

    def decompose_circuit(self) -> None:
        self._store_address_bits()
        self._route_bus_query()
        self._restore_address_bits()
