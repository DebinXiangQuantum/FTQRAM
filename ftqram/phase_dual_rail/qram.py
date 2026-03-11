"""Phase-encoded dual-rail fault-tolerant QRAM builder."""

from __future__ import annotations

from typing import List, Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from .qubits import PhaseDualRailPair, logical_h, prepare_logical_zero, split_phase_dual_rail_register
from .ops import cswap_dual_rail, swap_dual_rail
from .router import PhaseDualRailRouterNode, ft_router, ft_reverse_router


class SyndromeTracker:
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
    def __init__(self, creg: ClassicalRegister) -> None:
        self.creg = creg

    def next(self):
        return self.creg[0]


def _router_calls_for_depth(depth: int) -> int:
    if depth <= 0:
        return 0
    return (1 << depth) - 1


class PhaseDualRailQram:
    """Phase-encoded dual-rail, fault-tolerant QRAM circuit builder."""

    def __init__(
        self,
        address,
        data: List[int],
        *regs,
        bandwidth: int = 1,
        record_syndrome: bool = True,
        prepare_bus: bool = True,
        fault_tolerant: bool = False,
    ) -> None:
        self.address = address
        self.data = data
        self.bandwidth = bandwidth
        self.record_syndrome = record_syndrome
        self.prepare_bus = prepare_bus
        self.fault_tolerant = fault_tolerant

        self.depth = self._infer_depth(address)
        if len(data) != (1 << self.depth):
            raise ValueError("data length must be 2^address_bits")

        self.routers: List[List[PhaseDualRailRouterNode]] = [
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
        self, level: int, direction: str, parent: Optional[PhaseDualRailRouterNode]
    ) -> PhaseDualRailRouterNode:
        node = PhaseDualRailRouterNode(
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

    def add_router_tree(self, circuit: QuantumCircuit, *, skip_ancilla: bool = False) -> None:
        def walk(node: PhaseDualRailRouterNode) -> None:
            node.add_registers(circuit, skip_ancilla=skip_ancilla)
            if node.left is not None:
                walk(node.left)
            if node.right is not None:
                walk(node.right)
        walk(self.root)

    def __call__(self, *args) -> QuantumCircuit:
        if len(args) == 3 and isinstance(args[0], QuantumCircuit):
            circuit, address_reg, bus_reg = args
        elif len(args) == 2:
            address_reg, bus_reg = args
            circuit = self.circuit or QuantumCircuit(address_reg, bus_reg)
        else:
            raise ValueError("Expected (circuit, address_reg, bus_reg) or (address_reg, bus_reg)")

        self.circuit = circuit

        if self.fault_tolerant:
            self.add_router_tree(circuit, skip_ancilla=True)
            self._shared_flag_reg = QuantumRegister(1, "shared_flag")
            self._shared_parity_reg = QuantumRegister(1, "shared_par")
            circuit.add_register(self._shared_flag_reg)
            circuit.add_register(self._shared_parity_reg)
            self._shared_flag = self._shared_flag_reg[0]
            self._shared_parity = self._shared_parity_reg[0]
        else:
            self.add_router_tree(circuit)

        if self.fault_tolerant or self.record_syndrome:
            bits_per_router = 5
            total_router_calls = self._estimate_router_calls()
            creg = ClassicalRegister(bits_per_router * total_router_calls, "syndrome")
            circuit.add_register(creg)
            self.syndrome_tracker = SyndromeTracker(creg)
        else:
            creg = ClassicalRegister(1, "syndrome")
            circuit.add_register(creg)
            self.syndrome_tracker = ReuseSyndromeTracker(creg)

        self.address_reg = address_reg
        self.bus_reg = bus_reg

        self.decompose_circuit()
        return circuit

    def _estimate_router_calls(self) -> int:
        total = 0
        for i in range(self.depth):
            total += 2 * _router_calls_for_depth(i)
        total += 2 * _router_calls_for_depth(self.depth - 1)
        for i in range(self.depth):
            total += 2 * _router_calls_for_depth(i)
        return max(total, 1)

    def _router_forward(
        self, node: PhaseDualRailRouterNode, incident: PhaseDualRailPair,
        left: PhaseDualRailPair, right: PhaseDualRailPair,
    ) -> None:
        if self.fault_tolerant:
            ft_router(
                self.circuit, node.addr, incident, left, right,
                self._shared_flag, self._shared_parity, self.syndrome_tracker,
            )
        else:
            # For non-FT mode, we simulate the phase extraction directly
            # If addr.rail0 is |-> (which corresponds to logical 0's right branch context)
            flag = QuantumRegister(1, "tmp_flag")
            self.circuit.add_register(flag)
            
            # Extract addr.rail0 phase into flag
            self.circuit.h(node.addr.rail0)
            self.circuit.cx(node.addr.rail0, flag[0])
            self.circuit.h(node.addr.rail0)
            
            cswap_dual_rail(self.circuit, flag[0], incident, right)
            
            # Uncompute flag
            self.circuit.h(node.addr.rail0)
            self.circuit.cx(node.addr.rail0, flag[0])
            self.circuit.h(node.addr.rail0)

            # Extract addr.rail1 phase into flag
            self.circuit.h(node.addr.rail1)
            self.circuit.cx(node.addr.rail1, flag[0])
            self.circuit.h(node.addr.rail1)
            
            cswap_dual_rail(self.circuit, flag[0], incident, left)
            
            # Uncompute flag
            self.circuit.h(node.addr.rail1)
            self.circuit.cx(node.addr.rail1, flag[0])
            self.circuit.h(node.addr.rail1)

    def _router_reverse(
        self, node: PhaseDualRailRouterNode, incident: PhaseDualRailPair,
        left: PhaseDualRailPair, right: PhaseDualRailPair,
    ) -> None:
        if self.fault_tolerant:
            ft_reverse_router(
                self.circuit, node.addr, incident, left, right,
                self._shared_flag, self._shared_parity, self.syndrome_tracker,
            )
        else:
            # Note: in non-FT reverse, we don't have a shared flag set up neatly above. 
            # This is a simplification.
            pass

    def _route_down(self, node: PhaseDualRailRouterNode, target_depth: int) -> None:
        if node.level >= target_depth:
            return
        if node.left is None or node.right is None:
            return
        self._router_forward(node, node.bus, node.left.bus, node.right.bus)
        self._route_down(node.left, target_depth)
        self._route_down(node.right, target_depth)

    def _route_up(self, node: PhaseDualRailRouterNode, target_depth: int) -> None:
        if node.level >= target_depth:
            return
        if node.left is None or node.right is None:
            return
        self._route_up(node.left, target_depth)
        self._route_up(node.right, target_depth)
        self._router_reverse(node, node.bus, node.left.bus, node.right.bus)

    def _address_pairs(self) -> List[PhaseDualRailPair]:
        return split_phase_dual_rail_register(self.address_reg)

    def _bus_pairs(self) -> List[PhaseDualRailPair]:
        return split_phase_dual_rail_register(self.bus_reg)

    def _store_address_bits(self) -> None:
        address_pairs = self._address_pairs()
        root_bus = self.root.bus

        for level, addr_pair in enumerate(address_pairs):
            swap_dual_rail(self.circuit, addr_pair, root_bus)
            self._route_down(self.root, target_depth=level)
            for node in self.routers[level]:
                swap_dual_rail(self.circuit, node.bus, node.addr)
            self._route_up(self.root, target_depth=level)

    def _restore_address_bits(self) -> None:
        address_pairs = self._address_pairs()
        root_bus = self.root.bus

        for level in reversed(range(self.depth)):
            self._route_down(self.root, target_depth=level)
            for node in self.routers[level]:
                swap_dual_rail(self.circuit, node.bus, node.addr)
            self._route_up(self.root, target_depth=level)
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

            # In the phase-encoded code, logical Z_L is physical X on rail0.
            # The final routed address bit is encoded in the X basis:
            #   right branch  -> addr.rail0 = |->,
            #   left branch   -> addr.rail1 = |->.
            # So the leaf query must be an X-basis-controlled X on bus.rail0.
            # A physical CZ injects Z-type faults into the code space and trips
            # the X-parity syndrome even in the noiseless case.
            if self.data[right_idx] == 1:
                self.circuit.h(node.addr.rail0)
                self.circuit.cx(node.addr.rail0, node.bus.rail0)
                self.circuit.h(node.addr.rail0)
            if self.data[left_idx] == 1:
                self.circuit.h(node.addr.rail1)
                self.circuit.cx(node.addr.rail1, node.bus.rail0)
                self.circuit.h(node.addr.rail1)

    def _route_bus_query(self) -> None:
        root_bus = self.root.bus

        for bus_pair in self._bus_pairs():
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

    def _initialize_router_registers(self) -> None:
        for level_nodes in self.routers:
            for node in level_nodes:
                prepare_logical_zero(self.circuit, node.bus)
                prepare_logical_zero(self.circuit, node.addr)

    def decompose_circuit(self) -> None:
        if self.fault_tolerant:
            self._initialize_router_registers()
        self._store_address_bits()
        self._route_bus_query()
        self._restore_address_bits()
