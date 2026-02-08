"""Dual-rail implementation that mirrors the bucktele.py routing logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from qiskit import QuantumCircuit, QuantumRegister

from .ops import cswap_dual_rail, swap_dual_rail
from .qubits import DualRailPair, logical_h, logical_x, prepare_logical_zero


@dataclass
class DualRailRouterQubit:
    """Dual-rail router node compatible with bucktele-style routing."""

    index: int
    level: int
    direction: str
    parent: Optional["DualRailRouterQubit"]

    left_router: Optional["DualRailRouterQubit"] = None
    right_router: Optional["DualRailRouterQubit"] = None
    left: Optional["DualRailRouterQubit"] = None
    right: Optional["DualRailRouterQubit"] = None

    def __post_init__(self) -> None:
        self.qreg = QuantumRegister(2, self.reg_name)
        self.data: Optional[QuantumRegister] = None

    @property
    def address(self) -> str:
        if self.parent is None:
            return self.direction
        return self.parent.address + self.direction

    @property
    def reg_name(self) -> str:
        if self.address:
            return f"router_{self.level}_{self.address}"
        return f"router_{self.level}"

    @property
    def pair(self) -> DualRailPair:
        return DualRailPair(self.qreg[0], self.qreg[1], self.reg_name)

    def add_data_qubits(self, circuit: QuantumCircuit) -> None:
        self.data = QuantumRegister(1, f"{self.reg_name}_data")
        circuit.add_register(self.data)


class DualRailBucketQram:
    """Dual-rail QRAM that mirrors bucktele.py's bucket-brigade routing."""

    def __init__(self, address, data: List[int], *regs, bandwidth: int = 1) -> None:
        self.address = address
        self.data = data
        self.bandwidth = bandwidth
        self.circuit = QuantumCircuit(*regs) if regs else None
        self.depth = len(address[0]) if isinstance(address, list) and address else int(address)
        self.routers: List[List[DualRailRouterQubit]] = [
            [] for _ in range(self.depth)
        ]
        self.root = self._build_tree(0, "", None)
        self.incident: Optional[DualRailRouterQubit] = None

    def _build_tree(
        self, level: int, direction: str, parent: Optional[DualRailRouterQubit]
    ) -> DualRailRouterQubit:
        node = DualRailRouterQubit(
            index=len(self.routers[level]),
            level=level,
            direction=direction,
            parent=parent,
        )
        self.routers[level].append(node)

        node.left_router = DualRailRouterQubit(0, level, "l", node)
        node.right_router = DualRailRouterQubit(0, level, "r", node)

        if level < self.depth - 1:
            node.left = self._build_tree(level + 1, "0", node)
            node.right = self._build_tree(level + 1, "1", node)
        return node

    def add_router_tree(self, level: int, root: DualRailRouterQubit) -> None:
        self.circuit.add_register(root.qreg)
        if root.left_router is not None:
            self.circuit.add_register(root.left_router.qreg)
        if root.right_router is not None:
            self.circuit.add_register(root.right_router.qreg)
        # routers are already tracked during _build_tree
        if root.left is not None:
            self.add_router_tree(level + 1, root.left)
        if root.right is not None:
            self.add_router_tree(level + 1, root.right)

    def add_incident_qubits(self, incident: DualRailRouterQubit) -> None:
        self.incident = incident
        self.circuit.add_register(self.incident.qreg)

    def __call__(self, address_qubits: QuantumRegister, bus_qubits: QuantumRegister) -> QuantumCircuit:
        self.address_qubits = address_qubits
        self.bus_qubits = bus_qubits
        if self.circuit is None:
            self.circuit = QuantumCircuit(address_qubits, bus_qubits)

        # Build router tree and incident register
        self.routers = [[] for _ in range(self.depth)]
        self.root = self._build_tree(0, "", None)
        self.add_router_tree(0, self.root)

        incident = DualRailRouterQubit(0, 0, "inc", None)
        self.add_incident_qubits(incident)

        for node in self.routers[-1]:
            node.add_data_qubits(self.circuit)

        self.decompose_circuit(self.circuit)
        return self.circuit

    def _router(self, circuit: QuantumCircuit, router: DualRailPair, incident: DualRailPair, left: DualRailPair, right: DualRailPair) -> None:
        logical_x(circuit, router)
        cswap_dual_rail(circuit, router.rail0, incident, left)
        logical_x(circuit, router)
        cswap_dual_rail(circuit, router.rail0, incident, right)

    def _reverse_router(self, circuit: QuantumCircuit, router: DualRailPair, incident: DualRailPair, left: DualRailPair, right: DualRailPair) -> None:
        cswap_dual_rail(circuit, router.rail0, incident, right)
        logical_x(circuit, router)
        cswap_dual_rail(circuit, router.rail0, incident, left)
        logical_x(circuit, router)

    def _layers_router(
        self,
        circuit: QuantumCircuit,
        router_obj: DualRailRouterQubit,
        incident: DualRailPair,
        address_index: int,
        mid: DualRailRouterQubit,
    ) -> None:
        if router_obj.level == 0:
            swap_dual_rail(circuit, incident, mid.pair)
        if router_obj.level == 0 and address_index == 0:
            swap_dual_rail(circuit, router_obj.pair, mid.pair)
        else:
            if router_obj.level + 2 == address_index and address_index == len(self.address_qubits) // 2:
                self._router(circuit, router_obj.pair, mid.pair, router_obj.left.pair, router_obj.right.pair)
                return

            self._router(
                circuit,
                router_obj.pair,
                mid.pair,
                router_obj.left_router.pair,
                router_obj.right_router.pair,
            )
            if router_obj.level + 1 == address_index:
                if router_obj.left_router is not None and router_obj.left is not None:
                    swap_dual_rail(circuit, router_obj.left_router.pair, router_obj.left.pair)
                if router_obj.right_router is not None and router_obj.right is not None:
                    swap_dual_rail(circuit, router_obj.right_router.pair, router_obj.right.pair)
                return
            if router_obj.left is not None:
                self._layers_router(circuit, router_obj.left, router_obj.left_router.pair, address_index, router_obj.left_router)
            if router_obj.right is not None:
                self._layers_router(circuit, router_obj.right, router_obj.right_router.pair, address_index, router_obj.right_router)

    def _reverse_layers_router(
        self,
        circuit: QuantumCircuit,
        router_obj: DualRailRouterQubit,
        incident: DualRailPair,
        address_index: int,
        mid: DualRailRouterQubit,
    ) -> None:
        if address_index != 0:
            if router_obj.level + 1 > address_index:
                return
            if router_obj.right is not None:
                self._reverse_layers_router(circuit, router_obj.right, router_obj.right_router.pair, address_index, router_obj.right_router)
            if router_obj.left is not None:
                self._reverse_layers_router(circuit, router_obj.left, router_obj.left_router.pair, address_index, router_obj.left_router)
            if router_obj.level + 2 == address_index and address_index == len(self.address_qubits) // 2:
                self._router(circuit, router_obj.pair, mid.pair, router_obj.left.pair, router_obj.right.pair)
                return
            elif router_obj.level + 1 == address_index and address_index != len(self.address_qubits) // 2:
                if router_obj.left_router is not None and router_obj.left is not None:
                    swap_dual_rail(circuit, router_obj.left_router.pair, router_obj.left.pair)
                if router_obj.right_router is not None and router_obj.right is not None:
                    swap_dual_rail(circuit, router_obj.right_router.pair, router_obj.right.pair)
            if router_obj.right_router is not None:
                self._reverse_router(
                    circuit,
                    router_obj.pair,
                    mid.pair,
                    router_obj.left_router.pair,
                    router_obj.right_router.pair,
                )
        else:
            swap_dual_rail(circuit, router_obj.pair, mid.pair)
        if router_obj.level == 0:
            swap_dual_rail(circuit, incident, mid.pair)

    def decompose_circuit(self, circuit: QuantumCircuit) -> None:
        # Prepare bus in logical |+>
        bus_pairs = [DualRailPair(self.bus_qubits[0], self.bus_qubits[1], "bus0")]
        for pair in bus_pairs:
            prepare_logical_zero(circuit, pair)
            logical_h(circuit, pair)

        address_pairs = [
            DualRailPair(self.address_qubits[2 * i], self.address_qubits[2 * i + 1], f"addr{i}")
            for i in range(len(self.address_qubits) // 2)
        ]

        incidents = {i: address_pairs[i] for i in range(len(address_pairs))}
        incidents.update({i + len(address_pairs): bus_pairs[i] for i in range(len(bus_pairs))})

        for idx in range(len(incidents)):
            self._layers_router(circuit, self.routers[0][0], incidents[idx], idx, self.incident)
            circuit.barrier()

        # Memory interaction on leaf nodes
        for router_obj in self.routers[-1]:
            left_idx = int(router_obj.address + "0", 2)
            right_idx = int(router_obj.address + "1", 2)

            if self.data[right_idx] == 1:
                circuit.cz(router_obj.pair.rail0, router_obj.data[0])
            if self.data[left_idx] == 1:
                logical_x(circuit, router_obj.pair)
                circuit.cz(router_obj.pair.rail0, router_obj.data[0])
                logical_x(circuit, router_obj.pair)

        circuit.barrier()

        for idx in reversed(range(len(address_pairs) + 1)):
            self._reverse_layers_router(circuit, self.routers[0][0], incidents[idx], idx, self.incident)
            circuit.barrier()

        for pair in bus_pairs:
            logical_h(circuit, pair)
