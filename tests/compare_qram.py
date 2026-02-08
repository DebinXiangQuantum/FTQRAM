import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import Aer

from bucktele import Qram as BuckQram, RouterQubit
from ftqram.dual_rail import DualRailQram, logical_h, prepare_logical_zero, split_dual_rail_register


@dataclass
class CaseResult:
    name: str
    l1_distance: float
    max_diff: float
    invalid_rate: float
    passed: bool


class RegisterProxy:
    """Proxy to provide legacy _size attribute for bucktele.Qram."""

    def __init__(self, qreg: QuantumRegister) -> None:
        self._qreg = qreg
        self._size = len(qreg)

    def __len__(self) -> int:
        return len(self._qreg)

    def __getitem__(self, idx):
        return self._qreg[idx]


def build_bucktele_tree(qram: BuckQram, address_bits: int) -> None:
    """Create router tree and incident qubits for bucktele.Qram."""

    qram.routers = [[] for _ in range(address_bits)]
    counters = {level: 0 for level in range(address_bits)}

    def make_router(level: int, direction: str, parent) -> RouterQubit:
        idx = counters[level]
        counters[level] += 1
        node = RouterQubit(idx, level, direction, parent)

        node.left_router = RouterQubit(0, level, "l", node)
        node.right_router = RouterQubit(0, level, "r", node)

        if level < address_bits - 1:
            node.left = make_router(level + 1, "0", node)
            node.right = make_router(level + 1, "1", node)
        else:
            node.left = None
            node.right = None
        return node

    root = make_router(0, "", None)
    incident = RouterQubit(0, 0, "inc", None)

    qram.add_router_tree(0, root)
    qram.add_incident_qubits(incident)

    for node in qram.routers[-1]:
        node.add_data_qubits(qram.circuit)


def _bitstring_to_rev(bitstring: str) -> str:
    compact = bitstring.replace(" ", "")
    return compact[::-1]


def _reg_bit_indices(circuit: QuantumCircuit, reg: ClassicalRegister) -> List[int]:
    return [circuit.clbits.index(reg[i]) for i in range(len(reg))]


def _extract_bits_by_index(bitstring_rev: str, bit_indices: List[int]) -> List[str]:
    return [bitstring_rev[idx] for idx in bit_indices]


def _logical_from_dual_rail(rail0: str, rail1: str) -> Tuple[str, bool]:
    if rail0 == "0" and rail1 == "1":
        return "0", True
    if rail0 == "1" and rail1 == "0":
        return "1", True
    return "X", False


def _bucket_counts_to_logical(
    counts: Dict[str, int],
    circuit: QuantumCircuit,
    addr_reg: ClassicalRegister,
    bus_reg: ClassicalRegister,
) -> Dict[str, int]:
    addr_indices = _reg_bit_indices(circuit, addr_reg)
    bus_indices = _reg_bit_indices(circuit, bus_reg)

    logical_counts: Dict[str, int] = {}
    for bitstring, count in counts.items():
        rev = _bitstring_to_rev(bitstring)
        addr_bits = _extract_bits_by_index(rev, addr_indices)
        bus_bits = _extract_bits_by_index(rev, bus_indices)

        addr_str = "".join(reversed(addr_bits))
        bus_str = "".join(reversed(bus_bits))

        key = f"{addr_str}|{bus_str}"
        logical_counts[key] = logical_counts.get(key, 0) + count
    return logical_counts


def _dualrail_counts_to_logical(
    counts: Dict[str, int],
    circuit: QuantumCircuit,
    addr_reg: ClassicalRegister,
    bus_reg: ClassicalRegister,
    logical_bits: int,
) -> Tuple[Dict[str, int], float]:
    addr_indices = _reg_bit_indices(circuit, addr_reg)
    bus_indices = _reg_bit_indices(circuit, bus_reg)

    logical_counts: Dict[str, int] = {}
    invalid = 0
    total = 0

    for bitstring, count in counts.items():
        rev = _bitstring_to_rev(bitstring)
        addr_bits = _extract_bits_by_index(rev, addr_indices)
        bus_bits = _extract_bits_by_index(rev, bus_indices)

        addr_logical = []
        valid = True
        for i in range(logical_bits):
            rail0 = addr_bits[2 * i]
            rail1 = addr_bits[2 * i + 1]
            bit, ok = _logical_from_dual_rail(rail0, rail1)
            if not ok:
                valid = False
                break
            addr_logical.append(bit)

        bus_logical = "X"
        if valid:
            rail0 = bus_bits[0]
            rail1 = bus_bits[1]
            bus_logical, ok = _logical_from_dual_rail(rail0, rail1)
            if not ok:
                valid = False

        total += count
        if not valid:
            invalid += count
            continue

        addr_str = "".join(reversed(addr_logical))
        key = f"{addr_str}|{bus_logical}"
        logical_counts[key] = logical_counts.get(key, 0) + count

    invalid_rate = invalid / total if total else 0.0
    return logical_counts, invalid_rate


def _normalize(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _compare_distributions(a: Dict[str, int], b: Dict[str, int]) -> Tuple[float, float]:
    pa = _normalize(a)
    pb = _normalize(b)
    keys = set(pa) | set(pb)
    l1 = 0.0
    max_diff = 0.0
    for k in keys:
        diff = abs(pa.get(k, 0.0) - pb.get(k, 0.0))
        l1 += diff
        max_diff = max(max_diff, diff)
    return l1, max_diff


def run_bucktele(address_bits: int, data: List[int], shots: int, seed: int) -> Dict[str, int]:
    address_list = [bin(i)[2:].zfill(address_bits) for i in range(2**address_bits)]

    addr_q = QuantumRegister(address_bits, "addr")
    bus_q = QuantumRegister(1, "bus")
    addr_c = ClassicalRegister(address_bits, "addr_c")
    bus_c = ClassicalRegister(1, "bus_c")

    qram = BuckQram(address_list, data, addr_q, bus_q, addr_c, bus_c, bandwidth=1)
    circuit = qram.circuit

    build_bucktele_tree(qram, address_bits)

    for i in range(address_bits):
        circuit.h(addr_q[i])

    # bucktele.py expects a legacy _size attribute on registers.
    addr_proxy = RegisterProxy(addr_q)
    bus_proxy = RegisterProxy(bus_q)

    qram(addr_proxy, bus_proxy)
    circuit.measure(bus_q, bus_c)
    circuit.measure(addr_q, addr_c)

    backend = Aer.get_backend("qasm_simulator")
    result = backend.run(circuit, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts(circuit)

    return _bucket_counts_to_logical(counts, circuit, addr_c, bus_c)


def run_dualrail(address_bits: int, data: List[int], shots: int, seed: int) -> Tuple[Dict[str, int], float]:
    address_list = [bin(i)[2:].zfill(address_bits) for i in range(2**address_bits)]

    addr_q = QuantumRegister(2 * address_bits, "addr_dr")
    bus_q = QuantumRegister(2, "bus_dr")
    circuit = QuantumCircuit(addr_q, bus_q)

    for pair in split_dual_rail_register(addr_q):
        prepare_logical_zero(circuit, pair)
        logical_h(circuit, pair)

    qram = DualRailQram(address_list, data, bandwidth=1, record_syndrome=True, prepare_bus=True)
    qram(circuit, addr_q, bus_q)

    addr_c = ClassicalRegister(2 * address_bits, "addr_c")
    bus_c = ClassicalRegister(2, "bus_c")
    circuit.add_register(addr_c)
    circuit.add_register(bus_c)
    circuit.measure(addr_q, addr_c)
    circuit.measure(bus_q, bus_c)

    backend = Aer.get_backend("qasm_simulator")
    result = backend.run(circuit, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts(circuit)

    return _dualrail_counts_to_logical(counts, circuit, addr_c, bus_c, address_bits)


def build_cases(address_bits_list: List[int], random_cases: int) -> List[Tuple[str, int, List[int]]]:
    cases = []
    rng = random.Random(7)
    for address_bits in address_bits_list:
        size = 2**address_bits
        cases.append((f"n={address_bits}-zeros", address_bits, [0] * size))
        cases.append((f"n={address_bits}-ones", address_bits, [1] * size))
        cases.append(
            (
                f"n={address_bits}-alt",
                address_bits,
                [i % 2 for i in range(size)],
            )
        )
        for j in range(random_cases):
            data = [rng.randint(0, 1) for _ in range(size)]
            cases.append((f"n={address_bits}-rand{j}", address_bits, data))
    return cases


def main() -> int:
    shots = int(os.environ.get("SHOTS", "32"))
    random_cases = int(os.environ.get("RANDOM_CASES", "0"))
    bits_env = os.environ.get("ADDRESS_BITS", "2")
    address_bits_list = [int(x) for x in bits_env.split(",") if x.strip()]

    tolerance_l1 = float(os.environ.get("TOL_L1", "0.15"))
    tolerance_max = float(os.environ.get("TOL_MAX", "0.12"))

    results: List[CaseResult] = []

    for idx, (name, address_bits, data) in enumerate(
        build_cases(address_bits_list, random_cases)
    ):
        seed = 100 + idx
        buck = run_bucktele(address_bits, data, shots, seed)
        dual, invalid_rate = run_dualrail(address_bits, data, shots, seed)

        l1, max_diff = _compare_distributions(buck, dual)
        passed = l1 <= tolerance_l1 and max_diff <= tolerance_max and invalid_rate <= 0.01
        results.append(
            CaseResult(
                name=name,
                l1_distance=l1,
                max_diff=max_diff,
                invalid_rate=invalid_rate,
                passed=passed,
            )
        )

        print(
            f"{name}: L1={l1:.4f} max={max_diff:.4f} invalid={invalid_rate:.4f} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    print(f"\nSummary: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
