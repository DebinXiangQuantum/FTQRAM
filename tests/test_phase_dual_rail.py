import os
import sys
from typing import Dict, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer, AerSimulator
from ftqram.phase_dual_rail import (
    PhaseDualRailQram,
    split_phase_dual_rail_register,
    prepare_logical_zero,
    logical_h,
)

def _bitstring_to_rev(bitstring: str) -> str:
    compact = bitstring.replace(" ", "")
    return compact[::-1]

def _reg_bit_indices(circuit: QuantumCircuit, reg: ClassicalRegister) -> list[int]:
    return [circuit.clbits.index(reg[i]) for i in range(len(reg))]

def _extract_bits_by_index(bitstring_rev: str, bit_indices: list[int]) -> list[str]:
    return [bitstring_rev[idx] for idx in bit_indices]

def _logical_from_phase_dual_rail(rail0: str, rail1: str) -> Tuple[str, bool]:
    # In X-basis measurement: |+> gives '0', |-> gives '1'
    # |0_L> = |+-> -> rail0=0, rail1=1
    if rail0 == "0" and rail1 == "1":
        return "0", True
    # |1_L> = |-+> -> rail0=1, rail1=0
    if rail0 == "1" and rail1 == "0":
        return "1", True
    return "X", False

def _parse_counts(counts: Dict[str, int], circuit: QuantumCircuit, addr_reg: ClassicalRegister, bus_reg: ClassicalRegister, logical_bits: int, syndrome_reg=None):
    addr_indices = _reg_bit_indices(circuit, addr_reg)
    bus_indices = _reg_bit_indices(circuit, bus_reg)
    if syndrome_reg:
        syn_indices = _reg_bit_indices(circuit, syndrome_reg)
    else:
        syn_indices = []

    logical_counts: Dict[str, int] = {}
    invalid = 0
    total = 0
    syn_errors = 0

    for bitstring, count in counts.items():
        rev = _bitstring_to_rev(bitstring)
        total += count
        
        # Check syndrome if available
        if syn_indices:
            syn_bits = _extract_bits_by_index(rev, syn_indices)
            if any(b == '1' for b in syn_bits):
                syn_errors += count
                invalid += count
                continue

        addr_bits = _extract_bits_by_index(rev, addr_indices)
        bus_bits = _extract_bits_by_index(rev, bus_indices)

        addr_logical = []
        valid = True
        for i in range(logical_bits):
            rail0 = addr_bits[2 * i]
            rail1 = addr_bits[2 * i + 1]
            bit, ok = _logical_from_phase_dual_rail(rail0, rail1)
            if not ok:
                valid = False
                break
            addr_logical.append(bit)

        bus_logical = "X"
        if valid:
            rail0 = bus_bits[0]
            rail1 = bus_bits[1]
            bus_logical, ok = _logical_from_phase_dual_rail(rail0, rail1)
            if not ok:
                valid = False

        if not valid:
            invalid += count
            continue

        addr_str = "".join(reversed(addr_logical))
        key = f"{addr_str}|{bus_logical}"
        logical_counts[key] = logical_counts.get(key, 0) + count

    invalid_rate = invalid / total if total else 0.0
    syn_error_rate = syn_errors / total if total else 0.0
    return logical_counts, invalid_rate, syn_error_rate

def _assert_bit_reversed_mapping(logical_counts: Dict[str, int], data: list[int], address_bits: int) -> None:
    for addr_val in range(1 << address_bits):
        addr_str = format(addr_val, f"0{address_bits}b")
        expected_bus = str(data[int(addr_str[::-1], 2)])
        wrong_bus = "1" if expected_bus == "0" else "0"

        assert logical_counts.get(f"{addr_str}|{expected_bus}", 0) > 0, (
            f"Missing expected logical outcome for address {addr_str}"
        )
        assert logical_counts.get(f"{addr_str}|{wrong_bus}", 0) == 0, (
            f"Observed wrong bus value for address {addr_str}"
        )

def test_no_noise():
    address_bits = 2
    data = [0, 1, 0, 1]
    shots = 1000

    addr_q = QuantumRegister(2 * address_bits, "addr_dr")
    bus_q = QuantumRegister(2, "bus_dr")
    circuit = QuantumCircuit(addr_q, bus_q)

    for pair in split_phase_dual_rail_register(addr_q):
        prepare_logical_zero(circuit, pair)
        logical_h(circuit, pair)

    qram = PhaseDualRailQram(address_bits, data, record_syndrome=True, fault_tolerant=True)
    qram(circuit, addr_q, bus_q)

    # To measure in X basis, apply H to all data qubits
    circuit.h(addr_q)
    circuit.h(bus_q)

    addr_c = ClassicalRegister(2 * address_bits, "addr_c")
    bus_c = ClassicalRegister(2, "bus_c")
    circuit.add_register(addr_c)
    circuit.add_register(bus_c)
    circuit.measure(addr_q, addr_c)
    circuit.measure(bus_q, bus_c)
    
    syn_creg = next((cr for cr in circuit.cregs if cr.name == "syndrome"), None)

    backend = Aer.get_backend("qasm_simulator")
    result = backend.run(circuit, shots=shots).result()
    counts = result.get_counts(circuit)

    logical_counts, invalid_rate, syn_error_rate = _parse_counts(counts, circuit, addr_c, bus_c, address_bits, syn_creg)
    
    print("Ideal simulation results:")
    for k, v in sorted(logical_counts.items()):
        print(f"  {k}: {v}")
    print(f"Invalid state rate: {invalid_rate}")
    print(f"Syndrome error rate: {syn_error_rate}")

    # Check correctness
    assert invalid_rate == 0.0, "Expected 0 invalid states in ideal simulation"
    assert syn_error_rate == 0.0, "Expected 0 syndrome errors in ideal simulation"
    _assert_bit_reversed_mapping(logical_counts, data, address_bits)


def test_no_noise_three_layers_mps():
    address_bits = 3
    data = [0, 0, 1, 1, 1, 0, 0, 1]
    shots = 128

    addr_q = QuantumRegister(2 * address_bits, "addr_dr")
    bus_q = QuantumRegister(2, "bus_dr")
    circuit = QuantumCircuit(addr_q, bus_q)

    for pair in split_phase_dual_rail_register(addr_q):
        prepare_logical_zero(circuit, pair)
        logical_h(circuit, pair)

    qram = PhaseDualRailQram(address_bits, data, record_syndrome=True, fault_tolerant=True)
    qram(circuit, addr_q, bus_q)

    circuit.h(addr_q)
    circuit.h(bus_q)

    addr_c = ClassicalRegister(2 * address_bits, "addr_c")
    bus_c = ClassicalRegister(2, "bus_c")
    circuit.add_register(addr_c)
    circuit.add_register(bus_c)
    circuit.measure(addr_q, addr_c)
    circuit.measure(bus_q, bus_c)

    backend = AerSimulator(method="matrix_product_state")
    transpiled = transpile(circuit, backend=backend, optimization_level=0)

    addr_c_t = next(cr for cr in transpiled.cregs if cr.name == "addr_c")
    bus_c_t = next(cr for cr in transpiled.cregs if cr.name == "bus_c")
    syn_creg_t = next(cr for cr in transpiled.cregs if cr.name == "syndrome")

    result = backend.run(transpiled, shots=shots, seed_simulator=7).result()
    counts = result.get_counts(transpiled)

    logical_counts, invalid_rate, syn_error_rate = _parse_counts(
        counts, transpiled, addr_c_t, bus_c_t, address_bits, syn_creg_t
    )

    assert invalid_rate == 0.0, "Expected 0 invalid states in 3-layer ideal simulation"
    assert syn_error_rate == 0.0, "Expected 0 syndrome errors in 3-layer ideal simulation"
    _assert_bit_reversed_mapping(logical_counts, data, address_bits)

if __name__ == "__main__":
    test_no_noise()
    test_no_noise_three_layers_mps()
    print("No-noise test completed successfully.")
