from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer

from ftqram.dual_rail import DualRailQram, logical_h, prepare_logical_zero, split_dual_rail_register


def run_dual_rail_demo():
    # Same address/data as bucktele.py
    address_bits = 3
    address_list = [bin(i)[2:].zfill(address_bits) for i in range(2**address_bits)]
    data = [0, 0, 1, 1, 1, 0, 0, 1]

    # Dual-rail registers: 2 qubits per logical bit
    address_reg = QuantumRegister(2 * address_bits, "addr_dr")
    bus_reg = QuantumRegister(2, "bus_dr")

    circuit = QuantumCircuit(address_reg, bus_reg)

    # Prepare address register in equal superposition |+>_L for each bit
    for pair in split_dual_rail_register(address_reg):
        prepare_logical_zero(circuit, pair)
        logical_h(circuit, pair)

    qram = DualRailQram(address_list, data, bandwidth=1, record_syndrome=True, prepare_bus=True)
    qram(circuit, address_reg, bus_reg)

    # Measurements
    addr_c = ClassicalRegister(2 * address_bits, "addr_c")
    bus_c = ClassicalRegister(2, "bus_c")
    circuit.add_register(addr_c)
    circuit.add_register(bus_c)
    circuit.measure(address_reg, addr_c)
    circuit.measure(bus_reg, bus_c)

    simulator = Aer.get_backend("qasm_simulator")
    result = simulator.run(circuit, shots=2000).result()
    counts = result.get_counts(circuit)

    print("Dual-rail FT-QRAM counts (raw):")
    print(counts)

    transpiled = transpile(circuit, basis_gates=["rx", "rz", "cz"], optimization_level=2)
    print("Transpiled depth:", transpiled.depth())
    print("Gate counts:", transpiled.count_ops())


if __name__ == "__main__":
    run_dual_rail_demo()
