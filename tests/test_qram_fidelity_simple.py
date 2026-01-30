"""
Simplified fidelity test using measurement-based process tomography
Practical approach that works within memory constraints
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib.util
spec = importlib.util.spec_from_file_location("dual_rail_qram", "ftqram/dual-rail-qram.py")
dual_rail_qram = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dual_rail_qram)

FlagBridgeQRAM = dual_rail_qram.FlagBridgeQRAM
initialize_dual_rail = dual_rail_qram.initialize_dual_rail

from qiskit import transpile, ClassicalRegister
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error
import numpy as np

def measure_pauli_expectation(circuit, qubit_indices, basis, noise_model=None, shots=1000):
    """
    Measure expectation value in specified Pauli basis
    basis: 'X', 'Y', or 'Z'
    """
    meas_circuit = circuit.copy()
    meas_circuit.remove_final_measurements(inplace=True)
    
    # Apply basis rotation
    for q_idx in qubit_indices:
        if basis == 'X':
            meas_circuit.h(q_idx)
        elif basis == 'Y':
            meas_circuit.sdg(q_idx)
            meas_circuit.h(q_idx)
    
    # Measure
    cr = ClassicalRegister(len(qubit_indices), 'meas')
    meas_circuit.add_register(cr)
    for i, q_idx in enumerate(qubit_indices):
        meas_circuit.measure(q_idx, cr[i])
    
    # Run
    sim = Aer.get_backend('qasm_simulator')
    if noise_model:
        job = sim.run(transpile(meas_circuit, sim), shots=shots, noise_model=noise_model)
    else:
        job = sim.run(transpile(meas_circuit, sim), shots=shots)
    
    result = job.result()
    counts = result.get_counts()
    
    # Calculate expectation values
    expectations = []
    for i in range(len(qubit_indices)):
        exp_val = 0
        total = 0
        for bitstring, count in counts.items():
            bits = bitstring.replace(' ', '')
            if len(bits) > i:
                bit = bits[-(i+1)]  # Count from right
                if bit == '0':
                    exp_val += count
                else:
                    exp_val -= count
                total += count
        expectations.append(exp_val / total if total > 0 else 0)
    
    return expectations

def calculate_fidelity_from_tomography(address_width, address_bits, expected_leaf, error_rate):
    """
    Calculate fidelity using process tomography
    For dual-rail encoding, ideal state is |01> or |10>
    """
    print(f"\n{'='*60}")
    print(f"Tomography Fidelity - Level {address_width}")
    print(f"Address: {address_bits}, Error rate: {error_rate:.4f}")
    print(f"{'='*60}")
    
    # Create circuit
    qram = FlagBridgeQRAM(address_width=address_width, data_values=None)
    initialize_dual_rail(qram.circuit, qram.reg_addr, address_bits)
    qram.build_circuit()
    
    # Get target leaf qubits
    leaf_layer = qram.tree_layers[-1]
    l0, l1 = leaf_layer.get_logical(expected_leaf)
    
    # Get qubit indices
    qubit_indices = [qram.circuit.find_bit(l0).index, qram.circuit.find_bit(l1).index]
    
    print(f"Circuit: {qram.circuit.num_qubits} qubits, {qram.circuit.depth()} depth")
    print(f"Measuring qubits: {qubit_indices}")
    
    # Create noise model
    noise_model = None
    if error_rate > 0:
        noise_model = NoiseModel()
        error_1q = depolarizing_error(error_rate, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'z', 'reset'])
        error_2q = depolarizing_error(error_rate * 2, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'swap'])
    
    # Measure in X, Y, Z bases
    print(f"\nPerforming tomography measurements...")
    
    exp_z = measure_pauli_expectation(qram.circuit, qubit_indices, 'Z', noise_model, shots=1000)
    exp_x = measure_pauli_expectation(qram.circuit, qubit_indices, 'X', noise_model, shots=1000)
    exp_y = measure_pauli_expectation(qram.circuit, qubit_indices, 'Y', noise_model, shots=1000)
    
    print(f"  Z-basis: <Z0>={exp_z[0]:.3f}, <Z1>={exp_z[1]:.3f}")
    print(f"  X-basis: <X0>={exp_x[0]:.3f}, <X1>={exp_x[1]:.3f}")
    print(f"  Y-basis: <Y0>={exp_y[0]:.3f}, <Y1>={exp_y[1]:.3f}")
    
    # For dual-rail |01>, ideal expectations:
    # Z0 = -1, Z1 = +1 (or vice versa for |10>)
    # Fidelity estimate from purity
    
    # Calculate state purity from Pauli expectations
    # Purity = Tr(rho^2) = (1 + sum(<P_i>^2)) / 2^n
    pauli_squares = []
    for exp in [exp_z, exp_x, exp_y]:
        pauli_squares.extend([e**2 for e in exp])
    
    purity = np.mean(pauli_squares)
    
    # Fidelity estimate: how close to pure state
    # For dual-rail, check if Z expectations are near ±1
    z_purity = (abs(exp_z[0]) + abs(exp_z[1])) / 2
    
    # Combined fidelity estimate
    fidelity_estimate = z_purity * purity
    
    print(f"\nResults:")
    print(f"  Z-basis purity: {z_purity:.4f}")
    print(f"  Overall purity: {purity:.4f}")
    print(f"  Fidelity estimate: {fidelity_estimate:.4f}")
    print(f"  Infidelity: {1-fidelity_estimate:.4f}")
    
    return {
        'level': address_width,
        'address': address_bits,
        'error_rate': error_rate,
        'fidelity': fidelity_estimate,
        'z_purity': z_purity,
        'purity': purity,
        'expectations': {'Z': exp_z, 'X': exp_x, 'Y': exp_y}
    }

def main():
    print("="*60)
    print("PRACTICAL FIDELITY TESTS")
    print("Using measurement-based process tomography")
    print("="*60)
    
    results = []
    
    # Test Level 1 with different error rates
    print("\n\n### LEVEL 1 FIDELITY TESTS ###")
    error_rates = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    
    for err_rate in error_rates:
        result = calculate_fidelity_from_tomography(1, [0], 0, err_rate)
        results.append(result)
    
    # Test Level 2
    print("\n\n### LEVEL 2 FIDELITY TESTS ###")
    for err_rate in [0.0, 0.001, 0.005, 0.01]:
        result = calculate_fidelity_from_tomography(2, [0, 1], 1, err_rate)
        results.append(result)
    
    # Summary
    print("\n\n" + "="*60)
    print("FIDELITY vs ERROR RATE SUMMARY")
    print("="*60)
    print(f"{'Level':<8} {'Error Rate':<12} {'Fidelity':<12} {'Z-Purity':<12} {'Infidelity':<12}")
    print("-"*60)
    
    for r in results:
        print(f"{r['level']:<8} {r['error_rate']:<12.4f} {r['fidelity']:<12.4f} {r['z_purity']:<12.4f} {1-r['fidelity']:<12.4f}")
    
    # Analysis
    print("\n\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    # Group by level
    level1_results = [r for r in results if r['level'] == 1]
    level2_results = [r for r in results if r['level'] == 2]
    
    print("\nLevel 1 Fidelity Degradation:")
    for r in level1_results:
        if r['error_rate'] > 0:
            degradation = (1 - r['fidelity']) / r['error_rate']
            print(f"  Error rate {r['error_rate']:.4f}: Fidelity {r['fidelity']:.4f}, Degradation rate: {degradation:.2f}")
    
    if level2_results:
        print("\nLevel 2 Fidelity Degradation:")
        for r in level2_results:
            if r['error_rate'] > 0:
                degradation = (1 - r['fidelity']) / r['error_rate']
                print(f"  Error rate {r['error_rate']:.4f}: Fidelity {r['fidelity']:.4f}, Degradation rate: {degradation:.2f}")
    
    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("""
1. Z-PURITY: Measures how well dual-rail encoding is preserved
   - Ideal: 1.0 (perfect |01> or |10> state)
   - Degradation indicates bit-flip or relaxation errors

2. OVERALL PURITY: Measures quantum coherence
   - Ideal: 1.0 (pure state)
   - Degradation indicates decoherence

3. FIDELITY ESTIMATE: Combined measure
   - Product of Z-purity and overall purity
   - Captures both classical and quantum errors

4. SCALING: Fidelity should decrease with:
   - Increasing error rate (linear for small errors)
   - Increasing circuit depth (more gates = more errors)
   - Level 2 has ~2x depth of Level 1
    """)

if __name__ == "__main__":
    main()
