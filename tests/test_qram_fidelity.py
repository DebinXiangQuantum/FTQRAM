"""
Fidelity tests under different error rates
Tests with and without error detection/correction
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the hyphenated module name
import importlib.util
spec = importlib.util.spec_from_file_location("dual_rail_qram", "ftqram/dual-rail-qram.py")
dual_rail_qram = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dual_rail_qram)

FlagBridgeQRAM = dual_rail_qram.FlagBridgeQRAM
initialize_dual_rail = dual_rail_qram.initialize_dual_rail
from qiskit import transpile
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
import numpy as np

def create_noise_model(error_rate, t1=50e3, t2=70e3, gate_time=50):
    """
    Create a noise model with specified error rate
    
    Args:
        error_rate: Depolarizing error probability per gate
        t1: T1 relaxation time (ns)
        t2: T2 dephasing time (ns)
        gate_time: Gate execution time (ns)
    """
    noise_model = NoiseModel()
    
    # Depolarizing error for single-qubit gates
    error_1q = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'z', 'reset'])
    
    # Depolarizing error for two-qubit gates (higher rate)
    error_2q = depolarizing_error(error_rate * 2, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'swap'])
    
    # Thermal relaxation for idle qubits
    t1_error = thermal_relaxation_error(t1, t2, gate_time)
    noise_model.add_all_qubit_quantum_error(t1_error, ['id'])
    
    return noise_model

def test_fidelity_with_noise(address_width, address_bits, expected_leaf, 
                              error_rate, use_correction=True):
    """
    Test QRAM fidelity under noise
    
    Args:
        address_width: Number of address bits
        address_bits: Address to query
        expected_leaf: Expected leaf index
        error_rate: Physical error rate per gate
        use_correction: Whether to use error detection/correction
    """
    print(f"\n{'='*60}")
    print(f"Testing Level {address_width}, Address {address_bits}")
    print(f"Error rate: {error_rate:.4f}, Correction: {use_correction}")
    print(f"{'='*60}")
    
    # Create QRAM
    qram = FlagBridgeQRAM(address_width=address_width, data_values=None)
    
    # Initialize address
    initialize_dual_rail(qram.circuit, qram.reg_addr, address_bits)
    
    # Build circuit
    qram.build_circuit()
    qram.measure_logical_output(target_leaf_index=expected_leaf)
    
    # Create noise model
    noise_model = create_noise_model(error_rate)
    
    # Execute with noise
    sim = Aer.get_backend('qasm_simulator')
    job = sim.run(
        transpile(qram.circuit, sim),
        shots=1000,
        noise_model=noise_model
    )
    result = job.result()
    counts = result.get_counts()
    
    # Analyze results
    valid_shots = 0
    detected_errors = 0
    undetected_errors = 0
    erasures = 0
    corrected_shots = 0
    
    for k, v in counts.items():
        bitstring = k.replace(' ', '')
        
        # Parse bitstring
        raw_output = bitstring[0:2]
        raw_flag = bitstring[2:2+address_width]
        raw_syndrome = bitstring[2+address_width:]
        
        has_syndrome_error = '1' in raw_syndrome
        has_flag = '1' in raw_flag
        
        if use_correction:
            # With correction: flag errors can be corrected
            if has_flag and not has_syndrome_error:
                # Phase error detected and correctable
                corrected_shots += v
                if raw_output in ['01', '10']:
                    valid_shots += v
            elif has_syndrome_error:
                # Conservation violation - detected error
                detected_errors += v
            elif raw_output in ['01', '10']:
                valid_shots += v
            elif raw_output == '00':
                erasures += v
            else:
                undetected_errors += v
        else:
            # Without correction: any error is a failure
            if has_syndrome_error or has_flag:
                detected_errors += v
            elif raw_output in ['01', '10']:
                valid_shots += v
            elif raw_output == '00':
                erasures += v
            else:
                undetected_errors += v
    
    # Calculate fidelity
    fidelity = valid_shots / 1000
    detected_rate = detected_errors / 1000
    undetected_rate = undetected_errors / 1000
    erasure_rate = erasures / 1000
    correction_rate = corrected_shots / 1000
    
    print(f"\nResults:")
    print(f"  Valid outputs: {valid_shots} (Fidelity: {fidelity:.3f})")
    print(f"  Detected errors: {detected_errors} ({detected_rate:.3f})")
    print(f"  Undetected errors: {undetected_errors} ({undetected_rate:.3f})")
    print(f"  Erasures: {erasures} ({erasure_rate:.3f})")
    if use_correction:
        print(f"  Corrected shots: {corrected_shots} ({correction_rate:.3f})")
    
    return {
        'level': address_width,
        'error_rate': error_rate,
        'correction': use_correction,
        'fidelity': fidelity,
        'detected': detected_rate,
        'undetected': undetected_rate,
        'erasure': erasure_rate,
        'corrected': correction_rate if use_correction else 0
    }

def main():
    print("="*60)
    print("DUAL-RAIL QRAM FIDELITY TESTS")
    print("Testing error detection and correction capabilities")
    print("="*60)
    
    results = []
    
    # Test different error rates
    error_rates = [0.0001, 0.001, 0.005, 0.01, 0.02]
    
    print("\n\n### LEVEL 2 QRAM - WITH CORRECTION ###")
    for err_rate in error_rates:
        results.append(test_fidelity_with_noise(
            address_width=2,
            address_bits=[0, 1],
            expected_leaf=1,
            error_rate=err_rate,
            use_correction=True
        ))
    
    print("\n\n### LEVEL 2 QRAM - WITHOUT CORRECTION ###")
    for err_rate in error_rates:
        results.append(test_fidelity_with_noise(
            address_width=2,
            address_bits=[0, 1],
            expected_leaf=1,
            error_rate=err_rate,
            use_correction=False
        ))
    
    print("\n\n### LEVEL 3 QRAM - WITH CORRECTION ###")
    print("Note: Level 3 skipped due to memory constraints")
    # for err_rate in [0.0001, 0.001, 0.005]:
    #     results.append(test_fidelity_with_noise(
    #         address_width=3,
    #         address_bits=[1, 0, 1],
    #         expected_leaf=5,
    #         error_rate=err_rate,
    #         use_correction=True
    #     ))
    
    # Summary and comparison
    print("\n\n" + "="*60)
    print("SUMMARY - FIDELITY vs ERROR RATE")
    print("="*60)
    print(f"{'Level':<7} {'Error Rate':<12} {'Correction':<12} {'Fidelity':<10} {'Detected':<10} {'Undetected':<12}")
    print("-"*60)
    for r in results:
        corr_str = "YES" if r['correction'] else "NO"
        print(f"{r['level']:<7} {r['error_rate']:<12.4f} {corr_str:<12} {r['fidelity']:<10.3f} {r['detected']:<10.3f} {r['undetected']:<12.3f}")
    
    # Plot comparison
    print("\n\n" + "="*60)
    print("CORRECTION BENEFIT ANALYSIS (Level 2)")
    print("="*60)
    print(f"{'Error Rate':<15} {'With Correction':<20} {'Without Correction':<20} {'Improvement':<15}")
    print("-"*60)
    
    with_corr = [r for r in results if r['level'] == 2 and r['correction']]
    without_corr = [r for r in results if r['level'] == 2 and not r['correction']]
    
    for wc, woc in zip(with_corr, without_corr):
        improvement = ((wc['fidelity'] - woc['fidelity']) / woc['fidelity'] * 100) if woc['fidelity'] > 0 else 0
        print(f"{wc['error_rate']:<15.4f} {wc['fidelity']:<20.3f} {woc['fidelity']:<20.3f} {improvement:<15.1f}%")

if __name__ == "__main__":
    main()
