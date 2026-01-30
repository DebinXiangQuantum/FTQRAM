"""
Basic functionality tests for Dual-Rail QRAM
Tests different QRAM levels and address/data combinations
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
import numpy as np

def test_qram_level(address_width, address_bits, expected_leaf):
    """Test QRAM at a specific level with given address"""
    print(f"\n{'='*60}")
    print(f"Testing QRAM Level {address_width} (2^{address_width} = {2**address_width} addresses)")
    print(f"Address bits: {address_bits} (Binary: {''.join(map(str, address_bits))})")
    print(f"Expected leaf index: {expected_leaf}")
    print(f"{'='*60}")
    
    # Create QRAM
    qram = FlagBridgeQRAM(address_width=address_width, data_values=None)
    
    # Initialize address
    initialize_dual_rail(qram.circuit, qram.reg_addr, address_bits)
    
    # Build circuit
    qram.build_circuit()
    qram.measure_logical_output(target_leaf_index=expected_leaf)
    
    print(f"Circuit constructed: {qram.circuit.num_qubits} qubits, {qram.circuit.depth()} depth")
    
    # Execute
    sim = Aer.get_backend('qasm_simulator')
    
    # For large circuits, use statevector simulator or increase coupling map
    try:
        job = sim.run(transpile(qram.circuit, sim), shots=1000)
    except Exception as e:
        # If circuit is too large, try without transpilation
        print(f"  Warning: Transpilation failed ({e}), running without optimization")
        job = sim.run(qram.circuit, shots=1000)
    
    result = job.result()
    counts = result.get_counts()
    
    # Build register bit map (Qiskit little-endian ordering)
    bit_map = {}
    current_idx = 0
    reversed_cregs = list(reversed(qram.circuit.cregs))
    for creg in reversed_cregs:
        bit_map[creg.name] = (current_idx, current_idx + creg.size)
        current_idx += creg.size
    
    # Analyze results
    valid_shots = 0
    errors = 0
    erasures = 0
    
    for bitstring, count in counts.items():
        clean_bits = bitstring.replace(' ', '')
        
        # Extract syndrome
        if 'syndrome' in bit_map:
            start, end = bit_map['syndrome']
            syndrome = clean_bits[start:end]
            has_error = '1' in syndrome
        else:
            has_error = False
        
        # Extract output
        if 'output_dual_rail' in bit_map:
            start, end = bit_map['output_dual_rail']
            raw_output = clean_bits[start:end]
        else:
            raw_output = '00'
        
        if has_error:
            errors += count
        elif raw_output in ['01', '10']:
            valid_shots += count
        else:
            erasures += count
    
    success_rate = (valid_shots / 1000) * 100
    error_rate = (errors / 1000) * 100
    erasure_rate = (erasures / 1000) * 100
    
    print(f"\nResults:")
    print(f"  Valid routed signals: {valid_shots} ({success_rate:.1f}%)")
    print(f"  Detected errors: {errors} ({error_rate:.1f}%)")
    print(f"  Erasures/Loss: {erasures} ({erasure_rate:.1f}%)")
    
    return {
        'level': address_width,
        'address': address_bits,
        'valid': valid_shots,
        'errors': errors,
        'erasures': erasures,
        'success_rate': success_rate
    }

def main():
    print("="*60)
    print("DUAL-RAIL QRAM BASIC FUNCTIONALITY TESTS")
    print("="*60)
    
    results = []
    
    # Test Level 1 (2 addresses)
    print("\n\n### LEVEL 1 TESTS (2 addresses) ###")
    results.append(test_qram_level(1, [0], 0))  # Address 0 -> Leaf 0
    results.append(test_qram_level(1, [1], 1))  # Address 1 -> Leaf 1
    
    # Test Level 2 (4 addresses)
    print("\n\n### LEVEL 2 TESTS (4 addresses) ###")
    results.append(test_qram_level(2, [0, 0], 0))  # Address 00 -> Leaf 0
    results.append(test_qram_level(2, [0, 1], 1))  # Address 01 -> Leaf 1
    results.append(test_qram_level(2, [1, 0], 2))  # Address 10 -> Leaf 2
    results.append(test_qram_level(2, [1, 1], 3))  # Address 11 -> Leaf 3
    
    # Test Level 3 (8 addresses) - Reduced due to memory constraints
    print("\n\n### LEVEL 3 TESTS (8 addresses) - SKIPPED ###")
    print("Note: Level 3 requires 40 qubits and exceeds simulator memory limits")
    print("Level 1 and 2 tests demonstrate the core functionality")
    
    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY OF ALL TESTS")
    print("="*60)
    print(f"{'Level':<8} {'Address':<12} {'Success Rate':<15} {'Errors':<10} {'Erasures':<10}")
    print("-"*60)
    for r in results:
        addr_str = ''.join(map(str, r['address']))
        print(f"{r['level']:<8} {addr_str:<12} {r['success_rate']:<14.1f}% {r['errors']:<10} {r['erasures']:<10}")
    
    avg_success = np.mean([r['success_rate'] for r in results])
    print("-"*60)
    print(f"Average success rate: {avg_success:.1f}%")

if __name__ == "__main__":
    main()
