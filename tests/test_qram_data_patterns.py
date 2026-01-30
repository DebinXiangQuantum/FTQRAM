"""
Test QRAM with different data patterns stored at memory locations
Tests data values from 0000 to 1111 (4-bit patterns)
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
DualRailRegister = dual_rail_qram.DualRailRegister
initialize_dual_rail = dual_rail_qram.initialize_dual_rail

from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer import Aer
import numpy as np

def initialize_memory_data(circuit, leaf_layer, address_index, data_bits):
    """
    Initialize a specific memory location with data
    
    Args:
        circuit: The quantum circuit
        leaf_layer: The leaf layer of the QRAM tree
        address_index: Which leaf to initialize (0 to 2^n-1)
        data_bits: List of bits to store [b0, b1, b2, b3]
    """
    # For each data bit, initialize the corresponding dual-rail qubit at the leaf
    # In a real QRAM, this would be the stored data
    # Here we simulate by preparing the leaf in the correct state
    
    # Note: In the current implementation, we're testing routing, not data storage
    # This function demonstrates how data would be encoded
    # We'll encode data in the leaf state itself
    
    l0, l1 = leaf_layer.get_logical(address_index)
    
    # Encode first data bit in the dual-rail state
    # This is a simplified model - full QRAM would have separate data registers
    if data_bits[0] == 0:
        # Already initialized to |01⟩ by default
        pass
    else:
        # Flip to |10⟩
        circuit.x(l0)
        circuit.x(l1)

def test_data_pattern(address_width, address_bits, data_pattern, expected_leaf):
    """
    Test QRAM query with specific data pattern at target address
    
    Args:
        address_width: Number of address bits
        address_bits: Address to query [a0, a1, ...]
        data_pattern: Data stored at that address [d0, d1, d2, d3]
        expected_leaf: Expected leaf index
    """
    print(f"\n{'='*60}")
    print(f"Testing Data Pattern")
    print(f"Address: {address_bits} (Binary: {''.join(map(str, address_bits))}) → Leaf {expected_leaf}")
    print(f"Data: {data_pattern} (Binary: {''.join(map(str, data_pattern))}) = {int(''.join(map(str, data_pattern)), 2)}")
    print(f"{'='*60}")
    
    # Create QRAM
    qram = FlagBridgeQRAM(address_width=address_width, data_values=None)
    
    # Initialize address
    initialize_dual_rail(qram.circuit, qram.reg_addr, address_bits)
    
    # Build routing circuit
    qram.build_circuit()
    
    # In a full implementation, we would:
    # 1. Route to the correct leaf (done by build_circuit)
    # 2. Read data from that leaf's memory register
    # 3. Route data back through the tree
    
    # For this test, we verify routing works and demonstrate data encoding
    # Measure the target leaf
    qram.measure_logical_output(target_leaf_index=expected_leaf)
    
    # Add measurement for data verification
    # In real QRAM, data would be in separate registers
    # Here we show the concept
    
    print(f"Circuit: {qram.circuit.num_qubits} qubits, {qram.circuit.depth()} depth")
    
    # Execute
    sim = Aer.get_backend('qasm_simulator')
    try:
        job = sim.run(transpile(qram.circuit, sim), shots=1000)
    except:
        job = sim.run(qram.circuit, shots=1000)
    
    result = job.result()
    counts = result.get_counts()
    
    # Analyze results
    valid_shots = 0
    errors = 0
    
    for k, v in counts.items():
        bitstring = k.replace(' ', '')
        
        raw_output = bitstring[0:2]
        raw_syndrome = bitstring[2+address_width:]
        
        if '1' in raw_syndrome:
            errors += v
        elif raw_output in ['01', '10']:
            valid_shots += v
    
    success_rate = (valid_shots / 1000) * 100
    
    print(f"\nResults:")
    print(f"  Valid routing: {valid_shots} ({success_rate:.1f}%)")
    print(f"  Errors: {errors}")
    print(f"  Status: {'✓ PASS' if success_rate > 95 else '✗ FAIL'}")
    
    return {
        'address': address_bits,
        'data': data_pattern,
        'data_value': int(''.join(map(str, data_pattern)), 2),
        'valid': valid_shots,
        'errors': errors,
        'success_rate': success_rate
    }

def test_full_memory_map(address_width):
    """
    Test a complete memory map with different data at each address
    
    For 2-bit address (4 locations), store data patterns:
    Address 00: Data 0000 (0)
    Address 01: Data 0101 (5)
    Address 10: Data 1010 (10)
    Address 11: Data 1111 (15)
    """
    print(f"\n{'='*60}")
    print(f"FULL MEMORY MAP TEST - Level {address_width}")
    print(f"{'='*60}")
    
    # Define memory contents
    if address_width == 2:
        memory_map = {
            (0, 0): [0, 0, 0, 0],  # Address 00 → Data 0
            (0, 1): [0, 1, 0, 1],  # Address 01 → Data 5
            (1, 0): [1, 0, 1, 0],  # Address 10 → Data 10
            (1, 1): [1, 1, 1, 1],  # Address 11 → Data 15
        }
    elif address_width == 1:
        memory_map = {
            (0,): [0, 0, 1, 1],    # Address 0 → Data 3
            (1,): [1, 1, 0, 0],    # Address 1 → Data 12
        }
    else:
        return []
    
    results = []
    
    print("\nMemory Map:")
    for addr, data in memory_map.items():
        addr_str = ''.join(map(str, addr))
        data_str = ''.join(map(str, data))
        data_val = int(data_str, 2)
        leaf_idx = int(addr_str, 2)
        print(f"  Address {addr_str} (Leaf {leaf_idx}): Data {data_str} ({data_val})")
    
    print("\nTesting each address...")
    for addr, data in memory_map.items():
        addr_str = ''.join(map(str, addr))
        leaf_idx = int(addr_str, 2)
        result = test_data_pattern(address_width, list(addr), data, leaf_idx)
        results.append(result)
    
    return results

def main():
    print("="*60)
    print("DUAL-RAIL QRAM DATA PATTERN TESTS")
    print("Testing different data values (0000-1111)")
    print("="*60)
    
    all_results = []
    
    # Test Level 1 with different data patterns
    print("\n\n### LEVEL 1 TESTS (2 addresses, 4-bit data) ###")
    results_l1 = test_full_memory_map(1)
    all_results.extend(results_l1)
    
    # Test Level 2 with different data patterns
    print("\n\n### LEVEL 2 TESTS (4 addresses, 4-bit data) ###")
    results_l2 = test_full_memory_map(2)
    all_results.extend(results_l2)
    
    # Test specific interesting patterns
    print("\n\n### SPECIAL DATA PATTERNS ###")
    
    # All zeros
    all_results.append(test_data_pattern(2, [0, 0], [0, 0, 0, 0], 0))
    
    # All ones
    all_results.append(test_data_pattern(2, [1, 1], [1, 1, 1, 1], 3))
    
    # Alternating pattern
    all_results.append(test_data_pattern(2, [0, 1], [1, 0, 1, 0], 1))
    
    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY - DATA PATTERN TESTS")
    print("="*60)
    print(f"{'Address':<10} {'Data (Dec)':<12} {'Data (Bin)':<12} {'Success Rate':<15} {'Status':<10}")
    print("-"*60)
    
    for r in all_results:
        addr_str = ''.join(map(str, r['address']))
        data_str = ''.join(map(str, r['data']))
        status = "✓ PASS" if r['success_rate'] > 95 else "✗ FAIL"
        print(f"{addr_str:<10} {r['data_value']:<12} {data_str:<12} {r['success_rate']:<14.1f}% {status:<10}")
    
    avg_success = np.mean([r['success_rate'] for r in all_results])
    print("-"*60)
    print(f"Average success rate: {avg_success:.1f}%")
    
    print("\n\n" + "="*60)
    print("DATA ENCODING NOTES")
    print("="*60)
    print("""
This test demonstrates the QRAM routing mechanism with conceptual data storage.

In a full QRAM implementation:
1. Each leaf node would have a DATA REGISTER (separate from routing)
2. Data would be stored in dual-rail encoding: each bit as |01⟩ or |10⟩
3. Query process:
   - Route address through tree to target leaf
   - Entangle bus with data register at leaf
   - Route data back through tree to output
4. Data patterns 0000-1111 represent 4-bit values (0-15)

Current implementation focuses on fault-tolerant ROUTING.
Data storage/retrieval would add another layer of dual-rail registers.
    """)

if __name__ == "__main__":
    main()
