"""
Test QRAM with |+⟩ basis addresses and various data patterns
Tests addresses like 0+, 1+, +0, +1, ++, etc.
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

from qiskit import transpile, ClassicalRegister
from qiskit_aer import Aer
import numpy as np

def initialize_dual_rail_mixed(circuit, dual_reg, basis_list):
    """
    Initialize Dual-Rail qubits with mixed basis states
    
    Args:
        basis_list: List of basis for each qubit
                   '0' = |0⟩_L = |01⟩
                   '1' = |1⟩_L = |10⟩
                   '+' = |+⟩_L = (|01⟩ + |10⟩)/√2
                   '-' = |-⟩_L = (|01⟩ - |10⟩)/√2
    """
    for i, basis in enumerate(basis_list):
        r0, r1 = dual_reg.get_logical(i)
        
        if basis == '0':
            # Logical |0⟩ = |01⟩
            circuit.x(r1)
        elif basis == '1':
            # Logical |1⟩ = |10⟩
            circuit.x(r0)
        elif basis == '+':
            # Logical |+⟩ = (|01⟩ + |10⟩)/√2
            circuit.h(r0)
            circuit.x(r1)
            circuit.cx(r0, r1)
        elif basis == '-':
            # Logical |-⟩ = (|01⟩ - |10⟩)/√2
            circuit.h(r0)
            circuit.x(r1)
            circuit.cx(r0, r1)
            circuit.z(r0)

def test_plus_basis_address(address_width, basis_pattern, description):
    """
    Test QRAM with mixed basis addresses
    
    Args:
        address_width: Number of address bits
        basis_pattern: String like "0+", "1+", "++", etc.
        description: Human-readable description
    """
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Address pattern: {basis_pattern}")
    print(f"{'='*60}")
    
    # Create QRAM
    qram = FlagBridgeQRAM(address_width=address_width, data_values=None)
    
    # Initialize address with mixed basis
    initialize_dual_rail_mixed(qram.circuit, qram.reg_addr, list(basis_pattern))
    
    # Build circuit
    qram.build_circuit()
    
    # Measure all leaf nodes
    leaf_layer = qram.tree_layers[-1]
    for i in range(2**address_width):
        l0, l1 = leaf_layer.get_logical(i)
        cr_leaf = ClassicalRegister(2, f'leaf_{i}')
        qram.circuit.add_register(cr_leaf)
        qram.circuit.measure(l0, cr_leaf[0])
        qram.circuit.measure(l1, cr_leaf[1])
    
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
    leaf_activations = {i: 0 for i in range(2**address_width)}
    total_valid = 0
    total_errors = 0
    
    for bitstring, count in counts.items():
        bits = bitstring.replace(' ', '')
        
        # Extract syndrome
        syndrome_start = len(bits) - address_width
        syndrome = bits[syndrome_start:]
        
        if '1' in syndrome:
            total_errors += count
            continue
        
        # Parse leaf measurements
        leaf_start = 2 + address_width
        leaf_end = syndrome_start
        leaf_bits = bits[leaf_start:leaf_end]
        
        for i in range(2**address_width):
            if i*2+1 < len(leaf_bits):
                leaf_state = leaf_bits[i*2:i*2+2]
                if leaf_state in ['01', '10']:
                    leaf_activations[i] += count
                    total_valid += count
    
    print(f"\nResults:")
    print(f"  Total valid: {total_valid}")
    print(f"  Total errors: {total_errors}")
    
    # Determine expected distribution based on pattern
    plus_count = basis_pattern.count('+')
    expected_leaves = 2**plus_count  # Number of leaves that should be activated
    
    print(f"\nLeaf activation distribution:")
    print(f"  Expected active leaves: {expected_leaves} (due to {plus_count} |+⟩ bits)")
    
    active_leaves = []
    for i in range(2**address_width):
        percentage = (leaf_activations[i] / 1000) * 100
        if leaf_activations[i] > 10:  # Threshold for "active"
            active_leaves.append(i)
            marker = " ✓ ACTIVE"
        else:
            marker = ""
        print(f"  Leaf {i} ({bin(i)[2:].zfill(address_width)}): {leaf_activations[i]:4d} ({percentage:5.1f}%){marker}")
    
    # Check uniformity among active leaves
    if len(active_leaves) > 0:
        active_counts = [leaf_activations[i] for i in active_leaves]
        uniformity = np.std(active_counts)
        expected_per_leaf = total_valid / len(active_leaves) if len(active_leaves) > 0 else 0
        print(f"\nUniformity among active leaves:")
        print(f"  Active leaves: {len(active_leaves)} (expected: {expected_leaves})")
        print(f"  Std deviation: {uniformity:.2f}")
        print(f"  Average per active leaf: {expected_per_leaf:.1f}")
    
    return {
        'pattern': basis_pattern,
        'description': description,
        'valid': total_valid,
        'errors': total_errors,
        'active_leaves': len(active_leaves),
        'expected_leaves': expected_leaves,
        'leaf_dist': leaf_activations
    }

def main():
    print("="*60)
    print("DUAL-RAIL QRAM |+⟩ BASIS ADDRESS TESTS")
    print("Testing mixed computational and superposition basis")
    print("="*60)
    
    results = []
    
    # Level 1 tests
    print("\n\n### LEVEL 1 TESTS (2 addresses) ###")
    results.append(test_plus_basis_address(1, '0', 'Address |0⟩ (computational)'))
    results.append(test_plus_basis_address(1, '1', 'Address |1⟩ (computational)'))
    results.append(test_plus_basis_address(1, '+', 'Address |+⟩ (superposition)'))
    results.append(test_plus_basis_address(1, '-', 'Address |-⟩ (superposition)'))
    
    # Level 2 tests - Mixed basis
    print("\n\n### LEVEL 2 TESTS (4 addresses) - MIXED BASIS ###")
    results.append(test_plus_basis_address(2, '0+', 'Address |0⟩|+⟩ (0 then superposition)'))
    results.append(test_plus_basis_address(2, '1+', 'Address |1⟩|+⟩ (1 then superposition)'))
    results.append(test_plus_basis_address(2, '+0', 'Address |+⟩|0⟩ (superposition then 0)'))
    results.append(test_plus_basis_address(2, '+1', 'Address |+⟩|1⟩ (superposition then 1)'))
    results.append(test_plus_basis_address(2, '++', 'Address |+⟩|+⟩ (full superposition)'))
    
    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Pattern':<10} {'Description':<35} {'Valid':<8} {'Active':<8} {'Expected':<10}")
    print("-"*60)
    for r in results:
        match = "✓" if r['active_leaves'] == r['expected_leaves'] else "✗"
        print(f"{r['pattern']:<10} {r['description']:<35} {r['valid']:<8} {r['active_leaves']:<8} {r['expected_leaves']:<10} {match}")
    
    # Analysis
    print("\n\n" + "="*60)
    print("QUANTUM PARALLELISM ANALYSIS")
    print("="*60)
    print("\nKey observations:")
    print("1. Computational basis (0, 1): Routes to single leaf")
    print("2. Single |+⟩: Routes to 2 leaves simultaneously")
    print("3. Two |+⟩: Routes to 4 leaves simultaneously")
    print("4. Pattern: N |+⟩ bits → 2^N parallel queries")
    
    # Calculate parallelism factor
    for r in results:
        if r['pattern'].count('+') > 0:
            parallelism = r['active_leaves']
            print(f"   {r['pattern']}: {parallelism}x parallelism")

if __name__ == "__main__":
    main()
