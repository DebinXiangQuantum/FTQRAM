"""
Test QRAM with superposition addresses (|+> basis)
Tests the quantum query capability
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
from qiskit import transpile
from qiskit_aer import Aer
import numpy as np

def initialize_dual_rail_superposition(circuit, dual_reg, superposition_type='plus'):
    """
    Initialize Dual-Rail qubits to |+> or |-> superposition
    |+>_L = (|01> + |10>) / sqrt(2)  - Equal superposition
    |->_L = (|01> - |10>) / sqrt(2)  - Minus superposition
    """
    for i in range(dual_reg.size):
        r0, r1 = dual_reg.get_logical(i)
        
        if superposition_type == 'plus':
            # Create |+>_L = (|01> + |10>) / sqrt(2)
            circuit.h(r0)
            circuit.x(r1)
            circuit.cx(r0, r1)
        elif superposition_type == 'minus':
            # Create |->_L = (|01> - |10>) / sqrt(2)
            circuit.h(r0)
            circuit.x(r1)
            circuit.cx(r0, r1)
            circuit.z(r0)  # Add phase
        elif superposition_type == 'mixed':
            # Alternate between |+> and |->
            if i % 2 == 0:
                circuit.h(r0)
                circuit.x(r1)
                circuit.cx(r0, r1)
            else:
                circuit.h(r0)
                circuit.x(r1)
                circuit.cx(r0, r1)
                circuit.z(r0)

def test_superposition_address(address_width, superposition_type='plus'):
    """Test QRAM with address in superposition"""
    print(f"\n{'='*60}")
    print(f"Testing QRAM Level {address_width} with {superposition_type.upper()} superposition address")
    print(f"Expected: Query all {2**address_width} addresses simultaneously")
    print(f"{'='*60}")
    
    # Create QRAM
    qram = FlagBridgeQRAM(address_width=address_width, data_values=None)
    
    # Initialize address in superposition
    initialize_dual_rail_superposition(qram.circuit, qram.reg_addr, superposition_type)
    
    # Build circuit
    qram.build_circuit()
    
    # Measure all leaf nodes to see superposition distribution
    leaf_layer = qram.tree_layers[-1]
    for i in range(2**address_width):
        l0, l1 = leaf_layer.get_logical(i)
        # Add separate classical registers for each leaf
        from qiskit import ClassicalRegister
        cr_leaf = ClassicalRegister(2, f'leaf_{i}')
        qram.circuit.add_register(cr_leaf)
        qram.circuit.measure(l0, cr_leaf[0])
        qram.circuit.measure(l1, cr_leaf[1])
    
    print(f"Circuit: {qram.circuit.num_qubits} qubits, {qram.circuit.depth()} depth")
    
    # Execute
    sim = Aer.get_backend('qasm_simulator')
    job = sim.run(transpile(qram.circuit, sim), shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    # Analyze distribution across leaves
    leaf_activations = {i: 0 for i in range(2**address_width)}
    total_valid = 0
    total_errors = 0
    
    for bitstring, count in counts.items():
        bits = bitstring.replace(' ', '')
        
        # Extract syndrome (rightmost address_width bits)
        syndrome_start = len(bits) - address_width
        syndrome = bits[syndrome_start:]
        
        has_error = '1' in syndrome
        if has_error:
            total_errors += count
            continue
        
        # Check which leaves are activated
        # Each leaf has 2 bits, starting from left after output
        output_bits = bits[0:2]
        flag_bits = bits[2:2+address_width]
        
        # Parse leaf measurements (after flag and before syndrome)
        leaf_start = 2 + address_width
        leaf_end = syndrome_start
        leaf_bits = bits[leaf_start:leaf_end]
        
        # Each leaf is 2 bits
        for i in range(2**address_width):
            if i*2+1 < len(leaf_bits):
                leaf_state = leaf_bits[i*2:i*2+2]
                if leaf_state in ['01', '10']:
                    leaf_activations[i] += count
                    total_valid += count
    
    print(f"\nResults:")
    print(f"  Total valid shots: {total_valid}")
    print(f"  Total errors: {total_errors}")
    print(f"\nLeaf activation distribution:")
    for i in range(2**address_width):
        percentage = (leaf_activations[i] / 1000) * 100 if total_valid > 0 else 0
        print(f"  Leaf {i} (Address {bin(i)[2:].zfill(address_width)}): {leaf_activations[i]} ({percentage:.1f}%)")
    
    # Check if distribution is uniform (for |+> superposition)
    expected_per_leaf = 1000 / (2**address_width)
    uniformity = np.std([leaf_activations[i] for i in range(2**address_width)])
    print(f"\nUniformity (std dev): {uniformity:.2f} (expected ~{np.sqrt(expected_per_leaf):.2f} for uniform)")
    
    return {
        'level': address_width,
        'type': superposition_type,
        'valid': total_valid,
        'errors': total_errors,
        'uniformity': uniformity,
        'leaf_dist': leaf_activations
    }

def main():
    print("="*60)
    print("DUAL-RAIL QRAM SUPERPOSITION ADDRESS TESTS")
    print("Testing quantum query capability with |+> basis")
    print("="*60)
    
    results = []
    
    # Test different levels with |+> superposition
    print("\n\n### |+> SUPERPOSITION TESTS ###")
    results.append(test_superposition_address(1, 'plus'))
    results.append(test_superposition_address(2, 'plus'))
    
    # Test with |-> superposition
    print("\n\n### |-> SUPERPOSITION TESTS ###")
    results.append(test_superposition_address(1, 'minus'))
    results.append(test_superposition_address(2, 'minus'))
    
    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Level':<8} {'Type':<10} {'Valid':<10} {'Errors':<10} {'Uniformity':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['level']:<8} {r['type']:<10} {r['valid']:<10} {r['errors']:<10} {r['uniformity']:<12.2f}")

if __name__ == "__main__":
    main()
