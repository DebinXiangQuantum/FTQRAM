"""
Debug test to understand routing behavior
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

from qiskit import transpile
from qiskit_aer import Aer

def debug_routing(address_width, address_bits, expected_leaf):
    """Debug routing with detailed output"""
    print(f"\n{'='*60}")
    print(f"DEBUG: Address {address_bits} → Expected Leaf {expected_leaf}")
    print(f"{'='*60}")
    
    qram = FlagBridgeQRAM(address_width=address_width, data_values=None)
    
    # Initialize address
    initialize_dual_rail(qram.circuit, qram.reg_addr, address_bits)
    
    print(f"\nAddress initialization:")
    for i, bit in enumerate(address_bits):
        if bit == 0:
            print(f"  Addr[{i}] = |0>_L = |01>: r1 active")
        else:
            print(f"  Addr[{i}] = |1>_L = |10>: r0 active")
    
    # Build circuit
    qram.build_circuit()
    
    # Measure ALL leaves to see where signal goes
    leaf_layer = qram.tree_layers[-1]
    from qiskit import ClassicalRegister
    
    for i in range(2**address_width):
        l0, l1 = leaf_layer.get_logical(i)
        cr = ClassicalRegister(2, f'leaf_{i}')
        qram.circuit.add_register(cr)
        qram.circuit.measure(l0, cr[0])
        qram.circuit.measure(l1, cr[1])
    
    print(f"\nCircuit: {qram.circuit.num_qubits} qubits, {qram.circuit.depth()} depth")
    
    # Execute
    sim = Aer.get_backend('qasm_simulator')
    try:
        job = sim.run(transpile(qram.circuit, sim), shots=100)
    except:
        job = sim.run(qram.circuit, shots=100)
    
    result = job.result()
    counts = result.get_counts()
    
    print(f"\nMeasurement results (100 shots):")
    print(f"Total outcomes: {len(counts)}")
    
    # Parse results
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
        bits = bitstring.replace(' ', '')
        print(f"\n  Outcome (count={count}): {bitstring}")
        
        # Parse syndrome
        syndrome_start = len(bits) - address_width
        syndrome = bits[syndrome_start:]
        print(f"    Syndrome: {syndrome} {'OK' if syndrome == '0'*address_width else 'ERROR'}")
        
        # Parse leaves
        leaf_start = 2 + address_width
        leaf_end = syndrome_start
        leaf_bits = bits[leaf_start:leaf_end]
        
        print(f"    Leaf states:")
        for i in range(2**address_width):
            if i*2+1 < len(leaf_bits):
                leaf_state = leaf_bits[i*2:i*2+2]
                if leaf_state == '01':
                    print(f"      Leaf {i}: |01> (Logical |0>) OK")
                elif leaf_state == '10':
                    print(f"      Leaf {i}: |10> (Logical |1>) OK")
                elif leaf_state == '00':
                    print(f"      Leaf {i}: |00> (Empty)")
                elif leaf_state == '11':
                    print(f"      Leaf {i}: |11> (Violation!)")

def main():
    print("="*60)
    print("ROUTING DEBUG TEST")
    print("="*60)
    
    # Test Level 1
    print("\n\n### LEVEL 1 DEBUG ###")
    debug_routing(1, [0], 0)
    debug_routing(1, [1], 1)
    
    # Test Level 2
    print("\n\n### LEVEL 2 DEBUG ###")
    debug_routing(2, [0, 0], 0)
    debug_routing(2, [0, 1], 1)
    debug_routing(2, [1, 0], 2)
    debug_routing(2, [1, 1], 3)

if __name__ == "__main__":
    main()
