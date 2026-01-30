
# Technical Specification: Dual-Rail Fault-Tolerant QRAM (Standard Qubit Edition)

**Version:** 1.0 (Standard Qubit)
**Constraint Compliance:** No Magic State Distillation for Error Correction, Fault-Tolerant, Grid Topology.
**Substrate:** Standard Qubits (Transmon/Ion) using Dual-Rail Encoding.

---

## 1. System Overview

### 1.1 Architecture Philosophy

In standard bucket-brigade QRAM, a single error on the router or bus destroys the superposition. This architecture replaces every single logical node with a **Dual-Rail Pair**. This encoding transforms the most dangerous errors—misdirection (routing to the wrong path) and loss—into **illegal parity states** (Violation) that can be detected by Clifford-only checks.

### 1.2 The Conservation Invariant

The system relies on the **Excitation Conservation Principle**.

* **Valid State:** Exactly one rail in a pair is active ( or ).
* **Invalid State:** Both inactive () or Both active ().

**The Core Check:**
For every routing step, we enforce:



Any single fault in the routing hardware will violate this equation, triggering an alarm.

---

## 2. Data Structures (The Logical Qubit)

To implement this in code (e.g., Qiskit, Cirq), we define the fundamental data object: the **Dual-Rail Qubit**.

### 2.1 Mathematical Definition

Each logical qubit  is formed by two physical qubits,  and .

* **Logical Zero ():** Rail 1 is active, Rail 0 is inactive.


* **Logical One ():** Rail 0 is active, Rail 1 is inactive.



**Developer Note:** Initialization always requires preparing the entangled state or the specific product state .

### 2.2 Error Translation Table

This architecture works by translating complex logical errors into simple violations.

| Physical Error | Effect on Dual-Rail  | Logical Result | Detection Mechanism |
| :--- | :--- | :--- | :--- |
| **Bit Flip ()** |  | **Violation** (Both Active) | **Parity Check** () |
| **Relaxation ()** |  | **Erasure** (Loss) | **Parity Check** () |
| **Phase Flip ()** |  | Logical Phase Error | **End-to-End Hashing** |
| **Leakage** |  | Leakage | **Leakage Gadget** |

*Note: Unlike the GKP version, standard qubits cannot suppress Phase Errors physically. You must rely on the **End-to-End Hashing Check** (Bell Pair verification) described in Section 4 to catch  errors.*

---

## 3. The Routing Module (Clifford-Only Logic)

The fundamental building block is the **Dual-Rail Router**.

### 3.1 The Circuit Logic

The routing operation must send the Bus data () to Left () or Right () based on Address ().

**Inputs:**

* Bus: 
* Address: 

**The Algorithm (No T-Gates for Checks):**
Instead of a complex CSWAP, we use **Rail-Controlled Transmission**.

1. **Routing Rail 0 (Right Path):**
* If Address Rail  is , SWAP Bus  into Right Output .
* *Implementation:* Two Toffoli gates (or CNOTs if  is effectively classical during setup).


2. **Routing Rail 1 (Left Path):**
* If Address Rail  is , SWAP Bus  into Left Output .



### 3.2 The Fault-Tolerant "Flag-Bridge" Implementation

Since Toffolis are not fault-tolerant by default (a phase error on control kicks back to target), we wrap the routing in a **Flag-Bridge**.

**Pseudocode for `FT_Router`:**

```python
def ft_router(bus, addr, left, right):
    # 1. Pre-Check: Verify Address is valid (One-Hot)
    if measure_parity(addr) != ODD: raise Fault("Address Invalid")

    # 2. Flagged Routing (Right Branch)
    # We use a helper 'flag' qubit to detect hook errors during the swap
    flag_qubit.reset()
    CNOT(control=addr[0], target=flag_qubit)
    
    # Execute Swap Conditioned on Flag
    Toffoli(flag_qubit, bus[0], right[0])
    Toffoli(flag_qubit, bus[1], right[1])
    
    # Uncompute Flag
    CNOT(control=addr[0], target=flag_qubit)
    
    # Measure Flag to detect Phase Kickback errors
    if measure(flag_qubit) == 1: raise Fault("Router Phase Error")

    # 3. Flagged Routing (Left Branch) - Repeat logic for addr[1] -> left
    # ... (Similar Block) ...

    # 4. Post-Check: Conservation
    # Ensure exactly ONE output path (Left or Right) has data
    if not verify_conservation(left, right):
        raise Fault("Routing Divergence")

```

---

## 4. Protection of the "Flying Bus"

In a standard qubit architecture, the bus is the most vulnerable component. We protect it using **Teleportation Chains**.

### 4.1 The Transport Protocol

We never physically move the data qubit . We **teleport** it between router nodes.

1. **Link Generation:** Create a Bell Pair  between Parent Node and Child Node.
2. **Link Verification (Purification):** Perform a **BBP (Bennett-Brassard-Popescu)** distillation step.
* *Cost:* Requires 2 Bell pairs to get 1 good one.
* *Benefit:* Pure Clifford. No T-gates. Ensures the "wire" is error-free.


3. **Teleportation:** Teleport the Dual-Rail Bus  to the next node.
4. **Arrival Check:** Immediately measure Parity  at the destination.
* If Parity is Even (00 or 11), the teleportation corrupted the qubit. **Abort.**



---

## 5. Control Software Strategy

### 5.1 The Controller Class

The software managing this QRAM must handle the non-deterministic nature of "Flag" alerts.

```python
class DualRail_QRAM_Controller:
    def query(self, address_bits):
        # Reset the Bus to |01> (Logical 0)
        bus = self.initialize_dual_rail()
        current_node = self.root

        for depth in range(TREE_HEIGHT):
            # 1. Transport Phase
            # Attempt teleportation to next router layer
            success = self.teleport_bus(bus, current_node)
            if not success: 
                self.log_error("Teleport Failure")
                return self.retry_query()

            # 2. Routing Phase
            local_addr = self.get_address_dual_rail(address_bits, depth)
            
            try:
                # Execute Flagged Router
                next_dest = self.ft_router(bus, local_addr, 
                                           current_node.left, 
                                           current_node.right)
            except FlagAlert:
                # Flag qubit flipped. We know exactly where the error is.
                # Apply Z-correction to the specific branch and retry check.
                self.apply_correction()
                if not self.verify_integrity(): return self.retry_query()
            
            current_node = next_dest

        # 3. Leaf Interaction
        return self.read_memory(current_node)

```

### 5.2 The Correction Lookup Table

Since we don't use full Quantum Error Correction (Surface Code), we use **Error Detection & Retry**.

| Syndrome | Flag Status | Diagnosis | Action |
| --- | --- | --- | --- |
| **Parity OK** | **Flag 0** | Good State | Continue |
| **Parity Fail** | **Flag 0** | Bit Flip / T1 | **Abort Query** (Restart) |
| **Parity OK** | **Flag 1** | Phase Error | The Flag caught a Z-error. The data is likely fine, but check integrity. |

---

## 6. Formal Math Summary

**1. The Conservation Operator:**
The fundamental observable measuring the "health" of the Dual-Rail state:


* Eigenvalue 0: States  (Invalid).
* Eigenvalue 1: States  (Valid).

**2. Pseudo-Threshold:**
Let  be the physical error rate of a CNOT.
The probability of a Logical Failure (Undetected Error)  is:



Where  is the number of fault pairs that can cancel each other out (typically ).

* If , then .
* This allows QRAM depths of  before a guaranteed failure.

---

## 7. Hardware Requirements

**Topology:**

* **Grid Layout:** Compatible.
* **Connectivity:** Nearest Neighbor.

**Qubit Count:**

* **Per Logical Node:**
* 2 Data Qubits (Dual-Rail Address)
* 2 Data Qubits (Dual-Rail Bus Buffer)
* 1 Flag Ancilla
* 1 Measurement Ancilla
* **Total:** 6 Physical Qubits per Router Node.



**Gate Set:**

* CNOT, H, Measure (Standard Clifford).
* Toffoli (or CCZ) - *Can be decomposed into Clifford+T, but checks are Clifford.*

This specification allows a developer to build the **Dual-Rail QRAM** on standard hardware simulators today, verifying the fault-tolerance properties without needing bosonic modes.


### Decoding and Error Correction
#### 1.1 Hilbert Space and Basis

We define the physical Hilbert space for a single Dual-Rail logical node as .
The basis states are denoted .

The **Logical Subspace**  is defined by the single-excitation constraint:


* Logical Zero: 
* Logical One: 

#### 1.2 Error Space ()

We consider the set of Pauli errors acting on the physical qubits.

* **Bit-Flip / Excitation Error ():**
* :  (Violation: Gain)
* :  (Violation: Gain)


* **Phase Error ():**
* : , but kicks phase if in superposition.
* :  (Logical Phase Flip).


* **Leakage/Erasure:**
*  (Decay):  (Violation: Loss).



#### 1.3 The Syndrome Operators

The decoding relies on measuring two commuting operators:

1. **Conservation Parity ():** Checks if the total excitation number is odd (valid) or even (invalid).


* Eigenvalue  (Even):   **ERROR**
* Eigenvalue  (Odd):   **VALID**


2. **Flag Operator ():** Measured on the ancilla system to detect "Hook Errors" (dangerous fault propagation).

---

### 2. Error Propagation Model (The "Hook")

To define the decoder, we must model how a generic fault  transforms into a syndrome.

Let the syndrome extraction circuit  involve a measurement ancilla  and a flag ancilla . The circuit measures  using CNOT gates.

**Circuit Sequence:**

1. Initialize .
2. 
3. **** (The Flag Bridge)
4. 
5. Measure  in -basis ( Syndrome ).
6. Measure  in -basis ( Flag ).

**The Critical Fault:**
Consider a Phase Error  occurring on the ancilla  between Step 2 and Step 4.

* **Mathematical Propagation:**
Let the state be . The error is .
Using the identity  (Phase Kickback):



* **Result:** The  on the ancilla kicks back to **Data Qubit 1 only** ().
* **Logical Effect:**  applies a phase  to the  state but not the  state. This is a **Logical Phase Flip ()**.
* **Flag Detection:** The  error on  commutes with the control of , but triggers the Flag  if placed correctly, or ensures distinguishability.
* *Correction:* The measurement of  catches this  fault.



---

### 3. The Formal Decoding Map

The decoder is a function  where  is a recovery operation.

Let  be the syndrome result ( Valid,  Invalid).
Let  be the flag result ( Hook Detected).

#### Case 1: The "Hook" Phase Correction

**Input:**  (Valid Parity),  (Flag Triggered).
**Formal Description:**
The flag  indicates a single fault occurred on the ancilla that propagated to the data as a partial phase error (Kickback).

* **Derived Error:**  (Phase error on Rail 1).
* **Correction Operator ():**



(Apply Pauli-Z to Rail 1).
* **Proof of Correction:**



The state is restored to the code space with correct phase.

#### Case 2: The "Conservation" Correction

**Input:**  (Invalid Parity), .
**Formal Description:**
The state is in the subspace .
We perform a **Non-Demolition Inspection Measurement**  (measure Rail 0 only).

* Let outcome be .
* Since  (Parity Even),  fully determines the state:
* If  (State ), then Rail 1 must be . State is .
* If  (State ), then Rail 1 must be . State is .



**Action Map:**

1. **If  (Erasure):**



(Information is information-theoretically lost; cannot unitary correct).
2. **If  (Violation):**



(The superposition path has branched into two; cloning is forbidden, state is corrupted).

*Note: While surface codes can sometimes correct erasures, in the QRAM tree structure without lateral connectivity, an erasure at depth  destroys the path information required for the query. Thus, Abort is the rigorous response.*

---

### 4. Decoding Algorithm Summary

We can condense the formal analysis into the following decision algorithm:

**Rigorous Reliability Statement:**
Let  be the noise channel. The logical channel after decoding is .
For any single physical fault  with weight 1:

1. If ,  outputs ABORT (detected). Logical Fidelity = 0 (but heralded).
2. If ,  applies correction . Logical Fidelity = 1.
3. If , the check commutes. Error passes through (handled by end-to-end checks).

Thus, the **Unheralded Logical Error Rate** is , satisfying the fault tolerance criterion.