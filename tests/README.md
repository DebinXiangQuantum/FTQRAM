# Dual-Rail QRAM Test Suite

Comprehensive testing suite for the Dual-Rail Fault-Tolerant QRAM implementation.

## Test Files

### 1. `test_qram_basic.py`
Tests basic functionality across different QRAM levels and address combinations.

**Tests:**
- Level 1 (2 addresses): Tests both addresses [0] and [1]
- Level 2 (4 addresses): Tests all combinations [0,0], [0,1], [1,0], [1,1]
- Level 3 (8 addresses): Tests selected addresses including corners

**Metrics:**
- Success rate (valid routing)
- Error detection rate
- Erasure/loss rate

### 2. `test_qram_superposition.py`
Tests quantum query capability with superposition addresses in |+⟩ and |-⟩ basis.

**Tests:**
- |+⟩ superposition: Equal superposition of all addresses
- |-⟩ superposition: Minus superposition
- Mixed superposition: Alternating phases

**Metrics:**
- Leaf activation distribution
- Uniformity of superposition
- Error rates in superposition queries

### 3. `test_qram_fidelity.py`
Tests fidelity under realistic noise conditions with and without error correction.

**Tests:**
- Multiple error rates: 0.0001, 0.001, 0.005, 0.01, 0.02
- With error detection and correction
- Without error correction (baseline)
- Different QRAM levels (2 and 3)

**Noise Model:**
- Depolarizing errors on gates
- Thermal relaxation (T1/T2)
- Realistic gate times

**Metrics:**
- Fidelity (valid output rate)
- Detected error rate
- Undetected error rate
- Erasure rate
- Correction benefit

## Running Tests

### Activate Environment
```bash
.venv\Scripts\activate
```

### Run Individual Tests
```bash
uv run tests/test_qram_basic.py
uv run tests/test_qram_superposition.py
uv run tests/test_qram_fidelity.py
```

### Run All Tests
```bash
uv run tests/run_all_tests.py
```

## Expected Results

### Basic Tests
- Success rate should be >95% for ideal simulator
- Errors should be minimal (<5%)
- All addresses should route correctly

### Superposition Tests
- Uniform distribution across all leaves
- Standard deviation close to √(shots/leaves)
- Demonstrates quantum parallelism

### Fidelity Tests
- Fidelity decreases with error rate
- Error correction improves fidelity significantly
- Detected errors increase with error rate
- Undetected errors remain low (fault-tolerance)

## Key Observations

1. **Dual-Rail Encoding**: Transforms bit-flip and relaxation errors into detectable parity violations
2. **Flag Mechanism**: Catches phase errors during routing
3. **Conservation Check**: Ensures exactly one rail is active (excitation conservation)
4. **Fault-Tolerance**: Single faults are detected, preventing error propagation

## Interpreting Results

### Bitstring Format
Results are in format: `Output(2) Flag(N) Syndrome(N)` where N = address_width

- **Output**: `01` or `10` = valid dual-rail state, `00` = erasure, `11` = violation
- **Flag**: `1` indicates phase error detected during routing
- **Syndrome**: `1` indicates conservation violation (bit-flip or relaxation)

### Valid Shot
- Output: `01` or `10`
- Syndrome: all `0`
- Flag: any (correctable if `1`)

### Detected Error
- Syndrome: contains `1`
- System can abort and retry

### Corrected Shot
- Flag: contains `1`
- Syndrome: all `0`
- Phase error was caught and corrected
