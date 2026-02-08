# Dual-Rail FT-QRAM vs Bucket-Brigade QRAM Comparison

Date: 2026-02-08

## Run Configuration
- Command: `uv run python tests/compare_qram.py`
- Defaults in script:
  - `SHOTS=32`
  - `ADDRESS_BITS=2`
  - `RANDOM_CASES=0`
  - `TOL_L1=0.15`
  - `TOL_MAX=0.12`

## Results
```
n=2-zeros: L1=1.0000 max=0.3438 invalid=1.0000 FAIL
n=2-ones:  L1=1.0000 max=0.3750 invalid=1.0000 FAIL
n=2-alt:   L1=1.0000 max=0.3125 invalid=1.0000 FAIL

Summary: 0/3 passed
```

## Notes
- The dual-rail run reported `invalid=1.0000` for all cases, indicating that all measured dual-rail outcomes were outside the valid one-hot rail subspace.
- This suggests a mismatch between the dual-rail preparation/measurement mapping and the bucket-brigade reference, or an issue in the routing/measurement conversion.

## How to Re-run (Larger Batch)
You can override defaults via environment variables:
- `SHOTS=256`
- `ADDRESS_BITS=2,3`
- `RANDOM_CASES=2`

Example:
```
SHOTS=256 ADDRESS_BITS=2,3 RANDOM_CASES=2 uv run python tests/compare_qram.py
```
