"""
Master test runner for all QRAM tests
"""
import subprocess
import sys
import time

def run_test(test_file, description):
    """Run a single test file and report results"""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=False,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed successfully in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} failed after {elapsed:.2f}s")
        return False
    except Exception as e:
        print(f"\n✗ {description} error: {e}")
        return False

def main():
    print("="*70)
    print("DUAL-RAIL FAULT-TOLERANT QRAM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Python: {sys.version}")
    print(f"Starting tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("tests/test_qram_basic.py", "Basic Functionality Tests (Different Levels & Addresses)"),
        ("tests/test_qram_plus_basis.py", "Mixed Basis Tests (0+, 1+, ++, etc.)"),
        ("tests/test_qram_data_patterns.py", "Data Pattern Tests (0000-1111)"),
        ("tests/test_qram_superposition.py", "Full Superposition Tests (|+> Basis)"),
        ("tests/test_qram_fidelity.py", "Fidelity Tests (Error Rates & Correction)"),
    ]
    
    results = []
    total_start = time.time()
    
    for test_file, description in tests:
        success = run_test(test_file, description)
        results.append((description, success))
    
    total_elapsed = time.time() - total_start
    
    # Final summary
    print("\n\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    for description, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {description}")
    
    print("-"*70)
    print(f"Total: {len(results)} tests, {passed} passed, {failed} failed")
    print(f"Total time: {total_elapsed:.2f}s")
    print("="*70)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
