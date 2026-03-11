"""
Logical error rate vs physical error rate analysis for QRAM.

Compares single-rail (bucktele) and dual-rail QRAM under depolarizing noise.
Uses multiprocessing for parallelism.
"""

import gc
import json
import math
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

from bucktele import Qram as BuckQram, RouterQubit
from ftqram.cat_rail import (
    CatQubitParameters,
    CatRailNoiseAnalyzer,
    CatRailPhaseProtectedQram,
)
from ftqram.dual_rail import (
    DualRailBucketQram,
    DualRailQram,
    logical_h,
    prepare_logical_zero,
    split_dual_rail_register,
)

try:
    from ftqram.cat_qubit import make_cat_biased_noise_model
except ImportError:
    make_cat_biased_noise_model = None


# ---------------------------------------------------------------------------
# Helper: build bucktele router tree
# ---------------------------------------------------------------------------

def _build_bucktele(address_bits, data):
    """Build a complete bucktele QRAM circuit in superposition and return
    (circuit, addr_c, bus_c, n_addr)."""
    address_list = [bin(i)[2:].zfill(address_bits) for i in range(2 ** address_bits)]

    addr_q = QuantumRegister(address_bits, "addr")
    bus_q = QuantumRegister(1, "bus")
    addr_c = ClassicalRegister(address_bits, "addr_c")
    bus_c = ClassicalRegister(1, "bus_c")

    qram = BuckQram(address_list, data, addr_q, bus_q, addr_c, bus_c, bandwidth=1)
    circuit = qram.circuit

    counters = {level: 0 for level in range(address_bits)}

    def make_router(level, direction, parent):
        idx = counters[level]
        counters[level] += 1
        node = RouterQubit(idx, level, direction, parent)
        node.left_router = RouterQubit(0, level, "l", node)
        node.right_router = RouterQubit(0, level, "r", node)
        if level < address_bits - 1:
            node.left = make_router(level + 1, "0", node)
            node.right = make_router(level + 1, "1", node)
        else:
            node.left = None
            node.right = None
        return node

    root = make_router(0, "", None)
    incident = RouterQubit(0, 0, "inc", None)
    qram.add_router_tree(0, root)
    qram.add_incident_qubits(incident)

    for i in range(address_bits):
        circuit.h(addr_q[i])

    qram(addr_q, bus_q)
    circuit.measure(bus_q, bus_c)
    circuit.measure(addr_q, addr_c)

    return circuit, addr_c, bus_c, address_bits


def _build_dualrail(address_bits, data):
    """Build a dual-rail QRAM circuit in superposition and return
    (circuit, addr_c, bus_c, n_logical)."""
    address_list = [bin(i)[2:].zfill(address_bits) for i in range(2 ** address_bits)]

    addr_q = QuantumRegister(2 * address_bits, "addr_dr")
    bus_q = QuantumRegister(2, "bus_dr")
    circuit = QuantumCircuit(addr_q, bus_q)

    for pair in split_dual_rail_register(addr_q):
        prepare_logical_zero(circuit, pair)
        logical_h(circuit, pair)

    qram = DualRailBucketQram(address_list, data, bandwidth=1)
    qram(circuit, addr_q, bus_q)

    addr_c = ClassicalRegister(2 * address_bits, "addr_c")
    bus_c = ClassicalRegister(2, "bus_c")
    circuit.add_register(addr_c)
    circuit.add_register(bus_c)
    circuit.measure(addr_q, addr_c)
    circuit.measure(bus_q, bus_c)

    return circuit, addr_c, bus_c, address_bits


def _build_dualrail_ft(address_bits, data):
    """Build a dual-rail FT QRAM circuit with syndrome recording and return
    (circuit, addr_c, bus_c, n_logical, syndrome_reg_name)."""
    addr_q = QuantumRegister(2 * address_bits, "addr_dr")
    bus_q = QuantumRegister(2, "bus_dr")
    circuit = QuantumCircuit(addr_q, bus_q)

    for pair in split_dual_rail_register(addr_q):
        prepare_logical_zero(circuit, pair)
        logical_h(circuit, pair)

    qram = DualRailQram(address_bits, data, record_syndrome=True, fault_tolerant=True)
    qram(circuit, addr_q, bus_q)

    addr_c = ClassicalRegister(2 * address_bits, "addr_c")
    bus_c = ClassicalRegister(2, "bus_c")
    circuit.add_register(addr_c)
    circuit.add_register(bus_c)
    circuit.measure(addr_q, addr_c)
    circuit.measure(bus_q, bus_c)

    return circuit, addr_c, bus_c, address_bits, "syndrome"


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

def _tensor_channel_1q(err_1q, num_qubits: int):
    """Build an n-qubit product channel from a 1-qubit channel."""
    err = err_1q
    for _ in range(num_qubits - 1):
        err = err.tensor(err_1q)
    return err


def make_noise_model(p_phys: float, noise_kind: str = "depolarizing", cat_bias: float = 1e4) -> NoiseModel:
    """Create a gate noise model.

    Supported kinds:
      - depolarizing: symmetric Pauli noise (default)
      - excitation_only: X-dominated rail-occupation faults
      - phase_only: Z-only dephasing faults
      - cat_biased: effective cat-qubit model with strong phase-flip bias
    """
    noise = NoiseModel()
    if p_phys <= 0:
        return noise
    if noise_kind == "depolarizing":
        err_1q = depolarizing_error(p_phys, 1)
        err_2q = depolarizing_error(p_phys, 2)
        err_3q = depolarizing_error(p_phys, 3)
    elif noise_kind == "excitation_only":
        err_1q = pauli_error([("I", 1.0 - p_phys), ("X", p_phys)])
        err_2q = _tensor_channel_1q(err_1q, 2)
        err_3q = _tensor_channel_1q(err_1q, 3)
    elif noise_kind == "phase_only":
        err_1q = pauli_error([("I", 1.0 - p_phys), ("Z", p_phys)])
        err_2q = _tensor_channel_1q(err_1q, 2)
        err_3q = _tensor_channel_1q(err_1q, 3)
    elif noise_kind == "cat_biased":
        if make_cat_biased_noise_model is None:
            raise RuntimeError(
                "NOISE_KIND=cat_biased requires ftqram.cat_qubit.make_cat_biased_noise_model, "
                "but that module is not available in this repository."
            )
        return make_cat_biased_noise_model(
            p_phys,
            bias=cat_bias,
            one_qubit_gates=["x", "h", "z", "s", "sdg", "sx", "rz", "rx", "ry"],
            two_qubit_gates=["cx", "cz", "swap", "H_L"],
            three_qubit_gates=["cswap"],
        )
    else:
        raise ValueError(f"unsupported NOISE_KIND={noise_kind!r}")

    noise.add_all_qubit_quantum_error(err_1q, ["x", "h", "z", "s", "sdg", "sx", "rz", "rx", "ry"])
    noise.add_all_qubit_quantum_error(err_2q, ["cx", "cz", "swap", "H_L"])
    noise.add_all_qubit_quantum_error(err_3q, ["cswap"])
    return noise


# ---------------------------------------------------------------------------
# Bit extraction helpers (same logic as compare_qram.py)
# ---------------------------------------------------------------------------

def _reg_bit_indices(circuit, reg):
    return [circuit.clbits.index(reg[i]) for i in range(len(reg))]


def _bitstring_to_rev(bitstring):
    return bitstring.replace(" ", "")[::-1]


def _extract_bits(rev, indices):
    return [rev[i] for i in indices]


def _logical_from_dual_rail(r0, r1):
    if r0 == "0" and r1 == "1":
        return "0", True
    if r0 == "1" and r1 == "0":
        return "1", True
    return "X", False


# ---------------------------------------------------------------------------
# Compute logical error rate from measurement counts
# ---------------------------------------------------------------------------

def logical_error_rate_bucktele(
    counts: Dict[str, int],
    circuit: QuantumCircuit,
    addr_c: ClassicalRegister,
    bus_c: ClassicalRegister,
    data: List[int],
    address_bits: int,
) -> float:
    """Fraction of shots where bus != data[tree_addr]."""
    addr_idx = _reg_bit_indices(circuit, addr_c)
    bus_idx = _reg_bit_indices(circuit, bus_c)

    total = 0
    errors = 0
    for bitstring, count in counts.items():
        rev = _bitstring_to_rev(bitstring)
        addr_bits_list = _extract_bits(rev, addr_idx)
        bus_bits_list = _extract_bits(rev, bus_idx)

        # addr_str in MSB-first (Qiskit convention after reversal)
        addr_str = "".join(reversed(addr_bits_list))
        bus_str = "".join(reversed(bus_bits_list))

        # Tree index: routing sends q[0] (LSB) to root (MSB of tree address)
        tree_idx = int(addr_str[::-1], 2)
        expected_bus = str(data[tree_idx])

        total += count
        if bus_str != expected_bus:
            errors += count

    return errors / total if total > 0 else 1.0


def logical_error_rate_dualrail(
    counts: Dict[str, int],
    circuit: QuantumCircuit,
    addr_c: ClassicalRegister,
    bus_c: ClassicalRegister,
    data: List[int],
    address_bits: int,
) -> Tuple[float, float]:
    """Returns (logical_error_rate, invalid_rate)."""
    addr_idx = _reg_bit_indices(circuit, addr_c)
    bus_idx = _reg_bit_indices(circuit, bus_c)

    total = 0
    errors = 0
    invalid = 0

    for bitstring, count in counts.items():
        rev = _bitstring_to_rev(bitstring)
        addr_bits_list = _extract_bits(rev, addr_idx)
        bus_bits_list = _extract_bits(rev, bus_idx)

        # Decode dual-rail address
        addr_logical = []
        valid = True
        for i in range(address_bits):
            r0 = addr_bits_list[2 * i]
            r1 = addr_bits_list[2 * i + 1]
            bit, ok = _logical_from_dual_rail(r0, r1)
            if not ok:
                valid = False
                break
            addr_logical.append(bit)

        # Decode dual-rail bus
        bus_logical = "X"
        if valid:
            r0 = bus_bits_list[0]
            r1 = bus_bits_list[1]
            bus_logical, ok = _logical_from_dual_rail(r0, r1)
            if not ok:
                valid = False

        total += count
        if not valid:
            invalid += count
            continue

        addr_str = "".join(reversed(addr_logical))
        tree_idx = int(addr_str[::-1], 2)
        expected_bus = str(data[tree_idx])

        if bus_logical != expected_bus:
            errors += count

    valid_total = total - invalid
    invalid_rate = invalid / total if total > 0 else 0.0
    error_rate = errors / valid_total if valid_total > 0 else float("nan")
    return error_rate, invalid_rate


def logical_error_rate_dualrail_ft(
    counts: Dict[str, int],
    circuit: QuantumCircuit,
    addr_c: ClassicalRegister,
    bus_c: ClassicalRegister,
    data: List[int],
    address_bits: int,
    syndrome_reg_name: str,
) -> Tuple[float, float, float]:
    """Returns (unheralded_error_rate, heralded_rate, invalid_rate).

    Syndrome post-selection: shots with any syndrome bit = 1 are heralded
    (detected) errors and discarded from the unheralded error rate.
    """
    addr_idx = _reg_bit_indices(circuit, addr_c)
    bus_idx = _reg_bit_indices(circuit, bus_c)
    syndrome_creg = next(r for r in circuit.cregs if r.name == syndrome_reg_name)
    syndrome_idx = _reg_bit_indices(circuit, syndrome_creg)

    total = 0
    heralded = 0
    errors = 0
    invalid = 0

    for bitstring, count in counts.items():
        rev = _bitstring_to_rev(bitstring)

        # Check syndrome: any bit = 1 -> heralded error -> discard
        syn_bits = _extract_bits(rev, syndrome_idx)
        total += count
        if any(b == "1" for b in syn_bits):
            heralded += count
            continue

        # Decode dual-rail address
        addr_bits_list = _extract_bits(rev, addr_idx)
        bus_bits_list = _extract_bits(rev, bus_idx)

        addr_logical = []
        valid = True
        for i in range(address_bits):
            r0 = addr_bits_list[2 * i]
            r1 = addr_bits_list[2 * i + 1]
            bit, ok = _logical_from_dual_rail(r0, r1)
            if not ok:
                valid = False
                break
            addr_logical.append(bit)

        bus_logical = "X"
        if valid:
            r0 = bus_bits_list[0]
            r1 = bus_bits_list[1]
            bus_logical, ok = _logical_from_dual_rail(r0, r1)
            if not ok:
                valid = False

        if not valid:
            invalid += count
            continue

        addr_str = "".join(reversed(addr_logical))
        tree_idx = int(addr_str[::-1], 2)
        expected_bus = str(data[tree_idx])

        if bus_logical != expected_bus:
            errors += count

    accepted = total - heralded - invalid
    heralded_rate = heralded / total if total > 0 else 0.0
    invalid_rate = invalid / total if total > 0 else 0.0
    unheralded_error_rate = errors / accepted if accepted > 0 else float("nan")
    return unheralded_error_rate, heralded_rate, invalid_rate


# ---------------------------------------------------------------------------
# Single simulation task (for multiprocessing)
# ---------------------------------------------------------------------------

@dataclass
class SimTask:
    p_phys: float
    address_bits: int
    data: List[int]
    shots: int
    seed: int
    mode: str  # "bucktele", "dualrail", "dualrail_ft", or "cat_rail"


@dataclass
class SimResult:
    p_phys: float
    address_bits: int
    mode: str
    logical_error_rate: float
    invalid_rate: float
    shots: int
    num_qubits: int
    heralded_rate: float = 0.0
    accepted_trials: int = 0


def _make_backend(p_phys: float, num_qubits: int, sim_method: str = "auto"):
    """Create an AerSimulator backend with optional noise and appropriate method."""
    noise_kind = os.environ.get("NOISE_KIND", "depolarizing").strip().lower()
    cat_bias = float(os.environ.get("CAT_BIAS", "10000"))
    noise = make_noise_model(p_phys, noise_kind=noise_kind, cat_bias=cat_bias) if p_phys > 0 else None

    # Choose simulation method based on circuit size
    if sim_method == "auto":
        method = "matrix_product_state" if num_qubits > 30 else "automatic"
    else:
        method = sim_method

    kwargs = {"method": method}
    if noise is not None:
        kwargs["noise_model"] = noise
    # MPS-specific: set bond dimension limit for large circuits
    if method == "matrix_product_state":
        kwargs["matrix_product_state_max_bond_dimension"] = 256
        kwargs["matrix_product_state_truncation_threshold"] = 1e-8
    return AerSimulator(**kwargs)


def run_single_task(task: SimTask) -> SimResult:
    """Run one simulation task. Designed to be called in a subprocess."""
    sim_method = os.environ.get("SIM_METHOD", "auto")

    if task.mode == "bucktele":
        circuit, addr_c, bus_c, n_addr = _build_bucktele(task.address_bits, task.data)
        backend = _make_backend(task.p_phys, circuit.num_qubits, sim_method)
        try:
            result = backend.run(circuit, shots=task.shots, seed_simulator=task.seed).result()
            counts = result.get_counts(circuit)
        except Exception as e:
            return SimResult(task.p_phys, task.address_bits, task.mode, -1.0, -1.0, task.shots, circuit.num_qubits)

        err = logical_error_rate_bucktele(counts, circuit, addr_c, bus_c, task.data, task.address_bits)
        return SimResult(task.p_phys, task.address_bits, task.mode, err, 0.0, task.shots, circuit.num_qubits)

    elif task.mode == "dualrail":
        circuit, addr_c, bus_c, n_logical = _build_dualrail(task.address_bits, task.data)
        backend = _make_backend(task.p_phys, circuit.num_qubits, sim_method)
        try:
            result = backend.run(circuit, shots=task.shots, seed_simulator=task.seed).result()
            counts = result.get_counts(circuit)
        except Exception as e:
            return SimResult(task.p_phys, task.address_bits, task.mode, -1.0, -1.0, task.shots, circuit.num_qubits)

        err, inv = logical_error_rate_dualrail(counts, circuit, addr_c, bus_c, task.data, task.address_bits)
        return SimResult(task.p_phys, task.address_bits, task.mode, err, inv, task.shots, circuit.num_qubits)

    elif task.mode == "dualrail_ft":
        circuit, addr_c, bus_c, n_logical, syn_name = _build_dualrail_ft(task.address_bits, task.data)
        backend = _make_backend(task.p_phys, circuit.num_qubits, sim_method)
        try:
            result = backend.run(circuit, shots=task.shots, seed_simulator=task.seed).result()
            counts = result.get_counts(circuit)
        except Exception as e:
            return SimResult(task.p_phys, task.address_bits, task.mode, -1.0, -1.0, task.shots, circuit.num_qubits)

        err, herald, inv = logical_error_rate_dualrail_ft(
            counts, circuit, addr_c, bus_c, task.data, task.address_bits, syn_name
        )
        return SimResult(task.p_phys, task.address_bits, task.mode, err, inv, task.shots, circuit.num_qubits, herald)

    elif task.mode == "cat_rail":
        alpha = float(os.environ.get("CAT_ALPHA", "2.8"))
        bit_flip_prefactor = float(os.environ.get("CAT_BIT_FLIP_PREFACTOR", "1.0"))
        qram = CatRailPhaseProtectedQram(task.address_bits, task.data)
        analyzer = CatRailNoiseAnalyzer(
            qram,
            CatQubitParameters(
                alpha=alpha,
                phase_flip_rate=task.p_phys,
                bit_flip_prefactor=bit_flip_prefactor,
            ),
        )
        point = analyzer.sweep([task.p_phys], trials=task.shots, seed=task.seed)[0]
        return SimResult(
            task.p_phys,
            task.address_bits,
            task.mode,
            point.cat_rail_error_rate,
            0.0,
            task.shots,
            0,
            point.cat_rail_abort_rate,
            point.accepted_trials,
        )

    raise ValueError(f"unsupported mode={task.mode!r}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    SHOTS = int(os.environ.get("SHOTS", "10000"))
    ADDRESS_BITS = int(os.environ.get("ADDRESS_BITS", "2"))
    MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "16"))
    SEED_BASE = int(os.environ.get("SEED_BASE", "0"))
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(ROOT_DIR, "results"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Physical error rates to sweep (can be overridden with P_PHYS env var, comma-separated)
    p_phys_env = os.environ.get("P_PHYS", "")
    if p_phys_env:
        p_phys_list = sorted(set(float(x) for x in p_phys_env.split(",") if x.strip()))
    else:
        p_phys_list = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

    # Data patterns to average over
    max_patterns = int(os.environ.get("MAX_PATTERNS", "0"))
    random_patterns = int(os.environ.get("RANDOM_PATTERNS", "4"))
    n = 2 ** ADDRESS_BITS
    data_patterns = [
        [0] * n,
        [1] * n,
        [i % 2 for i in range(n)],
        [(i + 1) % 2 for i in range(n)],
    ]
    # Add some random patterns
    rng = np.random.RandomState(42)
    for _ in range(random_patterns):
        data_patterns.append(rng.randint(0, 2, n).tolist())
    if max_patterns > 0:
        data_patterns = data_patterns[:max_patterns]

    SIM_METHOD = os.environ.get("SIM_METHOD", "auto")
    NOISE_KIND = os.environ.get("NOISE_KIND", "depolarizing").strip().lower()
    CAT_BIAS = float(os.environ.get("CAT_BIAS", "10000"))
    CAT_ALPHA = float(os.environ.get("CAT_ALPHA", "2.8"))
    CAT_BIT_FLIP_PREFACTOR = float(os.environ.get("CAT_BIT_FLIP_PREFACTOR", "1.0"))
    print(f"Configuration:")
    print(f"  ADDRESS_BITS = {ADDRESS_BITS}")
    print(f"  SHOTS = {SHOTS}")
    print(f"  MAX_WORKERS = {MAX_WORKERS}")
    print(f"  SEED_BASE = {SEED_BASE}")
    print(f"  SIM_METHOD = {SIM_METHOD}")
    print(f"  NOISE_KIND = {NOISE_KIND}")
    if NOISE_KIND == "cat_biased":
        print(f"  CAT_BIAS = {CAT_BIAS}")
    print(f"  CAT_ALPHA = {CAT_ALPHA}")
    print(f"  CAT_BIT_FLIP_PREFACTOR = {CAT_BIT_FLIP_PREFACTOR}")
    print(f"  # physical error rates = {len(p_phys_list)}")
    print(f"  # data patterns = {len(data_patterns)}")
    MODES = os.environ.get("MODES", "bucktele,dualrail,dualrail_ft,cat_rail").split(",")
    MODES = [m.strip() for m in MODES if m.strip()]
    print(f"  MODES = {MODES}")
    print(f"  # total tasks = {len(p_phys_list) * len(data_patterns) * len(MODES)}")
    print()

    # Checkpoint file for incremental saving
    run_tag = os.environ.get("RUN_TAG", "")
    suffix_parts = []
    if NOISE_KIND != "depolarizing":
        suffix_parts.append(NOISE_KIND)
    if run_tag:
        suffix_parts.append(run_tag)
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    checkpoint_file = os.path.join(OUTPUT_DIR, f"checkpoint_n{ADDRESS_BITS}{suffix}.json")

    # Load existing checkpoint if available
    completed_p: set = set()
    results: List[SimResult] = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
        for r in ckpt.get("results", []):
            results.append(SimResult(**r))
            completed_p.add(r["p_phys"])
        print(f"Resumed from checkpoint: {len(completed_p)} p_phys values already done")

    # Process each p_phys value, save after each completes
    t0 = time.time()
    seed_counter = 0
    total_tasks = len(p_phys_list) * len(data_patterns) * len(MODES)
    done_count = sum(1 for r in results)

    for p in p_phys_list:
        # Build tasks for this p_phys
        tasks_for_p: List[SimTask] = []
        for di, data in enumerate(data_patterns):
            for mode in MODES:
                tasks_for_p.append(SimTask(
                    p_phys=p,
                    address_bits=ADDRESS_BITS,
                    data=data,
                    shots=SHOTS,
                    seed=1000 + SEED_BASE + seed_counter,
                    mode=mode,
                ))
                seed_counter += 1

        if p in completed_p:
            print(f"  p={p:.4f}: already done (checkpoint), skipping")
            continue

        # Execute tasks for this p_phys
        # Use maxtasksperchild=1 to force a fresh process per task,
        # preventing memory accumulation (AerSimulator can use 8GB+ per FT task).
        batch_results = []
        workers = max(MAX_WORKERS, 1)
        with multiprocessing.Pool(processes=workers, maxtasksperchild=1) as pool:
            batch_results = pool.map(run_single_task, tasks_for_p)

        results.extend(batch_results)
        done_count += len(batch_results)
        elapsed = time.time() - t0
        print(f"  p={p:.4f}: done [{done_count}/{total_tasks}] elapsed={elapsed:.1f}s")

        # Save checkpoint after each p_phys
        with open(checkpoint_file, "w") as f:
            json.dump({"results": [asdict(r) for r in results]}, f)

    elapsed = time.time() - t0
    print(f"All tasks completed in {elapsed:.1f}s\n")

    # Aggregate: average logical error rate per (p_phys, mode)
    agg = defaultdict(list)  # (p_phys, mode) -> list of error rates
    agg_inv = defaultdict(list)
    agg_herald = defaultdict(list)
    agg_accept = defaultdict(list)

    for r in results:
        if r.logical_error_rate < 0:
            continue  # skip failed
        agg[(r.p_phys, r.mode)].append(r.logical_error_rate)
        agg_inv[(r.p_phys, r.mode)].append(r.invalid_rate)
        agg_herald[(r.p_phys, r.mode)].append(r.heralded_rate)
        agg_accept[(r.p_phys, r.mode)].append(r.accepted_trials)

    # Print table
    header = (
        f"{'p_phys':>10s}  {'buck_err':>10s}  {'dual_err':>10s}  {'dual_inv':>10s}"
        f"  {'ft_unherald':>11s}  {'ft_herald':>10s}  {'cat_err':>10s}  {'cat_abort':>10s}"
        f"  {'cat_accept':>10s}"
    )
    print(header)
    print("-" * len(header))

    table_data = []
    for p in p_phys_list:
        buck_errs = agg.get((p, "bucktele"), [])
        dual_errs = agg.get((p, "dualrail"), [])
        dual_invs = agg_inv.get((p, "dualrail"), [])
        ft_errs = agg.get((p, "dualrail_ft"), [])
        ft_heralds = agg_herald.get((p, "dualrail_ft"), [])
        cat_errs = agg.get((p, "cat_rail"), [])
        cat_aborts = agg_herald.get((p, "cat_rail"), [])
        cat_accepts = agg_accept.get((p, "cat_rail"), [])

        buck_mean = np.mean(buck_errs) if buck_errs else float("nan")
        dual_mean = np.mean(dual_errs) if dual_errs else float("nan")
        dual_inv_mean = np.mean(dual_invs) if dual_invs else float("nan")
        ft_mean = np.mean(ft_errs) if ft_errs else float("nan")
        ft_herald_mean = np.mean(ft_heralds) if ft_heralds else float("nan")
        cat_mean = np.mean(cat_errs) if cat_errs else float("nan")
        cat_abort_mean = np.mean(cat_aborts) if cat_aborts else float("nan")
        cat_accept_mean = np.mean(cat_accepts) if cat_accepts else float("nan")

        print(
            f"{p:10.4f}  {buck_mean:10.6f}  {dual_mean:10.6f}  {dual_inv_mean:10.6f}"
            f"  {ft_mean:11.6f}  {ft_herald_mean:10.6f}  {cat_mean:10.6f}  {cat_abort_mean:10.6f}"
            f"  {cat_accept_mean:10.1f}"
        )

        table_data.append({
            "p_phys": p,
            "bucktele_error": float(buck_mean),
            "dualrail_error": float(dual_mean),
            "dualrail_invalid": float(dual_inv_mean),
            "dualrail_ft_unheralded": float(ft_mean),
            "dualrail_ft_heralded": float(ft_herald_mean),
            "cat_rail_error": float(cat_mean),
            "cat_rail_abort": float(cat_abort_mean),
            "cat_rail_accepted_trials": float(cat_accept_mean),
            "buck_std": float(np.std(buck_errs)) if buck_errs else 0.0,
            "dual_std": float(np.std(dual_errs)) if dual_errs else 0.0,
            "ft_std": float(np.std(ft_errs)) if ft_errs else 0.0,
            "cat_std": float(np.std(cat_errs)) if cat_errs else 0.0,
            "cat_abort_std": float(np.std(cat_aborts)) if cat_aborts else 0.0,
        })

    # Log-log slope analysis
    _log_log_slope_analysis(table_data)

    # Save raw results
    results_file = os.path.join(OUTPUT_DIR, f"error_rate_n{ADDRESS_BITS}{suffix}.json")
    with open(results_file, "w") as f:
        json.dump({
            "config": {
                "address_bits": ADDRESS_BITS,
                "shots": SHOTS,
                "modes": MODES,
                "n_data_patterns": len(data_patterns),
                "p_phys_list": p_phys_list,
                "noise_kind": NOISE_KIND,
                "cat_bias": CAT_BIAS,
                "cat_alpha": CAT_ALPHA,
                "cat_bit_flip_prefactor": CAT_BIT_FLIP_PREFACTOR,
            },
            "table": table_data,
            "raw_results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Plot
    _plot_results(table_data, ADDRESS_BITS, OUTPUT_DIR, suffix)

    # Find threshold
    _find_threshold(table_data, ADDRESS_BITS)

    return 0


def _log_log_slope_analysis(table_data):
    """Fit log(p_L) vs log(p) for small p to extract the scaling exponent."""
    print("\n=== Log-log Slope Analysis ===")
    # Use points where p > 0 and error > 0 and p <= 0.01 for the fit
    for label, key in [("bucktele", "bucktele_error"),
                       ("dualrail", "dualrail_error"),
                       ("dualrail_ft", "dualrail_ft_unheralded"),
                       ("cat_rail", "cat_rail_error")]:
        log_p = []
        log_e = []
        for d in table_data:
            p = d["p_phys"]
            e = d.get(key, float("nan"))
            if p > 0 and not np.isnan(e) and e > 0 and p <= 0.02:
                log_p.append(np.log10(p))
                log_e.append(np.log10(e))
        if len(log_p) >= 2:
            coeffs = np.polyfit(log_p, log_e, 1)
            slope, intercept = coeffs
            print(f"  {label:20s}: slope = {slope:.3f} (exponent), intercept = {intercept:.3f}")
            print(f"    => p_L ~ {10**intercept:.2e} * p^{slope:.2f}")
        else:
            print(f"  {label:20s}: insufficient data for fit")


def _plot_results(table_data, address_bits, output_dir, suffix=""):
    """Plot logical error rate vs physical error rate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plot skipped: matplotlib is not installed in the active environment.")
        return

    p_vals = [d["p_phys"] for d in table_data]
    buck_vals = [d["bucktele_error"] for d in table_data]
    dual_vals = [d["dualrail_error"] for d in table_data]
    ft_vals = [d.get("dualrail_ft_unheralded", float("nan")) for d in table_data]
    cat_vals = [d.get("cat_rail_error", float("nan")) for d in table_data]
    cat_abort_vals = [d.get("cat_rail_abort", float("nan")) for d in table_data]
    buck_std = [d["buck_std"] for d in table_data]
    dual_std = [d["dual_std"] for d in table_data]
    ft_std = [d.get("ft_std", 0.0) for d in table_data]
    cat_std = [d.get("cat_std", 0.0) for d in table_data]
    cat_abort_std = [d.get("cat_abort_std", 0.0) for d in table_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Linear scale ---
    ax = axes[0]
    ax.errorbar(p_vals, buck_vals, yerr=buck_std, marker="o", label="Single-rail (bucktele)", capsize=3)
    ax.errorbar(p_vals, dual_vals, yerr=dual_std, marker="s", label="Dual-rail", capsize=3)
    ax.errorbar(p_vals, ft_vals, yerr=ft_std, marker="^", label="Dual-rail FT (unheralded)", capsize=3)
    ax.errorbar(p_vals, cat_vals, yerr=cat_std, marker="D", label="Cat-rail (accepted)", capsize=3)
    ax.errorbar(p_vals, cat_abort_vals, yerr=cat_abort_std, marker="x", linestyle="--", label="Cat-rail abort", capsize=3)
    ax.plot(p_vals, p_vals, "--", color="gray", alpha=0.5, label="p_logical = p_physical")
    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Error Rate")
    ax.set_title(f"QRAM Error Rates (n={address_bits}, linear)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Log-log scale ---
    ax = axes[1]
    # Filter out zeros for log scale
    p_nz = [p for p, b in zip(p_vals, buck_vals) if p > 0 and b > 0]
    b_nz = [b for p, b in zip(p_vals, buck_vals) if p > 0 and b > 0]
    d_nz = [d for p, d in zip(p_vals, dual_vals) if p > 0 and d > 0]
    p_d_nz = [p for p, d in zip(p_vals, dual_vals) if p > 0 and d > 0]
    ft_nz = [e for p, e in zip(p_vals, ft_vals) if p > 0 and e > 0 and not np.isnan(e)]
    p_ft_nz = [p for p, e in zip(p_vals, ft_vals) if p > 0 and e > 0 and not np.isnan(e)]
    cat_nz = [e for p, e in zip(p_vals, cat_vals) if p > 0 and e > 0 and not np.isnan(e)]
    p_cat_nz = [p for p, e in zip(p_vals, cat_vals) if p > 0 and e > 0 and not np.isnan(e)]

    if p_nz and b_nz:
        ax.plot(p_nz, b_nz, "o-", label="Single-rail (bucktele)")
    if p_d_nz and d_nz:
        ax.plot(p_d_nz, d_nz, "s-", label="Dual-rail")
    if p_ft_nz and ft_nz:
        ax.plot(p_ft_nz, ft_nz, "^-", label="Dual-rail FT (unheralded)")
    if p_cat_nz and cat_nz:
        ax.plot(p_cat_nz, cat_nz, "D-", label="Cat-rail (accepted)")
    p_ref = np.logspace(-4, -0.5, 50)
    ax.plot(p_ref, p_ref, "--", color="gray", alpha=0.5, label="p_L = p")
    ax.plot(p_ref, p_ref**2, ":", color="gray", alpha=0.5, label="p_L = p^2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Error Rate")
    ax.set_title(f"QRAM Error Rates (n={address_bits}, log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"error_rate_n{address_bits}{suffix}.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_file}")


def _find_threshold(table_data, address_bits):
    """Estimate the error correction threshold from the data."""
    print(f"\n=== Error Correction Analysis (n={address_bits}) ===")

    dual_points = [
        d for d in table_data
        if d["p_phys"] > 0
        and not np.isnan(d.get("bucktele_error", float("nan")))
        and not np.isnan(d.get("dualrail_error", float("nan")))
    ]

    if dual_points:
        # Check if dual-rail ever beats single-rail
        crossings = []
        for i in range(1, len(table_data)):
            d = table_data[i]
            d_prev = table_data[i - 1]
            if d["p_phys"] == 0:
                continue

            buck = d["bucktele_error"]
            dual = d["dualrail_error"]
            buck_prev = d_prev["bucktele_error"]
            dual_prev = d_prev["dualrail_error"]

            if (
                np.isnan(buck) or np.isnan(dual) or np.isnan(buck_prev) or np.isnan(dual_prev)
                or buck <= 0 or dual <= 0 or buck_prev <= 0 or dual_prev <= 0
            ):
                continue

            # Check for crossing: dual goes from below to above buck (or vice versa)
            diff_now = dual - buck
            diff_prev = dual_prev - buck_prev

            if diff_prev * diff_now < 0:  # sign change
                # Linear interpolation
                p_now = d["p_phys"]
                p_prev = d_prev["p_phys"]
                # diff_prev + (diff_now - diff_prev) * (p_cross - p_prev) / (p_now - p_prev) = 0
                p_cross = p_prev - diff_prev * (p_now - p_prev) / (diff_now - diff_prev)
                crossings.append(p_cross)
                print(f"  Crossing detected between p={p_prev:.4f} and p={p_now:.4f}")
                print(f"  Estimated threshold: p_th ≈ {p_cross:.6f}")

        if not crossings:
            # Check if dual-rail is always better or always worse
            better_count = 0
            worse_count = 0
            for d in dual_points:
                if d["dualrail_error"] < d["bucktele_error"]:
                    better_count += 1
                else:
                    worse_count += 1

            if better_count > worse_count:
                print("  Dual-rail is consistently better than single-rail.")
                print("  Threshold is above the highest tested p_phys.")
            elif worse_count > better_count:
                print("  Dual-rail is consistently worse than single-rail.")
                print("  No error correction advantage observed.")
                print("  (This may be due to the overhead of encoding increasing circuit depth)")
            else:
                print("  Results are inconclusive.")

        print("\n  Physical error rate regions:")
        for d in dual_points:
            advantage = "DUAL-RAIL BETTER" if d["dualrail_error"] < d["bucktele_error"] else "SINGLE-RAIL BETTER"
            print(
                f"    p={d['p_phys']:.4f}: buck={d['bucktele_error']:.6f} "
                f"dual={d['dualrail_error']:.6f} → {advantage}"
            )

    cat_points = [
        d for d in table_data
        if d["p_phys"] > 0
        and not np.isnan(d.get("bucktele_error", float("nan")))
        and not np.isnan(d.get("cat_rail_error", float("nan")))
    ]
    if not cat_points:
        return

    print("\n  Cat-rail accepted logical error vs native bucket-brigade:")
    for d in cat_points:
        advantage = "CAT-RAIL BETTER" if d["cat_rail_error"] < d["bucktele_error"] else "NATIVE BETTER"
        print(
            f"    p={d['p_phys']:.4f}: buck={d['bucktele_error']:.6f} "
            f"cat={d['cat_rail_error']:.6f} abort={d['cat_rail_abort']:.6f} → {advantage}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
