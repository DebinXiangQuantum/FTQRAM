"""Noise analysis for native and phase-protected cat-rail QRAM."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Sequence

from .code import CatQubitParameters, classify_cat_rail_faults
from .qram import CatRailPhaseProtectedQram


@dataclass(frozen=True)
class IntervalOutcome:
    abort: bool
    logical_flip: bool
    logical_phase: bool


@dataclass(frozen=True)
class TrialResult:
    aborted: bool
    logical_failure: bool


@dataclass(frozen=True)
class NoisePoint:
    phase_error_rate: float
    native_error_rate: float
    cat_rail_error_rate: float
    cat_rail_abort_rate: float
    accepted_trials: int


def _sample_pair_interval(
    phase_error_rate: float,
    cat_params: CatQubitParameters,
    rng: random.Random,
) -> IntervalOutcome:
    bit_flip_rate = cat_params.bit_flip_rate(phase_error_rate)
    z0 = rng.random() < phase_error_rate
    z1 = rng.random() < phase_error_rate
    x0 = rng.random() < bit_flip_rate
    x1 = rng.random() < bit_flip_rate
    classified = classify_cat_rail_faults(z0=z0, z1=z1, x0=x0, x1=x1)
    return IntervalOutcome(
        abort=classified["abort"],
        logical_flip=classified["logical_flip"],
        logical_phase=classified["logical_phase"],
    )


def native_linear_theory(num_locations: int, phase_error_rate: float) -> float:
    return 1.0 - (1.0 - phase_error_rate) ** num_locations


def cat_rail_quadratic_theory(
    num_pair_intervals: int,
    phase_error_rate: float,
    cat_params: CatQubitParameters,
) -> Dict[str, float]:
    abort_per_interval = 2.0 * phase_error_rate * (1.0 - phase_error_rate)
    logical_flip_per_interval = phase_error_rate * phase_error_rate
    logical_phase_per_interval = 2.0 * cat_params.bit_flip_rate(phase_error_rate)
    abort_rate = 1.0 - (1.0 - abort_per_interval) ** num_pair_intervals
    accepted_error = 1.0 - (1.0 - logical_flip_per_interval - logical_phase_per_interval) ** num_pair_intervals
    return {
        "abort_rate": abort_rate,
        "accepted_error_rate": accepted_error,
    }


class CatRailNoiseAnalyzer:
    """Monte-Carlo noise study for the phase-protected cat-rail QRAM."""

    def __init__(self, qram: CatRailPhaseProtectedQram, cat_params: CatQubitParameters | None = None) -> None:
        self.qram = qram
        self.cat_params = cat_params or CatQubitParameters()

    def _run_native_trial(self, address: str, phase_error_rate: float, rng: random.Random) -> TrialResult:
        for frame in self.qram.build_query_frames(address):
            for _location in frame.native_locations:
                if rng.random() < phase_error_rate:
                    return TrialResult(aborted=False, logical_failure=True)
        return TrialResult(aborted=False, logical_failure=False)

    def _run_cat_rail_trial(self, address: str, phase_error_rate: float, rng: random.Random) -> TrialResult:
        original_bits = [int(bit) for bit in address]
        external_bits = list(original_bits)
        router_bits: Dict[str, int] = {}
        bus = 0
        routed_back_wrong = False
        undetected_phase = False

        frames = self.qram.build_query_frames(address)

        for frame in frames:
            if frame.stage == "store":
                depth = frame.depth
                prefix = frame.prefix
                external_outcome = _sample_pair_interval(phase_error_rate, self.cat_params, rng)
                if external_outcome.abort:
                    return TrialResult(aborted=True, logical_failure=False)
                if external_outcome.logical_flip:
                    external_bits[depth] ^= 1
                undetected_phase = undetected_phase or external_outcome.logical_phase

                router_value = external_bits[depth]
                router_outcome = _sample_pair_interval(phase_error_rate, self.cat_params, rng)
                if router_outcome.abort:
                    return TrialResult(aborted=True, logical_failure=False)
                if router_outcome.logical_flip:
                    router_value ^= 1
                undetected_phase = undetected_phase or router_outcome.logical_phase
                router_bits[prefix] = router_value
                continue

            if frame.stage == "route":
                prefix = frame.prefix
                router_outcome = _sample_pair_interval(phase_error_rate, self.cat_params, rng)
                if router_outcome.abort:
                    return TrialResult(aborted=True, logical_failure=False)
                if router_outcome.logical_flip:
                    router_bits[prefix] ^= 1
                undetected_phase = undetected_phase or router_outcome.logical_phase

                bus_outcome = _sample_pair_interval(phase_error_rate, self.cat_params, rng)
                if bus_outcome.abort:
                    return TrialResult(aborted=True, logical_failure=False)
                if bus_outcome.logical_flip:
                    bus ^= 1
                undetected_phase = undetected_phase or bus_outcome.logical_phase
                continue

            if frame.stage == "memory":
                bus_outcome = _sample_pair_interval(phase_error_rate, self.cat_params, rng)
                if bus_outcome.abort:
                    return TrialResult(aborted=True, logical_failure=False)
                if bus_outcome.logical_flip:
                    bus ^= 1
                undetected_phase = undetected_phase or bus_outcome.logical_phase

                effective_address = "".join(str(router_bits[address[:depth]]) for depth in range(self.qram.address_bits))
                bus ^= self.qram.memory_bit(effective_address)
                continue

            if frame.stage == "unroute":
                prefix = frame.prefix
                before = router_bits[prefix]

                router_outcome = _sample_pair_interval(phase_error_rate, self.cat_params, rng)
                if router_outcome.abort:
                    return TrialResult(aborted=True, logical_failure=False)
                if router_outcome.logical_flip:
                    router_bits[prefix] ^= 1
                if router_bits[prefix] != before:
                    routed_back_wrong = True
                undetected_phase = undetected_phase or router_outcome.logical_phase

                bus_outcome = _sample_pair_interval(phase_error_rate, self.cat_params, rng)
                if bus_outcome.abort:
                    return TrialResult(aborted=True, logical_failure=False)
                if bus_outcome.logical_flip:
                    bus ^= 1
                undetected_phase = undetected_phase or bus_outcome.logical_phase
                continue

            if frame.stage == "restore":
                depth = frame.depth
                prefix = frame.prefix

                router_outcome = _sample_pair_interval(phase_error_rate, self.cat_params, rng)
                if router_outcome.abort:
                    return TrialResult(aborted=True, logical_failure=False)
                if router_outcome.logical_flip:
                    router_bits[prefix] ^= 1
                undetected_phase = undetected_phase or router_outcome.logical_phase

                restored_value = router_bits[prefix]
                address_outcome = _sample_pair_interval(phase_error_rate, self.cat_params, rng)
                if address_outcome.abort:
                    return TrialResult(aborted=True, logical_failure=False)
                if address_outcome.logical_flip:
                    restored_value ^= 1
                undetected_phase = undetected_phase or address_outcome.logical_phase
                external_bits[depth] = restored_value
                continue

            raise RuntimeError(f"Unknown frame stage: {frame.stage}")

        ideal_bus = self.qram.memory_bit(address)
        logical_failure = (
            undetected_phase
            or routed_back_wrong
            or external_bits != original_bits
            or bus != ideal_bus
        )
        return TrialResult(aborted=False, logical_failure=logical_failure)

    def sweep(
        self,
        phase_error_rates: Sequence[float],
        trials: int = 4000,
        seed: int = 7,
    ) -> List[NoisePoint]:
        results: List[NoisePoint] = []

        for index, phase_error_rate in enumerate(phase_error_rates):
            rng = random.Random(seed + index)
            native_failures = 0
            cat_failures = 0
            cat_aborts = 0
            accepted = 0

            addresses = self.qram.addresses
            for trial in range(trials):
                address = addresses[trial % len(addresses)]

                native_result = self._run_native_trial(address, phase_error_rate, rng)
                native_failures += int(native_result.logical_failure)

                cat_result = self._run_cat_rail_trial(address, phase_error_rate, rng)
                cat_aborts += int(cat_result.aborted)
                if not cat_result.aborted:
                    accepted += 1
                    cat_failures += int(cat_result.logical_failure)

            results.append(
                NoisePoint(
                    phase_error_rate=phase_error_rate,
                    native_error_rate=native_failures / float(trials),
                    cat_rail_error_rate=cat_failures / float(accepted) if accepted else 0.0,
                    cat_rail_abort_rate=cat_aborts / float(trials),
                    accepted_trials=accepted,
                )
            )

        return results


def fit_log_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    pairs = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0.0 and y > 0.0]
    if len(pairs) < 2:
        return float("nan")
    sum_x = sum(x for x, _ in pairs)
    sum_y = sum(y for _, y in pairs)
    sum_xx = sum(x * x for x, _ in pairs)
    sum_xy = sum(x * y for x, y in pairs)
    n = float(len(pairs))
    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0.0:
        return float("nan")
    return (n * sum_xy - sum_x * sum_y) / denominator
