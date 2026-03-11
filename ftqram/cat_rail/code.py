"""Phase-protected dual-rail code on biased cat qubits.

The physical layer is the effective two-level cat-qubit model:

- physical phase flips ``Z_C`` are the dominant faults,
- physical bit flips ``X_C`` are exponentially suppressed in ``|alpha|^2``.

The outer code is a dual-rail repetition code in the ``X_C`` basis.  This is
the key modification relative to ``Core.md``: a single physical phase fault is
turned into a detectable syndrome event, while the cat substrate suppresses the
remaining physical bit-flip channel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

SQRT2_INV = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True)
class CatQubitParameters:
    """Effective biased-noise parameters for a Kerr-cat qubit.

    ``phase_flip_rate`` is the dominant physical error probability per checked
    interval.  The residual physical bit-flip probability follows the standard
    biased-cat scaling ``~exp(-2|alpha|^2)``.
    """

    alpha: float = 2.5
    phase_flip_rate: float = 1e-3
    bit_flip_prefactor: float = 1.0

    def bit_flip_rate(self, phase_flip_rate: float | None = None) -> float:
        pz = self.phase_flip_rate if phase_flip_rate is None else phase_flip_rate
        return self.bit_flip_prefactor * pz * math.exp(-2.0 * self.alpha * self.alpha)


def plus_state() -> Tuple[complex, complex]:
    return (SQRT2_INV, SQRT2_INV)


def minus_state() -> Tuple[complex, complex]:
    return (SQRT2_INV, -SQRT2_INV)


def phase_dual_rail_codewords() -> Dict[str, Tuple[complex, complex, complex, complex]]:
    """Return the exact two-qubit codewords in the computational basis.

    Basis order: ``|00>``, ``|01>``, ``|10>``, ``|11>``.
    """

    return {
        "0": (0.5, -0.5, 0.5, -0.5),
        "1": (0.5, 0.5, -0.5, -0.5),
        "e+": (0.5, 0.5, 0.5, 0.5),
        "e-": (0.5, -0.5, -0.5, 0.5),
    }


def valid_syndrome_eigenvalue() -> int:
    """The valid code space satisfies ``X \otimes X = -1``."""

    return -1


def phase_dual_rail_logical_ops() -> Dict[str, str]:
    """Human-readable logical operator representatives on the outer code."""

    return {
        "X_L": "Z0 Z1",
        "Z_L": "X0 = -X1 on the code space",
        "S": "X0 X1",
    }


def classify_cat_rail_faults(
    z0: bool,
    z1: bool,
    x0: bool,
    x1: bool,
) -> Dict[str, bool]:
    """Classify one checked interval of the phase-dual-rail code.

    ``Z`` faults are physical phase flips on the two cat rails and are the
    dominant channel.  ``X`` faults are the exponentially-suppressed residual
    cat-qubit bit flips.
    """

    return {
        "abort": z0 ^ z1,
        "logical_flip": z0 and z1,
        "logical_phase": x0 ^ x1,
    }


def interval_probabilities(
    phase_flip_rate: float,
    cat_params: CatQubitParameters,
) -> Dict[str, float]:
    """Closed-form leading probabilities for one checked pair interval."""

    pz = phase_flip_rate
    px = cat_params.bit_flip_rate(phase_flip_rate)
    return {
        "abort": 2.0 * pz * (1.0 - pz),
        "logical_flip": pz * pz,
        "logical_phase": 2.0 * px * (1.0 - px),
    }
