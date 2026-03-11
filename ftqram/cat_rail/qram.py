"""Bucket-brigade QRAM model and phase-protected cat-rail realization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Distribution = Dict[str, float]


@dataclass(frozen=True)
class QueryFrame:
    """One checked logical frame in the FT QRAM schedule."""

    stage: str
    depth: int
    prefix: str
    protected_pairs: Tuple[str, ...]
    native_locations: Tuple[str, ...]


def _all_addresses(address_bits: int) -> List[str]:
    return [format(index, f"0{address_bits}b") for index in range(1 << address_bits)]


class BucketBrigadeReferenceQram:
    """Reference bucket-brigade QRAM oracle.

    The logical action is the standard classical-memory QRAM map

    ``U_M |a>|b> = |a>|b XOR m_a>``.

    This class is used as the noiseless reference corresponding to the routing
    schedule encoded in ``bucktele.py``.
    """

    def __init__(self, address_bits: int, data: Sequence[int]) -> None:
        self.address_bits = address_bits
        self.data = [int(bit) for bit in data]
        if len(self.data) != (1 << self.address_bits):
            raise ValueError("data length must equal 2**address_bits")

    @property
    def addresses(self) -> List[str]:
        return _all_addresses(self.address_bits)

    def memory_bit(self, address: str) -> int:
        return self.data[int(address, 2)]

    def query_basis(self, address: str, bus_in: int = 0) -> Tuple[str, int]:
        if len(address) != self.address_bits:
            raise ValueError("address width mismatch")
        return address, bus_in ^ self.memory_bit(address)

    def distribution(
        self,
        amplitudes: Mapping[str, complex] | None = None,
        bus_in: int = 0,
    ) -> Distribution:
        if amplitudes is None:
            weight = 1.0 / float(1 << self.address_bits)
            amplitudes = {address: complex(weight**0.5, 0.0) for address in self.addresses}

        output: Distribution = {}
        for address, amplitude in amplitudes.items():
            _, bus_out = self.query_basis(address, bus_in=bus_in)
            key = f"{address}|{bus_out}"
            output[key] = output.get(key, 0.0) + abs(amplitude) ** 2
        return output


class CatRailPhaseProtectedQram(BucketBrigadeReferenceQram):
    """Phase-protected dual-rail QRAM built on biased cat qubits.

    The logical noiseless map matches ``BucketBrigadeReferenceQram`` exactly.
    The distinction appears in the checked schedule and the noise model.
    """

    def build_query_frames(self, address: str) -> List[QueryFrame]:
        if len(address) != self.address_bits:
            raise ValueError("address width mismatch")

        frames: List[QueryFrame] = []

        for depth in range(self.address_bits):
            prefix = address[:depth]
            frames.append(
                QueryFrame(
                    stage="store",
                    depth=depth,
                    prefix=prefix,
                    protected_pairs=(f"addr[{depth}]", f"router[{prefix}].addr"),
                    native_locations=(f"addr[{depth}]", f"router[{prefix}]"),
                )
            )

        for depth in range(self.address_bits):
            prefix = address[:depth]
            frames.append(
                QueryFrame(
                    stage="route",
                    depth=depth,
                    prefix=prefix,
                    protected_pairs=(f"router[{prefix}].addr", "bus"),
                    native_locations=(f"router[{prefix}]", "bus"),
                )
            )

        frames.append(
            QueryFrame(
                stage="memory",
                depth=self.address_bits,
                prefix=address,
                protected_pairs=("bus",),
                native_locations=("bus",),
            )
        )

        for depth in reversed(range(self.address_bits)):
            prefix = address[:depth]
            frames.append(
                QueryFrame(
                    stage="unroute",
                    depth=depth,
                    prefix=prefix,
                    protected_pairs=(f"router[{prefix}].addr", "bus"),
                    native_locations=(f"router[{prefix}]", "bus"),
                )
            )
        for depth in reversed(range(self.address_bits)):
            prefix = address[:depth]
            frames.append(
                QueryFrame(
                    stage="restore",
                    depth=depth,
                    prefix=prefix,
                    protected_pairs=(f"router[{prefix}].addr", f"addr[{depth}]"),
                    native_locations=(f"router[{prefix}]", f"addr[{depth}]"),
                )
            )
        return frames

    def protected_pair_intervals(self, address: str) -> int:
        return sum(len(frame.protected_pairs) for frame in self.build_query_frames(address))

    def native_sensitive_locations(self, address: str) -> int:
        return sum(len(frame.native_locations) for frame in self.build_query_frames(address))

    def verify_noiseless_equivalence(self) -> bool:
        return self.distribution() == BucketBrigadeReferenceQram(
            self.address_bits,
            self.data,
        ).distribution()


def distributions_close(a: Distribution, b: Distribution, atol: float = 1e-12) -> bool:
    keys = set(a) | set(b)
    return all(abs(a.get(key, 0.0) - b.get(key, 0.0)) <= atol for key in keys)


def build_memory_cases(address_bits: Iterable[int]) -> List[Tuple[str, int, List[int]]]:
    cases: List[Tuple[str, int, List[int]]] = []
    for bits in address_bits:
        size = 1 << bits
        cases.append((f"n={bits}-zeros", bits, [0] * size))
        cases.append((f"n={bits}-ones", bits, [1] * size))
        cases.append((f"n={bits}-alt", bits, [index & 1 for index in range(size)]))
        cases.append(
            (
                f"n={bits}-weight-half",
                bits,
                [1 if index < size // 2 else 0 for index in range(size)],
            )
        )
    return cases
