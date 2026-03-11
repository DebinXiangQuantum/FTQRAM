import math
import unittest

from ftqram.cat_rail import (
    BucketBrigadeReferenceQram,
    CatQubitParameters,
    CatRailPhaseProtectedQram,
    build_memory_cases,
    classify_cat_rail_faults,
    distributions_close,
    interval_probabilities,
    phase_dual_rail_codewords,
)


class CatRailCodeTests(unittest.TestCase):
    def test_codewords_are_normalized(self):
        for vector in phase_dual_rail_codewords().values():
            norm = sum(abs(amplitude) ** 2 for amplitude in vector)
            self.assertAlmostEqual(norm, 1.0)

    def test_single_phase_fault_is_detected(self):
        self.assertEqual(
            classify_cat_rail_faults(z0=True, z1=False, x0=False, x1=False),
            {"abort": True, "logical_flip": False, "logical_phase": False},
        )
        self.assertEqual(
            classify_cat_rail_faults(z0=False, z1=True, x0=False, x1=False),
            {"abort": True, "logical_flip": False, "logical_phase": False},
        )

    def test_double_phase_fault_is_logical_flip(self):
        self.assertEqual(
            classify_cat_rail_faults(z0=True, z1=True, x0=False, x1=False),
            {"abort": False, "logical_flip": True, "logical_phase": False},
        )

    def test_single_cat_bit_flip_is_logical_phase(self):
        self.assertEqual(
            classify_cat_rail_faults(z0=False, z1=False, x0=True, x1=False),
            {"abort": False, "logical_flip": False, "logical_phase": True},
        )
        self.assertEqual(
            classify_cat_rail_faults(z0=False, z1=False, x0=True, x1=True),
            {"abort": False, "logical_flip": False, "logical_phase": False},
        )

    def test_interval_probabilities_are_quadratic_in_phase_channel(self):
        params = CatQubitParameters(alpha=3.0)
        p = interval_probabilities(1e-3, params)
        self.assertAlmostEqual(p["logical_flip"], 1e-6)
        self.assertLess(p["logical_phase"], p["logical_flip"])


class CatRailQramTests(unittest.TestCase):
    def test_noiseless_distribution_matches_bucket_brigade_reference(self):
        for _name, address_bits, data in build_memory_cases([1, 2, 3]):
            reference = BucketBrigadeReferenceQram(address_bits, data)
            protected = CatRailPhaseProtectedQram(address_bits, data)
            self.assertTrue(distributions_close(reference.distribution(), protected.distribution()))

    def test_schedule_counts_are_consistent(self):
        qram = CatRailPhaseProtectedQram(3, [0, 1, 0, 1, 1, 0, 1, 0])
        frames = qram.build_query_frames("101")
        self.assertEqual(len(frames), 4 * qram.address_bits + 1)
        self.assertEqual(qram.native_sensitive_locations("101"), 8 * qram.address_bits + 1)
        self.assertEqual(qram.protected_pair_intervals("101"), 8 * qram.address_bits + 1)


if __name__ == "__main__":
    unittest.main()
