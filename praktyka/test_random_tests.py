import math
import unittest

from random_tests import (gammaincc, generate_bits, frequency_test,
                          block_frequency_test, runs_test, serial_test)

class TestRandomTests(unittest.TestCase):
    def test_gamma(self):
        for x in [0.01, 0.5, 2.0, 10.0]:
            self.assertAlmostEqual(gammaincc(1.0, x), math.exp(-x), places=11)

    def test_zeros_fail(self):
        self.assertLess(frequency_test("0" * 100000), 0.01)

    def test_alternating(self):
        data = "01" * 50000
        self.assertGreater(frequency_test(data), 0.01)
        self.assertLess(runs_test(data), 0.01)

    def test_periodic_0011(self):
        data = "0011" * 25000
        self.assertGreater(frequency_test(data), 0.01)
        self.assertGreater(block_frequency_test(data), 0.01)
        self.assertGreater(runs_test(data), 0.01)
        self.assertTrue(all(x >= 0.01 for x in serial_test(data, 2)))

    def test_reproducible(self):
        self.assertEqual(generate_bits("prng", 100000, 42),
                         generate_bits("prng", 100000, 42))

if __name__ == "__main__":
    unittest.main()
