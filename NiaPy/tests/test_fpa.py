from unittest import TestCase

from NiaPy.algorithms.basic import FlowerPollinationAlgorithm


class FPATestCase(TestCase):
    def setUp(self):
        def Fun(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val

        self.fpa_custom = FlowerPollinationAlgorithm(
            10, 20, 10000, 0.5, -2.0, 2.0, Fun)

    def test_custom_works_fine(self):
        self.assertTrue(self.fpa_custom.move_flower())
