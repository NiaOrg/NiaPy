from unittest import TestCase

from NiaPy.algorithms.modified import HybridBatAlgorithm


class HBATestCase(TestCase):
    def setUp(self):
        def Fun(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val

        self.hba_custom = HybridBatAlgorithm(
            10, 40, 1000, 0.5, 0.5, 0.0, 2.0, -2, 2, Fun)

    def test_custom_works_fine(self):
        self.assertTrue(self.hba_custom.move_bat())
