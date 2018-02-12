from unittest import TestCase

from NiaPy.algorithms.basic import DifferentialEvolutionAlgorithm


class DETestCase(TestCase):
    def setUp(self):
        def Fun(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val

        self.de_custom = DifferentialEvolutionAlgorithm(
            10, 40, 10000, 0.5, 0.9, 0.0, 2.0, Fun)

    def test_Custom_works_fine(self):
        self.assertTrue(self.de_custom.run())
