from unittest import TestCase

from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolutionAlgorithm


class MyBenchmark(object):

    def __init__(self):
        self.Lower = -5.12
        self.Upper = 5.12

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate


class jDETestCase(TestCase):
    def setUp(self):
        self.jde_custom = SelfAdaptiveDifferentialEvolutionAlgorithm(
            10, 40, 10000, 0.5, 0.9, 0.1, MyBenchmark())
        self.jde_griewank = SelfAdaptiveDifferentialEvolutionAlgorithm(
            10, 40, 10000, 0.5, 0.9, 0.1, 'griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.jde_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.jde_griewank.run())
