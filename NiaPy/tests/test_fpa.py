# pylint: disable=old-style-class
from unittest import TestCase

from NiaPy.algorithms.basic import FlowerPollinationAlgorithm


class MyBenchmark:

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


class FPATestCase(TestCase):

    def setUp(self):
        self.fpa_custom = FlowerPollinationAlgorithm(NP=10, D=20, nFES=1000, p=0.5, benchmark=MyBenchmark())
        self.fpa_griewank = FlowerPollinationAlgorithm(D=10, NP=20, nFES=1000, p=0.5, benchmark='griewank')
        self.fpa_beta_griewank = FlowerPollinationAlgorithm(D=10, NP=20, nFES=1000, p=0.5, beta=1.2, benchmark='griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.fpa_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.fpa_griewank.run())

    def test_griewank_works_fine_with_beta(self):
        self.assertTrue(self.fpa_beta_griewank.run())
