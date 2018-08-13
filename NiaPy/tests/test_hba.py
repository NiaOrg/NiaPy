# pylint: disable=old-style-class
from unittest import TestCase

from NiaPy.algorithms.modified import HybridBatAlgorithm


class MyBenchmark:

    def __init__(self):
        self.Lower = -11
        self.Upper = 11

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate


class HBATestCase(TestCase):
    def setUp(self):
        self.hba_custom = HybridBatAlgorithm(D=10, NP=40, nFES=1000, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, benchmark=MyBenchmark())
        self.hba_griewank = HybridBatAlgorithm(D=10, NP=40, nFES=1000, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, benchmark='griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.hba_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.hba_griewank.run())
