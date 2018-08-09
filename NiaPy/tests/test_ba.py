# pylint: disable=old-style-class
from unittest import TestCase
from NiaPy.algorithms.basic import BatAlgorithm


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


class BATestCase(TestCase):
    def setUp(self):
        self.ba_custom = BatAlgorithm(D=10, NP=20, nFES=1000, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, benchmark=MyBenchmark())
        self.ba_griewank = BatAlgorithm(NP=10, D=40, nFES=1000, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, benchmark='griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.ba_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.ba_griewank.run())
