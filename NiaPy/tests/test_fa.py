# pylint: disable=old-style-class
from unittest import TestCase

from NiaPy.algorithms.basic import FireflyAlgorithm


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


class FATestCase(TestCase):

    def setUp(self):
        self.fa = FireflyAlgorithm(D=10, NP=20, nFES=1000, alpha=0.5, betamin=0.2, gamma=1.0, benchmark=MyBenchmark())

        self.fa_griewank = FireflyAlgorithm(D=10, NP=20, nFES=1000, alpha=0.5, betamin=0.2, gamma=1.0, benchmark='griewank')

    def test_works_fine(self):
        self.assertTrue(self.fa.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.fa_griewank.run())
