from unittest import TestCase

from NiaPy.algorithms.basic import DifferentialEvolutionAlgorithm


class MyBenchmark(object):

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


class DETestCase(TestCase):
    def setUp(self):
        self.de_custom = DifferentialEvolutionAlgorithm(D=10, NP=40, nFES=1000, F=0.5, CR=0.9, benchmark=MyBenchmark())
        self.de_griewank = DifferentialEvolutionAlgorithm(NP=10, D=40, nFES=1000, CR=0.5, F=0.9, benchmark='griewank')

    def test_Custom_works_fine(self):
        self.assertTrue(self.de_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.de_griewank.run())
