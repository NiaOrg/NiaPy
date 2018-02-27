from unittest import TestCase

import NiaPy


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


class RunnerTestCase(TestCase):
    def setUp(self):
        self.algorithms = ['DifferentialEvolutionAlgorithm',
                           'GreyWolfOptimizer']
        self.benchmarks = ['griewank', MyBenchmark()]

    def test_runner_works_fine(self):
        self.assertTrue(NiaPy.Runner(10, 40, 1000, 3,
                                     self.algorithms, self.benchmarks).run())
