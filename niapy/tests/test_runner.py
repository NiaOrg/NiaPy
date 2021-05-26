# encoding=utf8
from unittest import TestCase

import numpy as np
import niapy
from niapy.benchmarks import Benchmark


class MyBenchmark(Benchmark):
    def __init__(self, dimension):
        super().__init__(dimension, -11, 11)

    def _evaluate(self, x):
        return np.sum(x ** 2)


class RunnerTestCase(TestCase):
    def setUp(self):
        self.algorithms = ['DifferentialEvolution', 'GreyWolfOptimizer', 'GeneticAlgorithm']
        self.benchmarks = ['griewank', MyBenchmark(7)]

    def test_runner(self):
        self.assertTrue(niapy.Runner(7, 100, 2, self.algorithms, self.benchmarks).run())

    def test_runner_bad_algorithm_throws(self):
        self.assertRaises(KeyError, lambda: niapy.Runner(4, 10, 3, ['EvolutionStrategy'], self.benchmarks).run())

    def test_runner_bad_benchmark_throws(self):
        self.assertRaises(KeyError, lambda: niapy.Runner(4, 10, 3, ['EvolutionStrategy1p1'], ['TesterMan']).run())

    def test_runner_bad_export_throws(self):
        self.assertRaises(TypeError,
                          lambda: niapy.Runner(4, 10, 3, ['GreyWolfOptimizer'], self.benchmarks).run(export="pandas"))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
