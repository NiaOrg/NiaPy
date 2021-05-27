# encoding=utf8
from unittest import TestCase

import numpy as np
import niapy
from niapy.problems import Problem


class MyProblem(Problem):
    def __init__(self, dimension):
        super().__init__(dimension, -11, 11)

    def _evaluate(self, x):
        return np.sum(x ** 2)


class RunnerTestCase(TestCase):
    def setUp(self):
        self.algorithms = ['DifferentialEvolution', 'GreyWolfOptimizer', 'GeneticAlgorithm']
        self.problems = ['griewank', MyProblem(7)]

    def test_runner(self):
        self.assertTrue(niapy.Runner(7, 100, 2, self.algorithms, self.problems).run())

    def test_runner_bad_algorithm_throws(self):
        self.assertRaises(KeyError, lambda: niapy.Runner(4, 10, 3, ['EvolutionStrategy'], self.problems).run())

    def test_runner_bad_problem_throws(self):
        self.assertRaises(KeyError, lambda: niapy.Runner(4, 10, 3, ['EvolutionStrategy1p1'], ['TesterMan']).run())

    def test_runner_bad_export_throws(self):
        self.assertRaises(TypeError,
                          lambda: niapy.Runner(4, 10, 3, ['GreyWolfOptimizer'], self.problems).run(export="pandas"))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
