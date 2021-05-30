# encoding=utf8
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from niapy.algorithms.basic import MonkeyKingEvolutionV1, MonkeyKingEvolutionV2, MonkeyKingEvolutionV3
from niapy.algorithms.basic.mke import MkeSolution
from niapy.task import Task
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class MkeSolutionTestCase(TestCase):
    def setUp(self):
        self.D = 20
        self.x, self.task = default_rng().uniform(-2, 2, self.D), Task(problem=MyProblem(self.D))
        self.sol1, self.sol2, self.sol3 = MkeSolution(x=self.x, e=False), MkeSolution(task=self.task), MkeSolution(
            x=self.x, e=False)

    def test_uPersonalBest(self):
        self.sol2.update_personal_best()
        self.assertTrue(np.array_equal(self.sol2.x, self.sol2.x_pb))
        self.assertEqual(self.sol2.f_pb, self.sol2.f)
        self.sol3.evaluate(self.task)
        self.sol3.x = np.full(self.task.dimension, -5.11)
        self.sol3.evaluate(self.task)
        self.sol3.update_personal_best()
        self.assertTrue(np.array_equal(self.sol3.x, self.sol3.x_pb))
        self.assertEqual(self.sol3.f_pb, self.sol3.f)


class MKEv1TestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MonkeyKingEvolutionV1

    def test_custom(self):
        mke_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        mke_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mke_custom, mke_customc, MyProblem())

    def test_griewank(self):
        mke_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        mke_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mke_griewank, mke_griewankc)


class MKEv2TestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MonkeyKingEvolutionV2

    def test_custom(self):
        mke_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        mke_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mke_custom, mke_customc, MyProblem())

    def test_griewank(self):
        mke_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        mke_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mke_griewank, mke_griewankc)


class MKEv3TestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MonkeyKingEvolutionV3

    def test_custom(self):
        mke_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        mke_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mke_custom, mke_customc, MyProblem())

    def test_griewank(self):
        mke_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        mke_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mke_griewank, mke_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
