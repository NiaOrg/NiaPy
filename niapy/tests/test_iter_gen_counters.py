# encoding=utf8

from unittest import TestCase

from niapy.algorithms.basic import BatAlgorithm, FireflyAlgorithm
from niapy.algorithms.basic import DifferentialEvolution
from niapy.problems import Sphere
from niapy.task import Task


class DETestCase(TestCase):

    def test_DE_evals(self):
        task = Task(max_evals=1000, problem=Sphere(10))
        algo = DifferentialEvolution(population_size=10, CR=0.9, F=0.5)
        algo.run_task(task)
        evals = task.evals
        self.assertEqual(1000, evals)

    def test_DE_iters(self):
        task = Task(max_iters=1000, problem=Sphere(10))
        algo = DifferentialEvolution(population_size=10, CR=0.9, F=0.5)
        algo.run_task(task)
        iters = task.iters
        self.assertEqual(1000, iters)


class BATestCase(TestCase):

    def test_BA_evals(self):
        task = Task(max_evals=1000, problem=Sphere(10))
        algo = BatAlgorithm(population_size=10)
        algo.run_task(task)
        evals = task.evals
        self.assertEqual(1000, evals)

    def test_BA_iters(self):
        task = Task(max_iters=1000, problem=Sphere(10))
        algo = BatAlgorithm(population_size=10)
        algo.run_task(task)
        iters = task.iters
        self.assertEqual(1000, iters)

    # 1000 BA iterations spend 10010 FES (10 + 10 * 1000)
    def test_BA_iters_to_fes(self):
        task = Task(max_iters=1000, problem=Sphere(10))
        algo = BatAlgorithm(population_size=10)
        algo.run_task(task)
        evals = task.evals
        self.assertEqual(10010, evals)


class FATestCase(TestCase):

    def test_FA_evals(self):
        task = Task(max_evals=1000, problem=Sphere(10))
        algo = FireflyAlgorithm(population_size=10)
        algo.run_task(task)
        evals = task.evals
        self.assertEqual(1000, evals)

    def test_FA_iters(self):
        task = Task(max_iters=1000, problem=Sphere(10))
        algo = FireflyAlgorithm(population_size=10)
        algo.run_task(task)
        iters = task.iters
        self.assertEqual(1000, iters)
