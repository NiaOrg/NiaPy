# encoding=utf8

from unittest import TestCase

from niapy.algorithms.basic import BatAlgorithm, FireflyAlgorithm
from niapy.algorithms.basic import DifferentialEvolution
from niapy.benchmarks import Sphere
from niapy.task import StoppingTask, OptimizationType


class DETestCase(TestCase):
    r"""Test cases for evaluating different stopping conditions.

        **Date:** November 2018

        **Author:** Iztok

        **Author:** This is a very important test!
        """

    def test_DE_evals(self):
        task = StoppingTask(max_evals=1000, dimension=10, optimization_type=OptimizationType.MINIMIZATION,
                            benchmark=Sphere())
        algo = DifferentialEvolution(population_size=40, CR=0.9, F=0.5)
        algo.run_task(task)
        evals = task.evals
        self.assertEqual(1000, evals)

    def test_DE_iters(self):
        task = StoppingTask(max_iters=1000, dimension=10, optimization_type=OptimizationType.MINIMIZATION,
                            benchmark=Sphere())
        algo = DifferentialEvolution(population_size=40, CR=0.9, F=0.5)
        algo.run_task(task)
        iters = task.iters
        self.assertEqual(1000, iters)


class BATestCase(TestCase):
    r"""Test cases for evaluating different stopping conditions.

        **Date:** November 2018

        **Author:** Iztok

        **Author:** This is a very important test!
        """

    def test_BA_evals(self):
        task = StoppingTask(max_evals=1000, dimension=10, optimization_type=OptimizationType.MINIMIZATION,
                            benchmark=Sphere())
        algo = BatAlgorithm(population_size=25)
        algo.run_task(task)
        evals = task.evals
        self.assertEqual(1000, evals)

    def test_BA_iters(self):
        task = StoppingTask(max_iters=1000, dimension=10, optimization_type=OptimizationType.MINIMIZATION,
                            benchmark=Sphere())
        algo = BatAlgorithm(population_size=25)
        algo.run_task(task)
        iters = task.iters
        self.assertEqual(1000, iters)

    # 1000 BA iterations spend 10010 FES (10 + 10 * 1000)
    def test_BA_iters_to_fes(self):
        task = StoppingTask(max_iters=1000, dimension=10, optimization_type=OptimizationType.MINIMIZATION,
                            benchmark=Sphere())
        algo = BatAlgorithm(population_size=10)
        algo.run_task(task)
        evals = task.evals
        self.assertEqual(10010, evals)


class FATestCase(TestCase):

    def test_FA_evals(self):
        task = StoppingTask(max_evals=1000, dimension=10, optimization_type=OptimizationType.MINIMIZATION,
                            benchmark=Sphere())
        algo = FireflyAlgorithm(population_size=25)
        algo.run_task(task)
        evals = task.evals
        self.assertEqual(1000, evals)

    def test_FA_iters(self):
        task = StoppingTask(max_iters=1000, dimension=10, optimization_type=OptimizationType.MINIMIZATION,
                            benchmark=Sphere())
        algo = FireflyAlgorithm(population_size=25)
        algo.run_task(task)
        iters = task.iters
        self.assertEqual(1000, iters)
