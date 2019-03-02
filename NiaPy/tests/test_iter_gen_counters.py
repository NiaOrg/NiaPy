# pylint: disable=line-too-long

from unittest import TestCase
from NiaPy.algorithms.basic import BatAlgorithm, FireflyAlgorithm
from NiaPy.util import StopingTask, OptimizationType
from NiaPy.algorithms.basic import DifferentialEvolution
from NiaPy.benchmarks import Sphere


class DETestCase(TestCase):

    r"""Test cases for evaluating different stopping conditions.

        **Date:** November 2018

        **Author:** Iztok

        **Author:** This is a very important test!
        """

    def test_DE_evals_fine(self):
        task = StopingTask(
            D=10,
            nFES=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = DifferentialEvolution(task=task, NP=40, CR=0.9, F=0.5)
        algo.run()
        evals = task.evals()
        self.assertEqual(1000, evals)

    def test_DE_iters_fine(self):
        task = StopingTask(
            D=10,
            nGEN=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = DifferentialEvolution(task=task, NP=40, CR=0.9, F=0.5)
        algo.run()
        iters = task.iters()
        self.assertEqual(1000, iters)


class BATestCase(TestCase):

    r"""Test cases for evaluating different stopping conditions.

        **Date:** November 2018

        **Author:** Iztok

        **Author:** This is a very important test!
        """

    def test_BA_evals_fine(self):
        task = StopingTask(
            D=10,
            nFES=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = BatAlgorithm(task=task, NP=25)
        algo.run()
        evals = task.evals()
        self.assertEqual(1000, evals)

    def test_BA_iters_fine(self):
        task = StopingTask(
            D=10,
            nGEN=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = BatAlgorithm(task=task, NP=25)
        algo.run()
        iters = task.iters()
        self.assertEqual(1000, iters)

    # 1000 BA iterations spend 10010 FES (10 + 10 * 1000)
    def test_BA_iters_to_fes(self):
        task = StopingTask(
            D=10,
            nGEN=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = BatAlgorithm(task=task, NP=10)
        algo.run()
        evals = task.evals()
        self.assertEqual(10000, evals)

class FATestCase(TestCase):

    def test_FA_evals_fine(self):
        task = StopingTask(
            D=10,
            nFES=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = FireflyAlgorithm(task=task, NP=25)
        algo.run()
        evals = task.evals()
        self.assertEqual(1000, evals)

    def test_FA_iters_fine(self):
        task = StopingTask(
            D=10,
            nGEN=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = FireflyAlgorithm(task=task, NP=25)
        algo.run()
        iters = task.iters()
        self.assertEqual(1000, iters)
