# encoding=utf8

from unittest import TestCase
from NiaPy.algorithms.basic import BatAlgorithm, FireflyAlgorithm
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.algorithms.basic import DifferentialEvolution
from NiaPy.benchmarks import Sphere


class DETestCase(TestCase):

    r"""Test cases for evaluating different stopping conditions.

        **Date:** November 2018

        **Author:** Iztok

        **Author:** This is a very important test!
        """

    def test_DE_evals_fine(self):
        task = StoppingTask(
            D=10,
            nFES=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = DifferentialEvolution(NP=40, CR=0.9, F=0.5)
        algo.runTask(task)
        evals = task.evals()
        self.assertEqual(1000, evals)

    def test_DE_iters_fine(self):
        task = StoppingTask(
            D=10,
            nGEN=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = DifferentialEvolution(NP=40, CR=0.9, F=0.5)
        algo.runTask(task)
        iters = task.iters()
        self.assertEqual(1000, iters)


class BATestCase(TestCase):

    r"""Test cases for evaluating different stopping conditions.

        **Date:** November 2018

        **Author:** Iztok

        **Author:** This is a very important test!
        """

    def test_BA_evals_fine(self):
        task = StoppingTask(
            D=10,
            nFES=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = BatAlgorithm(NP=25)
        algo.runTask(task)
        evals = task.evals()
        self.assertEqual(1000, evals)

    def test_BA_iters_fine(self):
        task = StoppingTask(
            D=10,
            nGEN=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = BatAlgorithm(NP=25)
        algo.runTask(task)
        iters = task.iters()
        self.assertEqual(1000, iters)

    # 1000 BA iterations spend 10010 FES (10 + 10 * 1000)
    def test_BA_iters_to_fes(self):
        task = StoppingTask(
            D=10,
            nGEN=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = BatAlgorithm(NP=10)
        algo.runTask(task)
        evals = task.evals()
        self.assertEqual(10000, evals)

class FATestCase(TestCase):

    def test_FA_evals_fine(self):
        task = StoppingTask(
            D=10,
            nFES=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = FireflyAlgorithm(NP=25)
        algo.runTask(task)
        evals = task.evals()
        self.assertEqual(1000, evals)

    def test_FA_iters_fine(self):
        task = StoppingTask(
            D=10,
            nGEN=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = FireflyAlgorithm(NP=25)
        algo.runTask(task)
        iters = task.iters()
        self.assertEqual(1000, iters)
