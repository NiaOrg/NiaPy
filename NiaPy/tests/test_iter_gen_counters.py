# pylint: disable=old-style-class, line-too-long

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import BatAlgorithm
from NiaPy.util import Task, TaskConvPrint, TaskConvPlot, TaskConvSave, OptimizationType, getDictArgs
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from NiaPy.algorithms.basic import DifferentialEvolution, MonkeyKingEvolutionV3
from NiaPy.benchmarks import Griewank, Sphere
from unittest import TestCase


class DETestCase(TestCase):
    r"""Test cases for evaluating different stopping conditions.

        **Date:** November 2018

        **Author:** Iztok

        **Author:** This is a very important test!
        """

    # add scenario description
    def test_DE_evals_fine(self):
        task = Task(
            D=10,
            nFES=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = DifferentialEvolution(task=task, NP=40, CR=0.9, F=0.5)
        best = algo.run()
        evals = algo.task.evals()
        self.assertEqual(evals, 1000)

     # add scenario description
    def test_DE_iters_fine(self):
        task = Task(
            D=10,
            nGEN=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = DifferentialEvolution(task=task, NP=40, CR=0.9, F=0.5)
        best = algo.run()
        iters = algo.task.iters()
        self.assertEqual(iters, 1000)


class BATestCase(TestCase):
    r"""Test cases for evaluating different stopping conditions.

        **Date:** November 2018

        **Author:** Iztok

        **Author:** This is a very important test!
        """

    # add scenario description
    def test_BA_evals_fine(self):
        task = Task(
            D=10,
            nFES=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = BatAlgorithm(task=task, NP=25)
        best = algo.run()
        evals = algo.task.evals()
        self.assertEqual(evals, 1000)

    # add scenario description 
    def test_BA_iters_fine(self):
        task = Task(
            D=10,
            nGEN=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = BatAlgorithm(task=task, NP=25)
        best = algo.run()
        iters = algo.task.iters()
        self.assertEqual(iters, 1000)
        

    # 1000 BA iterations spends 10010 FES (10 + 10 * 1000)
    def test_BA_iters_to_fes(self):
        task = Task(
            D=10,
            nGEN=1000,
            optType=OptimizationType.MINIMIZATION,
            benchmark=Sphere())
        algo = BatAlgorithm(task=task, NP=10)
        best = algo.run()
        evals = algo.task.evals()
        self.assertEqual(evals, 10010)        
