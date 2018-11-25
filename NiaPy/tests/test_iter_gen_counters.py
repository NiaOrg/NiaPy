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
    
    #add scenario description
    def test_DE_evals_fine(self):
        task = Task(D=10, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
        algo = DifferentialEvolution(task=task, NP=40, CR=0.9, F=0.5)   
        best = algo.run()
        evals = algo.task.evals()
        self.assertEqual(evals, 1000)
