# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy import Runner
from NiaPy.util import Task, TaskConvPrint, TaskConvPlot, OptimizationType, getDictArgs
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from NiaPy.algorithms.basic import DifferentialEvolution, MonkeyKingEvolutionV3
from NiaPy.benchmarks import Griewank, Sphere

#1 Number of function evaluations (nFES) as a stopping criteria
for i in range(10):
	task = Task(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = DifferentialEvolution(task=task, NP=40, CR=0.9, F=0.5)
	best = algo.run()
	print (best)

#1 Number of generations (iterations) as a stopping criteria
for i in range(10):
	task = Task(D=10, nGEN=100, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = DifferentialEvolution(task=task, NP=40, CR=0.5)
	best = algo.run()
	print (best)

#3 Reference value as a stopping criteria
for i in range(10):
	task = Task(D=10, refPoint=[50.0, None], optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = DifferentialEvolution(task=task, NP=40, CR=0.5)
	best = algo.run()
	print (best)

