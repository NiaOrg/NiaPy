# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy import Runner
from NiaPy.util import Task, TaskConvPrint, TaskConvPlot, TaskConvSave, OptimizationType, getDictArgs
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from NiaPy.algorithms.basic import DifferentialEvolution, MonkeyKingEvolutionV3
from NiaPy.benchmarks import Griewank, Sphere

#1 Number of function evaluations (nFES) as a stopping criteria
for i in range(1):
	task = TaskConvSave(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = DifferentialEvolution(task=task, NP=40, CR=0.9, F=0.5)
	best = algo.run()
	evals, x_f = algo.task.return_conv()
	
	print evals # print evals
	
	print x_f # print values 

