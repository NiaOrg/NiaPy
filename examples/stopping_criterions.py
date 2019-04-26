# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.algorithms.basic import DifferentialEvolution
from NiaPy.benchmarks import Griewank, Sphere

# 1 Number of function evaluations (nFES) as a stopping criteria
for i in range(10):
	task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = DifferentialEvolution(NP=40, CR=0.9, F=0.5)
	best = algo.run(task)
	print('%s -> %s' % (best[0].x , best[1]))

print ('---------------------------------------')

# 2 Number of generations (iterations) as a stopping criteria
for i in range(10):
	task = StoppingTask(D=10, nGEN=1000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = DifferentialEvolution(NP=40, CR=0.9, F=0.5)
	best = algo.run(task)
	print('%s -> %s' % (best[0].x , best[1]))

print ('---------------------------------------')

# 3 Reference value as a stopping criteria
for i in range(10):
	task = StoppingTask(D=10, refValue=50.0, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = DifferentialEvolution(NP=40, CR=0.9, F=0.5)
	best = algo.run(task)
	print('%s -> %s' % (best[0].x , best[1]))
