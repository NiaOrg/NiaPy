# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.algorithms.basic import DifferentialEvolution 
from NiaPy.benchmarks import Sphere

# Storing improvements during the evolutionary cycle
for i in range(1):
	task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = DifferentialEvolution(NP=40, CR=0.9, F=0.5)
	best = algo.run(task)
	evals, x_f = task.return_conv()
	print(evals)  # print function evaluations
	print(x_f)  # print values
