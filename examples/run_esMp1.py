# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import EvolutionStrategyMp1
from NiaPy.util import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

#we will run Differential Evolution for 5 independent runs
for i in range(5):
	task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = EvolutionStrategyMp1()
	best = algo.run(task=task)
	print('%s -> %f' % (best[0].x, best[1]))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
