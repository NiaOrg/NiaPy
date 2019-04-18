# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.other import MultipleTrajectorySearch
from NiaPy.util import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

# we will run Nelder Mead algorithm for 5 independent runs
for i in range(5):
	task = StoppingTask(D=10, nGEN=40, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = MultipleTrajectorySearch()
	best = algo.run(task=task)
	print('%s -> %s' % (best[0], best[1]))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
