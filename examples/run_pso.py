# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from niapy.algorithms.basic import ParticleSwarmAlgorithm
from niapy.task import StoppingTask
from niapy.benchmarks import Griewank

# we will run ParticleSwarmAlgorithm for 5 independent runs
algo = ParticleSwarmAlgorithm(NP=100, vMin=-4.0, vMax=4.0)
for i in range(5):
	task = StoppingTask(D=10, nFES=10000, benchmark=Griewank(Lower=-600, Upper=600))
	best = algo.run(task=task)
	print('%s -> %f' % (best[0], best[1]))
print(algo.getParameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
