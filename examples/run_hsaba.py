# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import numpy as np
import matplotlib.pyplot as plt

from NiaPy.algorithms.modified import HybridSelfAdaptiveBatAlgorithm, SelfAdaptiveBatAlgorithm, AdaptiveBatAlgorithm, HybridBatAlgorithm
from NiaPy.algorithms.modified.jade import CrossRandCurr2Pbest
from NiaPy.algorithms.basic.de import CrossBest2, CrossBest1, CrossRand1
from NiaPy.algorithms.basic import BatAlgorithm, ParticleSwarmAlgorithm
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Griewank, Sphere
from NiaPy.algorithms.statistics import friedmanNemenyi

bmark = Griewank
seed = 4321
algsParams = {
	'seed': 4321,
	'NP': 100,
	'A': 0.5,
	'r': 0.2,
	'F': 0.01,
	'CR': 0.9,
	'Qmin': 0.0,
	'Qmax': 2.0,
}
# we will run Bat Algorithm for 5 independent runs
algs = [BatAlgorithm(**algsParams), AdaptiveBatAlgorithm(**algsParams), SelfAdaptiveBatAlgorithm(**algsParams), HybridBatAlgorithm(**algsParams), HybridSelfAdaptiveBatAlgorithm(CrossMutt=CrossBest1, **algsParams)]
names = ['BA', 'ABA', 'SABA', 'HBA', 'HSABA']
res = [[] for _ in algs]

for i in range(25):
	print('---------------------------', i + 1, '---------------------------')
	tasks = [StoppingTask(D=10, nFES=10000, benchmark=bmark(Upper=600, Lower=-600), logger=False) for _ in algs]
	# task = StoppingTask(D=10, nFES=10000, benchmark=Sphere(Upper=600, Lower=-600), seed=seed)
	for i, a in enumerate(algs):
		best = a.run(tasks[i])
		if a.bad_run(): print(a.exception)
		print('%s => %s' % (a.Name[0], best[1]))
		res[i].append(best[1])

for a in algs: print(a.getParameters())

# friedmanNemenyi test
friedmanNemenyi(np.asarray(res), names)
plt.show()

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
