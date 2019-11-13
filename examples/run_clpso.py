# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import ComprehensiveLearningParticleSwarmOptimizer
from NiaPy.benchmarks import Sphere
from NiaPy.task import StoppingTask

# we will run ParticleSwarmAlgorithm for 5 independent runs
algo = ComprehensiveLearningParticleSwarmOptimizer(NP=50, C1=.3, C2=1.0, m=5, w=0.86, vMin=-2, vMax=2)
for i in range(5):
	task = StoppingTask(D=25, nFES=20000, benchmark=Sphere())
	best = algo.run(task=task)
	print('%s -> %f' % (best[0], best[1]))
print(algo.getParameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
