# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import CenterParticleSwarmOptimization
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere

# we will run ParticleSwarmAlgorithm for 5 independent runs
algo = CenterParticleSwarmOptimization(NP=51, C1=1.3, C2=2.0, w=0.86, vMin=-1, vMax=1)
for i in range(5):
	task = StoppingTask(D=10, nFES=10000, benchmark=Sphere())
	best = algo.run(task=task)
	print('%s -> %f' % (best[0], best[1]))
print(algo.getParameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
