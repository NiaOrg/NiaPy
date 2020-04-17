# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import OppositionVelocityClampingParticleSwarmOptimization
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere

# we will run ParticleSwarmAlgorithm for 5 independent runs
algo = OppositionVelocityClampingParticleSwarmOptimization(NP=50, C1=0.83, C2=1.6, w=0.6, vMin=-1.5, vMax=1.5)
for i in range(5):
	task = StoppingTask(D=10, nFES=1000, benchmark=Sphere())
	best = algo.run(task=task)
	print('%s -> %f' % (best[0], best[1]))
# print(algo.getParameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
