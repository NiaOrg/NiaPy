# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import ParticleSwarmAlgorithm
from NiaPy.util import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere
from numpy import random as rand, apply_along_axis

def customInit(task, NP, rnd=rand):
	pop = task.Lower + rnd.rand(NP, task.D) * 2
	fpop = apply_along_axis(task.eval, 1, pop)
	return pop, fpop


# we will run Particle Swarm Algorithm with custom Init function for 5 independent runs
for i in range(5):
	task = StoppingTask(D=10, nFES=1000, optType=OptimizationType.MINIMIZATION, InitPopFunc=customInit, benchmark=Sphere())
	algo = ParticleSwarmAlgorithm(NP=10, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4)
	best = algo.run(task=task)
	print(best)
