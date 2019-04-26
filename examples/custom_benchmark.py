# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import ParticleSwarmAlgorithm
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Benchmark

class MyBenchmark(Benchmark):
	def __init__(self):
		Benchmark.__init__(self, -10, 10)

	def function(self):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val += sol[i] ** 2
			return val
		return evaluate


# we will run Particle Swarm Algorithm with on custom benchmark
for i in range(1):
    task = StoppingTask(D=10, nGEN=1000, optType=OptimizationType.MINIMIZATION, benchmark=MyBenchmark())
    algo = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4)
    best = algo.run(task=task)
    print('%s -> %s ' % (best[0], best[1]))
