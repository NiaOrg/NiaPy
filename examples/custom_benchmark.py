# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')

from niapy.task import StoppingTask, OptimizationType
from niapy.benchmarks import Benchmark
from niapy.algorithms.basic import ParticleSwarmAlgorithm


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
    task = StoppingTask(max_iters=1000, dimension=10, optimization_type=OptimizationType.MINIMIZATION,
                        benchmark=MyBenchmark())
    algo = ParticleSwarmAlgorithm(population_size=40, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4)
    best = algo.run(task=task)
    print('%s -> %s ' % (best[0], best[1]))
