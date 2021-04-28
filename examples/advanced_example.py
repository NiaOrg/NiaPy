# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')

from niapy.task import StoppingTask, OptimizationType
from niapy.benchmarks import Benchmark
from niapy.algorithms.basic import GreyWolfOptimizer

# our custom benchmark class
class MyBenchmark(Benchmark):
	def __init__(self):
		Benchmark.__init__(self, -10, 10)

	def function(self):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val += sol[i] ** 2
			return val
		return evaluate


# we will run 10 repetitions of Grey Wolf Optimizer against our custom MyBenchmark benchmark function
for i in range(10):
    task = StoppingTask(D=20, nGEN=100, optType=OptimizationType.MINIMIZATION, benchmark=MyBenchmark())

    # parameter is population size
    algo = GreyWolfOptimizer(NP=20)

    # running algorithm returns best found minimum
    best = algo.run(task)

    # printing best minimum
    print(best[-1])
