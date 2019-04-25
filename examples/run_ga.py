# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import GeneticAlgorithm
from NiaPy.algorithms.basic.ga import MutationUros, CrossoverUros
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

# we will run Fireworks Algorithm for 5 independent runs
for i in range(5):
	task = StoppingTask(D=10, nFES=4000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = GeneticAlgorithm(NP=100, Crossover=CrossoverUros, Mutation=MutationUros, Cr=0.45, Mr=0.9)
	best = algo.run(task=task)
	print('%s -> %s' % (best[0].x, best[1]))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
