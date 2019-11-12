# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import GeneticAlgorithm
from NiaPy.algorithms.basic.ga import UniformCrossover, UniformMutation
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere

# we will run Genetic Algorithm for 5 independent runs
for i in range(5):
	task = StoppingTask(D=10, nFES=10000, benchmark=Sphere())
	algo = GeneticAlgorithm(NP=100, Crossover=UniformCrossover, Mutation=UniformMutation, Cr=0.45, Mr=0.9)
	best = algo.run(task=task)
	print('%s -> %s' % (best[0], best[1]))
