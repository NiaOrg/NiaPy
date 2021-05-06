# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import GeneticAlgorithm
from niapy.algorithms.basic.ga import uniform_crossover, uniform_mutation
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere

# we will run Genetic Algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(max_evals=10000, dimension=10, benchmark=Sphere())
    algo = GeneticAlgorithm(population_size=100, Crossover=uniform_crossover, Mutation=uniform_mutation, Cr=0.45, Mr=0.9)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))
