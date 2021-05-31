# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import GeneticAlgorithm
from niapy.algorithms.basic.ga import uniform_crossover, uniform_mutation
from niapy.task import Task
from niapy.problems import Sphere

# we will run Genetic Algorithm for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    algo = GeneticAlgorithm(population_size=100, crossover=uniform_crossover, mutation=uniform_mutation, crossover_rate=0.45, mutation_rate=0.9)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))
