# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import Task
from niapy.problems import Sphere
from niapy.algorithms.basic import MultiPopulationDifferentialEvolution

# we will run Multi-population based Differential Evolution for 5 independent runs
# population_size is the full population (3N); each iteration evolves N = population_size // 3 individuals
algo = MultiPopulationDifferentialEvolution(population_size=90)
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.get_parameters())
