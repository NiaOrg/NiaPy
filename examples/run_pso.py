# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import ParticleSwarmAlgorithm
from niapy.task import Task
from niapy.problems import Griewank

# we will run ParticleSwarmAlgorithm for 5 independent runs
algo = ParticleSwarmAlgorithm(population_size=100, min_velocity=-4.0, max_velocity=4.0)
for i in range(5):
    task = Task(problem=Griewank(dimension=10, lower=-600, upper=600), max_evals=10000)
    best = algo.run(task=task)
    print('%s -> %f' % (best[0], best[1]))
print(algo.get_parameters())
