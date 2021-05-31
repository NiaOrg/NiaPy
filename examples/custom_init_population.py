# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import ParticleSwarmAlgorithm
from niapy.task import Task
from niapy.problems import Sphere
import numpy as np


def my_init(task, population_size, rng, **_kwargs):
    pop = 0.2 + rng.random((population_size, task.dimension)) * task.range
    fpop = np.apply_along_axis(task.eval, 1, pop)
    return pop, fpop


# we will run Particle Swarm Algorithm with custom Init function for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=1000)
    algo = ParticleSwarmAlgorithm(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, initialization_function=my_init)
    best = algo.run(task=task)
    print(best)
