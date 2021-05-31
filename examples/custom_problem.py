# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')

import numpy as np
from niapy.task import Task
from niapy.problems import Problem
from niapy.algorithms.basic import ParticleSwarmAlgorithm


class MyProblem(Problem):
    def __init__(self, dimension, lower=-10, upper=10, *args, **kwargs):
        super().__init__(dimension, lower, upper, *args, **kwargs)

    def _evaluate(self, x):
        return np.sum(x ** 2)


# we will run Particle Swarm Algorithm on custom problem
task = Task(problem=MyProblem(dimension=10), max_iters=1000)
algo = ParticleSwarmAlgorithm(population_size=40, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4)
best = algo.run(task=task)
print('%s -> %s ' % (best[0], best[1]))
