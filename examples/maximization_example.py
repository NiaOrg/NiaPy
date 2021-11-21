# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import ParticleSwarmAlgorithm
from niapy.task import Task, OptimizationType
from niapy.problems import Sphere

# we will run ParticleSwarmAlgorithm for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000, optimization_type=OptimizationType.MAXIMIZATION)
    algo = ParticleSwarmAlgorithm(population_size=40, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4)
    best = algo.run(task=task)
    print(best[-1])
