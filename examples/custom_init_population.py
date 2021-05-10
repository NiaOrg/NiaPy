# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import ParticleSwarmAlgorithm
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere
from numpy import random as rand, apply_along_axis


def MyInit(task, NP, rnd=rand, **kwargs):
    pop = 0.2 + rnd.rand(NP, task.dimension) * task.range
    fpop = apply_along_axis(task.eval, 1, pop)
    return pop, fpop


# we will run Particle Swarm Algorithm with custom Init function for 5 independent runs
for i in range(5):
    task = StoppingTask(max_evals=1000, dimension=10, benchmark=Sphere())
    algo = ParticleSwarmAlgorithm(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, initialization_function=MyInit)
    best = algo.run(task=task)
    print(best)
