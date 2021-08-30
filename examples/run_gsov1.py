# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import GlowwormSwarmOptimizationV1
from niapy.task import Task
from niapy.problems import Sphere

for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    algo = GlowwormSwarmOptimizationV1()
    best = algo.run(task)
    print(best)
