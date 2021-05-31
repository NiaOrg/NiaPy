# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.modified import ParameterFreeBatAlgorithm

from niapy.task import Task
from niapy.problems import Sphere

algo = ParameterFreeBatAlgorithm()

for i in range(10):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.get_parameters())
