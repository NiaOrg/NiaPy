# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import Task
from niapy.problems import Sphere
from niapy.algorithms.other import RandomSearch


task = Task(problem=Sphere(dimension=5), max_iters=5000)
algo = RandomSearch()
best = algo.run(task=task)
print(best)
