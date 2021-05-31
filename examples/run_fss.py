# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import FishSchoolSearch
from niapy.task import Task
from niapy.problems import Sphere

# we will run Fish School Search for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    algo = FishSchoolSearch(population_size=20)
    best = algo.run(task)
    print('%s -> %f' % (best[0], best[1]))
