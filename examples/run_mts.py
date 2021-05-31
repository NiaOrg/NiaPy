# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.other import MultipleTrajectorySearch
from niapy.task import Task
from niapy.problems import Sphere

# we will run Nelder Mead algorithm for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_iters=40)
    algo = MultipleTrajectorySearch()
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
