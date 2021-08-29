# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.modified import SelfAdaptiveBatAlgorithm
from niapy.task import Task
from niapy.problems import Griewank

# we will run Bat Algorithm for 5 independent runs
algo = SelfAdaptiveBatAlgorithm()
for i in range(5):
    task = Task(problem=Griewank(dimension=10, lower=-600, upper=600), max_iters=10000)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.get_parameters())
