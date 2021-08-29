# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import SineCosineAlgorithm
from niapy.task import Task
from niapy.problems import Sphere

# we will run Sine Cosine Algorithm algorithm for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    algo = SineCosineAlgorithm(population_size=30, a=7, r_min=0.1, r_max=3)
    best = algo.run(task=task)
    print(best)
