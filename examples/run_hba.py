# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.modified import HybridBatAlgorithm
from niapy.task import Task
from niapy.problems import Sphere

# we will run Hybrid Bat Algorithm for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    algo = HybridBatAlgorithm(population_size=40, loudness=0.5, pulse_rate=0.5, differential_weight=0.5, crossover_probability=0.9, min_frequency=0.0, max_frequency=2.0)
    best = algo.run(task)
    print(best)
