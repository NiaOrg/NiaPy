# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import FlowerPollinationAlgorithm
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere

# we will run Flower Pollination Algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(max_evals=10000, dimension=10, benchmark=Sphere())
    algo = FlowerPollinationAlgorithm(population_size=20, p=0.5)
    best = algo.run(task=task)
    print(best)
