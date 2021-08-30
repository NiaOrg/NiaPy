# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import DynamicFireworksAlgorithmGauss
from niapy.task import Task
from niapy.problems import Sphere

# we will run Fireworks Algorithm for 5 independent runs
algo = DynamicFireworksAlgorithmGauss(population_size=70, amplitude_init=0.1, amplitude_final=0.9)
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_iters=50)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.get_parameters())
