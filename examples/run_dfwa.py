# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import StoppingTask
from niapy.problems import Sphere
from niapy.algorithms.basic import DynamicFireworksAlgorithm

# we will run Fireworks Algorithm for 5 independent runs
algo = DynamicFireworksAlgorithm(population_size=70, amplitude_init=0.1, amplitude_final=0.9)
for i in range(5):
    task = StoppingTask(problem=Sphere(dimension=10), max_iters=80)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.get_parameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
