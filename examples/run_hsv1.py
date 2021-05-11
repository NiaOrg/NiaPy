# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import HarmonySearchV1
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere

# we will run Bat Algorithm for 5 independent runs
algo = HarmonySearchV1()
for i in range(5):
    task = StoppingTask(max_iters=1000, dimension=10, benchmark=Sphere())
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.get_parameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
