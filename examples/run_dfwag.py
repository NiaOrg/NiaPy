# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import DynamicFireworksAlgorithmGauss
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere

# we will run Fireworks Algorithm for 5 independent runs
algo = DynamicFireworksAlgorithmGauss(N=70, Ainit=0.1, Afinal=0.9)
for i in range(5):
    task = StoppingTask(max_iters=50, dimension=10, benchmark=Sphere())
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.get_parameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
