# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from niapy.algorithms.basic import GlowwormSwarmOptimizationV1
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere

for i in range(5):
    task = StoppingTask(D=10, nFES=10000, benchmark=Sphere())
    algo = GlowwormSwarmOptimizationV1()
    best = algo.run(task)
    print(best)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
