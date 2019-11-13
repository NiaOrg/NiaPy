# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import GlowwormSwarmOptimizationV3
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere

for i in range(5):
    task = StoppingTask(D=10, nFES=10000, benchmark=Sphere())
    algo = GlowwormSwarmOptimizationV3()
    best = algo.run(task)
    print(best)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
