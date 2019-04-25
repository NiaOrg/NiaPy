# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import AgingNpMultiMutationDifferentialEvolution
from NiaPy.algorithms.basic.de import CrossCurr2Best1, CrossBest2
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

# we will run Differential Evolution for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=5000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = AgingNpMultiMutationDifferentialEvolution(NP=10, F=0.2, CR=0.65, strategies=(CrossCurr2Best1, CrossBest2), delta_np=0.05, omega=0.9)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0].x, best[1]))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
