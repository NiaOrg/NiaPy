# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
from NiaPy.algorithms.basic import CovarianceMatrixAdaptionEvolutionStrategy
from NiaPy.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

import sys
sys.path.append('../')
# End of fix

# we will run CMA-ES for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=1000, optType=OptimizationType.MINIMIZATION, logger=True, benchmark=Sphere())
    algo = CovarianceMatrixAdaptionEvolutionStrategy(NP=20)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))

