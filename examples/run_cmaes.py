# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
from NiaPy.algorithms.basic import CovarianceMatrixAdaptionEvolutionStrategy
from NiaPy.benchmarks import Sphere
from NiaPy.task import StoppingTask

import sys
sys.path.append('../')
# End of fix

# we will run CMA-ES for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=1000, logger=True, benchmark=Sphere())
    algo = CovarianceMatrixAdaptionEvolutionStrategy(NP=20)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))

