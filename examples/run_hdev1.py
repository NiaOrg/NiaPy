# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.modified import DifferentialEvolutionMTSv1
from NiaPy.algorithms.basic.de import CrossBest2
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere

# we will run Differential Evolution for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, benchmark=Sphere())
    algo = DifferentialEvolutionMTSv1(NP=50, F=0.5, CR=0.9, CrossMutt=CrossBest2, NoLsTests=5, NoLs=3, NoEnabled=4)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
