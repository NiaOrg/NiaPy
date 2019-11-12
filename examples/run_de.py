# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Griewank
from NiaPy.algorithms.basic import DifferentialEvolution

# we will run Differential Evolution for 5 independent runs
algo = DifferentialEvolution(NP=50, F=0.5, CR=0.9)
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, benchmark=Griewank(Lower=-600, Upper=600), logger=True)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.getParameters())

