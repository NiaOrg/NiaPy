# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import StoppingTask
from niapy.benchmarks import Griewank
from niapy.algorithms.basic import DifferentialEvolution

# we will run Differential Evolution for 5 independent runs
algo = DifferentialEvolution(population_size=50, F=0.5, CR=0.9)
for i in range(5):
    task = StoppingTask(max_evals=10000, enable_logging=True, dimension=10, benchmark=Griewank(Lower=-600, Upper=600))
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.get_parameters())
