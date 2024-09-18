# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.modified import ImprovedLpsrSuccessHistoryAdaptiveDifferentialEvolution
from niapy.task import Task
from niapy.problems import Griewank

# we will run iL-SHADE algorithm for 5 independent runs
algo = ImprovedLpsrSuccessHistoryAdaptiveDifferentialEvolution(population_size=40, extern_arc_rate=2.6, pbest_start=0.2,
                                                   pbest_end=0.1, hist_mem_size=6)
for i in range(5):
    task = Task(problem=Griewank(dimension=10, lower=-600, upper=600), max_evals=10000, enable_logging=True)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.get_parameters())