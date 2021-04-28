# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from niapy.task import StoppingTask
from niapy.benchmarks import Sphere
from niapy.algorithms.other import SimulatedAnnealing
from niapy.algorithms.other.sa import coolLinear

# we will run Simulated Annealing for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nGEN=1000, benchmark=Sphere())
    algo = SimulatedAnnealing(coolingMethod=coolLinear)
    best = algo.run(task=task)
    print(best)
