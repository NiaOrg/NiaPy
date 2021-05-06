# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import StoppingTask
from niapy.benchmarks import Sphere
from niapy.algorithms.other import SimulatedAnnealing
from niapy.algorithms.other.sa import cool_linear

# we will run Simulated Annealing for 5 independent runs
for i in range(5):
    task = StoppingTask(max_iters=1000, dimension=10, benchmark=Sphere())
    algo = SimulatedAnnealing(cooling_nethod=cool_linear)
    best = algo.run(task=task)
    print(best)
