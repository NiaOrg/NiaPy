# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import Task
from niapy.problems import Sphere
from niapy.algorithms.basic import CatSwarmOptimization

task = Task(problem=Sphere(dimension=10), max_evals=1000, enable_logging=True)
algo = CatSwarmOptimization()
best = algo.run(task=task)
print('%s -> %s' % (best[0], best[1]))
# plot a convergence graph
task.plot_convergence(x_axis='evals')
