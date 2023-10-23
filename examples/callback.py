from niapy.callbacks import Callback
from niapy.algorithms.basic import BatAlgorithm
from niapy.task import Task
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix


class PrintMeanFitness(Callback):
    def __init__(self):
        super().__init__()

    def after_iteration(self, population, fitness, best_x, best_fitness, **params):
        print(fitness.mean())


ba = BatAlgorithm(callbacks=[PrintMeanFitness()])
griewank = Task('griewank', max_evals=1000)

ba.run(griewank)
