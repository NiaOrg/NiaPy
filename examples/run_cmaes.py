# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
from NiaPy.task.task import Task, TaskConvPrint, TaskConvPlot, OptimizationType
from NiaPy.util import getDictArgs
from NiaPy import Runner
import logging
import random
import sys
sys.path.append('../')
# End of fix


logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

# For reproducive results
random.seed(1234)


class MinMB(object):
    def __init__(self):
        self.Lower = -11
        self.Upper = 11

    def function(self):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate


class MaxMB(MinMB):
    def function(self):
        f = MinMB.function(self)
        def e(D, sol): return -f(D, sol)
        return e


def simple_example(alg, runs=10, D=10, nFES=50000, nGEN=10000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB, **kn):
    for i in range(runs):
        task = Task(D=D, nFES=nFES, nGEN=nGEN, optType=optType, benchmark=optFunc())
        algo = alg(seed=seed, task=task)
        best = algo.run()
        logger.info('%s %s' % (best[0], best[1]))


def logging_example(alg, D=10, nFES=50000, nGEN=100000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB, **kn):
    task = TaskConvPrint(D=D, nFES=nFES, nGEN=nGEN, optType=optType, benchmark=optFunc())
    algo = alg(seed=seed, task=task)
    best = algo.run()
    logger.info('%s %s' % (best[0], best[1]))


def plot_example(alg, D=10, nFES=50000, nGEN=100000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB, **kn):
    task = TaskConvPlot(D=D, nFES=nFES, nGEN=nGEN, optType=optType, benchmark=optFunc())
    algo = alg(seed=seed, task=task)
    best = algo.run()
    logger.info('%s %s' % (best[0], best[1]))
    input('Press [enter] to continue')


def getOptType(otype):
    if otype == OptimizationType.MINIMIZATION:
        return MinMB
    elif otype == OptimizationType.MAXIMIZATION:
        return MaxMB
    else:
        return None


if __name__ == '__main__':
    pargs, algo = getDictArgs(sys.argv[1:]), Runner.getAlgorithm('CovarianceMaatrixAdaptionEvolutionStrategy')
    optFunc = getOptType(pargs['optType'])
    if not pargs['runType']:
        simple_example(algo, optFunc=optFunc, **pargs)
    elif pargs['runType'] == 'log':
        logging_example(algo, optFunc=optFunc, **pargs)
    elif pargs['runType'] == 'plot':
        plot_example(algo, optFunc=optFunc, **pargs)
