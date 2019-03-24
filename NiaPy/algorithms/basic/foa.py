# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, redefined-builtin, line-too-long, no-self-use, arguments-differ, no-else-return, bad-continuation
import logging
from numpy import where, apply_along_axis, zeros, append, ndarray, delete, arange, argmin, absolute
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.util import OptimizationType

__all__ = ['ForestOptimizationAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')


class ForestOptimizationAlgorithm(Algorithm):
    r"""Implementation of Forest Optimization Algorithm.

    **Algorithm:** Forest Optimization Algorithm

    **Date:** 2019

    **Authors:** Luka Pečnik

    **License:** MIT

    **Reference paper:** Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi, Forest Optimization Algorithm, Expert Systems with Applications, Volume 41, Issue 15, 2014, Pages 6676-6687, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2014.05.009.
    """
    Name = ['ForestOptimizationAlgorithm', 'FOA']

    @staticmethod
    def typeParameters(): return {
        'NP': lambda x: isinstance(x, int) and x > 0,
        'lt': lambda x: isinstance(x, int) and x > 0,
        'al': lambda x: isinstance(x, int) and x > 0,
        'lsc': lambda x: isinstance(x, int) and x > 0,
        'gsc': lambda x: isinstance(x, int) and x > 0,
        'tr': lambda x: isinstance(x, (float, int)) and x >= 0 and x <= 1,
    }

    def setParameters(self, NP=20, lt=1, al=1, lsc=1, gsc=1, tr=0, **ukwargs):
        r"""Set the parameters of the algorithm.

        **Arguments:**

        NP {integer} -- population size

        lt {integer} -- life time of trees

        al {integer} -- area limit

        lsc {integer} -- local seeding changes

        gsc {integer} -- global seeding changes

        tr {float} -- percentage of candidate population for global seeding (transfer rate) [0, 1]
        """
        self.NP, self.lt, self.al, self.lsc, self.gsc, self.tr = NP, lt, al, lsc, gsc, tr
        if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

    def repair(self, x, lower, upper):
        """Truncate exceeded dimensions to the limits."""
        ir = where(x < lower)
        x[ir] = lower[ir]
        ir = where(x > upper)
        x[ir] = upper[ir]
        return x

    def localSeeding(self, task, trees, dx):
        """Local optimum search stage."""
        n = trees.shape[0]
        deltas = self.uniform(-dx, dx, (n, self.lsc))
        deltas = append(deltas, zeros((n, task.D - self.lsc)), axis=1)
        perms = self.rand([deltas.shape[0], deltas.shape[1]]).argsort(1)
        deltas = deltas[arange(deltas.shape[0])[:, None], perms]
        trees[:, :-1] += deltas
        trees[:, :-1] = apply_along_axis(self.repair, 1, trees[:, :-1], task.Lower, task.Upper)
        return trees

    def globalSeeding(self, task, candidates, size):
        """Global optimum search stage that should prevent getting stuck in a local optimum."""
        seeds = candidates[self.randint(len(candidates), D=size), :-1]
        deltas = self.uniform(task.benchmark.Lower, task.benchmark.Upper, (size, self.gsc))
        deltas = append(deltas, zeros((size, task.D - self.gsc)), axis=1)
        perms = self.rand([deltas.shape[0], deltas.shape[1]]).argsort(1)
        deltas = deltas[arange(deltas.shape[0])[:, None], perms]

        deltas = deltas.flatten()
        seeds = seeds.flatten()
        seeds[deltas != 0] = deltas[deltas != 0]

        return append(seeds.reshape(size, task.D), zeros((size, 1)), axis=1)

    def removeLifeTimeExceeded(self, trees, candidates):
        """Remove dead trees."""
        lifeTimeExceeded = where(trees[:, -1] > self.lt)
        candidates = trees[lifeTimeExceeded]
        trees = delete(trees, lifeTimeExceeded, axis=0)
        return trees, candidates

    def survivalOfTheFittest(self, task, trees, candidates):
        """Evaluation and filtering of current population."""
        evaluations = apply_along_axis(task.eval, 1, trees[:, :-1])
        ei = evaluations.argsort()
        candidates = append(candidates, trees[ei[self.al:]], axis=0)
        trees = trees[ei[:self.al]]
        evaluations = evaluations[ei[:self.al]]
        return trees, candidates, evaluations

    def getBest(self, trees, evaluations):
        """Get currently best individual."""
        ib = argmin(evaluations)
        return trees[ib], evaluations[ib]

    def runTask(self, task):
        """Run."""
        if self.gsc > task.D:
            logger.info('GSC argument bigger than dimension of the task. Value truncated to dimension.')
            self.gsc = task.D
        if self.lsc > task.D:
            logger.info('LSC argument bigger than dimension of the task. Value truncated to dimension.')
            self.lsc = task.D

        forest = self.uniform(task.Lower, task.Upper, [self.NP, task.D])
        z = zeros((self.NP, 1))
        forest = append(forest, z, axis=1)

        evaluations = apply_along_axis(task.eval, 1, forest[:, :-1])
        bestTree, bestTreeEvaluation = self.getBest(forest, evaluations)

        dx = absolute(task.benchmark.Upper) / 5

        while not task.stopCondI():
            candidatePopulation = ndarray((0, task.D + 1))
            zeroAgeTrees = forest[forest[:, -1] == 0]

            localSeeds = self.localSeeding(task, zeroAgeTrees, dx)
            forest[:, -1] += 1

            forest, candidatePopulation = self.removeLifeTimeExceeded(forest, candidatePopulation)
            forest = append(forest, localSeeds, axis=0)
            forest, candidatePopulation, evaluations = self.survivalOfTheFittest(task, forest, candidatePopulation)

            gsn = int(self.tr * len(candidatePopulation))
            if gsn > 0:
                globalSeeds = self.globalSeeding(task, candidatePopulation, gsn)
                forest = append(forest, globalSeeds, axis=0)
                gste = apply_along_axis(task.eval, 1, globalSeeds[:, :-1])
                evaluations = append(evaluations, gste)
            
            bestTree, bestTreeEvaluation = self.getBest(forest, evaluations)
            ib = argmin(evaluations)
            forest[ib, -1] = 0
        
        return bestTree, bestTreeEvaluation

# TODO delete this
class MyBenchmark(object):
    def __init__(self):
        # define lower bound of benchmark function
        self.Lower = -11
        # define upper bound of benchmark function
        self.Upper = 11

    # function which returns evaluate function
    def function(self):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate

algorithm = ForestOptimizationAlgorithm(nFES=200000, NP=10, D=10, lt=10, lsc=2, gsc=2, al=10, tr=0.3, benchmark=MyBenchmark())
print(algorithm.run())
