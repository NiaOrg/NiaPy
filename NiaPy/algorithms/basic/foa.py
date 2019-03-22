# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, redefined-builtin, line-too-long, no-self-use, arguments-differ, no-else-return, bad-continuation
import logging
from numpy import random as rand, where, apply_along_axis, zeros, append, ndarray, array, delete, argsort, arange, argmin, inf
from NiaPy.algorithms.algorithm import Algorithm

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
            'tr': lambda x: isinstance(x, float) and x >= 0 and x <= 1,
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
        ir = where(x < lower)
        x[ir] = lower[ir]
        ir = where(x > upper)
        x[ir] = upper[ir]
        return x

    # opisi funkcij
    def localSeeding(self, task, trees):
        n = trees.shape[0]
        newTrees = ndarray((n * self.lsc, task.D + 1))
        c = 0
        for i in range(0, n):
            for j in range(0, self.lsc):
                newTree = trees[i]
                di = rand.randint(1, task.D + 1)
                delta = rand.uniform(-task.benchmark.Upper, task.benchmark.Upper)
                newTree[di] += delta
                newTrees[c] = newTree
                c += 1
        newTrees[:,1:] = apply_along_axis(self.repair, 1, newTrees[:,1:], task.Lower, task.Upper)    #repair exceeded values (the first one is tree age)
        return newTrees

    def globalSeeding(self, task, candidates, size):
        seeds = candidates[rand.randint(len(candidates), size=size), :]
        inds = arange(1, task.D + 1)
        for i in range(size):
            gsi = rand.permutation(inds)
            delta = self.uniform(task.benchmark.Lower, task.benchmark.Upper, self.gsc)
            seeds[i, gsi[:self.gsc]] = delta[:]
        return seeds

    def removeLifeTimeExceeded(self, trees, candidates):
        lifeTimeExceeded = where(trees[:,0] > self.lt)    #get dead tree indices
        candidates = append(candidates, trees[lifeTimeExceeded], axis=0)    #move them to candidate population
        trees = delete(trees, lifeTimeExceeded, axis=0)    #delete dead trees from population
        return trees, candidates

    def survivalOfTheFittest(self, task, trees, candidates):
        evaluations = apply_along_axis(task.eval, 1, trees[:,1:])    #evaluate population
        ei = evaluations.argsort()    #get sorted indices
        candidates = append(candidates, trees[self.al:], axis=0)
        trees = trees[ei[:self.al]]    #sort according to the indices and remove the number of trees that exceed the area limit argument
        evaluations = evaluations[ei[:self.al]]
        return trees, candidates, evaluations

    def getBest(self, bestTree, bestTreeEvaluation, trees, evaluations):
        ib = argmin(evaluations)
        return trees[ib], evaluations[ib]

    def runTask(self, task):
        if self.gsc > task.D:
            logger.info('GSC argument bigger than dimension of the task. Value changed to equal the dimension.')
            self.gsc = task.D

        forest = self.uniform(task.Lower, task.Upper, [self.NP, task.D])
        z = zeros((self.NP, 1))
        forest = append(z, forest, axis=1)

        evaluations = apply_along_axis(task.eval, 1, forest[:,1:])
        bestTree, bestTreeEvaluation = self.getBest(None, task.optType.value * inf, forest, evaluations)

        while not task.stopCondI():
            candidatePopulation = ndarray((0, task.D + 1))
            zeroAgeTrees = forest[forest[:,0] == 0]

            # local seeding on trees with age 0
            localSeeds = self.localSeeding(task, zeroAgeTrees)
            forest[:,0] += 1    #increase tree age

            # population limiting
            forest, candidatePopulation = self.removeLifeTimeExceeded(forest, candidatePopulation)
            forest = append(forest, localSeeds, axis=0)
            forest, candidatePopulation, evaluations = self.survivalOfTheFittest(task, forest, candidatePopulation)

            # global seeding
            gsn = int(self.tr * len(candidatePopulation))
            if gsn > 0:
                globalSeeds = self.globalSeeding(task, candidatePopulation, gsn)
                forest = append(forest, globalSeeds, axis=0)
                gste = apply_along_axis(task.eval, 1, globalSeeds[:,1:])    #evaluate global seeds
                evaluations = append(evaluations, gste)
            
            # update the best tree so far
            bestTree, bestTreeEvaluation = self.getBest(bestTree, bestTreeEvaluation, forest, evaluations)
            ib = argmin(evaluations)
            forest[ib, 0] = 0
        
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


algorithm = ForestOptimizationAlgorithm(nFES=100000, NP=10, D=9, lt=4, lsc=2, al=10, tr=0.2, benchmark=MyBenchmark())
best = algorithm.run()
print(best)
