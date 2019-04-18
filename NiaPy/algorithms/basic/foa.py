# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, redefined-builtin, line-too-long, no-self-use, arguments-differ, no-else-return, bad-continuation
import logging
from numpy import where, apply_along_axis, zeros, append, ndarray, delete, arange, argmin, absolute
from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['ForestOptimizationAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

class ForestOptimizationAlgorithm(Algorithm):
    r"""Implementation of Forest Optimization Algorithm.

    Algorithm:
        Forest Optimization Algorithm

    Date:
        2019

    Authors:
        Luka Pečnik

    License:
        MIT

    Reference paper:
        Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi, Forest Optimization Algorithm, Expert Systems with Applications, Volume 41, Issue 15, 2014, Pages 6676-6687, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2014.05.009.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        lt (int): Life time of trees parameter.
        al (int): Area limit parameter.
        lsc (int): Local seeding changes parameter.
        gsc (int): Global seeding changes parameter.
        tr (float): Transfer rate parameter.

    See Also:
        * :class:`NiaPy.algorithms.Algorithm`
    """
    Name = ['ForestOptimizationAlgorithm', 'FOA']

    @staticmethod
    def algorithmInfo():
        return r"""
        Description: Forest Optimization Algorithm is inspired by few trees in the forests which can survive for several decades, while other trees could live for a limited period.
        Authors: Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi
        Year: 2014
    Main reference: Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi, Forest Optimization Algorithm, Expert Systems with Applications, Volume 41, Issue 15, 2014, Pages 6676-6687, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2014.05.009.
    """

    @staticmethod
    def typeParameters():
        r"""Get dictionary with functions for checking values of parameters.

        Returns:
            Dict[str, Callable]:
                * lt (Callable[[int], bool]): Checks if life time parameter has a proper value.
                * al (Callable[[int], bool]): Checks if area limit parameter has a proper value.
                * lsc (Callable[[int], bool]): Checks if local seeding changes parameter has a proper value.
                * gsc (Callable[[int], bool]): Checks if global seeding changes parameter has a proper value.
                * tr (Callable[[float], bool]): Checks if transfer rate parameter has a proper value.

        See Also:
            * :func:`NiaPy.algorithms.algorithm.Algorithm.typeParameters`
        """
        d = Algorithm.typeParameters()
        d.update({
            'lt': lambda x: isinstance(x, int) and x > 0,
            'al': lambda x: isinstance(x, int) and x > 0,
            'lsc': lambda x: isinstance(x, int) and x > 0,
            'gsc': lambda x: isinstance(x, int) and x > 0,
            'tr': lambda x: isinstance(x, float) and 0 <= x <= 1,
        })
        return d

    def setParameters(self, NP=10, lt=3, al=10, lsc=1, gsc=1, tr=0.3, **ukwargs):
        r"""Set the parameters of the algorithm.

        Args:
            NP (Optional[int]): Population size.
            lt (Optional[int]): Life time parameter.
            al (Optional[int]): Area limit parameter.
            lsc (Optional[int]): Local seeding changes parameter.
            gsc (Optional[int]): Global seeding changes parameter.
            tr (Optional[float]): Transfer rate parameter.
            ukwargs (Dict[str, Any]): Additional arguments.

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.setParameters`
        """
        Algorithm.setParameters(self, NP=NP)
        self.lt, self.al, self.lsc, self.gsc, self.tr = lt, al, lsc, gsc, tr
        if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

    def repair(self, x, lower, upper):
        r"""Truncate exceeded dimensions to the limits.

        Args:
            x (numpy.ndarray): Individual to repair.
            lower (numpy.ndarray): Lower limits for dimensions.
            upper (numpy.ndarray): Upper limits for dimensions.

        Returns:
            numpy.ndarray: Repaired individual.
        """
        ir = where(x < lower)
        x[ir] = lower[ir]
        ir = where(x > upper)
        x[ir] = upper[ir]
        return x

    def localSeeding(self, task, trees, dx):
        r"""Local optimum search stage.

        Args:
            task (Task): Optimization task.
            trees (numpy.ndarray): Zero age trees for local seeding.
            dx (float): A small value used in local seeding stage.

        Returns:
            numpy.ndarray: Resulting zero age trees.
        """
        n = trees.shape[0]
        deltas = self.uniform(-dx, dx, (n, self.lsc))
        deltas = append(deltas, zeros((n, task.D - self.lsc)), axis=1)
        perms = self.rand([deltas.shape[0], deltas.shape[1]]).argsort(1)
        deltas = deltas[arange(deltas.shape[0])[:, None], perms]
        trees[:, :-1] += deltas
        trees[:, :-1] = apply_along_axis(self.repair, 1, trees[:, :-1], task.Lower, task.Upper)
        return trees

    def globalSeeding(self, task, candidates, size):
        r"""Global optimum search stage that should prevent getting stuck in a local optimum.

        Args:
            task (Task): Optimization task.
            candidates (numpy.ndarray): Candidate population for global seeding.
            size (int): Number of trees to produce.

        Returns:
            numpy.ndarray: Resulting trees.
        """
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
        r"""Remove dead trees.

        Args:
            trees (numpy.ndarray): Population to test.
            candidates (numpy.ndarray): Candidate population array to be updated.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                1. Alive trees.
                2. New candidate population.
        """
        lifeTimeExceeded = where(trees[:, -1] > self.lt)
        candidates = trees[lifeTimeExceeded]
        trees = delete(trees, lifeTimeExceeded, axis=0)
        return trees, candidates

    def survivalOfTheFittest(self, task, trees, candidates):
        r"""Evaluate and filter current population.

        Args:
            task (Task): Optimization task.
            trees (numpy.ndarray): Population to evaluate.
            candidates (numpy.ndarray): Candidate population array to be updated.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray[float]]:
                1. Trees sorted by fitness value.
                2. Updated candidate population.
                3. Population fitness values.
        """
        evaluations = apply_along_axis(task.eval, 1, trees[:, :-1])
        ei = evaluations.argsort()
        candidates = append(candidates, trees[ei[self.al:]], axis=0)
        trees = trees[ei[:self.al]]
        evaluations = evaluations[ei[:self.al]]
        return trees, candidates, evaluations

    def initPopulation(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * dx (float): A small value used in local seeding stage.

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.initPopulation`
        """
        Trees, Evaluations, _ = Algorithm.initPopulation(self, task)
        z = zeros((self.NP, 1))
        Trees = append(Trees, z, axis=1)
        dx = absolute(task.benchmark.Upper) / 5
        return Trees, Evaluations, {'dx': dx}

    def runIteration(self, task, Trees, Evaluations, xb, fxb, dx, **dparams):
        r"""Core function of Forest Optimization Algorithm.

        Args:
            task (Task): Optimization task.
            Trees (numpy.ndarray): Current population.
            Evaluations (numpy.ndarray[float]): Current population function/fitness values.
            xb (numpy.ndarray): Global best individual.
            fxb (float): Global best individual fitness/function value.
            dx (float): A small value used in local seeding stage.
            **dparams (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * dx (float): A small value used in local seeding stage.
        """
        candidatePopulation = ndarray((0, task.D + 1))
        zeroAgeTrees = Trees[Trees[:, -1] == 0]

        localSeeds = self.localSeeding(task, zeroAgeTrees, dx)
        Trees[:, -1] += 1

        Trees, candidatePopulation = self.removeLifeTimeExceeded(Trees, candidatePopulation)
        Trees = append(Trees, localSeeds, axis=0)
        Trees, candidatePopulation, Evaluations = self.survivalOfTheFittest(task, Trees, candidatePopulation)

        gsn = int(self.tr * len(candidatePopulation))
        if gsn > 0:
            globalSeeds = self.globalSeeding(task, candidatePopulation, gsn)
            Trees = append(Trees, globalSeeds, axis=0)
            gste = apply_along_axis(task.eval, 1, globalSeeds[:, :-1])
            Evaluations = append(Evaluations, gste)

        ib = argmin(Evaluations)
        Trees[ib, -1] = 0
        return Trees, Evaluations, {'dx': dx}
