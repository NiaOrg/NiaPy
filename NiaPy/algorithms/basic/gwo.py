# encoding=utf8

import numpy as np

from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['GreyWolfOptimizer']

class GreyWolfOptimizer(Algorithm):
    r"""Implementation of Grey wolf optimizer.

    Algorithm:
        Grey wolf optimizer

    Date:
        2018

    Author:
        Iztok Fister Jr. and Klemen Berkovič

    License:
        MIT

    Reference paper:
        * Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis. "Grey wolf optimizer." Advances in engineering software 69 (2014): 46-61.
        * Grey Wolf Optimizer (GWO) source code version 1.0 (MATLAB) from MathWorks

    Attributes:
        Name (List[str]): List of strings representing algorithm names.

    See Also:
        * :class:`NiaPy.algorithms.Algorithm`
    """
    Name = ['GreyWolfOptimizer', 'GWO']

    @staticmethod
    def algorithmInfo():
    	r"""Get algorithm information.

    	Returns:
    		str: Algorithm information.

    	See Also:
    		* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
    	"""
    	return r"""Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis. "Grey wolf optimizer." Advances in engineering software 69 (2014): 46-61."""

    @staticmethod
    def typeParameters(): return {
        'NP': lambda x: isinstance(x, int) and x > 0
    }

    def setParameters(self, NP=25, **ukwargs):
        r"""Set the algorithm parameters.

        Arguments:
            NP (int): Number of individuals in population

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.setParameters`
        """
        Algorithm.setParameters(self, NP=NP, **ukwargs)

    def initPopulation(self, task):
        r"""Initialize population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations fitness/function values.
                3. Additional arguments:
                    * A (): TODO

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.initPopulation`
        """
        pop, fpop, d = Algorithm.initPopulation(self, task)
        si = np.argsort(fpop)
        A, A_f, B, B_f, D, D_f = np.copy(pop[si[0]]), fpop[si[0]], np.copy(pop[si[1]]), fpop[si[1]], np.copy(pop[si[2]]), fpop[si[2]]
        d.update({'A': A, 'A_f': A_f, 'B': B, 'B_f': B_f, 'D': D, 'D_f': D_f})
        return pop, fpop, d

    def runIteration(self, task, pop, fpop, xb, fxb, A, A_f, B, B_f, D, D_f, **dparams):
        r"""Core funciton of GreyWolfOptimizer algorithm.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current population.
            fpop (numpy.ndarray): Current populations function/fitness values.
            xb (numpy.ndarray):
            fxb (float):
            A (numpy.ndarray):
            A_f (float):
            B (numpy.ndarray):
            B_f (float):
            D (numpy.ndarray):
            D_f (float):
            **dparams (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. Additional arguments:
                    * A (): TODO
        """
        a = 2 - task.Evals * (2 / task.nFES)
        for i, w in enumerate(pop):
            A1, C1 = 2 * a * self.rand(task.D) - a, 2 * self.rand(task.D)
            X1 = A - A1 * np.fabs(C1 * A - w)
            A2, C2 = 2 * a * self.rand(task.D) - a, 2 * self.rand(task.D)
            X2 = B - A2 * np.fabs(C2 * B - w)
            A3, C3 = 2 * a * self.rand(task.D) - a, 2 * self.rand(task.D)
            X3 = D - A3 * np.fabs(C3 * D - w)
            pop[i] = task.repair((X1 + X2 + X3) / 3, self.Rand)
            fpop[i] = task.eval(pop[i])
        for i, f in enumerate(fpop):
            if f < A_f: A, A_f = pop[i].copy(), f
            elif A_f < f < B_f: B, B_f = pop[i].copy(), f
            elif B_f < f < D_f: D, D_f = pop[i].copy(), f
        xb, fxb = self.getBest(A, A_f, xb, fxb)
        return pop, fpop, xb, fxb, {'A': A, 'A_f': A_f, 'B': B, 'B_f': B_f, 'D': D, 'D_f': D_f}
