# coding=utf-8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

__all__ = ['BeesAlgorithm']

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')


class BeesAlgorithm(Algorithm):
    r"""Implementation of Bees algorithm.

    Algorithm:
        The Bees algorithm

    Date:
        2019

    Authors:
        Rok Potočnik

    License:
        MIT

    Reference paper:
        DT Pham, A Ghanbarzadeh, E Koc, S Otri, S Rahim, and M Zaidi. The bees algorithm-a novel tool for complex optimisation problems. In Proceedings of the 2nd Virtual International Conference on Intelligent Production Machines and Systems (IPROMS 2006), pages 454–459, 2006

    Attributes:
        population_size (Optional[int]): Number of scout bees parameter.
        m (Optional[int]): Number of sites selected out of n visited sites parameter.
        e (Optional[int]): Number of best sites out of m selected sites parameter.
        nep (Optional[int]): Number of bees recruited for best e sites parameter.
        nsp (Optional[int]): Number of bees recruited for the other selected sites parameter.
        ngh (Optional[float]): Initial size of patches parameter.

    See Also:
        * :func:`niapy.algorithms.Algorithm.set_parameters`

    """

    Name = ['BeesAlgorithm', 'BEA']

    @staticmethod
    def info():
        r"""Get information about algorithm.

        Returns:
            str: Algorithm information

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""DT Pham, A Ghanbarzadeh, E Koc, S Otri, S Rahim, and M Zaidi. The bees algorithm-a novel tool for complex optimisation problems. In Proceedings of the 2nd Virtual International Conference on Intelligent Production Machines and Systems (IPROMS 2006), pages 454–459, 2006"""

    def __init__(self, population_size=40, m=5, e=4, ngh=1, nep=4, nsp=2, *args, **kwargs):
        """Initialize BeesAlgorithm.

        Args:
            population_size (Optional[int]): Number of scout bees parameter.
            m (Optional[int]): Number of sites selected out of n visited sites parameter.
            e (Optional[int]): Number of best sites out of m selected sites parameter.
            nep (Optional[int]): Number of bees recruited for best e sites parameter.
            nsp (Optional[int]): Number of bees recruited for the other selected sites parameter.
            ngh (Optional[float]): Initial size of patches parameter.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.m = m
        self.e = e
        self.ngh = ngh
        self.nep = nep
        self.nsp = nsp

    def set_parameters(self, population_size=40, m=5, e=4, ngh=1, nep=4, nsp=2, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Number of scout bees parameter.
            m (Optional[int]): Number of sites selected out of n visited sites parameter.
            e (Optional[int]): Number of best sites out of m selected sites parameter.
            nep (Optional[int]): Number of bees recruited for best e sites parameter.
            nsp (Optional[int]): Number of bees recruited for the other selected sites parameter.
            ngh (Optional[float]): Initial size of patches parameter.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.m = m
        self.e = e
        self.ngh = ngh
        self.nep = nep
        self.nsp = nsp

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm Parameters.

        """
        d = super().get_parameters()
        d.update({
            'm': self.m,
            'e': self.e,
            'ngh': self.ngh,
            'nep': self.nep,
            'nsp': self.nsp
        })
        return d

    def bee_dance(self, x, task, ngh):
        r"""Bees Dance. Search for new positions.

        Args:
            x (numpy.ndarray): One individual from the population.
            task (Task): Optimization task.
            ngh (float): A small value for patch search.

        Returns:
            Tuple[numpy.ndarray, float]:
                1. New individual.
                2. New individual fitness/function values.

        """
        ind = self.integers(task.dimension)
        y = x.copy()
        y[ind] = x[ind] + self.uniform(-ngh, ngh)
        y = task.repair(y)
        y_fitness = task.eval(y)
        return y, y_fitness

    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        population, population_fitness, _ = super().init_population(task)

        sorted_indices = np.argsort(population_fitness)
        population_fitness = population_fitness[sorted_indices]
        population = population[sorted_indices, :]

        return population, population_fitness, {'ngh': self.ngh}

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Forest Optimization Algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray[float]): Current population.
            population_fitness (numpy.ndarray[float]): Current population function/fitness values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individual fitness/function value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution.
                4. New global best fitness/objective value.
                5. Additional arguments:
                    * ngh (float): A small value used for patches.

        """
        ngh = params.pop('ngh')

        for ies in range(self.e):
            best_bee_pos = None
            best_bee_cost = np.inf
            for ieb in range(self.nep):
                new_bee_pos, new_bee_cost = self.bee_dance(population[ies, :], task, ngh)
                if new_bee_cost < best_bee_cost:
                    best_bee_cost = new_bee_cost
                    best_bee_pos = new_bee_pos
            if best_bee_cost < population_fitness[ies]:
                population[ies, :] = best_bee_pos
                population_fitness[ies] = best_bee_cost
        for ies in range(self.e, self.m):
            best_bee_pos = None
            best_bee_cost = np.inf
            for ieb in range(self.nsp):
                new_bee_pos, new_bee_cost = self.bee_dance(population[ies, :], task, ngh)
                if new_bee_cost < best_bee_cost:
                    best_bee_cost = new_bee_cost
                    best_bee_pos = new_bee_pos
            if best_bee_cost < population_fitness[ies]:
                population[ies, :] = best_bee_pos
                population_fitness[ies] = best_bee_cost
        for ies in range(self.m, self.population_size):
            population[ies, :] = self.uniform(task.lower, task.upper, task.dimension)
            population_fitness[ies] = task.eval(population[ies, :])
        sorted_indices = np.argsort(population_fitness)
        population_fitness = population_fitness[sorted_indices]
        population = population[sorted_indices, :]
        ngh = ngh * 0.95
        return population, population_fitness, population[0].copy(), population_fitness[0], {'ngh': ngh}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
