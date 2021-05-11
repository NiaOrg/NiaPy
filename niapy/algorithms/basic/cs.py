# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm
from niapy.util import levy_flight

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CuckooSearch']


class CuckooSearch(Algorithm):
    r"""Implementation of Cuckoo behaviour and levy flights.

    Algorithm:
        Cuckoo Search

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference:
        Yang, Xin-She, and Suash Deb. "Cuckoo search via Lévy flights."
        Nature & Biologically Inspired Computing, 2009. NaBIC 2009. World Congress on. IEEE, 2009.

    Attributes:
        Name (List[str]): list of strings representing algorithm names.
        population_size (int): Population size.
        pa (float): Proportion of worst nests.
        alpha (float): Scale factor for levy flight.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['CuckooSearch', 'CS']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Yang, Xin-She, and Suash Deb. "Cuckoo search via Lévy flights."
        Nature & Biologically Inspired Computing, 2009. NaBIC 2009. World Congress on. IEEE, 2009."""

    def __init__(self, population_size=50, pa=0.2, alpha=0.5, *args, **kwargs):
        r"""Initialize CuckooSearch.

        Args:
            population_size (int): Population size :math:`\in [1, \infty)`
            pa (float): factor :math:`\in [0, 1]`
            alpha (float): Levy flight scale factor.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.pa = pa
        self.num_abandoned = int(pa * self.population_size)
        self.alpha = alpha

    def set_parameters(self, population_size=50, pa=0.2, alpha=0.5, **kwargs):
        r"""Set the arguments of an algorithm.

        Args:
            population_size (int): Population size :math:`\in [1, \infty)`
            pa (float): factor :math:`\in [0, 1]`
            alpha (float): Levy flight scale factor.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.pa = pa
        self.num_abandoned = int(pa * self.population_size)
        self.alpha = alpha

    def get_parameters(self):
        """Get parameters of the algorithm."""
        d = super().get_parameters()
        d.update({
            'pa': self.pa,
            'alpha': self.alpha
        })
        return d

    def abandon_nests(self, pop, fpop, task):
        r"""Abandon nests.

        Args:
            pop (numpy.ndarray): Current population
            fpop (numpy.ndarray[float]): Current population fitness/funcion values
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float]]:
                1. New population
                2. New population fitness/function values

        """
        si = np.argsort(fpop)[:int(self.num_abandoned):-1]
        pop[si] = task.lower + self.random(task.dimension) * task.range
        fpop[si] = np.apply_along_axis(task.eval, 1, pop[si])
        return pop, fpop

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of CuckooSearch algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations fitness/function values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individual function/fitness values.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations fitness/function values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments.

        """
        i = self.integers(self.population_size)
        new_nests = task.repair(population[i] + levy_flight(alpha=self.alpha, size=task.dimension, rng=self.rng),
                                rng=self.rng)
        new_nests_fitness = task.eval(new_nests)
        j = self.integers(self.population_size)
        while i == j:
            j = self.integers(self.population_size)
        if new_nests_fitness <= population_fitness[j]:
            population[j] = new_nests
            population_fitness[j] = new_nests_fitness
        population, population_fitness = self.abandon_nests(population, population_fitness, task)
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
