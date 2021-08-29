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
        pa (float): Probability of a nest being abandoned.

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

    def __init__(self, population_size=25, pa=0.25, *args, **kwargs):
        r"""Initialize CuckooSearch.

        Args:
            population_size (int): Population size.
            pa (float): Probability of a nest being abandoned.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.pa = pa

    def set_parameters(self, population_size=50, pa=0.2, **kwargs):
        r"""Set the arguments of an algorithm.

        Args:
            population_size (int): Population size.
            pa (float): Probability of a nest being abandoned.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.pa = pa

    def get_parameters(self):
        """Get parameters of the algorithm."""
        d = super().get_parameters()
        d.update({
            'pa': self.pa,
        })
        return d

    def get_cuckoos(self, population, best_x, task):
        step_size = levy_flight(self.rng, size=population.shape) * (population - best_x)
        new_population = population + step_size * self.standard_normal(population.shape)
        return task.repair(new_population, rng=self.rng)

    def empty_nests(self, population, task):
        abandoned = self.random(population.shape) > self.pa
        i = self.rng.permutation(self.population_size)
        j = self.rng.permutation(self.population_size)
        step_size = self.random() * (population[i] - population[j])
        return task.repair(population + step_size * abandoned, rng=self.rng)

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
        new_nests = self.get_cuckoos(population, best_x, task)
        new_fitness = np.apply_along_axis(task.eval, 1, new_nests)

        replace = new_fitness < population_fitness
        population[replace] = new_nests[replace]
        population_fitness[replace] = new_fitness[replace]
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)

        new_nests = self.empty_nests(population, task)
        new_fitness = np.apply_along_axis(task.eval, 1, new_nests)

        replace = new_fitness < population_fitness
        population[replace] = new_nests[replace]
        population_fitness[replace] = new_fitness[replace]
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)

        return population, population_fitness, best_x, best_fitness, {}
