# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['ParameterFreeBatAlgorithm']


class ParameterFreeBatAlgorithm(Algorithm):
    r"""Implementation of Parameter-free Bat algorithm.

    Algorithm:
        Parameter-free Bat algorithm

    Date:
        2020

    Authors:
        Iztok Fister Jr.
        This implementation is based on the implementation of basic BA from niapy

    License:
        MIT

    Reference paper:
        Iztok Fister Jr., Iztok Fister, Xin-She Yang. Towards the development of a parameter-free bat algorithm . In: FISTER Jr., Iztok (Ed.), BRODNIK, Andrej (Ed.). StuCoSReC : proceedings of the 2015 2nd Student Computer Science Research Conference. Koper: University of Primorska, 2015, pp. 31-34.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['ParameterFreeBatAlgorithm', 'PLBA']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Iztok Fister Jr., Iztok Fister, Xin-She Yang. Towards the development of a parameter-free bat algorithm . In: FISTER, Iztok (Ed.), BRODNIK, Andrej (Ed.). StuCoSReC : proceedings of the 2015 2nd Student Computer Science Research Conference. Koper: University of Primorska, 2015, pp. 31-34."""

    def __init__(self, *args, **kwargs):
        """Initialize ParameterFreeBatAlgorithm."""
        super().__init__(80, *args, **kwargs)
        self.loudness = 0.9
        self.pulse_rate = 0.1

    def set_parameters(self, **kwargs):
        r"""Set the parameters of the algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=80, **kwargs)
        self.loudness = 0.9
        self.pulse_rate = 0.1

    def init_population(self, task):
        r"""Initialize the initial population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * velocities (numpy.ndarray[float]): Velocities

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        population, fitness, d = Algorithm.init_population(self, task)
        velocities = np.zeros((self.population_size, task.dimension))
        d.update({'velocities': velocities})
        return population, fitness, d

    def local_search(self, best, task, **_kwargs):
        r"""Improve the best solution according to the Yang (2010).

        Args:
            best (numpy.ndarray): Global best individual.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New solution based on global best individual.

        """
        return task.repair(best + 0.001 * self.normal(0, 1, task.dimension))

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Parameter-free Bat Algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population
            population_fitness (numpy.ndarray[float]): Current population fitness/function values
            best_x (numpy.ndarray): Current best individual
            best_fitness(float): Current best individual function/fitness value
            params (Dict[str, Any]): Additional algorithm arguments

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. New global best solution
                4. New global best fitness/objective value
                5. Additional arguments:
                    * velocities (numpy.ndarray): Velocities

        """
        velocities = params.pop('velocities')
        upper, lower = task.upper, task.lower
        for i in range(self.population_size):
            frequency = ((upper[0] - lower[0]) / float(self.population_size)) * self.normal(0, 1)
            velocities[i] += (population[i] - best_x) * frequency
            if self.random() > self.pulse_rate:
                solution = self.local_search(best=best_x, task=task, i=i, Sol=population)
            else:
                solution = task.repair(population[i] + velocities[i], rng=self.rng)
            new_fitness = task.eval(solution)
            if (new_fitness <= population_fitness[i]) and (self.random() < self.loudness):
                population[i], population_fitness[i] = solution, new_fitness
            if new_fitness <= best_fitness:
                best_x, best_fitness = solution.copy(), new_fitness
        return population, population_fitness, best_x, best_fitness, {'velocities': velocities}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
