# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm
from niapy.util.distances import euclidean

__all__ = ['GravitationalSearchAlgorithm']

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')


class GravitationalSearchAlgorithm(Algorithm):
    r"""Implementation of Gravitational Search Algorithm.

    Algorithm:
        Gravitational Search Algorithm

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://doi.org/10.1016/j.ins.2009.03.004

    Reference paper:
        Esmat Rashedi, Hossein Nezamabadi-pour, Saeid Saryazdi, GSA: A Gravitational Search Algorithm, Information Sciences, Volume 179, Issue 13, 2009, Pages 2232-2248, ISSN 0020-0255

    Attributes:
        Name (List[str]): List of strings representing algorithm name.

    See Also:
        * :class:`niapy.algorithms.algorithm.Algorithm`

    """

    Name = ['GravitationalSearchAlgorithm', 'GSA']

    @staticmethod
    def info():
        r"""Get algorithm information.

        Returns:
            str: Algorithm information.

        """
        return r"""Esmat Rashedi, Hossein Nezamabadi-pour, Saeid Saryazdi, GSA: A Gravitational Search Algorithm, Information Sciences, Volume 179, Issue 13, 2009, Pages 2232-2248, ISSN 0020-0255"""

    def __init__(self, population_size=40, g0=2.467, epsilon=1e-17, *args, **kwargs):
        """Initialize GravitationalSearchAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            g0 (Optional[float]): Starting gravitational constant.
            epsilon (Optional[float]): Small number.

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.g0 = g0
        self.epsilon = epsilon

    def set_parameters(self, population_size=40, g0=2.467, epsilon=1e-17, **kwargs):
        r"""Set the algorithm parameters.

        Args:
            population_size (Optional[int]): Population size.
            g0 (Optional[float]): Starting gravitational constant.
            epsilon (Optional[float]): Small number.

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.g0 = g0
        self.epsilon = epsilon

    def get_parameters(self):
        r"""Get algorithm parameters values.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.get_parameters`

        """
        d = super().get_parameters()
        d.update({
            'g0': self.g0,
            'epsilon': self.epsilon
        })
        return d

    def gravity(self, t):
        r"""Get new gravitational constant.

        Args:
            t (int): Time (Current iteration).

        Returns:
            float: New gravitational constant.

        """
        return self.g0 / t

    def init_population(self, task):
        r"""Initialize staring population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations fitness/function values.
                3. Additional arguments:
                    * velocities (numpy.ndarray[float]): Velocities

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.init_population`

        """
        population, fitness, _ = super().init_population(task)
        velocities = np.zeros((self.population_size, task.dimension))
        return population, fitness, {'velocities': velocities}

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of GravitationalSearchAlgorithm algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations fitness/function values.
            best_x (numpy.ndarray): Global best solution.
            best_fitness (float): Global best fitness/function value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New populations fitness/function values.
                3. New global best solution
                4. New global best solutions fitness/objective value
                5. Additional arguments:
                    * velocities (numpy.ndarray): Velocities.

        """
        velocities = params.pop('velocities')

        ib, iw = np.argmin(population_fitness), np.argmax(population_fitness)
        m = (population_fitness - population_fitness[iw]) / (population_fitness[ib] - population_fitness[iw])
        m = m / np.sum(m)
        forces = np.asarray([[self.gravity((task.iters + 1)) * ((m[i] * m[j]) / (euclidean(population[i], population[j]) + self.epsilon)) * (
                population[j] - population[i]) for j in range(len(m))] for i in range(len(m))])
        total_force = np.sum(self.random((self.population_size, task.dimension)) * forces, axis=1)
        a = total_force.T / (m + self.epsilon)
        velocities = self.random((self.population_size, task.dimension)) * velocities + a.T
        population = np.apply_along_axis(task.repair, 1, population + velocities, self.rng)
        population_fitness = np.apply_along_axis(task.eval, 1, population)
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {'velocities': velocities}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
