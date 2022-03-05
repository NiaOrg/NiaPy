# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm
from niapy.util.distances import euclidean

__all__ = ['FireflyAlgorithm']

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')


class FireflyAlgorithm(Algorithm):
    r"""Implementation of Firefly algorithm.

    Algorithm:
        Firefly algorithm

    Date:
        2016

    Authors:
        Iztok Fister Jr, Iztok Fister and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Fister, I., Fister Jr, I., Yang, X. S., & Brest, J. (2013).
        A comprehensive review of firefly algorithms. Swarm and Evolutionary Computation, 13, 34-46.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        alpha (float): Randomness strength.
        beta0 (float): Attractiveness constant.
        gamma (float): Absorption coefficient.
        theta (float): Randomness reduction factor.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['FireflyAlgorithm', 'FA']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Fister, I., Fister Jr, I., Yang, X. S., & Brest, J. (2013). A comprehensive review of firefly algorithms. Swarm and Evolutionary Computation, 13, 34-46."""

    def __init__(self, population_size=20, alpha=1, beta0=1, gamma=0.01, theta=0.97, *args, **kwargs):
        """Initialize FireflyAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            alpha (Optional[float]): Randomness strength 0--1 (highly random).
            beta0 (Optional[float]): Attractiveness constant.
            gamma (Optional[float]): Absorption coefficient.
            theta (Optional[float]): Randomness reduction factor.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.theta = theta

    def set_parameters(self, population_size=20, alpha=1, beta0=1, gamma=0.01, theta=0.97, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            alpha (Optional[float]): Randomness strength 0--1 (highly random).
            beta0 (Optional[float]): Attractiveness constant.
            gamma (Optional[float]): Absorption coefficient.
            theta (Optional[float]): Randomness reduction factor.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.theta = theta

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        params = super().get_parameters()
        params.update({
            'alpha': self.alpha,
            'beta0': self.beta0,
            'gamma': self.gamma,
            'theta': self.theta,
        })
        return params

    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * alpha (float): Randomness strength.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        fireflies, intensity, _ = super().init_population(task)
        return fireflies, intensity, {'alpha': self.alpha}

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Firefly Algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current population function/fitness values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individual fitness/function value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution
                4. New global best solutions fitness/objective value
                5. Additional arguments:
                    * alpha (float): Randomness strength.

        See Also:
            * :func:`niapy.algorithms.basic.FireflyAlgorithm.move_ffa`

        """
        alpha = params.pop('alpha') * self.theta

        for i in range(self.population_size):
            for j in range(self.population_size):
                if population_fitness[i] >= population_fitness[j]:
                    r = euclidean(population[i], population[j])
                    beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                    steps = alpha * (self.random(task.dimension) - 0.5) * task.range
                    population[i] += beta * (population[j] - population[i]) + steps
                    population[i] = task.repair(population[i])
                    population_fitness[i] = task.eval(population[i])
                    best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)

        return population, population_fitness, best_x, best_fitness, {'alpha': alpha}
