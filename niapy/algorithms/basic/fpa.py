# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm
from niapy.util import levy_flight

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['FlowerPollinationAlgorithm']


class FlowerPollinationAlgorithm(Algorithm):
    r"""Implementation of Flower Pollination algorithm.

    Algorithm:
        Flower Pollination algorithm

    Date:
        2018

    Authors:
        Dusan Fister, Iztok Fister Jr. and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Yang, Xin-She. "Flower pollination algorithm for global optimization. International conference on unconventional computing and natural computation. Springer, Berlin, Heidelberg, 2012.

    References URL:
        Implementation is based on the following MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/45112-flower-pollination-algorithm?requestedDomain=true

    Attributes:
        Name (List[str]): List of strings representing algorithm names.
        p (float): Switch probability.
        beta (float): Beta parameter of the Levy distribution (Should be between 0.3 and 1.99).

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['FlowerPollinationAlgorithm', 'FPA']

    @staticmethod
    def info():
        r"""Get default information of algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Yang, Xin-She. "Flower pollination algorithm for global optimization. International conference on unconventional computing and natural computation. Springer, Berlin, Heidelberg, 2012."""

    def __init__(self, population_size=25, p=0.8, beta=1.5, *args, **kwargs):
        """Initialize FlowerPollinationAlgorithm.

        Args:
            population_size (int): Population size.
            p (float): Switch probability.
            beta (float): Beta parameter of the Levy distribution (Should be between 0.3 and 1.99).

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.p = p
        self.beta = beta

    def set_parameters(self, population_size=25, p=0.8, beta=1.5, **kwargs):
        r"""Set core parameters of FlowerPollinationAlgorithm algorithm.

        Args:
            population_size (int): Population size.
            p (float): Switch probability.
            beta (float): Beta parameter of the Levy distribution (Should be between 0.3 and 1.99).

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.p = p
        self.beta = beta

    def init_population(self, task):
        """Initialize population."""
        pop, fpop, d = super().init_population(task)
        d.update({'solutions': np.zeros((self.population_size, task.dimension))})
        return pop, fpop, d

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of FlowerPollinationAlgorithm algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current population fitness/function values.
            best_x (numpy.ndarray): Global best solution.
            best_fitness (float): Global best solution function/fitness value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New populations fitness/function values.
                3. New global best solution.
                4. New global best solution fitness/objective value.
                5. Additional arguments.

        """
        solutions = params.pop('solutions')

        for i in range(self.population_size):
            if self.random() > self.p:
                step_size = levy_flight(beta=self.beta, size=task.dimension, rng=self.rng)
                solutions[i] += step_size * (population[i] - best_x)
            else:
                j, k = self.rng.choice(self.population_size, size=2, replace=False)
                solutions[i] += self.random() * (population[j] - population[k])
            solutions[i] = task.repair(solutions[i], rng=self.rng)
            f_i = task.eval(solutions[i])
            if f_i <= population_fitness[i]:
                population[i], population_fitness[i] = solutions[i], f_i
            if f_i <= best_fitness:
                best_x, best_fitness = solutions[i].copy(), f_i
        return population, population_fitness, best_x, best_fitness, {'solutions': solutions}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
