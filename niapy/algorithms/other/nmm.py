# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['NelderMeadMethod']


class NelderMeadMethod(Algorithm):
    r"""Implementation of Nelder Mead method or downhill simplex method or amoeba method.

    Algorithm:
        Nelder Mead Method

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

    Attributes:
        Name (List[str]): list of strings representing algorithm name
        alpha (float): Reflection coefficient parameter
        gamma (float): Expansion coefficient parameter
        rho (float): Contraction coefficient parameter
        sigma (float): Shrink coefficient parameter

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['NelderMeadMethod', 'NMM']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""No info"""

    def __init__(self, population_size=None, alpha=0.1, gamma=0.3, rho=-0.2, sigma=-0.2, *args, **kwargs):
        """Initialize NelderMeadMethod.

        Args:
            population_size (Optional[int]): Number of individuals.
            alpha (Optional[float]): Reflection coefficient parameter
            gamma (Optional[float]): Expansion coefficient parameter
            rho (Optional[float]): Contraction coefficient parameter
            sigma (Optional[float]): Shrink coefficient parameter

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, initialization_function=kwargs.pop('initialization_function', self.init_pop), *args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

    def set_parameters(self, population_size=None, alpha=0.1, gamma=0.3, rho=-0.2, sigma=-0.2, **kwargs):
        r"""Set the arguments of an algorithm.

        Args:
            population_size (Optional[int]): Number of individuals.
            alpha (Optional[float]): Reflection coefficient parameter
            gamma (Optional[float]): Expansion coefficient parameter
            rho (Optional[float]): Contraction coefficient parameter
            sigma (Optional[float]): Shrink coefficient parameter

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, initialization_function=kwargs.pop('initialization_function', self.init_pop), **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

    def get_parameters(self):
        d = Algorithm.get_parameters(self)
        d.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'rho': self.rho,
            'sigma': self.sigma
        })
        return d

    def init_pop(self, task, population_size, **_kwargs):
        r"""Init starting population.

        Args:
            population_size (int): Number of individuals in population.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float]]:
                1. New initialized population.
                2. New initialized population fitness/function values.

        """
        population_size = task.dimension if population_size is None or population_size < task.dimension else population_size
        population = self.uniform(task.lower, task.upper, (population_size, task.dimension))
        population_fitness = np.apply_along_axis(task.eval, 1, population)
        return population, population_fitness

    def method(self, population, population_fitness, task):
        r"""Run the main function.

        Args:
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray[float]): Current population function/fitness values.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float]]:
                1. New population.
                2. New population fitness/function values.

        """
        x0 = np.sum(population[:-1], axis=0) / (len(population) - 1)
        xr = x0 + self.alpha * (x0 - population[-1])
        rs = task.eval(xr)
        if population_fitness[0] >= rs < population_fitness[-2]:
            population[-1], population_fitness[-1] = xr, rs
            return population, population_fitness
        if rs < population_fitness[0]:
            xe = x0 + self.gamma * (x0 - population[-1])
            re = task.eval(xe)
            if re < rs:
                population[-1], population_fitness[-1] = xe, re
            else:
                population[-1], population_fitness[-1] = xr, rs
            return population, population_fitness
        xc = x0 + self.rho * (x0 - population[-1])
        rc = task.eval(xc)
        if rc < population_fitness[-1]:
            population[-1], population_fitness[-1] = xc, rc
            return population, population_fitness
        new_population = population[0] + self.sigma * (population[1:] - population[0])
        new_population_fitness = np.apply_along_axis(task.eval, 1, new_population)
        population[1:], population_fitness[1:] = new_population, new_population_fitness
        return population, population_fitness

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core iteration function of NelderMeadMethod algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations fitness/function values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best function/fitness value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution
                4. New global best solutions fitness/objective value
                5. Additional arguments.

        """
        sorted_indices = np.argsort(population_fitness)
        population, population_fitness = population[sorted_indices], population_fitness[sorted_indices]
        population, population_fitness = self.method(population, population_fitness, task)
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
