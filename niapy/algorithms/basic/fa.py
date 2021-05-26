# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

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
        alpha (float): Step size.
        beta_min (float): Minimum value for beta.
        gamma (float): Absorption coefficient.

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

    def __init__(self, population_size=20, alpha=1, beta_min=1, gamma=2, *args, **kwargs):
        """Initialize FireflyAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            alpha (Optional[float]): Step size.
            beta_min (Optional[float]): Minimum value of beta.
            gamma (Optional[float]): Absorption coefficient.
            kwargs (Dict[str, Any]): Additional arguments.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma = gamma

    def set_parameters(self, population_size=20, alpha=1, beta_min=1, gamma=2, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            alpha (Optional[float]): Step size.
            beta_min (Optional[float]): Minimum value of beta.
            gamma (Optional[float]): Absorption coefficient.
            kwargs (Dict[str, Any]): Additional arguments.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        Algorithm.set_parameters(self, population_size=population_size, **kwargs)
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma = gamma

    @staticmethod
    def alpha_new(a, alpha):
        r"""Optionally recalculate the new alpha value.

        Args:
            a (float):
            alpha (float):

        Returns:
            float: New value of parameter alpha.

        """
        delta = 1.0 - pow(pow(10.0, -4.0) / 0.9, 1.0 / float(a))
        return (1 - delta) * alpha

    def move_ffa(self, i, fireflies, intensity, o_fireflies, alpha, task):
        r"""Move fireflies.

        Args:
            i (int): Index of current individual.
            fireflies (numpy.ndarray):
            intensity (numpy.ndarray):
            o_fireflies (numpy.ndarray):
            alpha (float):
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, bool]:
                1. New individual
                2. ``True`` if individual was moved, ``False`` if individual was not moved

        """
        moved = False
        for j in range(self.population_size):
            r = np.sum((fireflies[i] - fireflies[j]) ** 2) ** (1 / 2)
            if intensity[i] <= intensity[j]:
                continue
            beta = (1.0 - self.beta_min) * np.exp(-self.gamma * r ** 2.0) + self.beta_min
            tmp_f = alpha * (self.random(task.dimension) - 0.5) * task.range
            fireflies[i] = task.repair(fireflies[i] * (1.0 - beta) + o_fireflies[j] * beta + tmp_f, rng=self.rng)
            moved = True
        return fireflies[i], moved

    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * alpha (float): Step size.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        fireflies, intensity, _ = Algorithm.init_population(self, task)
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
                    * alpha (float): Step size.

        See Also:
            * :func:`niapy.algorithms.basic.FireflyAlgorithm.move_ffa`

        """
        alpha = params.pop('alpha')

        alpha = self.alpha_new(task.max_evals / self.population_size, alpha)
        sorted_index = np.argsort(population_fitness)
        tmp = [self.move_ffa(i, population[sorted_index], population_fitness[sorted_index], population, alpha, task)
               for i in range(self.population_size)]
        population = np.asarray([tmp[i][0] for i in range(len(tmp))])
        moved = np.asarray([tmp[i][1] for i in range(len(tmp))])
        population_fitness[np.where(moved)] = np.apply_along_axis(task.eval, 1, population[np.where(moved)])
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {'alpha': alpha}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
