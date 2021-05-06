# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic.SineCosineAlgorithm')
logger.setLevel('INFO')

__all__ = ['SineCosineAlgorithm']


class SineCosineAlgorithm(Algorithm):
    r"""Implementation of sine cosine algorithm.

    Algorithm:
        Sine Cosine Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://www.sciencedirect.com/science/article/pii/S0950705115005043

    Reference paper:
        Seyedali Mirjalili, SCA: A Sine Cosine Algorithm for solving optimization problems, Knowledge-Based Systems, Volume 96, 2016, Pages 120-133, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2015.12.022.

    Attributes:
        Name (List[str]): List of string representing algorithm names.
        a (float): Parameter for control in :math:`r_1` value
        r_min (float): Minimum value for :math:`r_3` value
        r_max (float): Maximum value for :math:`r_3` value

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['SineCosineAlgorithm', 'SCA']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Seyedali Mirjalili, SCA: A Sine Cosine Algorithm for solving optimization problems, Knowledge-Based Systems, Volume 96, 2016, Pages 120-133, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2015.12.022."""

    def __init__(self, population_size=25, a=3, r_min=0, r_max=2, *args, **kwargs):
        """Initialize SineCosineAlgorithm.

        Args:
            population_size (Optional[int]): Number of individual in population
            a (Optional[float]): Parameter for control in :math:`r_1` value
            r_min (Optional[float]): Minimum value for :math:`r_3` value
            r_max (Optional[float]): Maximum value for :math:`r_3` value

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.a = a
        self.r_min = r_min
        self.r_max = r_max

    def set_parameters(self, population_size=25, a=3, r_min=0, r_max=2, **kwargs):
        r"""Set the arguments of an algorithm.

        Args:
            population_size (Optional[int]): Number of individual in population
            a (Optional[float]): Parameter for control in :math:`r_1` value
            r_min (Optional[float]): Minimum value for :math:`r_3` value
            r_max (Optional[float]): Maximum value for :math:`r_3` value

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.a = a
        self.r_min = r_min
        self.r_max = r_max

    def get_parameters(self):
        r"""Get algorithm parameters values.

        Returns:
            Dict[str, Any]:

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.get_parameters`

        """
        d = super().get_parameters()
        d.update({
            'a': self.a,
            'r_min': self.r_min,
            'r_max': self.r_max
        })
        return d

    def next_position(self, x, best_x, r1, r2, r3, r4, task):
        r"""Move individual to new position in search space.

        Args:
            x (numpy.ndarray): Individual represented with components.
            best_x (numpy.ndarray): Best individual represented with components.
            r1 (float): Number dependent on algorithm iteration/generations.
            r2 (float): Random number in range of 0 and 2 * PI.
            r3 (float): Random number in range [r_min, r_max].
            r4 (float): Random number in range [0, 1].
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New individual that is moved based on individual ``x``.

        """
        return task.repair(x + r1 * (np.sin(r2) if r4 < 0.5 else np.cos(r2)) * np.fabs(r3 * best_x - x), rng=self.rng)

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Sine Cosine Algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population individuals.
            population_fitness (numpy.ndarray[float]): Current population individuals function/fitness values.
            best_x (numpy.ndarray): Current best solution to optimization task.
            best_fitness (float): Current best function/fitness value.
            params (Dict[str, Any]): Additional parameters.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New populations fitness/function values.
                3. New global best solution.
                4. New global best fitness/objective value.
                5. Additional arguments.

        """
        r1 = self.a - (task.iters + 1) * (self.a / (task.iters + 1))
        r2 = self.uniform(0, 2 * np.pi)
        r3 = self.uniform(self.r_min, self.r_max)
        r4 = self.random()
        population = np.apply_along_axis(self.next_position, 1, population, best_x, r1, r2, r3, r4, task)
        population_fitness = np.apply_along_axis(task.eval, 1, population)
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
