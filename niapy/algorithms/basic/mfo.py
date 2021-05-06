# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['MothFlameOptimizer']


class MothFlameOptimizer(Algorithm):
    r"""MothFlameOptimizer of Moth flame optimizer.

    Algorithm:
        Moth flame optimizer

    Date:
        2018

    Author:
        Kivanc Guckiran and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Mirjalili, Seyedali. "Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm." Knowledge-Based Systems 89 (2015): 228-249.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.

    See Also:
        * :class:`niapy.algorithms.algorithm.Algorithm`

    """

    Name = ['MothFlameOptimizer', 'MFO']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Mirjalili, Seyedali. "Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm." Knowledge-Based Systems 89 (2015): 228-249."""

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of MothFlameOptimizer algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current population fitness/function values.
            best_x (numpy.ndarray): Current population best individual.
            best_fitness (float): Current best individual.
            **params (Dict[str, Any]): Additional parameters

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution.
                4. New global best fitness/objective value.
                5. Additional arguments:
                    * best_flames (numpy.ndarray): Best individuals.
                    * best_flame_fitness (numpy.ndarray): Best individuals fitness/function values.
                    * previous_population (numpy.ndarray): Previous population.
                    * previous_fitness (numpy.ndarray): Previous population fitness/function values.

        """
        # Previous positions
        # Create sorted population
        indexes = np.argsort(population_fitness)
        sorted_population = population[indexes]
        # Some parameters
        flame_no, a = round(self.population_size - (task.iters + 1) * ((self.population_size - 1) / task.max_iters)), -1 + (task.iters + 1) * (
                    (-1) / task.max_iters)
        for i in range(self.population_size):
            for j in range(task.dimension):
                distance_to_flame, b, t = abs(sorted_population[i, j] - population[i, j]), 1, (a - 1) * self.random() + 1
                if i <= flame_no:
                    population[i, j] = distance_to_flame * np.exp(b * t) * np.cos(2 * np.pi * t) + sorted_population[i, j]
                else:
                    population[i, j] = distance_to_flame * np.exp(b * t) * np.cos(2 * np.pi * t) + sorted_population[
                        flame_no, j]
        population = np.apply_along_axis(task.repair, 1, population, self.rng)
        population_fitness = np.apply_along_axis(task.eval, 1, population)
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
