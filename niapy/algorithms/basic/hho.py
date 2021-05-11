# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm
from niapy.util import levy_flight

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['HarrisHawksOptimization']


class HarrisHawksOptimization(Algorithm):
    r"""Implementation of Harris Hawks Optimization algorithm.

    Algorithm:
        Harris Hawks Optimization

    Date:
        2020

    Authors:
        Francisco Jose Solis-Munoz

    License:
        MIT

    Reference paper:
        Heidari et al. "Harris hawks optimization: Algorithm and applications". Future Generation Computer Systems. 2019. Vol. 97. 849-872.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        levy (float): Levy factor.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['HarrisHawksOptimization', 'HHO']

    def __init__(self, population_size=40, levy=0.01, *args, **kwargs):
        """Initialize HarrisHawksOptimization.

        Args:
            population_size (Optional[int]): Population size.
            levy (Optional[float]): Levy factor.

        """
        super().__init__(population_size, *args, **kwargs)
        self.levy = levy

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Heidari et al. "Harris hawks optimization: Algorithm and applications". Future Generation Computer Systems. 2019. Vol. 97. 849-872."""

    def set_parameters(self, population_size=40, levy=0.01, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            levy (Optional[float]): Levy factor.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.levy = levy

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        d = super().get_parameters()
        d.update({
            'levy': self.levy
        })
        return d

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Harris Hawks Optimization.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population
            population_fitness (numpy.ndarray[float]): Current population fitness/function values
            best_x (numpy.ndarray): Current best individual
            best_fitness (float): Current best individual function/fitness value
            params (Dict[str, Any]): Additional algorithm arguments

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. New global best solution
                4. New global best fitness/objective value

        """
        # Decreasing energy factor
        decreasing_energy_factor = 2 * (1 - (task.iters + 1) / task.max_iters)
        mean_sol = np.mean(population)
        # Update population
        for i in range(self.population_size):
            jumping_energy = self.rng.uniform(0, 2)
            decreasing_energy_random = self.rng.uniform(-1, 1)
            escaping_energy = decreasing_energy_factor * decreasing_energy_random
            escaping_energy_abs = np.abs(escaping_energy)
            random_number = self.rng.random()
            if escaping_energy >= 1 and random_number >= 0.5:
                # 0. Exploration: Random tall tree
                rhi = self.rng.integers(self.population_size)
                random_agent = population[rhi]
                population[i] = random_agent - self.rng.random() * np.abs(random_agent - 2 * self.rng.random() * population[i])
            elif escaping_energy_abs >= 1 and random_number < 0.5:
                # 1. Exploration: Family members mean
                population[i] = (best_x - mean_sol) - self.rng.random() * self.rng.uniform(task.lower, task.upper)
            elif escaping_energy_abs >= 0.5 and random_number >= 0.5:
                # 2. Exploitation: Soft besiege
                population[i] = \
                    (best_x - population[i]) - \
                    escaping_energy * \
                    np.abs(jumping_energy * best_x - population[i])
            elif escaping_energy_abs < 0.5 <= random_number:
                # 3. Exploitation: Hard besiege
                population[i] = \
                    best_x - \
                    escaping_energy * \
                    np.abs(best_x - population[i])
            elif escaping_energy_abs >= 0.5 > random_number:
                # 4. Exploitation: Soft besiege with progressive rapid dives
                cand1 = task.repair(best_x - escaping_energy * np.abs(jumping_energy * best_x - population[i]), rng=self.rng)
                random_vector = self.rng.random(task.dimension)
                cand2 = task.repair(cand1 + random_vector * levy_flight(alpha=self.levy, size=task.dimension, rng=self.rng),
                                    rng=self.rng)
                if task.eval(cand1) < population_fitness[i]:
                    population[i] = cand1
                elif task.eval(cand2) < population_fitness[i]:
                    population[i] = cand2
            elif escaping_energy_abs < 0.5 and random_number < 0.5:
                # 5. Exploitation: Hard besiege with progressive rapid dives
                cand1 = task.repair(best_x - escaping_energy * np.abs(jumping_energy * best_x - mean_sol), rng=self.rng)
                random_vector = self.rng.random(task.dimension)
                cand2 = task.repair(cand1 + random_vector * levy_flight(alpha=self.levy, size=task.dimension, rng=self.rng),
                                    rng=self.rng)
                if task.eval(cand1) < population_fitness[i]:
                    population[i] = cand1
                elif task.eval(cand2) < population_fitness[i]:
                    population[i] = cand2
            # Repair agent (from population) values
            population[i] = task.repair(population[i], rng=self.rng)
            # Eval population
            population_fitness[i] = task.eval(population[i])
        # Get best of population
        best_index = np.argmin(population_fitness)
        xb_cand = population[best_index].copy()
        fxb_cand = population_fitness[best_index].copy()
        if fxb_cand < best_fitness:
            best_fitness = fxb_cand
            best_x = xb_cand.copy()
        return population, population_fitness, best_x, best_fitness, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
