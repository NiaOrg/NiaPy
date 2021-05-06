# encoding=utf8

import numpy as np

from niapy.algorithms.algorithm import Algorithm

__all__ = ['GreyWolfOptimizer']


class GreyWolfOptimizer(Algorithm):
    r"""Implementation of Grey wolf optimizer.

    Algorithm:
        Grey wolf optimizer

    Date:
        2018

    Author:
        Iztok Fister Jr. and Klemen Berkovič

    License:
        MIT

    Reference paper:
        * Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis. "Grey wolf optimizer." Advances in engineering software 69 (2014): 46-61.
        * Grey Wolf Optimizer (GWO) source code version 1.0 (MATLAB) from MathWorks

    Attributes:
        Name (List[str]): List of strings representing algorithm names.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['GreyWolfOptimizer', 'GWO']

    @staticmethod
    def info():
        r"""Get algorithm information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis. "Grey wolf optimizer." Advances in engineering software 69 (2014): 46-61."""

    def init_population(self, task):
        r"""Initialize population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations fitness/function values.
                3. Additional arguments:
                    * alpha (numpy.ndarray): Alpha of the pack (Best solution)
                    * alpha_fitness (float): Best fitness.
                    * beta (numpy.ndarray): Beta of the pack (Second best solution)
                    * beta_fitness (float): Second best fitness.
                    * delta (numpy.ndarray): Delta of the pack (Third best solution)
                    * delta_fitness (float): Third best fitness.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        pop, fpop, d = super().init_population(task)
        si = np.argsort(fpop)
        alpha = np.copy(pop[si[0]])
        alpha_fitness = fpop[si[0]]
        beta = np.copy(pop[si[1]])
        beta_fitness = fpop[si[1]]
        delta = np.copy(pop[si[2]])
        delta_fitness = fpop[si[2]]
        d.update({
            'alpha': alpha,
            'alpha_fitness': alpha_fitness,
            'beta': beta,
            'beta_fitness': beta_fitness,
            'delta': delta,
            'delta_fitness': delta_fitness
        })
        return pop, fpop, d

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of GreyWolfOptimizer algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations function/fitness values.
            best_x (numpy.ndarray):
            best_fitness (float):
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. Additional arguments:
                    * alpha (numpy.ndarray): Alpha of the pack (Best solution)
                    * alpha_fitness (float): Best fitness.
                    * beta (numpy.ndarray): Beta of the pack (Second best solution)
                    * beta_fitness (float): Second best fitness.
                    * delta (numpy.ndarray): Delta of the pack (Third best solution)
                    * delta_fitness (float): Third best fitness.

        """
        alpha = params.pop('alpha')
        alpha_fitness = params.pop('alpha_fitness')
        beta = params.pop('beta')
        beta_fitness = params.pop('beta_fitness')
        delta = params.pop('delta')
        delta_fitness = params.pop('delta_fitness')

        a = 2 - task.evals * (2 / task.max_evals)
        for i, w in enumerate(population):
            a1, c1 = 2 * a * self.random(task.dimension) - a, 2 * self.random(task.dimension)
            x1 = alpha - a1 * np.fabs(c1 * alpha - w)
            a2, c2 = 2 * a * self.random(task.dimension) - a, 2 * self.random(task.dimension)
            x2 = beta - a2 * np.fabs(c2 * beta - w)
            a3, c3 = 2 * a * self.random(task.dimension) - a, 2 * self.random(task.dimension)
            x3 = delta - a3 * np.fabs(c3 * delta - w)
            population[i] = task.repair((x1 + x2 + x3) / 3, rng=self.rng)
            population_fitness[i] = task.eval(population[i])
        for i, f in enumerate(population_fitness):
            if f < alpha_fitness:
                alpha, alpha_fitness = population[i].copy(), f
            elif alpha_fitness < f < beta_fitness:
                beta, beta_fitness = population[i].copy(), f
            elif beta_fitness < f < delta_fitness:
                delta, delta_fitness = population[i].copy(), f
        best_x, best_fitness = self.get_best(alpha, alpha_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {'alpha': alpha, 'alpha_fitness': alpha_fitness,
                                                                      'beta': beta, 'beta_fitness': beta_fitness,
                                                                      'delta': delta, 'delta_fitness': delta_fitness}
