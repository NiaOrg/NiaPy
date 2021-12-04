# encoding=utf8

import numpy as np

import random

from niapy.algorithms.algorithm import Algorithm

__all__ = ['SquirrelSearch']

class SquirrelSearch(Algorithm):
    r"""Implementation of Squirrel Search Algorithm.

    Algorithm:
        Squirrel Search Algorithm

    Date:
        2019

    Author:
        Mohit Jain, Vijander Singh, Asha Rani

    License:
        MIT

    Reference paper:
        * 2.	Jain, Mohit, et al. “A Novel Nature-Inspired Algorithm for Optimization: Squirrel Search Algorithm.” Swarm and Evolutionary Computation, vol. 44, 2019, pp. 148–175., doi:10.1016/j.swevo.2018.02.013.

    Attributes:
        Name (List[str]): List of strings representing algorithm names.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['SquirrelSearch', 'SSA']

    @staticmethod
    def info():
        r"""Get algorithm information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""No info"""

    def __init__(self, population_size=50, lower_limit=1, upper_limit=100, dimension=2, *args, **kwargs):
        """Initialize SSA.

        Args:
            population_size (Optional[int]): Population size.
            lower_limit (Optional[float]): lower limit of uniform dist.
            upper_limit (Optional[float]): upper_limit of uniform dist.
            dimension (Optional[float]): dimension of each location.

        See Also:
            :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.dimension = dimension

    def set_parameters(self, population_size=50, lower_limit=1, upper_limit=100, dimension=2, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            lower_limit (Optional[float]): lower limit of uniform dist.
            upper_limit (Optional[float]): upper_limit of uniform dist.
            dimension (Optional[float]): dimension of each location.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.dimension = dimension

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        d = super().get_parameters()
        d.update({
            'lower_limit': self.lower_limit,
            'upper_limit': self.upper_limit,
            'dimension': self.dimension
        })
        return d

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
                    * beta_a (numpy.ndarray): Beta_a of the pack (Second best solution)
                    * beta_a_fitness (float): Second best fitness.
                    * beta_b (numpy.ndarray): Beta_b of the pack (Third best solution)
                    * beta_b_fitness (float): Third best fitness.
                    * beta_c (numpy.ndarray): Beta_c of the pack (Fourth best solution)
                    * beta_c_fitness (float): Third best fitness.
        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`
        """
        pop, fpop, d = super().init_population(task)
        si = np.argsort(fpop)
        alpha = np.copy(pop[si[0]])
        alpha_fitness = fpop[si[0]]
        beta_a = np.copy(pop[si[1]])
        beta_a_fitness = fpop[si[1]]
        beta_b = np.copy(pop[si[2]])
        beta_b_fitness = fpop[si[2]]
        beta_c = np.copy(pop[si[3]])
        beta_c_fitness = fpop[si[3]]
        d.update({
            'alpha': alpha,
            'alpha_fitness': alpha_fitness,
            'beta_a': beta_a,
            'beta_a_fitness': beta_a_fitness,
            'beta_b': beta_b,
            'beta_b_fitness': beta_b_fitness,
            'beta_c': beta_c,
            'beta_c_fitness': beta_c_fitness
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
                dg: 0.8 (default gliding distance)
                Gc: 1.9 (default gliding constant)
        """
        alpha = params.pop('alpha')
        alpha_fitness = params.pop('alpha_fitness')
        beta_a = params.pop('beta_a')
        beta_a_fitness = params.pop('beta_a_fitness')
        beta_b = params.pop('beta_b')
        beta_b_fitness = params.pop('beta_b_fitness')
        beta_c = params.pop('beta_c')
        beta_c_fitness = params.pop('beta_c_fitness')

        a = 2 - task.evals * (2 / task.max_evals)
        for i, w in enumerate(population):
            if population_fitness[i]==beta_a_fitness:
                a1=beta_a+0.8*1.9*(alpha-beta_a)
                population[i] = task.repair(a1, rng=self.rng)
            elif population_fitness[i]==beta_b_fitness:
                a2=beta_b+0.8*1.9*(alpha-beta_b)
                population[i] = task.repair(a2, rng=self.rng)
            elif population_fitness[i]==beta_c_fitness:
                a3=beta_c+0.8*1.9*(alpha-beta_c)
                population[i] = task.repair(a3, rng=self.rng)
            else:
                k1 = random.randint(0, 1)
                k2 = random.randint(0, 1)
                k3 = random.randint(0, 1)
                a4=k1*(w+0.8*1.9*(alpha-w))+(1-k1)*(k2*(w+0.8*1.9*(beta_a-w))+(1-k2)*(k3*(w+0.8*1.9*(beta_b-w))+(1-k3)*(w+0.8*1.9*(beta_c-w))))
                population[i] = task.repair(a4, rng=self.rng)
            population_fitness[i] = task.eval(population[i])
        for i, f in enumerate(population_fitness):
            if f < alpha_fitness:
                alpha, alpha_fitness = population[i].copy(), f
            elif alpha_fitness < f < beta_a_fitness:
                beta_a, beta_a_fitness = population[i].copy(), f
            elif beta_a_fitness < f < beta_b_fitness:
                beta_b, beta_b_fitness = population[i].copy(), f
            elif beta_b_fitness < f < beta_c_fitness:
                beta_c, beta_c_fitness = population[i].copy(), f
        best_x, best_fitness = self.get_best(alpha, alpha_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {'alpha': alpha, 'alpha_fitness': alpha_fitness,
                                                                      'beta_a': beta_a, 'beta_a_fitness': beta_a_fitness,
                                                                      'beta_b': beta_b, 'beta_b_fitness': beta_b_fitness,
                                                                      'beta_c': beta_c, 'beta_c_fitness': beta_c_fitness}
