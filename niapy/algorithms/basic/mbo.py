# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

__all__ = ['MonarchButterflyOptimization']

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')


class MonarchButterflyOptimization(Algorithm):
    r"""Implementation of Monarch Butterfly Optimization.

    Algorithm:
        Monarch Butterfly Optimization

    Date:
        2019

    Authors:
        Jan Banko

    License:
        MIT

    Reference paper:
        Wang, G. G., Deb, S., & Cui, Z. (2019). Monarch butterfly optimization. Neural computing and applications, 31(7), 1995-2014.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        PAR (float): Partition.
        PER (float): Period.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['MonarchButterflyOptimization', 'MBO']

    @staticmethod
    def info():
        r"""Get information of the algorithm.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.info`

        """
        return r"""Description: Monarch butterfly optimization algorithm is inspired by the migration behaviour of the monarch butterflies in nature.
        Authors: Wang, Gai-Ge & Deb, Suash & Cui, Zhihua.
        Year: 2015
        Main reference: Wang, G. G., Deb, S., & Cui, Z. (2019). Monarch butterfly optimization. Neural computing and applications, 31(7), 1995-2014."""

    def __init__(self, population_size=20, partition=5.0 / 12.0, period=1.2, *args, **kwargs):
        """Initialize MonarchButterflyOptimization.

        Args:
            population_size (Optional[int]): Population size.
            partition (Optional[int]): Partition.
            period (Optional[int]): Period.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.partition = partition
        self.period = period
        self.keep = 2
        self.bar = partition
        self.np1 = int(np.ceil(partition * population_size))
        self.np2 = population_size - self.np1

    def set_parameters(self, population_size=20, partition=5.0 / 12.0, period=1.2, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            partition (Optional[int]): Partition.
            period (Optional[int]): Period.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.partition = partition
        self.period = period
        self.keep = 2
        self.bar = partition
        self.np1 = int(np.ceil(partition * population_size))
        self.np2 = population_size - self.np1

    def get_parameters(self):
        r"""Get parameters values for the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        d = Algorithm.get_parameters(self)
        d.update({
            'partition': self.partition,
            'period': self.period,
            'keep': self.keep,
            'bar': self.bar,
            'np1': self.np1,
            'np2': self.np2
        })
        return d

    def levy(self, _step_size, dimension):
        r"""Calculate levy flight.

        Args:
            _step_size (float): Size of the walk step.
            dimension (int): Number of dimensions.

        Returns:
            numpy.ndarray: Calculated values for levy flight.

        """
        delta_x = np.array([np.sum(np.tan(np.pi * self.uniform(0.0, 1.0, 10))) for _ in range(dimension)])
        return delta_x

    def migration_operator(self, dimension, np1, np2, butterflies):
        r"""Apply the migration operator.

        Args:
            dimension (int): Number of dimensions.
            np1 (int): Number of butterflies in Land 1.
            np2 (int): Number of butterflies in Land 2.
            butterflies (numpy.ndarray): Current butterfly population.

        Returns:
            numpy.ndarray: Adjusted butterfly population.

        """
        pop1 = np.copy(butterflies[:np1])
        pop2 = np.copy(butterflies[np1:])
        for k1 in range(np1):
            for i in range(dimension):
                r1 = self.random() * self.period
                if r1 <= self.partition:
                    r2 = self.integers(np1 - 1)
                    butterflies[k1, i] = pop1[r2, i]
                else:
                    r3 = self.integers(np2 - 1)
                    butterflies[k1, i] = pop2[r3, i]
        return butterflies

    def adjusting_operator(self, t, max_t, dimension, np1, np2, butterflies, best):
        r"""Apply the adjusting operator.

        Args:
            t (int): Current generation.
            max_t (int): Maximum generation.
            dimension (int): Number of dimensions.
            np1 (int): Number of butterflies in Land 1.
            np2 (int): Number of butterflies in Land 2.
            butterflies (numpy.ndarray): Current butterfly population.
            best (numpy.ndarray): The best butterfly currently.

        Returns:
            numpy.ndarray: Adjusted butterfly population.

        """
        pop2 = np.copy(butterflies[np1:])
        for k2 in range(np1, np1 + np2):
            scale = 1.0 / ((t + 1) ** 2)
            step_size = np.ceil(self.rng.exponential(2 * max_t))
            delta_x = self.levy(step_size, dimension)
            for i in range(dimension):
                if self.uniform(0.0, 1.0) >= self.partition:
                    butterflies[k2, i] = best[i]
                else:
                    r4 = self.integers(np2 - 1)
                    butterflies[k2, i] = pop2[r4, 1]
                    if self.uniform(0.0, 1.0) > self.bar:
                        butterflies[k2, i] += scale * (delta_x[i] - 0.5)
        return butterflies

    @staticmethod
    def evaluate_and_sort(task, butterflies):
        r"""Evaluate and sort the butterfly population.

        Args:
            task (Task): Optimization task
            butterflies (numpy.ndarray): Current butterfly population.

        Returns:
            numpy.ndarray: Tuple[numpy.ndarray, float, numpy.ndarray]:
                1. Best butterfly according to the evaluation.
                2. The best fitness value.
                3. Butterfly population.

        """
        fitness = np.apply_along_axis(task.eval, 1, butterflies)
        indices = np.argsort(fitness)
        butterflies = butterflies[indices]
        fitness = fitness[indices]

        return fitness, butterflies

    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * current_best (numpy.ndarray): Current generation's best individual.

        See Also:
             * :func:`niapy.algorithms.Algorithm.init_population`

        """
        population, fitness, _ = super().init_population(task)
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        return population, fitness, {'current_best': population[0]}

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Forest Optimization Algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray[float]): Current population function/fitness values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individual fitness/function value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * current_best (numpy.ndarray): Current generation's best individual.

        """
        current_best = params.pop('current_best')

        elite = np.copy(population[:self.keep])
        max_t = task.max_iters if not np.isinf(task.max_iters) else task.max_evals / self.population_size
        population = np.apply_along_axis(task.repair, 1,
                                         self.migration_operator(task.dimension, self.np1, self.np2, population))
        population = np.apply_along_axis(task.repair, 1,
                                         self.adjusting_operator(task.iters, max_t, task.dimension, self.np1, self.np2,
                                                                 population, current_best))
        population_fitness, population = self.evaluate_and_sort(task, population)
        current_best = population[0]
        population[-self.keep:] = elite
        population_fitness, population = self.evaluate_and_sort(task, population)
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {'current_best': current_best}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
