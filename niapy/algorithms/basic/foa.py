# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

__all__ = ['ForestOptimizationAlgorithm']

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')


class ForestOptimizationAlgorithm(Algorithm):
    r"""Implementation of Forest Optimization Algorithm.

    Algorithm:
        Forest Optimization Algorithm

    Date:
        2019

    Authors:
        Luka Pečnik

    License:
        MIT

    Reference paper:
        Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi, Forest Optimization Algorithm, Expert Systems with Applications, Volume 41, Issue 15, 2014, Pages 6676-6687, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2014.05.009.

    References URL:
        Implementation is based on the following MATLAB code: https://github.com/cominsys/FOA

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        lifetime (int): Life time of trees parameter.
        area_limit (int): Area limit parameter.
        local_seeding_changes (int): Local seeding changes parameter.
        global_seeding_changes (int): Global seeding changes parameter.
        transfer_rate (float): Transfer rate parameter.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['ForestOptimizationAlgorithm', 'FOA']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi, Forest Optimization Algorithm, Expert Systems with Applications, Volume 41, Issue 15, 2014, Pages 6676-6687, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2014.05.009."""

    def __init__(self, population_size=10, lifetime=3, area_limit=10, local_seeding_changes=1, global_seeding_changes=1,
                 transfer_rate=0.3, *args, **kwargs):
        """Initialize ForestOptimizationAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            lifetime (Optional[int]): Life time parameter.
            area_limit (Optional[int]): Area limit parameter.
            local_seeding_changes (Optional[int]): Local seeding changes parameter.
            global_seeding_changes (Optional[int]): Global seeding changes parameter.
            transfer_rate (Optional[float]): Transfer rate parameter.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.lifetime = lifetime
        self.area_limit = area_limit
        self.local_seeding_changes = local_seeding_changes
        self.global_seeding_changes = global_seeding_changes
        self.transfer_rate = transfer_rate
        self.dx = None

    def set_parameters(self, population_size=10, lifetime=3, area_limit=10, local_seeding_changes=1,
                       global_seeding_changes=1, transfer_rate=0.3, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            lifetime (Optional[int]): Life time parameter.
            area_limit (Optional[int]): Area limit parameter.
            local_seeding_changes (Optional[int]): Local seeding changes parameter.
            global_seeding_changes (Optional[int]): Global seeding changes parameter.
            transfer_rate (Optional[float]): Transfer rate parameter.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.lifetime = lifetime
        self.area_limit = area_limit
        self.local_seeding_changes = local_seeding_changes
        self.global_seeding_changes = global_seeding_changes
        self.transfer_rate = transfer_rate
        self.dx = None

    def get_parameters(self):
        r"""Get parameters values of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        d = Algorithm.get_parameters(self)
        d.update({
            'lifetime': self.lifetime,
            'area_limit': self.area_limit,
            'local_seeding_changes': self.local_seeding_changes,
            'global_seeding_changes': self.global_seeding_changes,
            'transfer_rate': self.transfer_rate
        })
        return d

    def local_seeding(self, task, trees):
        r"""Local optimum search stage.

        Args:
            task (Task): Optimization task.
            trees (numpy.ndarray): Zero age trees for local seeding.

        Returns:
            numpy.ndarray: Resulting zero age trees.

        """
        seeds = np.repeat(trees, self.local_seeding_changes, axis=0)
        for i in range(seeds.shape[0]):
            indices = self.rng.choice(task.dimension, self.local_seeding_changes, replace=False)
            seeds[i, indices] += self.uniform(-self.dx[indices], self.dx[indices])
            seeds[i] = task.repair(seeds[i], rng=self.rng)
        return seeds

    def global_seeding(self, task, candidates, size):
        r"""Global optimum search stage that should prevent getting stuck in a local optimum.

        Args:
            task (Task): Optimization task.
            candidates (numpy.ndarray): Candidate population for global seeding.
            size (int): Number of trees to produce.

        Returns:
            numpy.ndarray: Resulting trees.

        """
        seeds = candidates[self.rng.choice(len(candidates), size, replace=False)]
        for i in range(seeds.shape[0]):
            indices = self.rng.choice(task.dimension, self.global_seeding_changes, replace=False)
            seeds[i, indices] = self.uniform(task.lower[indices], task.upper[indices])
        return seeds

    def remove_lifetime_exceeded(self, trees, age):
        r"""Remove dead trees.

        Args:
            trees (numpy.ndarray): Population to test.
            age (numpy.ndarray[int32]): Age of trees.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray[int32]]:
                1. Alive trees.
                2. New candidate population.
                3. Age of trees.

        """
        life_time_exceeded = np.where(age > self.lifetime)
        candidates = trees[life_time_exceeded]
        trees = np.delete(trees, life_time_exceeded, axis=0)
        age = np.delete(age, life_time_exceeded, axis=0)
        return trees, candidates, age

    def survival_of_the_fittest(self, task, trees, candidates, age):
        r"""Evaluate and filter current population.

        Args:
            task (Task): Optimization task.
            trees (numpy.ndarray): Population to evaluate.
            candidates (numpy.ndarray): Candidate population array to be updated.
            age (numpy.ndarray[int32]): Age of trees.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray[float], numpy.ndarray[int32]]:
                1. Trees sorted by fitness value.
                2. Updated candidate population.
                3. Population fitness values.
                4. Age of trees

        """
        evaluations = np.apply_along_axis(task.eval, 1, trees)
        ei = evaluations.argsort()
        candidates = np.append(candidates, trees[ei[self.area_limit:]], axis=0)
        trees = trees[ei[:self.area_limit]]
        age = age[ei[:self.area_limit]]
        evaluations = evaluations[ei[:self.area_limit]]
        return trees, candidates, evaluations, age

    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * age (numpy.ndarray[int32]): Age of trees.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        trees, fitness, _ = Algorithm.init_population(self, task)
        age = np.zeros(self.population_size, dtype=np.int32)
        self.dx = np.absolute(task.upper) / 5.0
        return trees, fitness, {'age': age}

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
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * age (numpy.ndarray[int32]): Age of trees.

        """
        age = params.pop('age')

        zero_age_trees = population[age == 0]
        local_seeds = self.local_seeding(task, zero_age_trees)
        age += 1
        population, candidate_population, age = self.remove_lifetime_exceeded(population, age)
        population = np.append(population, local_seeds, axis=0)
        age = np.append(age, np.zeros(len(local_seeds), dtype=np.int32))
        population, candidate_population, population_fitness, age = self.survival_of_the_fittest(task, population, candidate_population, age)
        gsn = int(self.transfer_rate * len(candidate_population))
        if gsn > 0:
            global_seeds = self.global_seeding(task, candidate_population, gsn)
            population = np.append(population, global_seeds, axis=0)
            age = np.append(age, np.zeros(len(global_seeds), dtype=np.int32))
            global_seeds_fitness = np.apply_along_axis(task.eval, 1, global_seeds)
            population_fitness = np.append(population_fitness, global_seeds_fitness)
        ib = np.argmin(population_fitness)
        age[ib] = 0
        if population_fitness[ib] < best_fitness:
            best_x, best_fitness = population[ib].copy(), population_fitness[ib]
        return population, population_fitness, best_x, best_fitness, {'age': age}
