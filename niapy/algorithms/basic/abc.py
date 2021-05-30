# encoding=utf8
import copy
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm, Individual, default_individual_init

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['ArtificialBeeColonyAlgorithm']


class SolutionABC(Individual):
    r"""Representation of solution for Artificial Bee Colony Algorithm.

    Date:
        2018

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.algorithms.Individual`

    """

    pass


class ArtificialBeeColonyAlgorithm(Algorithm):
    r"""Implementation of Artificial Bee Colony algorithm.

    Algorithm:
        Artificial Bee Colony algorithm

    Date:
        2018

    Author:
        Uros Mlakar and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Karaboga, D., and Bahriye B. "A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm." Journal of global optimization 39.3 (2007): 459-471.

    Arguments
        Name (List[str]): List containing strings that represent algorithm names
        limit (Union[float, numpy.ndarray[float]]): Maximum number of cycles without improvement.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['ArtificialBeeColonyAlgorithm', 'ABC']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Karaboga, D., and Bahriye B. "A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm." Journal of global optimization 39.3 (2007): 459-471."""

    def __init__(self, population_size=10, limit=100, *args, **kwargs):
        """Initialize ArtificialBeeColonyAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            limit (Optional[int]): Maximum number of cycles without improvement.

        See Also:
            :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, initialization_function=default_individual_init, individual_type=SolutionABC,
                         *args, **kwargs)
        self.limit = limit
        self.food_number = self.population_size // 2

    def set_parameters(self, population_size=10, limit=100, **kwargs):
        r"""Set the parameters of Artificial Bee Colony Algorithm.

        Args:
            population_size(Optional[int]): Population size.
            limit (Optional[int]): Maximum number of cycles without improvement.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, initialization_function=default_individual_init,
                               individual_type=SolutionABC, **kwargs)
        self.food_number = self.population_size // 2
        self.limit = limit

    def get_parameters(self):
        """Get parameters."""
        params = super().get_parameters()
        params.update({
            'limit': self.limit
        })
        return params

    def calculate_probabilities(self, foods):
        r"""Calculate the probes.

        Args:
            foods (numpy.ndarray): Current population.

        Returns:
            numpy.ndarray: Probabilities.

        """
        probs = np.asarray([1.0 / (foods[i].f + 0.01) for i in range(self.food_number)])
        return probs / np.sum(probs)

    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. Additional arguments:
                    * trials (numpy.ndarray): Number of cycles without improvement.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        foods, fpop, _ = super().init_population(task)
        trials = np.zeros(self.food_number, dtype=np.int32)
        return foods, fpop, {'trials': trials}

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of  the algorithm.

        Args:
            task (Task): Optimization task
            population (numpy.ndarray): Current population
            population_fitness (numpy.ndarray[float]): Function/fitness values of current population
            best_x (numpy.ndarray): Current best individual
            best_fitness (float): Current best individual fitness/function value
            params (Dict[str, Any]): Additional parameters

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. New global best solution
                4. New global best fitness/objective value
                5. Additional arguments:
                    * trials (numpy.ndarray): Number of cycles without improvement.

        """
        trials = params.pop('trials')

        for i in range(self.food_number):
            new_solution = copy.deepcopy(population[i])
            param2change = int(self.random() * task.dimension)
            neighbor = int(self.food_number * self.random())
            new_solution.x[param2change] = population[i].x[param2change] + self.uniform(-1, 1) * (
                    population[i].x[param2change] - population[neighbor].x[param2change])
            new_solution.evaluate(task, rng=self.rng)
            if new_solution.f < population[i].f:
                population[i] = new_solution
                trials[i] = 0
                if new_solution.f < best_fitness:
                    best_x = new_solution.x.copy()
                    best_fitness = new_solution.f
            else:
                trials[i] += 1
        probabilities, t, s = self.calculate_probabilities(population), 0, 0
        while t < self.food_number:
            if self.random() < probabilities[s]:
                t += 1
                solution = copy.deepcopy(population[s])
                param2change = int(self.random() * task.dimension)
                neighbor = int(self.food_number * self.random())
                while neighbor == s:
                    neighbor = int(self.food_number * self.random())
                solution.x[param2change] = population[s].x[param2change] + self.uniform(-1, 1) * (
                        population[s].x[param2change] - population[neighbor].x[param2change])
                solution.evaluate(task, rng=self.rng)
                if solution.f < population[s].f:
                    population[s] = solution
                    trials[s] = 0
                    if solution.f < best_fitness:
                        best_x = solution.x.copy()
                        best_fitness = solution.f
                else:
                    trials[s] += 1
            s += 1
            if s == self.food_number:
                s = 0
        mi = np.argmax(trials)
        if trials[mi] >= self.limit:
            population[mi], trials[mi] = SolutionABC(task=task, rng=self.rng), 0
            if population[mi].f < best_fitness:
                best_x, best_fitness = population[mi].x.copy(), population[mi].f
        return population, np.asarray([f.f for f in population]), best_x, best_fitness, {'trials': trials}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
