# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['BatAlgorithm']


class BatAlgorithm(Algorithm):
    r"""Implementation of Bat algorithm.

    Algorithm:
        Bat algorithm

    Date:
        2015

    Authors:
        Iztok Fister Jr., Marko Burjek and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Yang, Xin-She. "A new metaheuristic bat-inspired algorithm." Nature inspired cooperative strategies for optimization (NICSO 2010). Springer, Berlin, Heidelberg, 2010. 65-74.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        loudness (float): Initial loudness.
        pulse_rate (float): Initial pulse rate.
        alpha (float): Parameter for controlling loudness decrease.
        gamma (float): Parameter for controlling pulse rate increase.
        min_frequency (float): Minimum frequency.
        max_frequency (float): Maximum frequency.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['BatAlgorithm', 'BA']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Yang, Xin-She. "A new metaheuristic bat-inspired algorithm." Nature inspired cooperative strategies for optimization (NICSO 2010). Springer, Berlin, Heidelberg, 2010. 65-74."""

    def __init__(self, population_size=40, loudness=1.0, pulse_rate=1.0, alpha=0.97, gamma=0.1, min_frequency=0.0,
                 max_frequency=2.0, *args, **kwargs):
        """Initialize BatAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            loudness (Optional[float]): Initial loudness.
            pulse_rate (Optional[float]): Initial pulse rate.
            alpha (Optional[float]): Parameter for controlling loudness decrease.
            gamma (Optional[float]): Parameter for controlling pulse rate increase.
            min_frequency (Optional[float]): Minimum frequency.
            max_frequency (Optional[float]): Maximum frequency.

        See Also:
            :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

    def set_parameters(self, population_size=20, loudness=1.0, pulse_rate=1.0, alpha=0.97, gamma=0.1, min_frequency=0.0,
                       max_frequency=2.0, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            loudness (Optional[float]): Initial loudness.
            pulse_rate (Optional[float]): Initial pulse rate.
            alpha (Optional[float]): Parameter for controlling loudness decrease.
            gamma (Optional[float]): Parameter for controlling pulse rate increase.
            min_frequency (Optional[float]): Minimum frequency.
            max_frequency (Optional[float]): Maximum frequency.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        d = super().get_parameters()
        d.update({
            'loudness': self.loudness,
            'pulse_rate': self.pulse_rate,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'min_frequency': self.min_frequency,
            'max_frequency': self.max_frequency
        })
        return d

    def init_population(self, task):
        r"""Initialize the starting population.

        Parameters:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * velocities (numpy.ndarray[float]): Velocities.
                    * alpha (float): Previous iterations loudness.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        population, fitness, d = super().init_population(task)
        velocities = np.zeros((self.population_size, task.dimension))
        d.update({'velocities': velocities, 'loudness': self.loudness})
        return population, fitness, d

    def local_search(self, best, loudness, task, **kwargs):
        r"""Improve the best solution according to the Yang (2010).

        Args:
            best (numpy.ndarray): Global best individual.
            loudness (float): Current loudness.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New solution based on global best individual.

        """
        return task.repair(best + 0.1 * self.standard_normal(task.dimension) * loudness)

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Bat Algorithm.

        Parameters:
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
                5. Additional arguments:
                    * velocities (numpy.ndarray): Velocities.
                    * alpha (float): Previous iterations loudness.

        """
        velocities = params.pop('velocities')
        loudness = params.pop('loudness') * self.alpha

        pulse_rate = self.pulse_rate * (1 - np.exp(-self.gamma * task.iters))

        for i in range(self.population_size):
            frequency = self.min_frequency + (self.max_frequency - self.min_frequency) * self.random()
            velocities[i] += (population[i] - best_x) * frequency
            if self.random() < pulse_rate:
                solution = self.local_search(best=best_x, loudness=loudness, task=task, i=i, population=population)
            else:
                solution = task.repair(population[i] + velocities[i], rng=self.rng)
            new_fitness = task.eval(solution)
            if (new_fitness <= population_fitness[i]) and (self.random() > loudness):
                population[i], population_fitness[i] = solution, new_fitness
            if new_fitness <= best_fitness:
                best_x, best_fitness = solution.copy(), new_fitness
        return population, population_fitness, best_x, best_fitness, {'velocities': velocities, 'loudness': loudness}
