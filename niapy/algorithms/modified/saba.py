# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['AdaptiveBatAlgorithm', 'SelfAdaptiveBatAlgorithm']


class AdaptiveBatAlgorithm(Algorithm):
    r"""Implementation of Adaptive bat algorithm.

    Algorithm:
        Adaptive bat algorithm

    Date:
        April 2019

    Authors:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        epsilon (float): Scaling factor.
        alpha (float): Constant for updating loudness.
        pulse_rate (float): Pulse rate.
        min_frequency (float): Minimum frequency.
        max_frequency (float): Maximum frequency.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['AdaptiveBatAlgorithm', 'ABA']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO"""

    def __init__(self, population_size=100, starting_loudness=0.5, epsilon=0.001, alpha=1.0, pulse_rate=0.5,
                 min_frequency=0.0, max_frequency=2.0, *args, **kwargs):
        """Initialize AdaptiveBatAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            starting_loudness (Optional[float]): Starting loudness.
            epsilon (Optional[float]): Scaling factor.
            alpha (Optional[float]): Constant for updating loudness.
            pulse_rate (Optional[float]): Pulse rate.
            min_frequency (Optional[float]): Minimum frequency.
            max_frequency (Optional[float]): Maximum frequency.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.starting_loudness = starting_loudness
        self.epsilon = epsilon
        self.alpha = alpha
        self.pulse_rate = pulse_rate
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

    def set_parameters(self, population_size=100, starting_loudness=0.5, epsilon=0.001, alpha=1.0, pulse_rate=0.5,
                       min_frequency=0.0, max_frequency=2.0, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            starting_loudness (Optional[float]): Starting loudness.
            epsilon (Optional[float]): Scaling factor.
            alpha (Optional[float]): Constant for updating loudness.
            pulse_rate (Optional[float]): Pulse rate.
            min_frequency (Optional[float]): Minimum frequency.
            max_frequency (Optional[float]): Maximum frequency.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.starting_loudness = starting_loudness
        self.epsilon = epsilon
        self.alpha = alpha
        self.pulse_rate = pulse_rate
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

    def get_parameters(self):
        r"""Get algorithm parameters.

        Returns:
            Dict[str, Any]: Arguments values.

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.get_parameters`

        """
        d = super().get_parameters()
        d.update({
            'starting_loudness': self.starting_loudness,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'pulse_rate': self.pulse_rate,
            'min_frequency': self.min_frequency,
            'max_frequency': self.max_frequency
        })
        return d

    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * loudness (float): Loudness.
                    * velocities (numpy.ndarray[float]): Velocity.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        population, fitness, d = super().init_population(task)
        loudness = np.full(self.population_size, self.starting_loudness)
        velocities = np.zeros((self.population_size, task.dimension))
        d.update({'loudness': loudness, 'velocities': velocities})
        return population, fitness, d

    def local_search(self, best, loudness, task, **kwargs):
        r"""Improve the best solution according to the Yang (2010).

        Args:
            best (numpy.ndarray): Global best individual.
            loudness (float): Loudness.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New solution based on global best individual.

        """
        return task.repair(best + self.epsilon * loudness * self.normal(0, 1, task.dimension), rng=self.rng)

    def update_loudness(self, loudness):
        r"""Update loudness when the prey is found.

        Args:
            loudness (float): Loudness.

        Returns:
            float: New loudness.

        """
        new_loudness = loudness * self.alpha
        return new_loudness if new_loudness > 1e-13 else self.starting_loudness

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Bat Algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population
            population_fitness (numpy.ndarray[float]): Current population fitness/function values
            best_x (numpy.ndarray): Current best individual
            best_fitness (float): Current best individual function/fitness value
            params (Dict[str, Any]): Additional algorithm arguments

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. Additional arguments:
                    * loudness (numpy.ndarray[float]): Loudness.
                    * velocities (numpy.ndarray[float]): Velocities.

        """
        loudness = params.pop('loudness')
        velocities = params.pop('velocities')

        for i in range(self.population_size):
            frequency = self.min_frequency + (self.max_frequency - self.min_frequency) * self.random()
            velocities[i] += (population[i] - best_x) * frequency
            if self.random() > self.pulse_rate:
                solution = self.local_search(best=best_x, loudness=loudness[i], task=task, i=i, Sol=population)
            else:
                solution = task.repair(population[i] + velocities[i], rng=self.rng)
            new_fitness = task.eval(solution)
            if (new_fitness <= population_fitness[i]) and (self.random() < loudness[i]):
                population[i], population_fitness[i] = solution, new_fitness
            if new_fitness <= best_fitness:
                best_x, best_fitness, loudness[i] = solution.copy(), new_fitness, self.update_loudness(loudness[i])
        return population, population_fitness, best_x, best_fitness, {'loudness': loudness, 'velocities': velocities}


class SelfAdaptiveBatAlgorithm(AdaptiveBatAlgorithm):
    r"""Implementation of Hybrid bat algorithm.

    Algorithm:
        Self Adaptive Bat Algorithm

    Date:
        April 2019

    Author:
        Klemen Berkovič

    License:
        MIT

    Reference paper:
        Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She. "A Hybrid Bat Algorithm". Elektrotehniški vestnik, 2013. 1-7.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        A_l (Optional[float]): Lower limit of loudness.
        A_u (Optional[float]): Upper limit of loudness.
        r_l (Optional[float]): Lower limit of pulse rate.
        r_u (Optional[float]): Upper limit of pulse rate.
        tao_1 (Optional[float]): Learning rate for loudness.
        tao_2 (Optional[float]): Learning rate for pulse rate.

    See Also:
        * :class:`niapy.algorithms.basic.BatAlgorithm`

    """

    Name = ['SelfAdaptiveBatAlgorithm', 'SABA']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She. "A Hybrid Bat Algorithm". Elektrotehniški vestnik, 2013. 1-7."""

    def __init__(self, min_loudness=0.9, max_loudness=1.0, min_pulse_rate=0.001, max_pulse_rate=0.1, tao_1=0.1,
                 tao_2=0.1, *args, **kwargs):
        """Initialize SelfAdaptiveBatAlgorithm.

        Args:
            min_loudness (Optional[float]): Lower limit of loudness.
            max_loudness (Optional[float]): Upper limit of loudness.
            min_pulse_rate (Optional[float]): Lower limit of pulse rate.
            max_pulse_rate (Optional[float]): Upper limit of pulse rate.
            tao_1 (Optional[float]): Learning rate for loudness.
            tao_2 (Optional[float]): Learning rate for pulse rate.

        See Also:
            * :func:`niapy.algorithms.modified.AdaptiveBatAlgorithm.__init__`

        """
        super().__init__(*args, **kwargs)
        self.min_loudness = min_loudness
        self.max_loudness = max_loudness
        self.min_pulse_rate = min_pulse_rate
        self.max_pulse_rate = max_pulse_rate
        self.tao_1 = tao_1
        self.tao_2 = tao_2

    def set_parameters(self, min_loudness=0.9, max_loudness=1.0, min_pulse_rate=0.001, max_pulse_rate=0.1, tao_1=0.1, tao_2=0.1, **kwargs):
        r"""Set core parameters of HybridBatAlgorithm algorithm.

        Args:
            min_loudness (Optional[float]): Lower limit of loudness.
            max_loudness (Optional[float]): Upper limit of loudness.
            min_pulse_rate (Optional[float]): Lower limit of pulse rate.
            max_pulse_rate (Optional[float]): Upper limit of pulse rate.
            tao_1 (Optional[float]): Learning rate for loudness.
            tao_2 (Optional[float]): Learning rate for pulse rate.

        See Also:
            * :func:`niapy.algorithms.modified.AdaptiveBatAlgorithm.set_parameters`

        """
        super().set_parameters(**kwargs)
        self.min_loudness = min_loudness
        self.max_loudness = max_loudness
        self.min_pulse_rate = min_pulse_rate
        self.max_pulse_rate = max_pulse_rate
        self.tao_1 = tao_1
        self.tao_2 = tao_2

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Parameters of the algorithm.

        See Also:
            * :func:`niapy.algorithms.modified.AdaptiveBatAlgorithm.get_parameters`

        """
        d = AdaptiveBatAlgorithm.get_parameters(self)
        d.update({
            'min_loudness': self.min_loudness,
            'max_loudness': self.max_loudness,
            'min_pulse_rate': self.min_pulse_rate,
            'max_pulse_rate': self.max_pulse_rate,
            'tao_1': self.tao_1,
            'tao_2': self.tao_2
        })
        return d

    def init_population(self, task):
        population, fitness, d = super().init_population(task)
        pulse_rates = np.full(self.population_size, self.pulse_rate)
        d.update({'pulse_rates': pulse_rates})
        return population, fitness, d

    def self_adaptation(self, loudness, pulse_rate):
        r"""Adaptation step.

        Args:
            loudness (float): Current loudness.
            pulse_rate (float): Current pulse rate.

        Returns:
            Tuple[float, float]:
                1. New loudness.
                2. Nwq pulse rate.

        """
        return self.min_loudness + self.random() * (
                self.max_loudness - self.min_loudness) if self.random() < self.tao_1 else loudness, self.min_pulse_rate + self.random() * (
                self.max_pulse_rate - self.min_pulse_rate) if self.random() < self.tao_2 else pulse_rate

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Bat Algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population
            population_fitness (numpy.ndarray[float]): Current population fitness/function values
            best_x (numpy.ndarray): Current best individual
            best_fitness (float): Current best individual function/fitness value
            params (Dict[str, Any]): Additional algorithm arguments

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. Additional arguments:
                    * loudness (numpy.ndarray[float]): Loudness.
                    * pulse_rates (numpy.ndarray[float]): Pulse rate.
                    * velocities (numpy.ndarray[float]): Velocities.

        """
        loudness = params.pop('loudness')
        pulse_rates = params.pop('pulse_rates')
        velocities = params.pop('velocities')

        for i in range(self.population_size):
            loudness[i], pulse_rates[i] = self.self_adaptation(loudness[i], pulse_rates[i])
            frequency = self.min_frequency + (self.max_frequency - self.min_frequency) * self.random()
            velocities[i] += (population[i] - best_x) * frequency
            if self.random() > pulse_rates[i]:
                solution = self.local_search(best=best_x, loudness=loudness[i], task=task, i=i, population=population)
            else:
                solution = task.repair(population[i] + velocities[i], rng=self.rng)
            new_fitness = task.eval(solution)
            if (new_fitness <= population_fitness[i]) and (self.random() < (self.min_loudness - loudness[i]) / self.starting_loudness):
                population[i], population_fitness[i] = solution, new_fitness
            if new_fitness <= best_fitness:
                best_x, best_fitness = solution.copy(), new_fitness
        return population, population_fitness, best_x, best_fitness, {'loudness': loudness, 'pulse_rates': pulse_rates, 'velocities': velocities}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
