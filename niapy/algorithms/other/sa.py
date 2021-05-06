# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['SimulatedAnnealing', 'cool_delta', 'cool_linear']


def cool_delta(current_temperature, delta_temperature, **_kwargs):
    r"""Calculate new temperature by differences.

    Args:
        current_temperature (float):
        delta_temperature (float):

    Returns:
        float: New temperature.

    """
    return current_temperature - delta_temperature


def cool_linear(current_temperature, starting_temperature, max_evals, **_kwargs):
    r"""Calculate temperature with linear function.

    Args:
        current_temperature (float): Current temperature.
        starting_temperature (float):
        max_evals (int): Number of evaluations done.
        _kwargs (Dict[str, Any]): Additional arguments.

    Returns:
        float: New temperature.

    """
    return current_temperature - starting_temperature / max_evals


class SimulatedAnnealing(Algorithm):
    r"""Implementation of Simulated Annealing Algorithm.

    Algorithm:
        Simulated Annealing Algorithm

    Date:
        2018

    Authors:
        Jan Popič and Klemen Berkovič

    License:
        MIT

    Reference URL:

    Reference paper:

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        delta (float): Movement for neighbour search.
        starting_temperature (float); Starting temperature.
        delta_temperature (float): Change in temperature.
        cooling_method (Callable): Neighbourhood function.
        epsilon (float): Error value.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['SimulatedAnnealing', 'SA']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""None"""

    def __init__(self, delta=0.5, starting_temperature=2000, delta_temperature=0.8, cooling_method=cool_delta,
                 epsilon=1e-23, *args, **kwargs):
        """Initialize SimulatedAnnealing.

        Args:
            delta (Optional[float]): Movement for neighbour search.
            starting_temperature (Optional[float]); Starting temperature.
            delta_temperature (Optional[float]): Change in temperature.
            cooling_method (Optional[Callable]): Neighbourhood function.
            epsilon (Optional[float]): Error value.

        See Also
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        kwargs.pop('population_size', None)
        super().__init__(1, *args, **kwargs)
        self.delta = delta
        self.starting_temperature = starting_temperature
        self.delta_temperature = delta_temperature
        self.cooling_method = cooling_method
        self.epsilon = epsilon

    def set_parameters(self, delta=0.5, starting_temperature=2000, delta_temperature=0.8, cooling_method=cool_delta,
                       epsilon=1e-23, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            delta (Optional[float]): Movement for neighbour search.
            starting_temperature (Optional[float]); Starting temperature.
            delta_temperature (Optional[float]): Change in temperature.
            cooling_method (Optional[Callable]): Neighbourhood function.
            epsilon (Optional[float]): Error value.

        See Also
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        kwargs.pop('population_size', None)
        super().set_parameters(population_size=1, **kwargs)
        self.delta = delta
        self.starting_temperature = starting_temperature
        self.delta_temperature = delta_temperature
        self.cooling_method = cooling_method
        self.epsilon = epsilon

    def get_parameters(self):
        r"""Get algorithms parameters values.

        Returns:
            Dict[str, Any]:

        See Also
            * :func:`niapy.algorithms.Algorithm.get_parameters`

        """
        d = Algorithm.get_parameters(self)
        d.update({
            'delta': self.delta,
            'delta_temperature': self.delta_temperature,
            'starting_temperature': self.starting_temperature,
            'epsilon': self.epsilon
        })
        return d

    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, float, dict]:
            1. Initial solution
            2. Initial solutions fitness/objective value
            3. Additional arguments

        """
        x = task.lower + task.range * self.random(task.dimension)
        current_temperature, x_fit = self.starting_temperature, task.eval(x)
        return x, x_fit, {'current_temperature': current_temperature}

    def run_iteration(self, task, x, x_fit, best_x, best_fitness, **params):
        r"""Core function of the algorithm.

        Args:
            task (Task):
            x (numpy.ndarray):
            x_fit (float):
            best_x (numpy.ndarray):
            best_fitness (float):
            **params (dict): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, float, numpy.ndarray, float, dict]:
            1. New solution
            2. New solutions fitness/objective value
            3. New global best solution
            4. New global best solutions fitness/objective value
            5. Additional arguments

        """
        current_temperature = params.pop('current_temperature')
        c = task.repair(x - self.delta / 2 + self.random(task.dimension) * self.delta, rng=self.rng)
        c_fit = task.eval(c)
        delta_fit, r = c_fit - x_fit, self.random()
        if delta_fit < 0 or r < np.exp(delta_fit / current_temperature):
            x, x_fit = c, c_fit
        current_temperature = self.cooling_method(current_temperature, starting_temperature=self.starting_temperature,
                                                  delta_temperature=self.delta_temperature, max_evals=task.max_evals)
        best_x, best_fitness = self.get_best(x, x_fit, best_x, best_fitness)
        return x, x_fit, best_x, best_fitness, {'current_temperature': current_temperature}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
