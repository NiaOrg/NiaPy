# encoding=utf8
import logging

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['HillClimbAlgorithm']


def neighborhood(x, delta, task, rng):
    r"""Get neighbours of point.

    Args:
        x (numpy.ndarray): Point.
        delta (float): Standard deviation.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random generator.

    Returns:
        Tuple[numpy.ndarray, float]:
            1. New solution.
            2. New solutions function/fitness value.

    """
    new_x = x + rng.normal(0, delta, task.dimension)
    new_x = task.repair(new_x, rng)
    new_x_fitness = task.eval(new_x)
    return new_x, new_x_fitness


class HillClimbAlgorithm(Algorithm):
    r"""Implementation of iterative hill climbing algorithm.

    Algorithm:
        Hill Climbing Algorithm

    Date:
        2018

    Authors:
        Jan Popič

    License:
        MIT

    Reference URL:

    Reference paper:

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    Attributes:
        delta (float): Change for searching in neighborhood.
        neighborhood_function (Callable[numpy.ndarray, float, Task], Tuple[numpy.ndarray, float]]): Function for getting neighbours.

    """

    Name = ['HillClimbAlgorithm', 'HC']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            :func:`niapy.algorithms.algorithm.Algorithm.info`

        """
        return r"""TODO"""

    def __init__(self, delta=0.5, neighborhood_function=neighborhood, *args, **kwargs):
        """Initialize HillClimbAlgorithm.

        Args:
            * delta (Optional[float]): Change for searching in neighborhood.
            * neighborhood_function (Optional[Callable[numpy.ndarray, float, Task], Tuple[numpy.ndarray, float]]]): Function for getting neighbours.

        """
        kwargs.pop('population_size', None)
        super().__init__(1, *args, **kwargs)
        self.delta = delta
        self.neighborhood_function = neighborhood_function

    def set_parameters(self, delta=0.5, neighborhood_function=neighborhood, **kwargs):
        r"""Set the algorithm parameters/arguments.

        Args:
            * delta (Optional[float]): Change for searching in neighborhood.
            * neighborhood_function (Optional[Callable[numpy.ndarray, float, Task], Tuple[numpy.ndarray, float]]]): Function for getting neighbours.

        """
        kwargs.pop('population_size', None)
        super().set_parameters(population_size=1, **kwargs)
        self.delta = delta
        self.neighborhood_function = neighborhood_function

    def get_parameters(self):
        d = Algorithm.get_parameters(self)
        d.update({
            'delta': self.delta,
            'neighborhood_function': self.neighborhood_function
        })
        return d

    def init_population(self, task):
        r"""Initialize stating point.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, float, Dict[str, Any]]:
                1. New individual.
                2. New individual function/fitness value.
                3. Additional arguments.

        """
        x = task.lower + self.random(task.dimension) * task.range
        return x, task.eval(x), {}

    def run_iteration(self, task, x, fx, best_x, best_fitness, **params):
        r"""Core function of HillClimbAlgorithm algorithm.

        Args:
            task (Task): Optimization task.
            x (numpy.ndarray): Current solution.
            fx (float): Current solutions fitness/function value.
            best_x (numpy.ndarray): Global best solution.
            best_fitness (float): Global best solutions function/fitness value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, float, numpy.ndarray, float, Dict[str, Any]]:
                1. New solution.
                2. New solutions function/fitness value.
                3. Additional arguments.

        """
        lo, xn = False, task.lower + task.range * self.random(task.dimension)
        xn_f = task.eval(xn)
        while not lo:
            yn, yn_f = self.neighborhood_function(x, self.delta, task, rng=self.rng)
            if yn_f < xn_f:
                xn, xn_f = yn, yn_f
            else:
                lo = True or task.stopping_condition()
        best_x, best_fitness = self.get_best(xn, xn_f, best_x, best_fitness)
        return xn, xn_f, best_x, best_fitness, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
