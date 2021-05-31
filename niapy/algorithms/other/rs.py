# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['RandomSearch']


class RandomSearch(Algorithm):
    r"""Implementation of a simple Random Algorithm.

    Algorithm:
        Random Search

    Date:
        11.10.2020

    Authors:
        Iztok Fister Jr., Grega Vrbančič

    License:
        MIT

    Reference URL: https://en.wikipedia.org/wiki/Random_search

    Attributes:
        Name (List[str]): List of strings representing algorithm name.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['RandomSearch', 'RS']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""None"""

    def __init__(self, *args, **kwargs):
        """Initialize RandomSearch."""
        kwargs.pop('population_size', None)
        super().__init__(1, *args, **kwargs)
        self.candidates = None

    def set_parameters(self, **kwargs):
        r"""Set the algorithm parameters/arguments.

        See Also
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        kwargs.pop('population_size', None)
        Algorithm.set_parameters(self, population_size=1, **kwargs)
        self.candidates = None

    def get_parameters(self):
        r"""Get algorithms parameters values.

        Returns:
            Dict[str, Any]:
        See Also
            * :func:`niapy.algorithms.Algorithm.get_parameters`

        """
        d = Algorithm.get_parameters(self)
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
        if task.max_iters != np.inf:
            total_candidates = task.max_iters
        elif task.max_evals != np.inf:
            total_candidates = task.max_evals
        else:
            total_candidates = 0
        self.candidates = []
        x = None
        for i in range(total_candidates):
            while True:
                x = task.lower + task.range * self.random(task.dimension)
                if not np.any([np.all(a == x) for a in self.candidates]):
                    self.candidates.append(x)
                    break

        x_fit = task.eval(self.candidates[0])
        return x, x_fit, {}

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
        current_candidate = task.iters if task.max_iters != np.inf else task.evals
        x = self.candidates[current_candidate]
        x_fit = task.eval(x)
        best_x, best_fitness = self.get_best(x, x_fit, best_x, best_fitness)
        return x, x_fit, best_x, best_fitness, {}
