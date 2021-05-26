# encoding=utf8
import logging

from niapy.algorithms.basic import BatAlgorithm
from niapy.algorithms.basic.de import cross_best1

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['HybridBatAlgorithm']


class HybridBatAlgorithm(BatAlgorithm):
    r"""Implementation of Hybrid bat algorithm.

    Algorithm:
        Hybrid bat algorithm

    Date:
        2018

    Author:
        Grega Vrbančič and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She. "A Hybrid Bat Algorithm". Elektrotehniški vestnik, 2013. 1-7.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        F (float): Scaling factor.
        CR (float): Crossover.

    See Also:
        * :class:`niapy.algorithms.basic.BatAlgorithm`

    """

    Name = ['HybridBatAlgorithm', 'HBA']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She. "A Hybrid Bat Algorithm". Elektrotehniški vestnik, 2013. 1-7."""

    def __init__(self, differential_weight=0.50, crossover_probability=0.90, strategy=cross_best1, *args, **kwargs):
        """Initialize HybridBatAlgorithm.

        Args:
            differential_weight (Optional[float]): Differential weight.
            crossover_probability (Optional[float]): Crossover rate.
            strategy (Optional[Callable]): DE Crossover and mutation strategy.

        See Also:
            * :func:`niapy.algorithms.basic.BatAlgorithm.set_parameters`

        """
        super().__init__(*args, **kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.strategy = strategy

    def set_parameters(self, differential_weight=0.50, crossover_probability=0.90, strategy=cross_best1, **kwargs):
        r"""Set core parameters of HybridBatAlgorithm algorithm.

        Args:
            differential_weight (Optional[float]): Differential weight.
            crossover_probability (Optional[float]): Crossover rate.
            strategy (Callable): DE Crossover and mutation strategy.

        See Also:
            * :func:`niapy.algorithms.basic.BatAlgorithm.set_parameters`

        """
        super().set_parameters(**kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.strategy = strategy

    def local_search(self, best, task, i=None, population=None, **kwargs):
        r"""Improve the best solution.

        Args:
            best (numpy.ndarray): Global best individual.
            task (Task): Optimization task.
            i (int): Index of current individual.
            population (numpy.ndarray): Current best population.

        Returns:
            numpy.ndarray: New solution based on global best individual.

        """
        return task.repair(self.strategy(population, i, self.differential_weight, self.crossover_probability, self.rng, best), rng=self.rng)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
