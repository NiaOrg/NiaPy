# encoding=utf8
import logging

from niapy.algorithms.basic.de import cross_best1
from niapy.algorithms.modified.saba import SelfAdaptiveBatAlgorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['HybridSelfAdaptiveBatAlgorithm']


class HybridSelfAdaptiveBatAlgorithm(SelfAdaptiveBatAlgorithm):
    r"""Implementation of Hybrid self adaptive bat algorithm.

    Algorithm:
        Hybrid self adaptive bat algorithm

    Date:
        April 2019

    Author:
        Klemen Berkovič

    License:
        MIT

    Reference paper:
        Fister, Iztok, Simon Fong, and Janez Brest. "A novel hybrid self-adaptive bat algorithm." The Scientific World Journal 2014 (2014).

    Reference URL:
        https://www.hindawi.com/journals/tswj/2014/709738/cta/

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        F (float): Scaling factor for local search.
        CR (float): Probability of crossover for local search.
        CrossMutt (Callable[[numpy.ndarray, int, numpy.ndarray, float, float, numpy.random.Generator, Dict[str, Any]): Local search method based of Differential evolution strategy.

    See Also:
        * :class:`niapy.algorithms.basic.BatAlgorithm`

    """

    Name = ['HybridSelfAdaptiveBatAlgorithm', 'HSABA']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Fister, Iztok, Simon Fong, and Janez Brest. "A novel hybrid self-adaptive bat algorithm." The Scientific World Journal 2014 (2014)."""

    def __init__(self, differential_weight=0.9, crossover_probability=0.85, strategy=cross_best1, *args, **kwargs):
        """Initialize HybridSelfAdaptiveBatAlgorithm.

        Args:
            differential_weight (Optional[float]): Scaling factor for local search.
            crossover_probability (Optional[float]): Probability of crossover for local search.
            strategy (Optional[Callable[[numpy.ndarray, int, numpy.ndarray, float, float, numpy.random.Generator, Dict[str, Any], numpy.ndarray]]): Local search method based of Differential evolution strategy.

        See Also:
            * :func:`niapy.algorithms.basic.BatAlgorithm.__init__`

        """
        super().__init__(*args, **kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.strategy = strategy

    def set_parameters(self, differential_weight=0.9, crossover_probability=0.85, strategy=cross_best1, **kwargs):
        r"""Set core parameters of HybridBatAlgorithm algorithm.

        Args:
            differential_weight (Optional[float]): Scaling factor for local search.
            crossover_probability (Optional[float]): Probability of crossover for local search.
            strategy (Optional[Callable[[numpy.ndarray, int, numpy.ndarray, float, float, numpy.random.Generator, Dict[str, Any], numpy.ndarray]]): Local search method based of Differential evolution strategy.

        See Also:
            * :func:`niapy.algorithms.basic.BatAlgorithm.set_parameters`

        """
        super().set_parameters(**kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.strategy = strategy

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Parameters of the algorithm.

        See Also:
            * :func:`niapy.algorithms.modified.AdaptiveBatAlgorithm.get_parameters`

        """
        d = super().get_parameters()
        d.update({
            'differential_weight': self.differential_weight,
            'crossover_probability': self.crossover_probability
        })
        return d

    def local_search(self, best, loudness, task, i=None, population=None, **kwargs):
        r"""Improve the best solution.

        Args:
            best (numpy.ndarray): Global best individual.
            loudness (float): Loudness.
            task (Task): Optimization task.
            i (int): Index of current individual.
            population (numpy.ndarray): Current best population.

        Returns:
            numpy.ndarray: New solution based on global best individual.

        """
        return task.repair(self.strategy(population, i, self.differential_weight, self.crossover_probability, rng=self.rng, x_b=best), rng=self.rng)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
