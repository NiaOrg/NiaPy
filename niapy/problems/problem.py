# encoding=utf8

"""Implementation of problems utility function."""

from abc import ABC, abstractmethod
import logging
from niapy.util.array import full_array

logging.basicConfig()
logger = logging.getLogger('niapy.problems.problem')
logger.setLevel('INFO')

__all__ = ['Problem']


class Problem(ABC):
    r"""Class representing an optimization problem.

    Attributes:
        dimension (int): Dimension of the problem.
        lower (numpy.ndarray): Lower bounds of the problem.
        upper (numpy.ndarray): Upper bounds of the problem.

    """

    def __init__(self, dimension=1, lower=None, upper=None, *args, **kwargs):
        r"""Initialize Problem.

        Args:
            dimension (Optional[int]): Dimension of the problem.
            lower (Optional[Union[float, Iterable[float]]]): Lower bounds of the problem.
            upper (Optional[Union[float, Iterable[float]]]): Upper bounds of the problem.

        """
        self.dimension = dimension
        self.lower = full_array(lower, dimension)
        self.upper = full_array(upper, dimension)

    @abstractmethod
    def _evaluate(self, x):
        """Evaluate solution."""
        pass

    def evaluate(self, x):
        """Evaluate solution.

        Args:
            x (numpy.ndarray): Solution.

        Returns:
            float: Function value of `x`.

        """
        if x.shape[0] != self.dimension:
            raise ValueError('Dimensions do not match. {} != {}'.format(x.shape[0], self.dimension))

        return self._evaluate(x)

    def __call__(self, x):
        r"""Evaluate solution.

        Args:
            x (numpy.ndarray): Solution.

        Returns:
            float: Function value of `x`.

        See Also:
            :func:`niapy.problems.Problem.evaluate`

        """
        return self.evaluate(x)

    def name(self):
        """Get class name."""
        return self.__class__.__name__

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
