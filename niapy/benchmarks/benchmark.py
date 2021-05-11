# encoding=utf8

"""Implementation of benchmarks utility function."""

import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig()
logger = logging.getLogger('niapy.benchmarks.benchmark')
logger.setLevel('INFO')

__all__ = ['Benchmark']


class Benchmark:
    r"""Class representing benchmarks.

    Attributes:
        Name (List[str]): List of names representing benchmark names.
        lower (Union[int, float, list, numpy.ndarray]): Lower bounds.
        upper (Union[int, float, list, numpy.ndarray]): Upper bounds.

    """

    Name = ['Benchmark', 'BBB']

    def __init__(self, lower, upper):
        r"""Initialize benchmark.

        Args:
            lower (Union[int, float, list, numpy.ndarray]): Lower bounds.
            upper (Union[int, float, list, numpy.ndarray]): Upper bounds.

        """
        self.lower = lower
        self.upper = upper

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code

        """
        return r'''$f(x) = \infty$'''

    def function(self):
        r"""Get the optimization function.

        Returns:
            Callable[[int, Union[list, numpy.ndarray]], float]: Fitness function.

        """
        def fun(dimension, x):
            r"""Initialize benchmark.

            Args:
                dimension (int): Dimensionality of the problem.
                x (Union[int, float, list, numpy.ndarray]): Solution to the problem.

            Returns:
                float: Fitness value for the solution.

            """
            return np.inf

        return fun

    def __call__(self):
        r"""Get the optimization function.

        Returns:
            Callable[[int, Union[list, numpy.ndarray]], float]: Fitness function.

        """
        return self.function()

    def plot2d(self):
        r"""Plot 2D graph."""
        pass

    @staticmethod
    def __2d_fun(x, y, f):
        r"""Calculate function value.

        Args:
            x (float): First coordinate.
            y (float): Second coordinate.
            f (Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]): Evaluation function.

        Returns:
            float: Calculate functional value for given input.

        """
        return f(2, [x, y])

    def plot3d(self, scale=0.32):
        r"""Plot 3d scatter plot of benchmark function.

        Args:
            scale (float): Scale factor for points.

        """
        fig = plt.figure()
        ax = Axes3D(fig)
        func = self.function()
        xr, yr = np.arange(self.lower, self.upper, scale), np.arange(self.lower, self.upper, scale)
        x, y = np.meshgrid(xr, yr)
        z = np.vectorize(self.__2d_fun)(x, y, func)
        ax.plot_surface(x, y, z, rstride=8, cstride=8, alpha=0.3)
        ax.contourf(x, y, z, zdir='z', offset=-10, cmap=cm.coolwarm)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
