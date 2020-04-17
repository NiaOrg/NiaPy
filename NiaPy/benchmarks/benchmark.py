# encoding=utf8

"""Implementation of benchmarks utility function."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

__all__ = ['Benchmark']

class Benchmark:
    r"""Class representing benchmarks.

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of names representing benchmark names.
        Lower (Union[int, float, list, numpy.ndarray]): Lower bounds.
        Upper (Union[int, float, list, numpy.ndarray]): Upper bounds.
    """
    Name = ['Benchmark', 'BBB', 'benchmark', 'bbb']

    def __init__(self, Lower, Upper, **kwargs):
        r"""Initialize benchmark.

        Args:
            Lower (Union[int, float, list, numpy.ndarray]): Lower bounds.
            Upper (Union[int, float, list, numpy.ndarray]): Upper bounds.
            kwargs (Dict[str, Any]): Additional arguments.
        """
        self.Lower, self.Upper = Lower, Upper

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
        def fun(D, X, **kwargs):
            r"""Initialize benchmark.

            Args:
                D (int): Dimensionality of the problem.
                X (Union[int, float, list, numpy.ndarray]): Solution to the problem.
                kwargs (Dict[str, Any]): Additional arguments for the objective/utility/fitness function.

            Returns:
                float: Fitness value for the solution
            """
            return np.inf
        return fun

    def __call__(self):
        r"""Get the optimization function.

        Returns:
            Callable[[int, Union[list, numpy.ndarray]], float]: Fitness funciton.
        """
        return self.function()

    def plot2d(self):
        r"""Plot 2D graph."""
        pass

    def __2dfun(self, x, y, f):
        r"""Calculate function value.

        Args:
            x (float): First coordinate.
            y (float): Second coordinate.
            f (Callable[[int, Union[int, float, list, numpy.ndarray]], float]): Evaluation function.

        Returns:
            float: Calculate functional value for given input
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
        Xr, Yr = np.arange(self.Lower, self.Upper, scale), np.arange(self.Lower, self.Upper, scale)
        X, Y = np.meshgrid(Xr, Yr)
        Z = np.vectorize(self.__2dfun)(X, Y, func)
        ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
        ax.contourf(X, Y, Z, zdir='z', offset=-10, cmap=cm.coolwarm)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
