# encoding=utf8
"""Implementation of benchmarks utility function."""
import logging
from numpy import inf, arange, meshgrid, vectorize
from matplotlib import pyplot as plt
from matplotlib import cm

logging.basicConfig()
logger = logging.getLogger('NiaPy.benchmarks.benchmark')
logger.setLevel('INFO')

__all__ = ['Benchmark']


class Benchmark:
    r"""Base Benchmark interface class."""

    Name = ['Benchmark', 'BBB']

    def __init__(self, Lower, Upper, **kwargs):
        r"""Initialization of Benchmark object.

        Arguments:
            Lower {[type]} -- Lower bound.
            Upper {[type]} -- Upper bound.
        """

        self.Lower = Lower
        self.Upper = Upper

    def function(self):
        r"""Returns the optimization function."""

        def fun(D, X):
            return inf

        return fun

    def plot2d(self):
        """Plot."""

        pass

    def __2dfun(self, x, y, f):
        r"""Plot function.

        Arguments:
            x {[type]} -- x value
            y {[type]} -- y value
            f {[type]} -- function

        """

        return f(2, x, y)

    def plot3d(self, scale=0.32):
        """Plot 3d.

        Keyword Arguments:
            scale {float} -- scale (default: {0.32})
        """

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        func = self.function()
        Xr, Yr = arange(self.Lower, self.Upper, scale), arange(
            self.Lower, self.Upper, scale)
        X, Y = meshgrid(Xr, Yr)
        Z = vectorize(self.__2dfun)(X, Y, func)
        ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
        ax.contourf(X, Y, Z, zdir='z', offset=-10, cmap=cm.coolwarm)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
