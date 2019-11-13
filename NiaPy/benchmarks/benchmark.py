# encoding=utf8

"""Implementation of benchmarks utility function."""

import logging
from numpy import inf, arange, meshgrid, vectorize
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig()
logger = logging.getLogger('NiaPy.benchmarks.benchmark')
logger.setLevel('INFO')

__all__ = ['Benchmark']

class Benchmark:
	r"""Class representing benchmarks.

	Attributes:
		Name (List[str]): List of names representiong benchmark names.
		Lower (Union[int, float, list, numpy.ndarray]): Lower bounds.
		Upper (Union[int, float, list, numpy.ndarray]): Upper bounds.
	"""
	Name = ['Benchmark', 'BBB']

	def __init__(self, Lower, Upper, **kwargs):
		r"""Initialize benchmark.

		Args:
			Lower (Union[int, float, list, numpy.ndarray]): Lower bounds.
			Upper (Union[int, float, list, numpy.ndarray]): Upper bounds.
			**kwargs (Dict[str, Any]): Additional arguments.
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
			Callable[[int, Union[list, numpy.ndarray]], float]: Fitness funciton.
		"""
		def fun(D, X):
			r"""Initialize benchmark.

			Args:
				D (int): Dimesionality of the problem.
				X (Union[int, float, list, numpy.ndarray]): Solution to the problem.

			Retruns:
				float: Fitness value for the solution
			"""
			return inf
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
			f (Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]): Evaluation function.

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
		Xr, Yr = arange(self.Lower, self.Upper, scale), arange(self.Lower, self.Upper, scale)
		X, Y = meshgrid(Xr, Yr)
		Z = vectorize(self.__2dfun)(X, Y, func)
		ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
		ax.contourf(X, Y, Z, zdir='z', offset=-10, cmap=cm.coolwarm)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		plt.show()

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
