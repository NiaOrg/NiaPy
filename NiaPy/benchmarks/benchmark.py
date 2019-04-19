# encoding=utf8

"""Implementation of base benchmark class."""

import logging
from numpy import inf, arange, meshgrid, vectorize
from matplotlib import pyplot as plt
from matplotlib import cm

logging.basicConfig()
logger = logging.getLogger("NiaPy.benchmarks.benchmark")
logger.setLevel("INFO")

__all__ = ["Benchmark"]


class Benchmark:
	r"""Class representing benchmarks.

	Attributes:
			  Name (List[str]): List of names representing benchmark names.
			  lower (Union[int, float, List[Union[int, float]], numpy.ndarray[Union[int, float]]]): lower bounds.
			  upper (Union[int, float, List[Union[int, float]], numpy.ndarray[Union[int, float]]]): upper bounds.

	"""

	name = ["Benchmark", "BBB"]

	def __init__(self, lower, upper, **kwargs):
		r"""Initialize benchmark.

		Args:
				  lower (Union[int, float, list, numpy.ndarray]): lower bounds.
				  upper (Union[int, float, list, numpy.ndarray]): upper bounds.
				  **kwargs (Dict[str, Any]): Additional arguments.

		"""

		self._lower = lower
		self._upper = upper

	@property
	def lower(self):
		r"""Get lower bounds of optimization problem.

		Returns:
			 Union[int, float, List[Union[int, float]], numpy.ndarray[Union[int, float]]]: Lower bounds.
		"""
		return self._lower

	@property
	def upper(self):
		r"""Get upper bounds of optimization problem.

		Returns:
			 Union[int, float, List[Union[int, float]], numpy.ndarray[Union[int, float]]]: Upper bounds.
		"""
		return self._upper

	def function(self):
		r"""Get evaluation function.

		Returns:
				  Callable[[int, Union[list, numpy.ndarray]], float]): Evaluation function.

		"""

		def evaluate(D, sol):
			r"""Utility/Evaluation function.

			Args:
					  D (int): Number of coordinates.
					  sol (Union[list, numpy.ndarray]): Solution to evaluate.

			Returns:
					  float: Function value.

			"""

			return inf

		return evaluate

	def plot2d(self):
		"""Plot."""
		pass

	def __2dfun(self, x, y, f):
		r"""Calculate function value.

		Args:
				  x (float): First coordinate.
				  y (float): Second coordinate.
				  f (Callable[[int, List[float]], float]): Evaluation function.

		Returns:
				  float: Calculate functional value for given input

		"""

		return f(2, x, y)

	def plot3d(self, scale=0.32):
		r"""Plot 3d scatter plot of benchmark function.

		Args:
				  scale (float): Scale factor for points.

		"""

		fig = plt.figure()
		ax = fig.gca(projection="3d")
		func = self.function()
		Xr, Yr = arange(self.lower, self.upper, scale), arange(
			self.lower, self.upper, scale)
		X, Y = meshgrid(Xr, Yr)
		Z = vectorize(self.__2dfun)(X, Y, func)
		ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
		ax.contourf(X, Y, Z, zdir="z", offset=-10, cmap=cm.coolwarm)
		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		ax.set_zlabel("Z")
		plt.show()
