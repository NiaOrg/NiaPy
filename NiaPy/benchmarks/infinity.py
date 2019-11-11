# encoding=utf8

"""Implementations of Infinity function."""

from numpy import sin
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Infinity']

class Infinity(Benchmark):
	r"""Implementations of Infinity function.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Function:
	**Infinity Function**

		:math:`f(\textbf{x}) = \sum_{i = 1}^D x_i^6 \left( \sin \left( \frac{1}{x_i} \right) + 2 \right)`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:**
		:math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = \sum_{i = 1}^D x_i^6 \left( \sin \left( \frac{1}{x_i} \right) + 2 \right)$

		Equation:
				\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D x_i^6 \left( \sin \left( \frac{1}{x_i} \right) + 2 \right) \end{equation}

		Domain:
				$-1 \leq x_i \leq 1$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_I.html#go_benchmark.Infinity
	"""
	Name = ['Infinity']

	def __init__(self, Lower=-1.0, Upper=1.0):
		r"""Initialize of Infinity benchmark.

		Args:
			Lower (Optional[float]): Lower bound of problem.
			Upper (Optional[float]): Upper bound of problem.

		See Also:
			:func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper)

	@staticmethod
	def latex_code():
		r"""Return the latex code of the problem.

		Returns:
			str: Latex code
		"""
		return r'''$f(\textbf{x}) = \sum_{i = 1}^D x_i^6 \left( \sin \left( \frac{1}{x_i} \right) + 2 \right)$'''

	def function(self):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function
		"""
		def f(D, X):
			r"""Fitness function.

			Args:
				D (int): Dimensionality of the problem
				sol (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

			Returns:
				float: Fitness value for the solution.
			"""
			val = 0.0
			for i in range(D): val += X[i] ** 6 * (sin(1 / X[i]) + 2)
			return val
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
