# encoding=utf8
"""Implementations of Zakharov function."""

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Zakharov']

class Zakharov(Benchmark):
	r"""Implementations of Zakharov functions.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Function:
	**Levy Function**

		:math:`f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-5, 10]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:**
		:math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (0, \cdots, 0)`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4$

		Equation:
				\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4 \end{equation}

		Domain:
				$-5 \leq x_i \leq 10$

	Reference:
		https://www.sfu.ca/~ssurjano/levy.html
	"""
	Name = ['Zakharov']

	def __init__(self, Lower=-5.0, Upper=10.0):
		r"""Initialize of Zakharov benchmark.

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
		return r'''$f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4$'''

	def function(self):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function
		"""
		def f(D, X):
			r"""Fitness function.

			Args:
				D (int): Dimensionality of the problem
				X (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

			Returns:
				float: Fitness value for the solution.
			"""
			v1, v2 = 0.0, 0.0
			for i in range(D): v1, v2 = v1 + X[i] ** 2, v2 + 0.5 * (i + 1) * X[i]
			return v1 + v2 ** 2 + v2 ** 4
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
