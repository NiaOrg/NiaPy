# encoding=utf8

"""Implementations of Levy function."""

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Trid']

class Trid(Benchmark):
	r"""Implementations of Trid functions.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Function:
	**Levy Function**

		:math:`f(\textbf{x}) = \sum_{i = 1}^D \left( x_i - 1 \right)^2 - \sum_{i = 2}^D x_i x_{i - 1}`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-D^2, D^2]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:**
		:math:`f(\textbf{x}^*) = \frac{-D(D + 4)(D - 1)}{6}` at :math:`\textbf{x}^* = (1 (D + 1 - 1), \cdots , i (D + 1 - i) , \cdots , D (D + 1 - D))`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = \sum_{i = 1}^D \left( x_i - 1 \right)^2 - \sum_{i = 2}^D x_i x_{i - 1}$

		Equation:
				\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \left( x_i - 1 \right)^2 - \sum_{i = 2}^D x_i x_{i - 1} \end{equation}

		Domain:
				$-D^2 \leq x_i \leq D^2$

	Reference:
		https://www.sfu.ca/~ssurjano/trid.html
	"""
	Name = ['Trid']

	def __init__(self, D=2):
		r"""Initialize of Trid benchmark.

		Args:
			Lower (Optional[float]): Lower bound of problem.
			Upper (Optional[float]): Upper bound of problem.

		See Also:
			:func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, -(D ** 2), D ** 2)

	@staticmethod
	def latex_code():
		r"""Return the latex code of the problem.

		Returns:
			str: Latex code
		"""
		return r'''$f(\textbf{x}) = \sum_{i = 1}^D \left( x_i - 1 \right)^2 - \sum_{i = 2}^D x_i x_{i - 1}$'''

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
			for i in range(D): v1 += (X[i] - 1) ** 2
			for i in range(1, D): v2 += X[i] * X[i - 1]
			return v1 - v2
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
