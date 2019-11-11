# encoding=utf8

"""Implementations of Michalewichz's function."""

from numpy import sin, pi
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Michalewichz']

class Michalewichz(Benchmark):
	r"""Implementations of Michalewichz's functions.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Function:
	**High Conditioned Elliptic Function**

		:math:`f(\textbf{x}) = \sum_{i=1}^D \left( 10^6 \right)^{ \frac{i - 1}{D - 1} } x_i^2`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [0, \pi]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:**
		at :math:`d = 2` :math:`f(\textbf{x}^*) = -1.8013` at :math:`\textbf{x}^* = (2.20, 1.57)`
		at :math:`d = 5` :math:`f(\textbf{x}^*) = -4.687658`
		at :math:`d = 10` :math:`f(\textbf{x}^*) = -9.66015`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m}$

		Equation:
				\begin{equation} f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m} \end{equation}

		Domain:
				$0 \leq x_i \leq \pi$

	Reference URL:
		https://www.sfu.ca/~ssurjano/michal.html
	"""
	Name = ['Michalewichz']

	def __init__(self, Lower=0.0, Upper=pi, m=10):
		r"""Initialize of Michalewichz benchmark.

		Args:
			Lower (Optional[float]): Lower bound of problem.
			Upper (Optional[float]): Upper bound of problem.

		See Also:
			:func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper)
		Michalewichz.m = m

	@staticmethod
	def latex_code():
		r"""Return the latex code of the problem.

		Returns:
			str: Latex code
		"""
		return r'''$f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m}$'''

	@classmethod
	def function(cls):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function
		"""
		def evaluate(D, X):
			r"""Fitness function.

			Args:
				D (int): Dimensionality of the problem
				sol (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

			Returns:
				float: Fitness value for the solution.
			"""
			v = 0.0
			for i in range(D): v += sin(X[i]) * sin(((i + 1) * X[i] ** 2) / pi) ** (2 * cls.m)
			return -v
		return evaluate

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
