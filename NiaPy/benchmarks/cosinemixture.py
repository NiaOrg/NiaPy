# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
"""Implementations of Cosine mixture functions."""

from numpy import cos, pi

__all__ = ['CosineMixture']

class CosineMixture:
	r"""Implementations of Cosine mixture function.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Function:
	**Cosine Mixture Function**

		:math:`f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

		**Global maximu:**
		:math:`f(x^*) = -0.1 D`, at :math:`x^* = (0.0,...,0.0)`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2$

		Equation:
				\begin{equation} f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2 \end{equation}

		Domain:
				$-1 \leq x_i \leq 1$

	Reference:
	http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CosineMixture
	"""
	def __init__(self, Lower=-1.0, Upper=1.0): self.Lower, self.Upper = Lower, Upper

	@classmethod
	def function(cls):
		def f(D, X):
			v1, v2 = 0.0, 0.0
			for i in range(D): v1, v2 = v1 + cos(5 * pi * X[i]), v2 + X[i] ** 2
			return -0.1 * v1 - v2
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
