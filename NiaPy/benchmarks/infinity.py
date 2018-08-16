# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
"""Implementations of Infinity function."""

from numpy import sin

__all__ = ['Infinity']

class Infinity:
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
	def __init__(self, Lower=-1.0, Upper=1.0): self.Lower, self.Upper = Lower, Upper

	@classmethod
	def function(cls):
		def f(D, X):
			val = 0.0
			for i in range(D): val += X[i] ** 6 * (sin(1 / X[i]) + 2)
			return val
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
