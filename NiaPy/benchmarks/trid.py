# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
"""Implementations of Levy function."""

__all__ = ['Trid']

class Trid:
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
	def __init__(self, D=2): self.Lower, self.Upper, = -D ** 2, D ** 2

	@classmethod
	def function(cls):
		def f(D, X):
			v1, v2 = 0.0, 0.0
			for i in range(D): v1 += (X[i] - 1) ** 2
			for i in range(1, D): v2 += X[i] * X[i - 1]
			return v1 - v2
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
