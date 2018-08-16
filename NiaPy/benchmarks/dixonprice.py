# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, old-style-class
"""Implementations of Dixon Price function."""

__all__ = ['DixonPrice']

class DixonPrice:
	r"""Implementations of Dixon Price function.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Function:
	**Levy Function**

		:math:`f(\textbf{x}) = (x_1 - 1)^2 + \sum_{i = 2}^D i (2x_i^2 - x_{i - 1})^2`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:**
		:math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (2^{-\frac{2^1 - 2}{2^1}}, \cdots , 2^{-\frac{2^i - 2}{2^i}} , \cdots , 2^{-\frac{2^D - 2}{2^D}})`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = (x_1 - 1)^2 + \sum_{i = 2}^D i (2x_i^2 - x_{i - 1})^2$

		Equation:
				\begin{equation} f(\textbf{x}) = (x_1 - 1)^2 + \sum_{i = 2}^D i (2x_i^2 - x_{i - 1})^2 \end{equation}

		Domain:
				$-10 \leq x_i \leq 10$

	Reference:
	https://www.sfu.ca/~ssurjano/dixonpr.html
	"""
	def __init__(self, Lower=-10.0, Upper=10): self.Lower, self.Upper, = Lower, Upper

	@classmethod
	def function(cls):
		def f(D, X):
			v = 0.0
			for i in range(2, D): v += i * (2 * X[i] ** 2 - X[i - 1]) ** 2
			return (X[0] - 1) ** 2 + v
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
