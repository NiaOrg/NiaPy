# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, multiple-statements, old-style-class
"""Implementations of HGBat functions."""

from math import fabs

__all__ = ['HGBat']

class HGBat:
	r"""Implementations of HGBat functions.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Function:
	**HGBat Function**

		:math:``f(\textbf{x}) = \left| \left( \sum_{i=1}^D x_i^2 \right)^2 - \left( \sum_{i=1}^D x_i \right)^2 \right|^{\frac{1}{2}} + \frac{0.5 \sum_{i=1}^D x_i^2 + \sum_{i=1}^D x_i}{D} + 0.5

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:**
		:math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
				$$f(\textbf{x}) = \left| \left( \sum_{i=1}^D x_i^2 \right)^2 - \left( \sum_{i=1}^D x_i \right)^2 \right|^{\frac{1}{2}} + \frac{0.5 \sum_{i=1}^D x_i^2 + \sum_{i=1}^D x_i}{D} + 0.5

		Equation:
				\begin{equation} f(\textbf{x}) = \left| \left( \sum_{i=1}^D x_i^2 \right)^2 - \left( \sum_{i=1}^D x_i \right)^2 \right|^{\frac{1}{2}} + \frac{0.5 \sum_{i=1}^D x_i^2 + \sum_{i=1}^D x_i}{D} + 0.5 \end{equation}

		Domain:
				$-100 \leq x_i \leq 100$

	Reference:
	http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf
	"""
	def __init__(self, Lower=-100.0, Upper=100.0): self.Lower, self.Upper = Lower, Upper

	@classmethod
	def function(cls):
		def f(D, x):
			val1, val2 = 0.0, 0.0
			for i in range(D): val1 += x[i] ** 2
			for i in range(D): val2 += x[i]
			return fabs(val1 ** 2 - val2 ** 2) ** (1 / 2) + (0.5 * val1 + val2) / D + 0.5
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
