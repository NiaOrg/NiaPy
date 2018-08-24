# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, old-style-class
"""Implementations of Shekel function."""

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Shekel']

class Shekel(Benchmark):
	r"""Implementations of Shekel functions.

	Date: 2018
	Author: Klemen Berkovič
	License: MIT

	Function:
	**Shekel Function**
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
	Name = ['Shekel']

	def __init__(self, C, Lower=-5.0, Upper=10.0):
		Benchmark.__init__(self, Lower, Upper)
		self.C = C

	@classmethod
	def function(cls): pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
