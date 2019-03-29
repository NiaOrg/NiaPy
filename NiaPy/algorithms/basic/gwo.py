# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, line-too-long, arguments-differ, bad-continuation
import logging
from numpy import fabs, inf, where, apply_along_axis
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['GreyWolfOptimizer']

class GreyWolfOptimizer(Algorithm):
	r"""Implementation of Grey wolf optimizer.

	Algorithm:
		Grey wolf optimizer

	Date:
		2018

	Author:
		Iztok Fister Jr. and Klemen Berkovič

	License:
		MIT

	Reference paper:
		* Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis. "Grey wolf optimizer." Advances in engineering software 69 (2014): 46-61.
		* Grey Wold Optimizer (GWO) source code version 1.0 (MATLAB) from MathWorks
	"""
	Name = ['GreyWolfOptimizer', 'GWO']

	@staticmethod
	def typeParameters(): return {
			'NP': lambda x: isinstance(x, int) and x > 0
	}

	def setParameters(self, NP=25, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			NP (int): Number of individuals in population

		See Also:
			:func:`Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP)
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def initPopulation(self, task):
		pop, fpop, = Algorithm.initPopulation(self, task)
		A, A_f, B, B_f, D, D_f = None, task.optType.value * inf, None, task.optType.value * inf, None, task.optType.value * inf
		for i, f in enumerate(fpop):
			if f < A_f: A, A_f = pop[i], f
			elif A_f < f < B_f: B, B_f = pop[i], f
			elif B_f < f < D_f: D, D_f = pop[i], f
		return pop, fpop, {'A':A, 'A_f':A_f, 'B':B, 'B_f':B_f, 'D':D, 'D_f':D_f}

	def runIteration(self, task, pop, fpop, xb, fxb, A, A_f, B, B_f, D, D_f, **dparams):
		r"""

		Args:
			task:
			pop:
			fpop:
			xb:
			fxb:
			A:
			A_f:
			B:
			B_f:
			D:
			D_f:
			**dparams:

		Returns:

		"""
		a = 2 - task.Evals * (2 / task.nFES)
		for i, w in enumerate(pop):
			A1, C1 = 2 * a * self.rand(task.D) - a, 2 * self.rand(task.D)
			X1 = A - A1 * fabs(C1 * A - w)
			A2, C2 = 2 * a * self.rand(task.D) - a, 2 * self.rand(task.D)
			X2 = B - A2 * fabs(C2 * B - w)
			A3, C3 = 2 * a * self.rand(task.D) - a, 2 * self.rand(task.D)
			X3 = D - A3 * fabs(C3 * D - w)
			pop[i] = task.repair((X1 + X2 + X3) / 3, self.Rand)
			fpop[i] = task.eval(pop[i])
		for i, f in enumerate(fpop):
			if f < A_f: A, A_f = pop[i], f
			elif A_f < f < B_f: B, B_f = pop[i], f
			elif B_f < f < D_f: D, D_f = pop[i], f
		return pop, fpop, {'A':A, 'A_f':A_f, 'B':B, 'B_f':B_f, 'D':D, 'D_f':D_f}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
