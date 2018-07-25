# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy
import logging
from numpy import apply_along_axis, argmin, argsort, fabs, inf, where
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['GreyWolfOptimizer']

class GreyWolfOptimizer(Algorithm):
	r"""Implementation of Grey wolf optimizer.

	**Algorithm:** Grey wolf optimizer

	**Date:** 2018

	**Author:** Iztok Fister Jr. and Klemen Berkovič

	**License:** MIT

	**Reference paper:**
	Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis.
	"Grey wolf optimizer." Advances in engineering software 69 (2014): 46-61.
	& Grey Wold Optimizer (GWO) source code version 1.0 (MATLAB) from MathWorks
	"""
	def __init__(self, **kwargs): super(GreyWolfOptimizer, self).__init__(name='GreyWolfOptimizer', sName='GWO', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP=25, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
		NP {integer} -- Number of individuals in population
		"""
		self.NP = NP
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def repair(self, x, task):
		"""Find limits."""
		ir = where(x > task.Upper)
		x[ir] = task.Upper[ir]
		ir = where(x < task.Lower)
		x[ir] = task.Lower[ir]
		return x

	def runTask(self, task):
		"""Run."""
		pop = task.Lower + task.bRange * self.rand.rand(self.NP, task.D)
		A, A_f, B, B_f, D, D_f = None, inf, None, inf, None, inf
		while not task.stopCond():
			for i in range(self.NP):
				pop[i] = self.repair(pop[i], task)
				f = task.eval(pop[i])
				if f < A_f: A, A_f = pop[i], f
				elif f > A_f and f < B_f: B, B_f = pop[i], f
				elif f > B_f and f < D_f: D, D_f = pop[i], f
			a = 2 - task.Evals * (2 / task.nFES)
			for i, w in enumerate(pop):
				A1, C1 = 2 * a * self.rand.rand(task.D) - a, 2 * self.rand.rand(task.D)
				X1 = A - A1 * fabs(C1 * A - w)
				A2, C2 = 2 * a * self.rand.rand(task.D) - a, 2 * self.rand.rand(task.D)
				X2 = B - A2 * fabs(C2 * B - w)
				A3, C3 = 2 * a * self.rand.rand(task.D) - a, 2 * self.rand.rand(task.D)
				X3 = D - A3 * fabs(C3 * D - w)
				pop[i] = (X1 + X2 + X3) / 3
		return A, A_f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
