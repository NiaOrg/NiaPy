# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, logging-not-lazy, attribute-defined-outside-init, arguments-differ, bad-continuation
import logging
from numpy import full, apply_along_axis, argmin
from NiaPy.algorithms.basic import BatAlgorithm
from NiaPy.algorithms.basic.de import CrossBest1

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['HybridBatAlgorithm']

class HybridBatAlgorithm(BatAlgorithm):
	r"""Implementation of Hybrid bat algorithm.

	**Algorithm:** Hybrid bat algorithm

	**Date:** 2018

	**Author:** Grega Vrbancic and Klemen Berkovič

	**License:** MIT

	**Reference paper:** Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She. "A Hybrid Bat Algorithm". Elektrotehniski vestnik, 2013. 1-7.
	"""
	Name = ['HybridBatAlgorithm', 'HBA']

	@staticmethod
	def typeParameters():
		d = BatAlgorithm.typeParameters()
		d['F'] = lambda x: isinstance(x, (int, float)) and x > 0
		d['CR'] = lambda x: isinstance(x, float) and 0 <= x <= 1
		return d

	def setParameters(self, F=0.78, CR=0.35, CrossMutt=CrossBest1, **ukwargs):
		r"""**__init__(self, D, NP, nFES, A, r, Qmin, Qmax, benchmark)**.

		**Arguments:**

		F {decimal} -- scaling factor

		CR {decimal} -- crossover
		"""
		BatAlgorithm.setParameters(self, **ukwargs)
		self.F, self.CR, self.CrossMutt = F, CR, CrossMutt
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		v, Sol = full([self.NP, task.D], 0.0), task.Lower + task.bRange * self.rand([self.NP, task.D])
		Fitness = apply_along_axis(task.eval, 1, Sol)
		ib = argmin(Fitness)
		best, f_min = Sol[ib], Fitness[ib]
		while not task.stopCondI():
			Q = self.Qmin + (self.Qmax - self.Qmin) * self.uniform(0, 1, self.NP)
			for i in range(self.NP):
				v[i] = v[i] + (Sol[i] - best) * Q[i]
				S = task.repair(Sol[i] + v[i], rnd=self.Rand)
				if self.rand() > self.r: S = task.repair(self.CrossMutt(Sol, i, best, self.F, self.CR, rnd=self.Rand), rnd=self.Rand)
				f_new = task.eval(S)
				if f_new <= Fitness[i] and self.rand() < self.A: Sol[i], Fitness[i] = S, f_new
				if f_new < f_min: best, f_min = S, f_new
		return best, f_min

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
