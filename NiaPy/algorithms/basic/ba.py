# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use
import logging
from numpy import full, apply_along_axis, argmin, where
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['BatAlgorithm']

class BatAlgorithm(Algorithm):
	r"""Implementation of Bat algorithm.

	**Algorithm:** Bat algorithm

	**Date:** 2015

	**Authors:** Iztok Fister Jr. and Marko Burjek

	**License:** MIT

	**Reference paper:**
	Yang, Xin-She. "A new metaheuristic bat-inspired algorithm."
	Nature inspired cooperative strategies for optimization (NICSO 2010).
	Springer, Berlin, Heidelberg, 2010. 65-74.
	"""
	def __init__(self, **kwargs):
		r"""**__init__(self, D, NP, nFES, A, r, Qmin, Qmax, benchmark)**.

		**See**:
		Algorithm.__init__(self, **kwargs)
		"""
		super(BatAlgorithm, self).__init__(name='BatAlgorithm', sName='BA', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP, A, r, Qmin, Qmax, **ukwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
		NP {integer} -- population size
		A {decimal} -- loudness
		r {decimal} -- pulse rate
		Qmin {decimal} -- minimum frequency
		Qmax {decimal} -- maximum frequency
		"""
		self.NP = NP  # population size
		self.A = A  # loudness
		self.r = r  # pulse rate
		self.Qmin = Qmin  # frequency min
		self.Qmax = Qmax  # frequency max
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def best_bat(self):
		"""Find the best bat."""
		j = argmin(self.Fitness)
		self.best = self.Sol[j]
		self.f_min = self.Fitness[j]

	def init_bat(self, task):
		"""Initialize population."""
		self.Q = full(self.NP, 0)
		self.v = full([self.NP, task.D], 0)
		self.Sol = task.Lower + task.bRange * self.rand.uniform(0, 1, [self.NP, task.D])
		self.Fitness = apply_along_axis(task.eval, 1, self.Sol)
		self.best = full(task.D, 0)
		self.best_bat()

	def repair(self, val, task):
		"""Keep it within bounds."""
		ir = where(val > task.Upper)
		val[ir] = task.Upper[ir]
		ir = where(val < task.Lower)
		val[ir] = task.Lower[ir]
		return val

	def move_bat(self, task):
		"""Move bats in search space."""
		S = full([self.NP, task.D], 0)
		self.init_bat(task)
		while not task.stopCond():
			for i in range(self.NP):
				self.Q[i] = self.Qmin + (self.Qmax - self.Qmin) * self.rand.uniform(0, 1)
				self.v[i] = self.v[i] + (self.Sol[i] - self.best) * self.Q[i]
				S[i] = self.Sol[i] + self.v[i]
				S[i] = self.repair(S[i], task)
				if self.rand.rand() > self.r:
					S[i] = self.best + 0.001 * self.rand.normal(0, 1, task.D)
					S[i] = self.repair(S[i], task)
				Fnew = task.eval(S[i])
				if (Fnew <= self.Fitness[i]) and (self.rand.rand() < self.A): self.Sol[i], self.Fitness[i] = S[i], Fnew
				if Fnew <= self.f_min: self.best, self.f_min = S[i], Fnew
		return self.best, self.f_min

	def runTask(self, task):
		"""Run algorithm with initialized parameters.

		Return:
		{decimal} -- coordinates of minimal found objective funciton
		{decimal} -- minimal value found of objective function
		"""
		return self.move_bat(task)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
