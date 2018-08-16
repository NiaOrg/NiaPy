# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, line-too-long, singleton-comparison, arguments-differ
import logging
from numpy import full, apply_along_axis, argmin
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['BatAlgorithm']

class BatAlgorithm(Algorithm):
	r"""Implementation of Bat algorithm.

	**Algorithm:** Bat algorithm

	**Date:** 2015

	**Authors:** Iztok Fister Jr., Marko Burjek and Klemen Berkovič

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
		if kwargs.get('name', None) == None: Algorithm.__init__(self, name=kwargs.get('name', 'BatAlgorithm'), sName=kwargs.get('sName', 'BA'), **kwargs)
		else: Algorithm.__init__(self, **kwargs)

	def setParameters(self, NP, A, r, Qmin, Qmax, **ukwargs):
		r"""Set the parameters of the algorithm.

		**Arguments:**

		NP {integer} -- population size

		A {decimal} -- loudness

		r {decimal} -- pulse rate

		Qmin {decimal} -- minimum frequency

		Qmax {decimal} -- maximum frequency
		"""
		self.NP, self.A, self.r, self.Qmin, self.Qmax = NP, A, r, Qmin, Qmax
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		r"""Run algorithm with initialized parameters.

		**Return:**

		{decimal} -- coordinates of minimal found objective function

		{decimal} -- minimal value found of objective function
		"""
		S, Q, v = full([self.NP, task.D], 0.0), full(self.NP, 0.0), full([self.NP, task.D], 0.0)
		Sol = task.Lower + task.bRange * self.uniform(0, 1, [self.NP, task.D])
		Fitness = apply_along_axis(task.eval, 1, Sol)
		j = argmin(Fitness)
		best, f_min = Sol[j], Fitness[j]
		while not task.stopCond():
			for i in range(self.NP):
				Q[i] = self.Qmin + (self.Qmax - self.Qmin) * self.uniform(0, 1)
				v[i] = v[i] + (Sol[i] - best) * Q[i]
				S[i] = Sol[i] + v[i]
				S[i] = task.repair(S[i])
				if self.rand() > self.r:
					S[i] = best + 0.001 * self.normal(0, 1, task.D)
					S[i] = task.repair(S[i])
				Fnew = task.eval(S[i])
				if (Fnew <= Fitness[i]) and (self.rand() < self.A): Sol[i], Fitness[i] = S[i], Fnew
				if Fnew <= f_min: best, f_min = S[i], Fnew
		return best, f_min

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
