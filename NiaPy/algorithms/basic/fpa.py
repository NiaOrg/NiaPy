# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, logging-not-lazy, no-self-use, attribute-defined-outside-init, arguments-differ
import logging
from scipy.special import gamma as Gamma
from numpy import where, argmin, sin, fabs, pi, apply_along_axis, full
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['FlowerPollinationAlgorithm']

class FlowerPollinationAlgorithm(Algorithm):
	r"""Implementation of Flower Pollination algorithm.

	**Algorithm:** Flower Pollination algorithm

	**Date:** 2018

	**Authors:** Dusan Fister, Iztok Fister Jr. and Klemen Berkovič

	**License:** MIT

	**Reference paper:** Yang, Xin-She. "Flower pollination algorithm for global optimization. International conference on unconventional computing and natural computation. Springer, Berlin, Heidelberg, 2012.
	Implementation is based on the following MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/45112-flower-pollination-algorithm?requestedDomain=true
	"""
	def __init__(self, **kwargs): Algorithm.__init__(self, name='FlowerPollinationAlgorithm', sName='FPA', **kwargs)

	def setParameters(self, NP=25, p=0.35, beta=1.5, **ukwargs):
		r"""**__init__(self, D, NP, nFES, p, benchmark)**.

		**Arguments:**

		NP {integer} -- population size

		p {decimal} -- probability switch

		beta {real} --
		"""
		self.NP, self.p, self.beta = NP, p, beta
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def repair(self, x, task):
		"""Find limits."""
		ir = where(x > task.Upper)
		x[ir] = task.Upper[ir]
		ir = where(x < task.Lower)
		x[ir] = task.Lower[ir]
		return x

	def levy(self):
		sigma = (Gamma(1 + self.beta) * sin(pi * self.beta / 2) / (Gamma(1 + self.beta) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
		return 0.01 * (self.normal(0, 1) * sigma / fabs(self.normal(0, 1)) ** (1 / self.beta))

	def runTask(self, task):
		Sol = self.uniform(task.Lower, task.Upper, [self.NP, task.D])
		Sol_f = apply_along_axis(task.eval, 1, Sol)
		ib = argmin(Sol_f)
		solb, solb_f = Sol[ib], Sol_f[ib]
		while not task.stopCond():
			for i in range(self.NP):
				S = full(task.D, 0.0)
				if self.uniform(0, 1) > self.p: S += self.levy() * (Sol[i] - solb)
				else:
					JK = self.Rand.permutation(self.NP)
					S += self.uniform(0, 1) * (Sol[JK[0]] - Sol[JK[1]])
				S = self.repair(S, task)
				f_i = task.eval(S)
				if f_i <= Sol_f[i]: Sol[i], Sol_f[i] = S, f_i
				if f_i <= solb_f: solb, solb_f = S, f_i
		return solb, solb_f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
