# encoding=utf8
import logging
from scipy.special import gamma as Gamma
from numpy import where, argmin, asarray, apply_along_axis, full, sin, fabs, pi
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

	**Reference paper:**
	Yang, Xin-She. "Flower pollination algorithm for global optimization. International conference on unconventional computing and natural computation. Springer, Berlin, Heidelberg, 2012.

	Implementation is based on the following MATLAB code:
	https://www.mathworks.com/matlabcentral/fileexchange/45112-flower-pollination-algorithm?requestedDomain=true
	"""
	def __init__(self, **kwargs): super(FlowerPollinationAlgorithm, self).__init__(name='FlowerPollinationAlgorithm', sName='FPA', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP=25, p=0.35, beta=1.5, **ukwargs):
		r"""**__init__(self, D, NP, nFES, p, benchmark)**.

		Arguments:
		NP {integer} -- population size
		p {decimal} -- probability switch
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

	def levy(self, task):
		sigma = (Gamma(1 + self.beta) * sin(pi * self.beta / 2) / (Gamma(1 + self.beta) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
		return 0.01 * (self.rand.normal(0, 1) * sigma / fabs(self.rand.normal(0, 1)) ** (1 / self.beta))

	def runTask(self, task):
		Sol = self.rand.uniform(task.Lower, task.Upper, [self.NP, task.D])
		Sol_f = apply_along_axis(task.eval, 1, Sol)
		ib = argmin(Sol_f)
		solb, solb_f = Sol[ib], Sol_f[ib]
		S, dS = full([self.NP, task.D], 0.0), full([self.NP, task.D], 0.0)
		while not task.stopCond():
			for i in range(self.NP):
				if self.rand.uniform(0, 1) > self.p: S[i] += self.levy(task) * (Sol[i] - solb)
				else: 
					JK = self.rand.permutation(self.NP)
					S[i] += self.rand.uniform(0, 1) * (Sol[JK[0]] - Sol[JK[1]])
				S[i] = self.repair(S[i], task)
				f_i = task.eval(S[i])
				if f_i <= Sol_f[i]: Sol[i], Sol_f[i] = S[i], f_i
				if f_i <= solb_f: solb, solb_f = S[i], f_i
		return solb, solb_f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
