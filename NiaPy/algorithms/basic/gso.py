# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, line-too-long, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, redefined-builtin
import logging
from scipy.spatial.distance import euclidean
from numpy import full, apply_along_axis, argmax, argmin, copy, sum, inf
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['GlowwormSwarmOptimization']

class GlowwormSwarmOptimization(Algorithm):
	r"""Implementation of glowwarm swarm optimization.

	**Algorithm:** Glowwarm Swarm Optimization Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** https://www.springer.com/gp/book/9783319515946
	**Reference paper:** Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.
	"""
	def __init__(self, **kwargs): super(GlowwormSwarmOptimization, self).__init__(name='GlowwormSwarmOptimization', sName='GSO', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, n=25, l0=5, nt=5, rs=40, rho=0.4, gamma=0.6, beta=0.08, s=0.03, d=euclidean, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		n {integer} -- number of glowworms in population
		l0 {real} -- initial luciferin quantity for each glowworm
		nt {real} --
		rs {real} -- maximum sensing range
		rho {real} -- luciferin decay constant
		gamma {real} -- luciferin enhancement constant
		beta {real} --
		s {real} --
		"""
		self.n, self.l0, self.nt, self.rs, self.rho, self.gamma, self.beta, self.s, self.d = n, l0, nt, rs, rho, gamma, beta, s, d
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def randMove(self, i):
		j = i
		while i == j: j = self.rand.randint(self.n)
		return j

	def getNeighbors(self, i, r, GS, L):
		N = list()
		for j, gw in enumerate(GS):
			if i != j and self.d(GS[i], gw) < r and L[i] > L[j]: N.append(j)
		return N

	def potentialShift(self, i, N, L):
		d, P = sum(L[N] - L[i]), list()
		for l in L[N]: P.append((l - L[i]) / d)
		return P

	def runTask(self, task):
		GS, L, R = self.rand.uniform(task.Lower, task.Upper, [self.n, task.D]), full(self.n, self.l0), full(self.n, self.rs)
		GS_f, xb, xb_f = full(self.n, inf), None, inf
		while not task.stopCondI():
			GSo, Ro, GS_f = copy(GS), copy(R), apply_along_axis(task.eval, 1, GS)
			L = (1 - self.rho) * L + self.gamma * GS_f
			for i, gw in enumerate(GSo):
				N = self.getNeighbors(i, Ro[i], GSo, L)
				P = self.potentialShift(i, N, L)
				j = self.randMove(i) if not P else N[argmax(P)]
				GS[i] = task.repair(gw + self.s * ((GSo[j] - gw) / euclidean(GSo[j], gw)))
				R[i] = min(self.rs, max(0, Ro[i] + self.beta * (self.nt - len(N))))
			ib = argmin(GS_f)
			if GS_f[ib] < xb_f: xb, xb_f = GSo[ib], GS_f[ib]
		return xb, xb_f 

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
