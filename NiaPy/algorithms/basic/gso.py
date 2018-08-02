# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, line-too-long, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, redefined-builtin, singleton-comparison, unused-argument, arguments-differ
import logging
from scipy.spatial.distance import euclidean
from numpy import full, apply_along_axis, argmax, argmin, copy, sum, inf, fmax, pi, where
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['GlowwormSwarmOptimization', 'GlowwormSwarmOptimizationV1', 'GlowwormSwarmOptimizationV2', 'GlowwormSwarmOptimizationV3']

class GlowwormSwarmOptimization(Algorithm):
	r"""Implementation of glowwarm swarm optimization.

	**Algorithm:** Glowwarm Swarm Optimization Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** https://www.springer.com/gp/book/9783319515946
	**Reference paper:** Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: Algorithm.__init__(self, name='GlowwormSwarmOptimization', sName='GSO', **kwargs)
		else: Algorithm.__init__(self, **kwargs)

	def setParameters(self, n=25, l0=5, nt=5, rho=0.4, gamma=0.6, beta=0.08, s=0.03, **ukwargs):
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
		self.n, self.l0, self.nt, self.rho, self.gamma, self.beta, self.s = n, l0, nt, rho, gamma, beta, s
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def randMove(self, i):
		j = i
		while i == j: j = self.rand.randint(self.n)
		return j

	def getNeighbors(self, i, r, GS, L):
		N = list()
		for j, gw in enumerate(GS):
			if i != j and euclidean(GS[i], gw) < r and L[i] > L[j]: N.append(j)
		return N

	def potentialShift(self, i, N, L):
		d, P = sum(L[N] - L[i]), list()
		for l in L[N]: P.append((l - L[i]) / d)
		return P

	def calcLuciferin(self, L, GS_f): return (1 - self.rho) * L + self.gamma * GS_f

	def rangeUpdate(self, R, N, rs): return R + self.beta * (self.nt - len(N))

	def runTask(self, task):
		rs = euclidean(full(task.D, 0), task.bRange)
		GS, GS_f, L, R = self.rand.uniform(task.Lower, task.Upper, [self.n, task.D]), full(self.n, inf), full(self.n, self.l0), full(self.n, rs)
		Mu, xb, xb_f = full(self.n, True), None, inf
		while not task.stopCondI():
			ie = where(Mu == True)
			GSo, Ro, GS_f[ie] = copy(GS), copy(R), apply_along_axis(task.eval, 1, GS[ie])
			L = self.calcLuciferin(L, GS_f)
			for i, gw in enumerate(GSo):
				N = self.getNeighbors(i, Ro[i], GSo, L)
				if N:
					Mu[i], P = self.potentialShift(i, N, L), True
					j = N[argmax(P)]
					GS[i] = task.repair(gw + self.s * ((GSo[j] - gw) / euclidean(GSo[j], gw)))
				else: Mu[i] = False
				R[i] = min(rs, max(0, self.rangeUpdate(Ro[i], N, rs)))
			ib = argmin(GS_f)
			if GS_f[ib] < xb_f: xb, xb_f = GSo[ib], GS_f[ib]
		return xb, xb_f

class GlowwormSwarmOptimizationV1(GlowwormSwarmOptimization):
	r"""Implementation of glowwarm swarm optimization.

	**Algorithm:** Glowwarm Swarm Optimization Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** https://www.springer.com/gp/book/9783319515946
	**Reference paper:** Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.
	"""
	def __init__(self, **kwargs): GlowwormSwarmOptimization.__init__(self, name='GlowwormSwarmOptimizationV1', sName='GSOv1', **kwargs)

	def setParameters(self, **kwargs):
		self.__setParams(**kwargs)
		GlowwormSwarmOptimization.setParameters(self, **kwargs)

	def __setParams(self, alpha=0.2, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		alpha {real} --
		"""
		self.alpha = alpha
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def calcLuciferin(self, L, GS_f): return fmax(0, (1 - self.rho) * L + self.gamma * GS_f)

	def rangeUpdate(self, R, N, rs): return rs / (1 + self.beta * (len(N) / (pi * rs ** 2)))

class GlowwormSwarmOptimizationV2(GlowwormSwarmOptimization):
	r"""Implementation of glowwarm swarm optimization.

	**Algorithm:** Glowwarm Swarm Optimization Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** https://www.springer.com/gp/book/9783319515946
	**Reference paper:** Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.
	"""
	def __init__(self, **kwargs): GlowwormSwarmOptimization.__init__(self, name='GlowwormSwarmOptimizationV2', sName='GSOv2', **kwargs)

	def setParameters(self, **kwargs):
		self.__setParams(alpha=kwargs.pop('alpha', 0.2), **kwargs)
		GlowwormSwarmOptimization.setParameters(self, **kwargs)

	def __setParams(self, alpha=0.2, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		beta1 {real} --
		s {real} --
		"""
		self.alpha = alpha
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def rangeUpdate(self, P, N, rs): return self.alpha + (rs - self.alpha) / (1 + self.beta * len(N))

class GlowwormSwarmOptimizationV3(GlowwormSwarmOptimization):
	r"""Implementation of glowwarm swarm optimization.

	**Algorithm:** Glowwarm Swarm Optimization Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** https://www.springer.com/gp/book/9783319515946
	**Reference paper:** Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.
	"""
	def __init__(self, **kwargs): GlowwormSwarmOptimization.__init__(self, name='GlowwormSwarmOptimizationV2', sName='GSOv2', **kwargs)

	def setParameters(self, **kwargs):
		self.__setParams(beta1=kwargs.pop('beta1', 0.2), **kwargs)
		GlowwormSwarmOptimization.setParameters(self, **kwargs)

	def __setParams(self, beta1=0.2, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		beta1 {real} --
		s {real} --
		"""
		self.beta1 = beta1
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def rangeUpdate(self, R, N, rs): return R + (self.beta * len(N)) if len(N) < self.nt else (-self.beta1 * len(N))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
