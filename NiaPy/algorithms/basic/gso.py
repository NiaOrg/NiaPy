# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, line-too-long, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, redefined-builtin, singleton-comparison, unused-argument, arguments-differ, no-else-return
import logging
from scipy.spatial.distance import euclidean
from numpy import full, apply_along_axis, argmin, copy, sum, inf, fmax, pi, where
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

		**Arguments:**

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
		while i == j: j = self.randint(self.n)
		return j

	def getNeighbors(self, i, r, GS, L):
		N = full(self.n, 0)
		for j, gw in enumerate(GS): N[j] = 1 if i != j and euclidean(GS[i], gw) <= r and L[i] >= L[j] else 0
		return N

	def probabilityes(self, i, N, L):
		d, P = sum(L[where(N == 1)] - L[i]), full(self.n, .0)
		for j in range(self.n): P[i] = ((L[j] - L[i]) / d) if N[j] == 1 else 0
		return P

	def moveSelect(self, pb, i):
		r, b_l, b_u = self.rand(), 0, 0
		for j in range(self.n):
			b_l, b_u = b_u, b_u + pb[i]
			if b_l < r < b_u: return j
		return self.randint(self.n)

	def calcLuciferin(self, L, GS_f): return (1 - self.rho) * L + self.gamma * GS_f

	def rangeUpdate(self, R, N, rs): return R + self.beta * (self.nt - sum(N))

	def getBest(self, GS, GS_f, xb, xb_f):
		ib = argmin(GS_f)
		if GS_f[ib] < xb_f: return GS[ib], GS_f[ib]
		else: return xb, xb_f

	def runTask(self, task):
		rs = euclidean(full(task.D, 0), task.bRange)
		GS, GS_f, L, R = self.uniform(task.Lower, task.Upper, [self.n, task.D]), full(self.n, inf), full(self.n, self.l0), full(self.n, rs)
		xb, xb_f = None, inf
		while not task.stopCondI():
			GSo, Ro, GS_f = copy(GS), copy(R), apply_along_axis(task.eval, 1, GS)
			xb, xb_f = self.getBest(GS, GS_f, xb, xb_f)
			L = self.calcLuciferin(L, GS_f)
			N = [self.getNeighbors(i, Ro[i], GSo, L) for i in range(self.n)]
			P = [self.probabilityes(i, N[i], L) for i in range(self.n)]
			j = [self.moveSelect(P[i], i) for i in range(self.n)]
			for i in range(self.n): GS[i] = task.repair(GSo[i] + self.s * ((GSo[j[i]] - GSo[i]) / (euclidean(GSo[j[i]], GSo[i]) + 1e-31)))
			for i in range(self.n): R[i] = max(0, min(rs, self.rangeUpdate(Ro[i], N[i], rs)))
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

		**Arguments:**

		alpha {real} --
		"""
		self.alpha = alpha
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def calcLuciferin(self, L, GS_f): return fmax(0, (1 - self.rho) * L + self.gamma * GS_f)

	def rangeUpdate(self, R, N, rs): return rs / (1 + self.beta * (sum(N) / (pi * rs ** 2)))

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

		**Arguments:**

		beta1 {real} --

		s {real} --
		"""
		self.alpha = alpha
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def rangeUpdate(self, P, N, rs): return self.alpha + (rs - self.alpha) / (1 + self.beta * sum(N))

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

		**Arguments:**

		beta1 {real} --

		s {real} --
		"""
		self.beta1 = beta1
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def rangeUpdate(self, R, N, rs): return R + (self.beta * sum(N)) if sum(N) < self.nt else (-self.beta1 * sum(N))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
