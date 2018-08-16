# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, singleton-comparison, multiple-statements, attribute-defined-outside-init, no-self-use, logging-not-lazy, unused-variable, arguments-differ, unused-argument, dangerous-default-value
import logging
from scipy.spatial.distance import euclidean
from numpy import apply_along_axis, argmin, full, inf, where, asarray, random as rand, sort, exp
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.benchmarks.utility import fullArray

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['AnarchicSocietyOptimization', 'Elitism', 'Sequential', 'Crossover']

def Elitism(x, xpb, xb, xr, MP_c, MP_s, MP_p, F, CR, task, rnd=rand):
	r"""Select the best of all three strategies."""
	xn = [task.repair(MP_C(x, F, CR, MP_c, rnd)), task.repair(MP_S(x, xr, xb, CR, MP_s, rnd)), task.repair(MP_P(x, xpb, CR, MP_p, rnd))]
	xn_f = apply_along_axis(task.eval, 1, xn)
	ib = argmin(xn_f)
	return xn[ib], xn_f[ib]

def Sequential(x, xpb, xb, xr, MP_c, MP_s, MP_p, F, CR, task, rnd=rand):
	r"""Sequentialy combines all three strategies."""
	xn = task.repair(MP_S(MP_P(MP_C(x, F, CR, MP_c, rnd), xpb, CR, MP_p, rnd), xr, xb, CR, MP_s, rnd))
	return xn, task.eval(xn)

def Crossover(x, xpb, xb, xr, MP_c, MP_s, MP_p, F, CR, task, rnd=rand):
	r"""Create a crossover over all three strategies."""
	xns = [task.repair(MP_C(x, F, CR, MP_c, rnd)), task.repair(MP_S(x, xr, xb, CR, MP_s, rnd)), task.repair(MP_P(x, xpb, CR, MP_p, rnd))]
	x = asarray([xns[rnd.randint(len(xns))][i] if rnd.rand() < CR else x[i] for i in range(len(x))])
	return x, task.eval(x)

def MP_C(x, F, CR, MP, rnd=rand):
	if MP < 0.5:
		b = sort(rnd.choice(len(x), 2, replace=False))
		x[b[0]:b[1]] = x[b[0]:b[1]] + F * rnd.normal(0, 1, b[1] - b[0])
		return x
	return asarray([x[i] + F * rnd.normal(0, 1) if rnd.rand() < CR else x[i] for i in range(len(x))])

def MP_S(x, xr, xb, CR, MP, rnd=rand):
	if MP < 0.25:
		b = sort(rnd.choice(len(x), 2, replace=False))
		x[b[0]:b[1]] = xb[b[0]:b[1]]
		return x
	if MP < 0.5:
		return asarray([xb[i] if rnd.rand() < CR else x[i] for i in range(len(x))])
	if MP < 0.75:
		b = sort(rnd.choice(len(x), 2, replace=False))
		x[b[0]:b[1]] = xr[b[0]:b[1]]
		return x

	return asarray([xr[i] if rnd.rand() < CR else x[i] for i in range(len(x))])

def MP_P(x, xpb, CR, MP, rnd=rand):
	if MP < 0.5:
		b = sort(rnd.choice(len(x), 2, replace=False))
		x[b[0]:b[1]] = xpb[b[0]:b[1]]
		return x
	return asarray([xpb[i] if rnd.rand() < CR else x[i] for i in range(len(x))])

class AnarchicSocietyOptimization(Algorithm):
	r"""Implementation of Anarchic Society Optimization algorithm.

	**Algorithm:** Particle Swarm Optimization algorithm

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference paper:** Ahmadi-Javid, Amir. "Anarchic Society Optimization: A human-inspired method." Evolutionary Computation (CEC), 2011 IEEE Congress on. IEEE, 2011.
	"""
	def __init__(self, **kwargs): Algorithm.__init__(self, name='ParticleSwarmAlgorithm', sName='PSO', **kwargs)

	def setParameters(self, NP=43, alpha=[1, 0.83], gamma=[1.17, 0.56], theta=[0.932, 0.832], d=euclidean, dn=euclidean, nl=1, F=1.2, CR=0.25, Combination=Elitism, **ukwargs):
		r"""Set the parameters for the algorith.

		**Arguments:**

		NP {integer} -- population size

		alpha {array} -- factor for fickleness index function $\in [0, 1]$

		gamma {array} -- factor for external irregularity index function $\in [0, \infty)$

		theta {array} -- factor for internal irregularity index function $\in [0, \infty)$

		d {function} -- function that takes two arguments that are function values and calcs the distance between them

		dn {function} -- function that takes two arguments that are points in function landscape and calcs the distance between them

		nl {real} -- normalized range for neighborhood search $\in (0, 1]$

		F {real} -- mutation parameter

		CR {real} -- crossover parameter $\in [0, 1]$

		Combination {function} -- Function that combines movment strategies
		"""
		self.NP, self.alpha, self.gamma, self.theta, self.d, self.dn, self.nl, self.F, self.CR, self.Combination = NP, alpha, gamma, theta, d, dn, nl, F, CR, Combination
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def init(self, task): return fullArray(self.alpha, self.NP), fullArray(self.gamma, self.NP), fullArray(self.theta, self.NP)

	def FI(self, x_f, xpb_f, xb_f, alpha):
		r"""Get fickleness index."""
		return 1 - alpha * xb_f / x_f - (1 - alpha) * xpb_f / x_f

	def EI(self, x_f, xnb_f, gamma):
		r"""Get external irregularity index."""
		return 1 - exp(-gamma * self.d(x_f, xnb_f))

	def II(self, x_f, xpb_f, theta):
		r"""Get internal irregularity index."""
		return 1 - exp(-theta * self.d(x_f, xpb_f))

	def getBestNeighbors(self, i, X, X_f, rs):
		nn = asarray([self.dn(X[i], X[j]) / rs for j in range(len(X))])
		return argmin(X_f[where(nn <= self.nl)])

	def uBestAndPBest(self, X, X_f, Xpb, Xpb_f):
		ix_pb = where(X_f < Xpb_f)
		Xpb[ix_pb], Xpb_f[ix_pb] = X[ix_pb], X_f[ix_pb]
		ib = argmin(Xpb_f)
		return Xpb, Xpb_f, Xpb[ib], Xpb_f[ib]

	def runTask(self, task):
		X, (alpha, gamma, theta), rs = self.uniform(task.Lower, task.Upper, [self.NP, task.D]), self.init(task), euclidean(full(task.D, 0.0), task.D)
		X_f = apply_along_axis(task.eval, 1, X)
		Xpb, Xpb_f, xb, xb_f = self.uBestAndPBest(X, X_f, full([self.NP, task.D], 0.0), full(self.NP, inf))
		while not task.stopCondI():
			Xin = [self.getBestNeighbors(i, X, X_f, rs) for i in range(len(X))]
			MP_c, MP_s, MP_p = [self.FI(X_f[i], Xpb_f[i], xb_f, alpha[i]) for i in range(len(X))], [self.EI(X_f[i], Xin[i], gamma[i]) for i in range(len(X))], [self.II(X_f[i], Xpb_f[i], theta[i]) for i in range(len(X))]
			Xtmp = asarray([self.Combination(X[i], Xpb[i], xb, X[self.randint(len(X), skip=[i])], MP_c[i], MP_s[i], MP_p[i], self.F, self.CR, task, self.Rand) for i in range(len(X))])
			X, X_f = asarray([Xtmp[i][0] for i in range(len(X))]), asarray([Xtmp[i][1] for i in range(len(X))])
			Xpb, Xpb_f, xb, xb_f = self.uBestAndPBest(X, X_f, Xpb, Xpb_f)
		return xb, xb_f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
