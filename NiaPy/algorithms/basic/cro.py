# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, singleton-comparison, multiple-statements, attribute-defined-outside-init, no-self-use, logging-not-lazy, unused-variable, arguments-differ, bad-continuation, redefined-builtin, unused-argument, consider-using-enumerate, expression-not-assigned
import logging
from scipy.spatial.distance import euclidean
from numpy import apply_along_axis, argsort, where, inf, random as rand, asarray, delete, sqrt, sum, unique, append
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CoralReefsOptimization']

def SexualCrossoverSimple(pop, p, task, rnd=rand, **kwargs):
	for i in range(len(pop) // 2): pop[i] = asarray([pop[i, d] if rnd.rand() < p else pop[i * 2, d] for d in range(task.D)])
	return pop, apply_along_axis(task.eval, 1, pop)

def BroodingSimple(pop, p, task, rnd=rand, **kwargs):
	for i in range(len(pop)): pop[i] = task.repair(asarray([pop[i, d] if rnd.rand() < p else task.Lower[d] + task.bRange[d] * rnd.rand() for d in range(task.D)]), rnd=rnd)
	return pop, apply_along_axis(task.eval, 1, pop)

def MoveCorals(pop, p, F, task, rnd=rand, **kwargs):
	for i in range(len(pop)): pop[i] = task.repair(asarray([pop[i, d] if rnd.rand() < p else pop[i, d] + F * rnd.rand() for d in range(task.D)]), rnd=rnd)
	return pop, apply_along_axis(task.eval, 1, pop)

class CoralReefsOptimization(Algorithm):
	r"""Implementation of Coral Reefs Optimization Algorithm.

	**Algorithm:** Coral Reefs Optimization Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference:** S. Salcedo-Sanz, J. Del Ser, I. Landa-Torres, S. Gil-López, and J. A. Portilla-Figueras, “The Coral Reefs Optimization Algorithm: A Novel Metaheuristic for Efficiently Solving Optimization Problems,” The Scientific World Journal, vol. 2014, Article ID 739768, 15 pages, 2014. https://doi.org/10.1155/2014/739768.
	"""
	Name = ['CoralReefsOptimization', 'CRO']
	N, phi, Fa, Fb, Fd, k = 0, 0, 0, 0, 0, 0
	P_F, P_Cr = 0.36, 0.5

	@staticmethod
	def typeParameters(): return {
			# TODO funkcije za testiranje
			'N': False,
			'phi': False,
			'Fa': False,
			'Fb': False,
			'Fd': False,
			'k': False
	}

	def setParameters(self, N=25, phi=10, Fa=0.5, Fb=0.5, Fd=0.3, k=25, P_Cr=0.5, P_F=0.36, SexualCrossover=SexualCrossoverSimple, Brooding=BroodingSimple, Distance=euclidean, **ukwargs):
		r"""Set the parameters of the algorithm.

		**Arguments:**
		N {integer} -- population size for population initialization
		Fb {real} -- value $\in [0, 1]$ for Brooding size
		Fd {real} -- value $\in [0, 1]$ for Depredation
		k {integer} -- trys for larvae setting
		SexualCrossover {function} -- Crossover function
		P_Cr {real} -- Crossover rate $\in [0, 1]$
		Brooding {function} -- Brooding function
		F {real} -- Mutation variable $\in [0, \inf)$
		P_F {real} -- Crossover rate $\in [0, 1]$
		"""
		self.N, self.phi, self.Fa, self.Fb, self.Fd, self.k, self.P_Cr, self.P_F = N, phi, Fa, Fb, Fd, k, P_Cr, P_F
		self.SexualCrossover, self.Brooding, self.Distance = SexualCrossover, Brooding, Distance
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def initRun(self, task):
		Fa, Fb, Fd = self.N * self.Fa, self.N * self.Fb, self.N * self.Fd
		if Fa % 2 != 0: Fa += 1
		Reef = task.Lower + self.rand([self.N, task.D]) * task.bRange
		Reef_f = apply_along_axis(task.eval, 1, Reef)
		return Reef, Reef_f, int(Fa), int(Fb), int(Fd)

	def asexualReprodution(self, Reef, Reef_f, Fa, task):
		I = argsort(Reef_f)[:Fa]
		Reefn, Reefn_f = self.Brooding(Reef[I], self.P_F, task, rnd=self.Rand)
		Reef, Reef_f = self.setting(Reef, Reef_f, Reefn, Reefn_f, task)
		return Reef, Reef_f

	def depredation(self, Reef, Reef_f, Fd):
		I = argsort(Reef_f)[::-1][:Fd]
		return delete(Reef, I), delete(Reef_f, I)

	def setting(self, X, X_f, Xn, Xn_f, task):
		def update(A):
			D = asarray([sqrt(sum((A - e) ** 2, axis=1)) for e in Xn])
			I = unique(where(D < self.phi)[0])
			if I: Xn[I], Xn_f[I] = MoveCorals(Xn[I], self.P_F, self.P_F, task, rnd=self.Rand)
		for i in range(self.k): update(X), update(Xn)
		D = asarray([sqrt(sum((X - e) ** 2, axis=1)) for e in Xn])
		I = unique(where(D >= self.phi)[0])
		return append(X, Xn[I], 0), append(X_f, Xn_f[I], 0)

	def runTask(self, task):
		Reef, Reef_f, Fa, Fb, Fd = self.initRun(task)
		while not task.stopCondI():
			I = self.Rand.choice(len(Reef), size=Fb, replace=False)
			Reefn_s, Reefn_s_f = self.SexualCrossover(Reef[I], self.P_Cr, task, rnd=self.Rand)
			Reefn_b, Reffn_b_f = self.Brooding(delete(Reef, I, 0), self.P_F, task, rnd=self.Rand)
			Reefn, Reefn_f = self.setting(Reef, Reef_f, append(Reefn_s, Reefn_b, 0), append(Reefn_s_f, Reffn_b_f, 0), task)
			Reef, Reef_f = self.asexualReprodution(Reef, Reef_f, Fa, task)
			if task.Iters % self.k == 0: Reef, Reef_f = self.depredation(Reef, Reef_f, Fd)
		return self.getBest(Reef, Reef_f, None, inf * task.optType.value)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
