# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, len-as-condition, singleton-comparison, arguments-differ
import logging
from math import ceil
from numpy import apply_along_axis, vectorize, argmin, inf, where, ones, tril
from NiaPy.algorithms.algorithm import Algorithm, Individual

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['MonkeyKingEvolutionV1', 'MonkeyKingEvolutionV2', 'MonkeyKingEvolutionV3']

class MkeSolution(Individual):
	def __init__(self, **kwargs):
		Individual.__init__(self, **kwargs)
		self.f_pb, self.x_pb = self.f, self.x
		self.MonkeyKing = False

	def uPersonalBest(self):
		if self.f < self.f_pb: self.x_pb, self.f_pb = self.x, self.f

class MonkeyKingEvolutionV1(Algorithm):
	r"""Implementation of monkey king evolution algorithm version 1.

	**Algorithm:** Monkey King Evolution version 1

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:** https://www.sciencedirect.com/science/article/pii/S0950705116000198

	**Reference paper:** Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009.
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: Algorithm.__init__(self, name='MonkeyKingEvolutionV1', sName='MKEv1', **kwargs)
		else: Algorithm.__init__(self, **kwargs)

	def setParameters(self, NP=40, F=0.7, R=0.3, C=3, FC=0.5, **ukwargs):
		r"""Set the algorithm parameters.

		**Arguments:**

		NP {integer} -- Size of population

		F {real} -- param

		R {real} -- param

		C {real} -- param

		FC {real} -- param
		"""
		self.NP, self.F, self.R, self.C, self.FC = NP, F, R, C, FC
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def repair(self, x, task):
		ir = where(x > task.Upper)
		x[ir] = task.Upper[ir]
		ir = where(x < task.Lower)
		x[ir] = task.Lower[ir]
		return x

	def moveP(self, x, x_pb, x_b, task): return x_pb + self.F * self.rand(task.D) * (x_b - x)

	def moveMK(self, x, task): return x + self.FC * self.rand([int(self.C * task.D), task.D]) * x

	def movePartice(self, p, p_b, task):
		p.x = self.repair(self.moveP(p.x, p.x_pb, p_b.x, task), task)
		p.evaluate(task)

	def moveMokeyKingPartice(self, p, task):
		p.MonkeyKing = False
		A = apply_along_axis(self.repair, 1, self.moveMK(p.x, task), task)
		A_f = apply_along_axis(task.eval, 1, A)
		ib = argmin(A_f)
		p.x, p.f = A[ib], A_f[ib]

	def movePopulation(self, pop, p_b, task):
		for p in pop:
			if p.MonkeyKing: self.moveMokeyKingPartice(p, task)
			else: self.movePartice(p, p_b, task)
			p.uPersonalBest()

	def runTask(self, task):
		pop = [MkeSolution(task=task) for i in range(self.NP)]
		p_b = pop[argmin([x.f for x in pop])]
		for i in self.Rand.choice(self.NP, int(self.R * len(pop)), replace=False): pop[i].MonkeyKing = True
		while not task.stopCond():
			self.movePopulation(pop, p_b, task)
			for i in self.Rand.choice(self.NP, int(self.R * len(pop)), replace=False): pop[i].MonkeyKing = True
			ib = argmin([x.f for x in pop])
			if pop[ib].f < p_b.f: p_b = pop[ib]
		return p_b.x, p_b.f

class MonkeyKingEvolutionV2(MonkeyKingEvolutionV1):
	r"""Implementation of monkey king evolution algorithm version 2.

	**Algorithm:** Monkey King Evolution version 2

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:**
	https://www.sciencedirect.com/science/article/pii/S0950705116000198

	**Reference paper:**
	Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009.
	"""
	def __init__(self, **kwargs): MonkeyKingEvolutionV1.__init__(self, name='MonkeyKingEvolutionV2', sName='MKEv2', **kwargs)

	def moveMK(self, x, dx, task): return x - self.FC * dx

	def moveMokeyKingPartice(self, p, pop, task):
		p.MonkeyKing = False
		p_b, p_f = p.x, p.f
		for _i in range(int(self.C * self.NP)):
			r = self.Rand.choice(self.NP, 2, replace=False)
			a = self.repair(self.moveMK(p.x, pop[r[0]].x - pop[r[1]].x, task), task)
			a_f = task.eval(a)
			if a_f < p_f: p_b, p_f = a, a_f
		p.x, p.f = p_b, p_f

	def movePopulation(self, pop, p_b, task):
		for p in pop:
			if p.MonkeyKing: self.moveMokeyKingPartice(p, pop, task)
			else: self.movePartice(p, p_b, task)
			p.uPersonalBest()

class MonkeyKingEvolutionV3(MonkeyKingEvolutionV1):
	r"""Implementation of monkey king evolution algorithm version 3.

	**Algorithm:** Monkey King Evolution version 3

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:**
	https://www.sciencedirect.com/science/article/pii/S0950705116000198

	**Reference paper:**
	Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009.
	"""
	def __init__(self, **kwargs): MonkeyKingEvolutionV1.__init__(self, name='MonkeyKingEvolutionV3', sName='MKEv3', **kwargs)

	def eval(self, X, x, x_f, task):
		X_f = apply_along_axis(task.eval, 1, X)
		igb = argmin(X_f)
		if X_f[igb] <= x_f: x, x_f = X[igb], X_f[igb]
		return x, x_f

	def neg(self, x): return 0.0 if x == 1 else 1.0

	def runTask(self, task):
		X = task.Lower + task.bRange * self.rand([self.NP, task.D])
		x_gb, x_f_gb = self.eval(X, None, inf, task)
		k, c = int(ceil(self.NP / task.D)), int(ceil(self.C * task.D))
		while not task.stopCond():
			X_gb = x_gb + self.FC * X[self.Rand.choice(len(X), c)] - X[self.Rand.choice(len(X), c)]
			x_gb, x_f_gb = self.eval(X_gb, x_gb, x_f_gb, task)
			M = ones([self.NP, task.D])
			for i in range(k): M[i * task.D:(i + 1) * task.D] = tril(M[i * task.D:(i + 1) * task.D])
			for i in range(self.NP): self.Rand.shuffle(M[i])
			X = M * X + vectorize(self.neg)(M) * x_gb
			x_gb, x_f_gb = self.eval(X, x_gb, x_f_gb, task)
		return x_gb, x_f_gb

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
