# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, len-as-condition
import logging
from math import ceil
from numpy import apply_along_axis, argmin, inf, where, asarray, ndarray, random as rand, ones, tril
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['MonkeyKingEvolutionV1', 'MonkeyKingEvolutionV2', 'MonkeyKingEvolutionV3']

class MkeSolution(object):
	def __init__(self, **kwargs):
		self.f, self.f_pb, self.x_pb = inf, inf, None
		self.MonkeyKing = False
		task = kwargs.get('task', None)
		rnd = kwargs.get('rand', rand)
		x = kwargs.get('x', [])
		if len(x) > 0: self.x = x if isinstance(x, ndarray) else asarray(x)
		else: self.generateSolution(task, rnd)

	def generateSolution(self, task, rnd): 
		self.x = task.Lower + task.bRange * rnd.rand(task.D)
		self.x_pb = self.x

	def evaluate(self, task): self.f = task.eval(self.x)

	def uPersonalBest(self):
		if self.f < self.f_pb: self.x_pb, self.f_pb = self.x, self.f

	def repair(self, task):
		ir = where(self.x > task.Upper)
		self.x[ir] = task.Upper[ir]
		ir = where(self.x < task.Lower)
		self.x[ir] = task.Lower[ir]

	def __eq__(self, other): return self.x == other.x and self.f == other.f

	def __len__(self): return len(self.x)

	def __getitem__(self, i): return self.x[i]

class MonkeyKingEvolutionV1(Algorithm):
	r"""Implementation of monkey king evolution algorithm version 1.

	**Algorithm:** Monkey King Evolution version 1

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:**
	https://www.sciencedirect.com/science/article/pii/S0950705116000198

	**Reference paper:**
	Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009.
	"""
	def __init__(self, **kwargs): 
		if kwargs.get('name', None) == None: super(MonkeyKingEvolutionV1, self).__init__(name='MonkeyKingEvolutionV1', sName='MKEv1', **kwargs)
		else: super(MonkeyKingEvolutionV1, self).__init__(**kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP=40, F=0.7, R=0.3, C=3, FC=0.5, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
		NP {integer} -- Size of population
		F {real} -- 
		R {real} --
		C {real} --
		FC {real} -- 
		"""
		self.NP, self.F, self.R, self.C, self.FC = NP, F, R, C, FC
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def repair(self, x, task):
		ir = where(x > task.Upper)
		x[ir] = task.Upper[ir]
		ir = where(x < task.Lower)
		x[ir] = task.Lower[ir]
		return x

	def moveP(self, x, x_pb, x_b, task): return x_pb + self.F * self.rand.rand(task.D) * (x_b - x)

	def moveMK(self, x, task): return x + self.FC * self.rand.rand(int(self.C * task.D), task.D) * x

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
		for p in pop: p.evaluate(task)
		p_b = pop[argmin([x.f for x in pop])]
		for i in self.rand.choice(self.NP, int(self.R * len(pop)), replace=False): pop[i].MonkeyKing = True
		while not task.stopCond():
			self.movePopulation(pop, p_b, task)
			for i in self.rand.choice(self.NP, int(self.R * len(pop)), replace=False): pop[i].MonkeyKing = True
			ib = argmin([x.f for x in pop])
			if pop[ib].f < p_b.f: p_b = pop[ib]
		return p_b.x, p_b.f

class MonkeyKingEvolutionV2(MonkeyKingEvolutionV1):
	r"""Implementation of monkey king evolution algorithm version 1.

	**Algorithm:** Monkey King Evolution version 1

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:**
	https://www.sciencedirect.com/science/article/pii/S0950705116000198

	**Reference paper:**
	Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009.
	"""
	def __init__(self, **kwargs): super(MonkeyKingEvolutionV2, self).__init__(name='MonkeyKingEvolutionV2', sName='MKEv2', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP=40, F=0.7, R=0.3, C=3, FC=2.0, **ukwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
		NP {integer} -- Size of population
		F {real} -- 
		R {real} --
		C {real} --
		FC {real} -- 
		"""
		self.NP, self.F, self.R, self.C, self.FC = NP, F, R, C, FC
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def moveMK(self, x, dx, task): return x - self.FC * dx

	def moveMokeyKingPartice(self, p, pop, task):
		p.MonkeyKing = False
		p_b, p_f = p.x, p.f
		for i in range(int(self.C * self.NP)):
			r = self.rand.choice(self.NP, 2, replace=False)
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
	r"""Implementation of monkey king evolution algorithm version 1.

	**Algorithm:** Monkey King Evolution version 1

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:**
	https://www.sciencedirect.com/science/article/pii/S0950705116000198

	**Reference paper:**
	Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009.
	"""
	def __init__(self, **kwargs): super(MonkeyKingEvolutionV3, self).__init__(name='MonkeyKingEvolutionV3', sName='MKEv3', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP, F, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
		NP {integer} -- Size of the population
		F {real} --
		"""
		self.NP, self.F = NP, F

	def moveM(self, x, m, b): return x * m + b

	def moveB(self, x, d): return x + self.F * d

	def movePopulation(self, pop, task):
		for p in pop:
			# TODO
			p.uPersonalBest()

	def runTask(self, task):
		pop = [MkeSolution(task=task) for i in range(self.NP)]
		for p in pop: p.evaluate(task)
		p_b = pop[argmin([x.f for x in pop])]
		K = ceil(self.NP / taks.D)
		while not task.stopCond():
			A = ones([self.NP, task.D])
			for i in range(K): A[i * task.D:(i + 1) * task.D] = tril(A[i * task.D:(i + 1) * taks.D])
			for i in range(self.NP): self.rand.shuffle(A[i])
			r = self.rand.choice(self.NP, 2, replace=False)
			dx = pop[r[0]].x - pop[f[1]].x
		return None, None

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
