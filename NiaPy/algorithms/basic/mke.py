# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy
import logging
from numpy import apply_along_axis, argmin, inf, full, where, random as rand
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

	def generateSolution(self, task, rnd): self.x = task.Lower + task.bRange * rnd.rand(task.D)

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
	**Reference paper:**
	"""
	def __init__(self, **kwargs): super(MonkeyKingEvolutionV1, self).__init__(name='MonkeyKingEvolutionV1', sName='MKEv1', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP=40, F=0.7, R=0.3, C=3, FC=0.5, **ukwargs):
		self.NP, self.F, self.R, self.C, self.FC = NP, F, R, C, FC
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def repair(self, x, task):
		# FIXME for reparing matirxes
		ir = where(x > task.Upper)
		x[ir] = task.Upper[ir]
		ir = where(x < task.Lower)
		x[ir] = task.Lower[ir]
		return x

	def movePartice(self, p, p_b, task): p.x = self.repair(p_b.x + self.F * self.rand.rand(task.D) * (p_b.x - p.x), task)

	def moveMokeyKingPartice(self, p, task):
		p.MonkeyKing = False
		A = self.repair(p.x + self.FC * self.rand.rand(int(self.C * task.D), task.D) * p.x, task)
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
			ib = argmin([x.f for x in pop])
			for i in self.rand.choice(self.NP, int(self.R * len(pop)), replace=False): pop[i].MonkeyKing = True
			if pop[ib].f < p_b.f: p_b = pop[ib]
		return p_b.x, p_b.f

class MonkeyKingEvolutionV2(Algorithm):
	r"""Implementation of monkey king evolution algorithm version 1.

	**Algorithm:** Monkey King Evolution version 1

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs): super(MonkeyKingEvolutionV1, self).__init__(name='MonkeyKingEvolutionV2', sName='MKEv2', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, F, R, C, FC, **ukwargs): pass

	def runTask(self, task): pass

class MonkeyKingEvolutionV3(Algorithm):
	r"""Implementation of monkey king evolution algorithm version 1.

	**Algorithm:** Monkey King Evolution version 1

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs): super(MonkeyKingEvolutionV1, self).__init__(name='MonkeyKingEvolutionV3', sName='MKEv3', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, F, R, C, FC, **ukwargs): pass

	def runTask(self, task): pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
