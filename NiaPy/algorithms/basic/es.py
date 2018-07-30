# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, unused-argument
import logging
from numpy import argmin, random as rand, full
from NiaPy.algorithms.algorithm import Algorithm, Individual

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['EvolutionStrategy1p1']

def PlusStrategy(pop_1, pop_2): return pop_1.copy().extend(pop_2)

def NormalStrategy(pop_1, pop_2): return pop_2

def TurnamentSelection(pop, ts, rnd=rand):
	comps = [pop[i] for i in rand.choice(len(pop), ts, replace=False)]
	return comps[argmin([c.f for c in comps])]

def ElitistSelection(xn, xn_f, x, x_f): return xn, xn_f if xn_f <= x_ else x, x_f

class IndividualES(Individual):
	def __init__(self, **kwargs):
		task, x = kwargs.get('task', None), kwargs.get('x', None)
		if task != None: self.rho = full(task.D, 1.0)
		elif x != None: self.rho = full(len(x), 1.0)
		super(IndividualES, self).__init__(**kwargs)

class EvolutionStrategy1p1(Algorithm):
	r"""Implementation of evolution strategy algorithm.

	**Algorithm:** (1 + 1) Evolution Strategy Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs): 
		if kwargs.get('name', None) == None: super(EvolutionStrategy1p1, self).__init__(name='(1+1)-EvolutionStrategy', sName='(1+1)-ES', **kwargs)
		else: super(EvolutionStrategy1p1, self).__init__(**kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, k=10, c_a=1.1, c_r=0.5, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		k {integer} --
		c_a {real} --
		c_r {real} --
		"""
		self.mu, self.k, self.c_a, self.c_r = 1, k, c_a, c_r
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def mutate(self, x, rho): return x + self.rand.normal(0, rho)

	def updateRho(self, rho, k):
		phi = k / self.k
		if phi < 0.2: return self.c_r * rho
		elif phi > 0.2: return self.c_a * rho
		else: return rho

	def runTask(self, task):
		c, ki = IndividualES(task=task, rand=self.rand), 0
		while not task.stopCondI():
			if task.Iters % self.k == 0: c.rho, ki = self.updateRho(c.rho, ki), 0
			cn = [task.repair(self.mutate(c.x, c.rho)) for _i in range(self.mu)]
			cn_f = [task.eval(cn[i]) for i in range(self.mu)]
			ib = argmin(cn_f)
			if cn_f[ib] < c.f: c.x, c.f, ki = cn[ib], cn_f[ib], ki + 1
		return c.x, c.f

class EvolutionStrategyMp1(EvolutionStrategy1p1):
	r"""Implementation of evolution strategy algorithm.

	**Algorithm:** ($\mu$ + 1) Evolution Strategy Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs): super(EvolutionStrategyMp1, self).__init__(name='(mu+1)-EvolutionStrategy', sName='(mu+1)-ES', **kwargs)

	def setParameters(self, **kwargs):
		super(EvolutionStrategyMp1, self).setParameters(**kwargs)
		self.__setParams(**kwargs)

	def __setParams(self, mu=40, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		mu {integer} -- number of parent population
		"""
		self.mu = mu
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
