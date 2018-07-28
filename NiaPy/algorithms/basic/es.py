# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, unused-argument
import logging
from numpy import argmin
from NiaPy.algorithms.algorithm import Algorithm, Individual
from NiaPy.algorithms.basic.ga import TurnamentSelection

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['EvolutionStrategy']

def PlusStrategy(pop_1, pop_2): return pop_1.copy().extend(pop_2)

def NormalStrategy(pop_1, pop_2): return pop_2

class EvolutionStrategy(Algorithm):
	r"""Implementation of evolution strategy algorithm.

	**Algorithm:** Evolution Strategy Algorithm

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:**

	**Reference paper:**
	"""
	def __init__(self, **kwargs):
		r"""Initialize Evolution Strategy algorithm class.

		**See**:
		Algorithm.__init__(self, **kwargs)
		"""
		super(EvolutionStrategy, self).__init__(name='EvolutionStrategy', sName='ES', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, mi=50, l=50, rho=0.0, n=5, Strategy=NormalStrategy, Selection=TurnamentSelection, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		mi {integer} -- number of parent population
		l {integer} -- or lambda, number of children population
		rho {real} -- parameter for gayssian distribution
		n {integer} -- or lambda, number of children population
		Strategy {function} --
		Selection {function} --
		"""
		self.mi, self.l, self.rho, self.n, self.Strategy, self.Selection = mi, l, rho, n, Strategy, Selection
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def mutate(self, x, task):
		x.x = x.x + self.rand.normal(self.rho, 1)
		x.repair(task)
		x.evaluate(task)
		return x

	def runTask(self, task):
		pop = [Individual(task=task, rand=self.rand) for _i in range(self.mi)]
		x_b = pop[argmin([i.f for i in pop])]
		while not task.stopCond():
			npop = self.Strategy(pop, [Individual(task=task, rand=self.rand, e=False) for _i in range(self.l)])
			npop = [self.mutate(i, task) for i in npop]
			pop = [self.Selection(npop, self.n, self.rand) for _i in range(self.mi)]
			x_pb = pop[argmin([i.f for i in pop])]
			if x_pb.f < x_b.f: x_b = x_pb
		return x_b.x, x_b.f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
