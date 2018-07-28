# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy
import logging
from numpy import apply_along_axis, argmin, where
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.algorithms.basic.ga import TurnamentSelection

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['EvolutionStrategy']

def PlusStrategy(pop_1, pop_2): return pop_1.append(pop_2)

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
		super(BareBonesFireworksAlgorithm, self).__init__(name='EvolutionStrategy', sName='ES', **kwargs)

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

	def repair(self, x, task):
		ir = where(x > task.Upper)
		x[ir] = task.Upper[ir]
		ir = where(x < task.Lower)
		x[ir] = task.Lower[ir]
		return x

	def mutate(self, x, task): return self.repair(x + self.rand.normal(self.rho, 1), task)

	def runTask(self, task):
		pop = task.Lower + task.bRange * self.rand.rand(self.mi, task.D)
		pop_f = apply_along_axis(task.eval, 1, pop)
		ib = argmin(pop_f)
		xb, xb_f = pop[ib], pop_f[ib]
		while not task.stopCond():
			npop = self.Strategy(pop, task.Lower + task.bRange * self.rand.rand(self.mi, task.D))
			npop = apply_along_axis(self.mutate, 1, npop)
			npop_f = apply_along_axis(task.eval, 1, npop)
			npop = self.Selection(npop, self.n, self.rand)
			ib = argmin(pop_f)
			xb, xb_f = pop[ib], pop_f[ib]
		return xb, xb_f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
