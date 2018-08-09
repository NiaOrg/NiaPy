# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, logging-not-lazy, attribute-defined-outside-init, line-too-long, arguments-differ
import logging
from numpy import argmin
from NiaPy.algorithms.algorithm import Individual
from NiaPy.algorithms.modified.jde import SolutionjDE, SelfAdaptiveDifferentialEvolutionAlgorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['DynNPSelfAdaptiveDifferentialEvolutionAlgorithm']

class DynNPSelfAdaptiveDifferentialEvolutionAlgorithm(SelfAdaptiveDifferentialEvolutionAlgorithm):
	r"""Implementation of Dynamic population size self-adaptive differential evolution algorithm.

	**Algorithm:** Dynamic population size self-adaptive differential evolution algorithm

	**Date:** 2018

	**Author:** Uros Mlakar and Klemen Berkvoič

	**License:** MIT

	**Reference URL:** https://link.springer.com/article/10.1007/s10489-007-0091-x

	**Reference paper:**
	Brest, Janez, and Mirjam Sepesy Maučec. "Population size reduction for the differential evolution algorithm." Applied Intelligence 29.3 (2008): 228-247.
	"""
	def __init__(self, **kwargs): SelfAdaptiveDifferentialEvolutionAlgorithm.__init__(self, name='SelfAdaptiveDifferentialEvolutionAlgorithm', sName='jDE', **kwargs)

	def setParameters(self, rp=0, pmax=4, **ukwargs):
		r"""Set the parameters of an algorithm.

		Arguments:
		rp {integer} -- small non-negative number which is added to value of genp (if it's not divisible)
		pmax {integer} -- number of population reductions
		"""
		SelfAdaptiveDifferentialEvolutionAlgorithm.setParameters(self, **ukwargs)
		self.rp, self.pmax = rp, pmax
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def AdaptiveGen(self, x):
		f = self.F_l + self.rand() * (self.F_u - self.F_l) if self.rand() < self.Tao1 else x.F
		cr = self.rand() if self.rand() < self.Tao2 else x.CR
		return SolutionjDE(x=x.x, F=f, CR=cr)

	def runTask(self, task):
		pop = [SolutionjDE(task=task, F=self.F, CR=self.CR) for _i in range(self.Np)]
		Gr = task.nFES // (self.pmax * self.Np) + self.rp
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [self.AdaptiveGen(pop[i]) for i in range(len(pop))]
			npop = [SolutionjDE(x=self.CrossMutt(npop, i, x_b, self.F, self.CR, self.Rand), F=npop[i].F, CR=npop[i].CR, e=False) for i in range(len(pop))]
			pop = [self.evalPopulation(npop[i], pop[i], task) for i in range(len(pop))]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
			if task.Iters == Gr:
				NP = int(len(pop) / 2)
				pop = [self.selectBetter(pop[i], pop[i + NP]) for i in range(NP)]
				Gr += task.nFES // (self.pmax * NP) + self.rp
		return x_b.x, x_b.f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
