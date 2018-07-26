# encoding=utf8
import logging
from numpy import argmin 
from NiaPy.algorithms.basic.de import SolutionDE, DifferentialEvolutionAlgorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['SelfAdaptiveDifferentialEvolutionAlgorithm']


class SolutionjDE(SolutionDE):
	def __init__(self, **kwargs):
		super(SolutionjDE, self).__init__(**kwargs)
		self.F, self.CR = kwargs.get('F', 2), kwargs.get('CR', 0.5)

class SelfAdaptiveDifferentialEvolutionAlgorithm(DifferentialEvolutionAlgorithm):
	r"""Implementation of Self-adaptive differential evolution algorithm.

	**Algorithm:** Self-adaptive differential evolution algorithm

	**Date:** 2018

	**Author:** Uros Mlakar and Klemen Berkvoič

	**License:** MIT

	**Reference paper:**
	Brest, J., Greiner, S., Boskovic, B., Mernik, M., Zumer, V.
	Self-adapting control parameters in differential evolution:
	A comparative study on numerical benchmark problems.
	IEEE transactions on evolutionary computation, 10(6), 646-657, 2006.
	"""
	def __init__(self, **kwargs): super(SelfAdaptiveDifferentialEvolutionAlgorithm, self).__init__(name='SelfAdaptiveDifferentialEvolutionAlgorithm', sName='jDE', **kwargs)

	def setParameters(self, **kwargs):
		super(SelfAdaptiveDifferentialEvolutionAlgorithm, self).setParameters(**kwargs)
		self.__setParams(**kwargs)

	def __setParams(self, F_l=0.0, F_u=2.0, Tao1=0.4, Tao2=0.6, **ukwargs):
		r"""Set the parameters of an algorithm.

		Arguments:
		F_l {decimal} -- scaling factor
		F_u {decimal} -- scaling factor
		Tao1 {decimal} --
		Tao2 {decimal} --
		"""
		self.F_l, self.F_u, self.Tao1, self.Tao2 = F_l, F_u, Tao1, Tao2
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def AdaptiveGen(self, x):
		f = self.F_l + self.rand.rand() * (self.F_u - self.F_l) if self.rand.rand() < self.Tao1 else x.F
		cr = self.rand.rand() if self.rand.rand() < self.Tao2 else x.CR
		return SolutionjDE(x=x.x, F=f, CR=cr)

	def runTask(self, task):
		pop = [SolutionjDE(task=task, F=self.F, CR=self.CR) for _i in range(self.Np)]
		pop = [self.evalPopulation(pop[i], pop[i], task) for i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCond():
			npop = [self.AdaptiveGen(pop[i]) for i in range(self.Np)]
			npop = [SolutionjDE(x=self.CrossMutt(npop, i, x_b, self.F, self.CR, self.rand), F=npop[i].F, CR=npop[i].CR) for i in range(self.Np)]
			pop = [self.evalPopulation(npop[i], pop[i], task) for i in range(self.Np)]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
