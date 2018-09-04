# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, logging-not-lazy, attribute-defined-outside-init, line-too-long, arguments-differ, singleton-comparison, bad-continuation
import logging
from numpy import argmin
from NiaPy.algorithms.algorithm import Individual
from NiaPy.algorithms.basic.de import DifferentialEvolutionAlgorithm
from NiaPy.algorithms.other.sa import SimulatedAnnealingBF
from NiaPy.algorithms.basic.hs import HarmonySearchB, HarmonySearchV1B
from NiaPy.algorithms.other.mts import MTS_LS1, MTS_LS2, MTS_LS3

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['SelfAdaptiveDifferentialEvolutionAlgorithm', 'DynNPSelfAdaptiveDifferentialEvolutionAlgorithm', 'SelfAdaptiveDifferentialEvolutionAlgorithmBestHarmonySearch', 'SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS1', 'SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS2', 'SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS3']

class SolutionjDE(Individual):
	def __init__(self, **kwargs):
		Individual.__init__(self, **kwargs)
		self.F, self.CR = kwargs.get('F', 2), kwargs.get('CR', 0.5)

class SelfAdaptiveDifferentialEvolutionAlgorithm(DifferentialEvolutionAlgorithm):
	r"""Implementation of Self-adaptive differential evolution algorithm.

	**Algorithm:** Self-adaptive differential evolution algorithm
	**Date:** 2018
	**Author:** Uros Mlakar and Klemen Berkvoič
	**License:** MIT
	**Reference paper:** Brest, J., Greiner, S., Boskovic, B., Mernik, M., Zumer, V.	Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems. IEEE transactions on evolutionary computation, 10(6), 646-657, 2006.
	"""
	Name = ['SelfAdaptiveDifferentialEvolutionAlgorithm', 'jDE']

	@staticmethod
	def typeParameters():
		d = DifferentialEvolutionAlgorithm.typeParameters()
		d['F_l'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['F_u'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['Tao1'] = lambda x: isinstance(x, (float, int)) and 0 <= x <= 1
		d['Tao2'] = lambda x: isinstance(x, (float, int)) and 0 <= x <= 1
		return d

	def setParameters(self, F_l=0.0, F_u=2.0, Tao1=0.4, Tao2=0.6, **ukwargs):
		r"""Set the parameters of an algorithm.

		Arguments:
		F_l {decimal} -- scaling factor lower limit
		F_u {decimal} -- scaling factor upper limit
		Tao1 {decimal} -- change rate for F parameter update
		Tao2 {decimal} -- change rate for CR parameter update
		"""
		DifferentialEvolutionAlgorithm.setParameters(self, **ukwargs)
		self.F_l, self.F_u, self.Tao1, self.Tao2 = F_l, F_u, Tao1, Tao2
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def AdaptiveGen(self, x):
		f = self.F_l + self.rand() * (self.F_u - self.F_l) if self.rand() < self.Tao1 else x.F
		cr = self.rand() if self.rand() < self.Tao2 else x.CR
		return SolutionjDE(x=x.x, F=f, CR=cr, e=False)

	def selectBetter(self, x, y): return x if x.f < y.f else y

	def evalPopulation(self, x, x_old, task):
		"""Evaluate element."""
		x.evaluate(task, rnd=self.Rand)
		return self.selectBetter(x, x_old)

	def runTask(self, task):
		pop = [SolutionjDE(task=task, F=self.F, CR=self.CR, rand=self.Rand) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [self.AdaptiveGen(pop[i]) for i in range(self.Np)]
			for i in range(self.Np): npop[i].x = self.CrossMutt(npop, i, x_b, self.F, self.CR, rnd=self.Rand)
			pop = [self.evalPopulation(npop[i], pop[i], task) for i in range(self.Np)]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

class DynNPSelfAdaptiveDifferentialEvolutionAlgorithm(SelfAdaptiveDifferentialEvolutionAlgorithm):
	r"""Implementation of Dynamic population size self-adaptive differential evolution algorithm.

	**Algorithm:** Dynamic population size self-adaptive differential evolution algorithm
	**Date:** 2018
	**Author:** Jan Popič
	**License:** MIT
	**Reference URL:** https://link.springer.com/article/10.1007/s10489-007-0091-x
	**Reference paper:** Brest, Janez, and Mirjam Sepesy Maučec. "Population size reduction for the differential evolution algorithm." Applied Intelligence 29.3 (2008): 228-247.
	"""
	Name = ['DynNPSelfAdaptiveDifferentialEvolutionAlgorithm', 'dynNPjDE']

	@staticmethod
	def typeParameters():
		d = SelfAdaptiveDifferentialEvolutionAlgorithm.typeParameters()
		d['rp'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['pmax'] = lambda x: isinstance(x, int) and x > 0
		return d

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
		return SolutionjDE(x=x.x, F=f, CR=cr, e=False)

	def runTask(self, task):
		pop = [SolutionjDE(task=task, e=True, F=self.F, CR=self.CR, rand=self.Rand) for _i in range(self.Np)]
		Gr = task.nFES // (self.pmax * self.Np) + self.rp
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [self.AdaptiveGen(pop[i]) for i in range(len(pop))]
			for i in range(len(npop)): npop[i].x = self.CrossMutt(npop, i, x_b, self.F, self.CR, rnd=self.Rand)
			pop = [self.evalPopulation(npop[i], pop[i], task) for i in range(len(npop))]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
			if task.Iters == Gr:
				NP = int(len(pop) / 2)
				pop = [pop[i] if pop[i].f < pop[i + NP].f else pop[i + NP] for i in range(NP)]
				Gr += task.nFES // (self.pmax * NP) + self.rp
		return x_b.x, x_b.f

class SelfAdaptiveDifferentialEvolutionAlgorithmBestSimulatedAnnealing(SelfAdaptiveDifferentialEvolutionAlgorithm):
	Name = ['SelfAdaptiveDifferentialEvolutionAlgorithmBestSimulatedAnnealing', 'jDEbSA']

	@staticmethod
	def typeParameters():
		d = SelfAdaptiveDifferentialEvolutionAlgorithm.typeParameters()
		d['SR'] = lambda x: isinstance(x, float) and 0 < x <= 1
		return d

	def setParameters(self, SR=0.189, delta=0.563, delta_t=0.564, T=2000,**ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
		SR {decimal} -- search reange for best (normalized)
		"""
		self.SR, self.delta, self.delta_t, self.T = SR, delta, delta_t, T
		SelfAdaptiveDifferentialEvolutionAlgorithm.setParameters(self, **ukwargs)
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		pop = [SolutionjDE(task=task, F=self.F, CR=self.CR, rand=self.Rand) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [self.AdaptiveGen(pop[i]) for i in range(self.Np)]
			for i in range(self.Np): npop[i].x = self.CrossMutt(npop, i, x_b, self.F, self.CR, rnd=self.Rand)
			pop = [self.evalPopulation(npop[i], pop[i], task) for i in range(self.Np)]
			ix_b = argmin([x.f for x in pop])
			tSR = (task.bRange * self.SR) / 2
			tLower, tUpper = pop[ix_b].x - tSR, pop[ix_b] + tSR
			xn = SimulatedAnnealingBF(task, tLower, tUpper, x=pop[ix_b].x, xfit=pop[ix_b].f, delta=self.delta, delta_t=self.delta_t, T=self.T, rnd=self.Rand)
			if xn[1] < pop[ix_b].f: pop[ix_b].x, pop[ix_b].f = xn
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

class SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS1(SelfAdaptiveDifferentialEvolutionAlgorithm):
	Name = ['SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS1', 'jDEbMTS1']

	def setParameters(self, **ukwargs): pass

	def runTask(self, task):
		# FIXME
		pass

class SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS2(SelfAdaptiveDifferentialEvolutionAlgorithm):
	Name = ['SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS2', 'jDEbMTS2']

	def setParameters(self, **ukwargs): pass

	def runTask(self, task):
		# FIXME
		pass

class SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS3(SelfAdaptiveDifferentialEvolutionAlgorithm):
	Name = ['SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS3', 'jDEbMTS3']

	def setParameters(self, **ukwargs): pass

	def runTask(self, task):
		# FIXME
		pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
