# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, logging-not-lazy, attribute-defined-outside-init, line-too-long, arguments-differ, singleton-comparison, bad-continuation
import logging
from numpy import argmin
from NiaPy.algorithms.algorithm import Individual, Algorithm
from NiaPy.algorithms.basic.de import DifferentialEvolutionAlgorithm
from NiaPy.algorithms.other.sa import SimulatedAnnealingF
from NiaPy.algorithms.basic.hs import HarmonySearchB, HarmonySearchV1B
from NiaPy.algorithms.other.mts import MTS_LS1, MTS_LS2, MTS_LS3

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['DifferentialEvolutionBestSimulatedAnnealing', 'DifferentialEvolutionBestHarmonySearch', 'DifferentialEvolutionPBestHarmonySearch', 'DifferentialEvolutionBestMTS1', 'DifferentialEvolutionBestMTS2', 'DifferentialEvolutionBestMTS3']

class DifferentialEvolutionBestSimulatedAnnealing(DifferentialEvolutionAlgorithm):
	Name = ['DifferentialEvolutionBestSimulatedAnnealing', 'DEbSA']

	@staticmethod
	def typeParameters():
		d = DifferentialEvolutionAlgorithm.typeParameters()
		d['SR'] = lambda x: isinstance(x, float) and 0 < x <= 1
		d['delta'] = lambda x: isinstance(x, (int, float)) and x > 0
		d['T'] = lambda x: isinstance(x, (int, float)) and x > 0
		d['deltaT'] = lambda x: isinstance(x, (int, float)) and x > 0
		return d

	def setParameters(self, SR=0.1, delta=1.5, delta_t=0.564, T=2000, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
		SR {decimal} -- search reange for best (normalized)
		delta {real} -- for SA
		T {real} -- for SA
		deltaT {real} -- for SA
		"""
		self.SR, self.delta, self.delta_t, self.T = SR, delta, delta_t, T
		DifferentialEvolutionAlgorithm.setParameters(self, **ukwargs)
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		pop = [Individual(task=task, e=True) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [Individual(x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand), task=task, e=True) for i in range(self.Np)]
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			ix_b = argmin([x.f for x in pop])
			# TODO dodaj zagon SA funkcije
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

class DifferentialEvolutionBestHarmonySearch(DifferentialEvolutionAlgorithm):
	Name = ['DifferentialEvolutionBestHarmonySearch', 'DEbHS']

	@staticmethod
	def typeParameters():
		d = DifferentialEvolutionAlgorithm.typeParameters()
		d['SR'] = lambda x: isinstance(x, float) and 0 < x <= 1
		return d

	def runTask(self, task):
		# FIXME add HS algorithm to the mix
		pop = [Individual(task=task, e=True) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [Individual(x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand), task=task, e=True) for i in range(self.Np)]
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

class DifferentialEvolutionPBestHarmonySearch(DifferentialEvolutionBestHarmonySearch):
	Name = ['DifferentialEvolutionPBestHarmonySearch', 'DEpbHS']

	@staticmethod
	def typeParameters():
		d = DifferentialEvolutionBestHarmonySearch.typeParameters()
		d['p'] = lambda x: isinstance(x, float) and 0 < x <= 1
		return d

	def setParameters(self, p=0.1, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
		p {decimal} -- procentage of best individuals in population
		"""
		DifferentialEvolutionBestHarmonySearch.setParameters(self, **ukwargs)
		self.p = p
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		# FIXME add HS algorithm to the mix with p% of best population
		pop = [Individual(task=task, e=True) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [Individual(x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand), task=task, e=True) for i in range(self.Np)]
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

class DifferentialEvolutionBestMTS1(DifferentialEvolutionBestSimulatedAnnealing):
	Name = ['DifferentialEvolutionBestMTS1', 'DEbMTS1']

	@staticmethod
	def typeParameters():
		d = DifferentialEvolutionBestHarmonySearch.typeParameters()
		d['NoLSRuns'] = lambda x: isinstance(x, int) and x >= 0
		return d

	def setParameters(self, NoLSRuns=25, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
		NP {integer} -- population size
		F {decimal} -- scaling factor
		CR {decimal} -- crossover rate
		SR {decimal} -- search reange for best (normalized)
		CrossMutt {function} -- crossover and mutation strategy
		"""
		self.Np, self.F, self.CR, self.SR, self.CrossMutt = NP, F, CR, SR, CrossMutt
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		# FIXME add MTS_LS1 algorithm to the mix
		pop = [Individual(task=task, e=True) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [Individual(x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand), task=task, e=True) for i in range(self.Np)]
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

class DifferentialEvolutionBestMTS2(DifferentialEvolutionBestSimulatedAnnealing):
	Name = ['DifferentialEvolutionBestMTS2', 'DEbMTS2']

	def runTask(self, task):
		# FIXME add MTS_LS2 algorithm to the mix
		pop = [Individual(task=task, e=True) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [Individual(x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand), task=task, e=True) for i in range(self.Np)]
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

class DifferentialEvolutionBestMTS3(DifferentialEvolutionBestSimulatedAnnealing):
	Name = ['DifferentialEvolutionBestMTS3', 'DEbMTS3']

	def runTask(self, task):
		# FIXME add MTS_LS3 algorithm to the mix
		pop = [Individual(task=task, e=True) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [Individual(x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand), task=task, e=True) for i in range(self.Np)]
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
