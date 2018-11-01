# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, logging-not-lazy, attribute-defined-outside-init, line-too-long, arguments-differ, singleton-comparison, bad-continuation, dangerous-default-value
import logging
from numpy import argmin, argsort, argmax, full
from NiaPy.algorithms.algorithm import Individual
from NiaPy.algorithms.basic.de import DifferentialEvolution, CrossBest1, CrossBest2, CrossRand1, CrossRand2, CrossCurr2Rand1
from NiaPy.algorithms.other.mts import MTS_LS1, MTS_LS1v1, MTS_LS2, MTS_LS3, MTS_LS3v1

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['DifferentialEvolutionMTS', 'DifferentialEvolutionMTSv1', 'DynNpDifferentialEvolutionMTS', 'DynNpDifferentialEvolutionMTSv1', 'MultiStratgyDifferentialEvolutionMTS', 'MultiStratgyDifferentialEvolutionMTSv1', 'DynNpMultiStrategyDifferentialEvolutionMTS', 'DynNpMultiStrategyDifferentialEvolutionMTSv1']

class MtsIndividual(Individual):
	def __init__(self, SR, grade=0, enable=True, improved=False, **kwargs):
		Individual.__init__(self, **kwargs)
		self.SR, self.grade, self.enable, self.improved = SR, grade, enable, improved

class DifferentialEvolutionMTS(DifferentialEvolution):
	r"""Implementation of Differential Evolution withm MTS local searches.

	**Algorithm:** Differential Evolution withm MTS local searches

	**Date:** 2018

	**Author:** Klemen Berkovič

	**License:** MIT
	"""
	Name = ['DifferentialEvolutionMTS', 'DEMTS']

	@staticmethod
	def typeParameters(): return DifferentialEvolution.typeParameters()

	def setParameters(self, NoGradingRuns=1, NoLs=2, NoEnabled=2, **ukwargs):
		r"""Set the algorithm parameters.

		**Arguments:**

		SR {real} -- Normalized search range
		"""
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.LSs, self.NoGradingRuns, self.NoLs, self.NoEnabled = [MTS_LS1, MTS_LS2, MTS_LS3], NoGradingRuns, NoLs, NoEnabled
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def GradingRun(self, x, xb, task):
		ls_grades, Xn, Xnb, SR = full(3, 0.0), [x.x, x.f] * len(self.LSs), [xb.x, xb.f], x.SR
		for _ in range(self.NoGradingRuns):
			improve = x.improved
			for k, LS in enumerate(self.LSs):
				xn, xn_f, xnb, xnb_f, improve, g, SR = LS(Xn[0], Xn[1], Xnb[0], Xnb[1], improve, SR, task, rnd=self.Rand)
				if Xn[1] > xn_f: Xn = [xn, xn_f]
				if Xnb[1] > xnb_f: Xnb = [xnb, xnb_f]
				ls_grades[k] += g
		x.x, x.f, x.SR, xb.x, xb.f, k = Xn[0], Xn[1], SR, Xnb[0], Xnb[1], argmax(ls_grades)
		return x, xb, k

	def LsRun(self, k, x, xb, task):
		XBn, grade = list(), 0
		for _ in range(self.NoLs):
			x.x, x.f, xnb, xnb_f, x.improved, grade, x.SR = self.LSs[k](x.x, x.f, xb.x, xb.f, x.improved, x.SR, task, rnd=self.Rand)
			x.grade += grade
			XBn.append((xnb, xnb_f))
		xb.x, xb.f = min(XBn, key=lambda x: x[1])
		return x, xb

	def LSprocedure(self, x, xb, task):
		if not x.enable: return x, xb
		x.enable, x.grade = False, 0
		x, xb, k = self.GradingRun(x, xb, task)
		x, xb = self.LsRun(k, x, xb, task)
		return x, xb

	def runTask(self, task):
		pop = [MtsIndividual(task.bcRange() * 0.06, task=task, rand=self.Rand, e=True) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [MtsIndividual(pop[i].SR, x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand), task=task, rand=self.Rand, e=True) for i in range(len(pop))]
			for i, e in enumerate(npop): npop[i], x_b = self.LSprocedure(e, x_b, task)
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			for i in argsort([x.grade for x in pop])[:self.NoEnabled]: pop[i].enable = True
		return x_b.x, x_b.f

class DifferentialEvolutionMTSv1(DifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution withm MTSv1 local searches.

	**Algorithm:** Differential Evolution withm MTSv1 local searches
	**Date:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	Name = ['DifferentialEvolutionMTSv1', 'DEMTSv1']

	def setParameters(self, **ukwargs):
		DifferentialEvolutionMTS.setParameters(self, **ukwargs)
		self.LSs = [MTS_LS1v1, MTS_LS2, MTS_LS3v1]

class DynNpDifferentialEvolutionMTS(DifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution withm MTS local searches dynamic and population size.

	**Algorithm:** Differential Evolution withm MTS local searches and dynamic population size
	**Date:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	Name = ['DynNpDifferentialEvolutionMTS', 'dynNpDEMTS']

	def setParameters(self, pmax=10, rp=3, **ukwargs):
		DifferentialEvolutionMTS.setParameters(self, **ukwargs)
		self.pmax, self.rp = pmax, rp

	def runTask(self, task):
		Gr = task.nFES // (self.pmax * self.Np) + self.rp
		pop = [MtsIndividual(task.bcRange() * 0.06, task=task, rand=self.Rand, e=True) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [MtsIndividual(pop[i].SR, x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand), task=task, rand=self.Rand, e=True) for i in range(len(pop))]
			for i, e in enumerate(npop): npop[i], x_b = self.LSprocedure(e, x_b, task)
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			for i in argsort([x.grade for x in pop])[:self.NoEnabled]: pop[i].enable = True
			if task.Iters == Gr and len(pop) > 3:
				NP = int(len(pop) / 2)
				pop = [pop[i] if pop[i].f < pop[i + NP].f else pop[i + NP] for i in range(NP)]
				Gr += task.nFES // (self.pmax * NP) + self.rp
		return x_b.x, x_b.f

class DynNpDifferentialEvolutionMTSv1(DynNpDifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution withm MTSv1 local searches and dynamic population size.

	**Algorithm:** Differential Evolution withm MTSv1 local searches and dynamic population size
	**Date:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	Name = ['DynNpDifferentialEvolutionMTSv1', 'dynNpDEMTSv1']

	def setParameters(self, pmax=10, rp=3, **ukwargs):
		DifferentialEvolutionMTS.setParameters(self, **ukwargs)
		self.LSs, self.pmax, self.rp = [MTS_LS1v1, MTS_LS2, MTS_LS3v1], pmax, rp

class MultiStratgyDifferentialEvolutionMTS(DifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution withm MTS local searches and multiple mutation strategys.

	**Algorithm:** Differential Evolution withm MTS local searches and multiple mutation strategys
	**Date:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	Name = ['MultiStratgyDifferentialEvolutionMTS', 'MSDEMTS']

	def setParameters(self, strategys=[CrossBest1, CrossRand1, CrossCurr2Rand1, CrossBest2, CrossRand2], **ukwargs):
		DifferentialEvolutionMTS.setParameters(self, **ukwargs)
		self.strategys = strategys

	def multiMutations(self, pop, i, x_b, task):
		L = [MtsIndividual(pop[i].SR, x=strategy(pop, i, x_b, self.F, self.CR, rnd=self.Rand), task=task, e=True, rand=self.Rand) for strategy in self.strategys]
		return L[argmin([x.f for x in L])]

	def runTask(self, task):
		pop = [MtsIndividual(task.bcRange() * 0.06, task=task, rand=self.Rand, e=True) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [self.multiMutations(pop, i, x_b, task) for i in range(len(pop))]
			for i, e in enumerate(npop): npop[i], x_b = self.LSprocedure(e, x_b, task)
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			for i in argsort([x.grade for x in pop])[:self.NoEnabled]: pop[i].enable = True
		return x_b.x, x_b.f

class MultiStratgyDifferentialEvolutionMTSv1(MultiStratgyDifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution withm MTSv1 local searches and multiple mutation strategys.

	**Algorithm:** Differential Evolution withm MTSv1 local searches and multiple mutation strategys
	**Date:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	Name = ['MultiStratgyDifferentialEvolutionMTSv1', 'MSDEMTSv1']

	def __init__(self, **kwargs):
		MultiStratgyDifferentialEvolutionMTS.__init__(self, **kwargs)
		self.LSs = [MTS_LS1v1, MTS_LS2, MTS_LS3v1]

class DynNpMultiStrategyDifferentialEvolutionMTS(MultiStratgyDifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution withm MTS local searches, multiple mutation strategys and dynamic population size.

	**Algorithm:** Differential Evolution withm MTS local searches, multiple mutation strategys and dynamic population size
	**Date:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	Name = ['DynNpMultiStrategyDifferentialEvolutionMTS', 'dynNpMSDEMTS']

	def setParameters(self, pmax=10, rp=7, **ukwargs):
		MultiStratgyDifferentialEvolutionMTS.setParameters(self, **ukwargs)
		self.pmax, self.rp = pmax, rp

	def runTask(self, task):
		Gr = task.nFES // (self.pmax * self.Np) + self.rp
		pop = [MtsIndividual(task.bcRange() * 0.06, task=task, rand=self.Rand, e=True) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [MtsIndividual(pop[i].SR, x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand), task=task, rand=self.Rand, e=True) for i in range(len(pop))]
			for i, e in enumerate(npop): npop[i], x_b = self.LSprocedure(e, x_b, task)
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			for i in argsort([x.grade for x in pop])[:self.NoEnabled]: pop[i].enable = True
			if task.Iters == Gr and len(pop) > 3:
				NP = int(len(pop) / 2)
				pop = [pop[i] if pop[i].f < pop[i + NP].f else pop[i + NP] for i in range(NP)]
				Gr += task.nFES // (self.pmax * NP) + self.rp
		return x_b.x, x_b.f

class DynNpMultiStrategyDifferentialEvolutionMTSv1(DynNpMultiStrategyDifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution withm MTSv1 local searches, multiple mutation strategys and dynamic population size.

	**Algorithm:** Differential Evolution withm MTSv1 local searches, multiple mutation strategys and dynamic population size
	**Date:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	Name = ['DynNpMultiStrategyDifferentialEvolutionMTSv1', 'dynNpMSDEMTSv1']

	def setParameters(self, **kwargs):
		DynNpMultiStrategyDifferentialEvolutionMTS.setParameters(self, **kwargs)
		self.LSs = [MTS_LS1v1, MTS_LS2, MTS_LS3v1]

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
