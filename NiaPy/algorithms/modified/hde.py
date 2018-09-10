# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, logging-not-lazy, attribute-defined-outside-init, line-too-long, arguments-differ, singleton-comparison, bad-continuation
import logging
from numpy import argmin, full
from NiaPy.algorithms.algorithm import Individual
from NiaPy.algorithms.basic.de import DifferentialEvolution
from numpy import argsort, argmin, argmax
from NiaPy.algorithms.other.mts import MTS_LS1, MTS_LS1v1, MTS_LS2, MTS_LS3, MTS_LS3v1

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['DifferentialEvolutionMTS', 'DifferentialEvolutionMTSv1']

class MtsIndividual(Individual):
	def __init__(self, SR, grade=0, enable=True, improved=False, **kwargs):
		Individual.__init__(self, **kwargs)
		self.SR, self.grade, self.enable, self.improved = SR, grade, enable, improved

class DifferentialEvolutionMTS(DifferentialEvolution):
	Name = ['DifferentialEvolutionMTS', 'DEMTS']

	@staticmethod
	def typeParameters(): return DifferentialEvolution.typeParameters()

	def setParameters(self, NoGradingRuns=3, NoLs=6, NoEnabled=15, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
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
	Name = ['DifferentialEvolutionMTSv1', 'DEMTSv1']

	def __init__(self, **kwargs):
		DifferentialEvolutionMTS.__init__(self, **kwargs)
		self.LSs = [MTS_LS1v1, MTS_LS2, MTS_LS3v1]

class DynNpDifferentialEvolutionMTS(DifferentialEvolutionMTS):
	Name = []
	pass

class MultiStratgyDifferentialEvolutionMTS(DifferentialEvolutionMTS):
	Name = []
	pass

class DynNpMultiStrategyDifferentialEvolutionMTS(DifferentialEvolutionMTS):
	Name = []
	pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
