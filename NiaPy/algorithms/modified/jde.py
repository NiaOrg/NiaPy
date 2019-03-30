# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, logging-not-lazy, attribute-defined-outside-init, line-too-long, arguments-differ, singleton-comparison, bad-continuation, dangerous-default-value, consider-using-enumerate
import logging
from numpy import argmin, apply_along_axis
from NiaPy.algorithms.algorithm import Individual
from NiaPy.algorithms.basic.de import DifferentialEvolution, CrossBest1, CrossRand1, CrossCurr2Best1, CrossBest2, CrossCurr2Rand1, proportional

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = [
	'SelfAdaptiveDifferentialEvolution',
	'DynNpSelfAdaptiveDifferentialEvolutionAlgorithm',
	'AgingSelfAdaptiveDifferentialEvolution',
	'MultiStrategySelfAdaptiveDifferentialEvolution',
	'DynNpMultiStrategySelfAdaptiveDifferentialEvolution'
]

def selectBetter(x, y): return x if x.f < y.f else y

class SolutionjDE(Individual):
	def __init__(self, F=2, CR=0.5, **kwargs):
		Individual.__init__(self, **kwargs)
		self.F, self.CR = F, CR

class SelfAdaptiveDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Self-adaptive differential evolution algorithm.

	Algorithm:
		Self-adaptive differential evolution algorithm

	Date:
		2018

	Author:
		Uros Mlakar and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Brest, J., Greiner, S., Boskovic, B., Mernik, M., Zumer, V. Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems. IEEE transactions on evolutionary computation, 10(6), 646-657, 2006.

	Attributes:
		Name (list of str): List of strings representing algorithm name
	"""
	Name = ['SelfAdaptiveDifferentialEvolution', 'jDE']

	@staticmethod
	def typeParameters():
		d = DifferentialEvolution.typeParameters()
		d['F_l'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['F_u'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['Tao1'] = lambda x: isinstance(x, (float, int)) and 0 <= x <= 1
		d['Tao2'] = lambda x: isinstance(x, (float, int)) and 0 <= x <= 1
		return d

	def setParameters(self, F_l=0.0, F_u=1.0, Tao1=0.4, Tao2=0.2, **ukwargs):
		r"""Set the parameters of an algorithm.

		Arguments:
			F_l (float): Scaling factor lower limit
			F_u (float): Scaling factor upper limit
			Tao1 (float): Change rate for F parameter update
			Tao2 (float): Change rate for CR parameter update

		See Also:
			:func:`DifferentialEvolution.setParameters`
		"""
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.IndividualType = SolutionjDE
		self.F_l, self.F_u, self.Tao1, self.Tao2 = F_l, F_u, Tao1, Tao2
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def initPop(self, task):
		r"""

		Args:
			task:

		Returns:

		"""
		return [self.IndividualType(task=task, e=True, F=self.F_l + self.rand() * (self.F_u - self.F_l), CR=self.rand(), rand=self.Rand) for _ in range(self.NP)]

	def AdaptiveGen(self, x):
		r"""

		Args:
			x:

		Returns:

		"""
		f = self.F_l + self.rand() * (self.F_u - self.F_l) if self.rand() < self.Tao1 else x.F
		cr = self.rand() if self.rand() < self.Tao2 else x.CR
		return self.IndividualType(x=x.x, F=f, CR=cr, e=False)

	def evolve(self, pop, xb, task):
		r"""

		Args:
			pop:
			xb:
			task:

		Returns:

		"""
		npop = [self.AdaptiveGen(e) for e in pop]
		for i, e in enumerate(npop): e.x = self.CrossMutt(npop, i, xb, e.F, e.CR, rnd=self.Rand)
		return npop

class AgingIndividualJDE(SolutionjDE):
	r"""

	"""
	def __init__(self, **kwargs):
		r"""

		Args:
			**kwargs:

		See Also:
			:func:`SolutionjDE.__init__`
		"""
		SolutionjDE.__init__(self, **kwargs)
		self.age = 0

class AgingSelfAdaptiveDifferentialEvolution(SelfAdaptiveDifferentialEvolution):
	r"""Implementation of Dynamic population size with aging self-adaptive differential evolution algorithm.

	Algorithm:
		Dynamic population size with aging self-adaptive differential evolution algorithm

	Date:
		2018

	Author:
		Jan Popič and Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://link.springer.com/article/10.1007/s10489-007-0091-x

	Reference paper:
		Brest, Janez, and Mirjam Sepesy Maučec. Population size reduction for the differential evolution algorithm. Applied Intelligence 29.3 (2008): 228-247.

	Attributes:
		Name (list of str): List of strings representing algorithm name
	"""
	Name = ['AgingSelfAdaptiveDifferentialEvolution', 'ANpjDE']

	@staticmethod
	def typeParameters():
		d = SelfAdaptiveDifferentialEvolution.typeParameters()
		# FIXME
		return d

	def setParameters(self, LT_min=1, LT_max=7, age=proportional, **ukwargs):
		r"""

		Args:
			LT_min:
			LT_max:
			age:
			**ukwargs:

      See Also:
			:func:`SelfAdaptiveDifferentialEvolution.setParameters`
		"""
		SelfAdaptiveDifferentialEvolution.setParameters(self, **ukwargs)
		self.LT_min, self.LT_max, self.age = LT_min, LT_max, age
		self.mu = abs(self.LT_max - self.LT_min) / 2
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

class DynNpSelfAdaptiveDifferentialEvolutionAlgorithm(SelfAdaptiveDifferentialEvolution):
	r"""Implementation of Dynamic population size self-adaptive differential evolution algorithm.

	Algorithm:
		Dynamic population size self-adaptive differential evolution algorithm

	Date:
		2018

	Author:
		Jan Popič and Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://link.springer.com/article/10.1007/s10489-007-0091-x

	Reference paper:
		Brest, Janez, and Mirjam Sepesy Maučec. Population size reduction for the differential evolution algorithm. Applied Intelligence 29.3 (2008): 228-247.

	Attributes:
		Name (list of str): List of strings representing algorithm name.
	"""
	Name = ['DynNpSelfAdaptiveDifferentialEvolutionAlgorithm', 'dynNPjDE']

	@staticmethod
	def typeParameters():
		d = SelfAdaptiveDifferentialEvolution.typeParameters()
		d['rp'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['pmax'] = lambda x: isinstance(x, int) and x > 0
		return d

	def setParameters(self, rp=0, pmax=10, **ukwargs):
		r"""Set the parameters of an algorithm.

		Arguments:
			rp (int): Small non-negative number which is added to value of genp (if it's not divisible)
			pmax (int): Number of population reductions
		"""
		SelfAdaptiveDifferentialEvolution.setParameters(self, **ukwargs)
		self.rp, self.pmax = rp, pmax
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def postSelection(self, pop, task):
		r"""

		Args:
			pop:
			task:

		Returns:
			array of Individual: New population
		"""
		Gr = task.nFES // (self.pmax * len(pop)) + self.rp
		nNP = len(pop) // 2
		if task.Iters == Gr and len(pop) > 3: pop = [pop[i] if pop[i].f < pop[i + nNP].f else pop[i + nNP] for i in range(nNP)]
		return pop

class MultiStrategySelfAdaptiveDifferentialEvolution(SelfAdaptiveDifferentialEvolution):
	r"""Implementation of self-adaptive differential evolution algorithm with multiple mutation strategys.

	Algorithm:
		Self-adaptive differential evolution algorithm with multiple mutation strategys

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (list of str): List of strings representing algorithm name
	"""
	Name = ['MultiStrategySelfAdaptiveDifferentialEvolution', 'MsjDE']

	def setParameters(self, strategys=[CrossCurr2Rand1, CrossCurr2Best1, CrossRand1, CrossBest1, CrossBest2], **kwargs):
		r"""

		Args:
			strategys:
			**kwargs:

      See Also:
			:func:`SelfAdaptiveDifferentialEvolution.setParameters`
		"""
		SelfAdaptiveDifferentialEvolution.setParameters(self, **kwargs)
		self.IndividualType = SolutionjDE
		self.strategys = strategys

	def multiMutations(self, pop, i, x_b, task):
		r"""

		Args:
			pop:
			i:
			x_b:
			task:

		Returns:

		"""
		L = [task.repair(strategy(pop, i, x_b, pop[i].F, pop[i].CR, rnd=self.Rand), rnd=self.Rand) for strategy in self.strategys]
		L_f = apply_along_axis(task.eval, 1, L)
		ib = argmin(L_f)
		return L[ib], L_f[ib]

	def runTask(self, task):
		#FIXME
		pop = [self.IndividualType(task=task, F=self.F, CR=self.CR, rand=self.Rand) for _i in range(self.NP)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [self.AdaptiveGen(pop[i]) for i in range(len(pop))]
			for i in range(len(npop)): npop[i].x, npop[i].f = self.multiMutations(npop, i, x_b, task)
			pop = [pop[i] if pop[i].f < npop[i].f else npop[i] for i in range(len(npop))]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

class DynNpMultiStrategySelfAdaptiveDifferentialEvolution(MultiStrategySelfAdaptiveDifferentialEvolution):
	r"""Implementation of Dynamic population size self-adaptive differential evolution algorithm with multiple mutation strategys.

	Algorithm:
		Dynamic population size self-adaptive differential evolution algorithm with multiple mutation strategys

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (list of str): List of strings representing algorithm name.
	"""
	Name = ['DynNpMultiStrategySelfAdaptiveDifferentialEvolution', 'dynNpMsjDE']

	def setParameters(self, pmax=10, rp=5, **kwargs):
		r"""

		Args:
			pmax:
			rp:
			**kwargs:

      See Also:
			:func:`MultiStrategySelfAdaptiveDifferentialEvolution.setParameters`
		"""
		MultiStrategySelfAdaptiveDifferentialEvolution.setParameters(self, **kwargs)
		self.pmax, self.rp = pmax, rp

	def runTask(self, task):
		#FIXME
		Gr = task.nFES // (self.pmax * self.NP) + self.rp
		pop = [SolutionjDE(task=task, F=self.F, CR=self.CR, rand=self.Rand) for _i in range(self.NP)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [self.AdaptiveGen(pop[i]) for i in range(len(pop))]
			for i, e in enumerate(npop): e.x, e.f = self.multiMutations(npop, i, x_b, task)
			pop = [pop[i] if pop[i].f < npop[i].f else npop[i] for i in range(len(npop))]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
			if task.Iters == Gr and len(pop) > 3:
				NP = int(len(pop) / 2)
				pop = [pop[i] if pop[i].f < pop[i + NP].f else pop[i + NP] for i in range(NP)]
				Gr += task.nFES // (self.pmax * NP) + self.rp
		return x_b.x, x_b.f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
