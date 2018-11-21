# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, unused-argument, no-self-use, no-self-use, attribute-defined-outside-init, logging-not-lazy, len-as-condition, singleton-comparison, arguments-differ, bad-continuation, dangerous-default-value
import logging
from numpy import random as rand, argmin, argmax, concatenate, mean, asarray
from NiaPy.algorithms.algorithm import Algorithm, Individual

__all__ = ['DifferentialEvolution', 'DynNpDifferentialEvolution', 'AgingNpDifferentialEvolution', 'MultiStrategyDifferentialEvolution', 'DynNpMultiStrategyDifferentialEvolution', 'CrossRand1', 'CrossBest2', 'CrossBest1', 'CrossBest2', 'CrossCurr2Rand1', 'CrossCurr2Best1']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

def CrossRand1(pop, ic, x_b, f, cr, rnd=rand):
	j = rnd.randint(len(pop[0]))
	r = rnd.choice(len(pop), 3, replace=not len(pop) >= 3)
	x = [pop[r[0]][i] + f * (pop[r[1]][i] - pop[r[2]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossBest1(pop, ic, x_b, f, cr, rnd=rand):
	j = rnd.randint(len(pop[0]))
	r = rnd.choice(len(pop), 2, replace=not len(pop) >= 2)
	x = [x_b[i] + f * (pop[r[0]][i] - pop[r[1]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossRand2(pop, ic, x_b, f, cr, rnd=rand):
	j = rnd.randint(len(pop[0]))
	r = rnd.choice(len(pop), 5, replace=not len(pop) >= 5)
	x = [pop[r[0]][i] + f * (pop[r[1]][i] - pop[r[2]][i]) + f * (pop[r[3]][i] - pop[r[4]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossBest2(pop, ic, x_b, f, cr, rnd=rand):
	j = rnd.randint(len(pop[0]))
	r = rnd.choice(len(pop), 4, replace=not len(pop) >= 4)
	x = [x_b[i] + f * (pop[r[0]][i] - pop[r[1]][i]) + f * (pop[r[2]][i] - pop[r[3]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossCurr2Rand1(pop, ic, x_b, f, cr, rnd=rand):
	j = rnd.randint(len(pop[0]))
	r = rnd.choice(len(pop), 4, replace=not len(pop) >= 4)
	x = [pop[ic][i] + f * (pop[r[0]][i] - pop[r[1]][i]) + f * (pop[r[2]][i] - pop[r[3]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossCurr2Best1(pop, ic, x_b, f, cr, rnd=rand):
	j = rnd.randint(len(pop[0]))
	r = rnd.choice(len(pop), 3, replace=not len(pop) >= 3)
	x = [pop[ic][i] + f * (x_b[i] - pop[r[0]][i]) + f * (pop[r[1]][i] - pop[r[2]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

class DifferentialEvolution(Algorithm):
	r"""Implementation of Differential evolution algorithm.

	**Algorithm:** Differential evolution algorithm

	**Date:** 2018

	**Author:** Uros Mlakar and Klemen Berkovič

	**License:** MIT

	**Reference paper:**
	Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and efficient heuristic for global optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359.
	"""
	Name = ['DifferentialEvolution', 'DE']

	@staticmethod
	def typeParameters(): return {
			'NP': lambda x: isinstance(x, int) and x > 0,
			'F': lambda x: isinstance(x, (float, int)) and 0 < x <= 2,
			'CR': lambda x: isinstance(x, float) and 0 <= x <= 1
			# TODO add constraint testing for mutation strategy method
	}

	def setParameters(self, NP=25, F=0.5, CR=0.9, CrossMutt=CrossRand1, **ukwargs):
		r"""Set the algorithm parameters.

		**Arguments:**

		NP {integer} -- population size

		F {decimal} -- scaling factor

		CR {decimal} -- crossover rate

		CrossMutt {function} -- crossover and mutation strategy
		"""
		self.Np, self.F, self.CR, self.CrossMutt = NP, F, CR, CrossMutt
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		pop = [Individual(task=task, e=True, rand=self.Rand) for _i in range(self.Np)]
		ib = argmin([x.f for x in pop])
		x_b, x_bf = pop[ib].x, pop[ib].f
		while not task.stopCondI():
			npop = [Individual(x=self.CrossMutt(pop, i, x_b, self.F, self.CR, rnd=self.Rand), task=task, e=True, rand=self.Rand) for i in range(self.Np)]
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			ib = argmin([x.f for x in pop])
			if x_bf > pop[ib].f: x_b, x_bf = pop[ib].x, pop[ib].f
		return x_b, x_bf

class DynNpDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Dynamic poulation size Differential evolution algorithm.

	**Algorithm:** Dynamic poulation size Differential evolution algorithm

	**Date:** 2018

	**Author:** Klemen Berkovič

	**License:** MIT
	"""
	Name = ['DynNpDifferentialEvolution', 'dynNpDE']

	@staticmethod
	def typeParameters():
		r = DifferentialEvolution.typeParameters()
		r['rp'] = lambda x: isinstance(x, (float, int)) and x > 0
		r['pmax'] = lambda x: isinstance(x, int) and x > 0
		return r

	def setParameters(self, pmax=50, rp=3, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
		NP {integer} -- population size
		F {decimal} -- scaling factor
		CR {decimal} -- crossover rate
		SR {decimal} -- search reange for best (normalized)
		CrossMutt {function} -- crossover and mutation strategy
		"""
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.pmax, self.rp = pmax, rp
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		Gr = task.nFES // (self.pmax * self.Np) + self.rp
		pop = [Individual(task=task, rand=self.Rand, e=True) for _i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCondI():
			npop = [Individual(x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand), task=task, rand=self.Rand, e=True) for i in range(len(pop))]
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
			if task.Iters == Gr and len(pop) > 3:
				NP = int(len(pop) / 2)
				pop = [pop[i] if pop[i].f < pop[i + NP].f else pop[i + NP] for i in range(NP)]
				Gr += task.nFES // (self.pmax * NP) + self.rp
		return x_b.x, x_b.f

def proportional(Lt_min, Lt_max, mu, x_f, avg, *kwargs): return min(Lt_min + mu * avg / x_f, Lt_max)

def linear(Lt_min, Lt_max, mu, x_f, avg, x_gw, x_gb, *kwargs): return Lt_min + 2 * mu * (x_f - x_gw) / (x_gb - x_gw)

def bilinear(Lt_min, Lt_max, mu, x_f, avg, x_gw, x_gb, *kwargs):
	if avg < x_f: return Lt_min + mu * (x_f - x_gw) / (x_gb - x_gw)
	return 0.5 * (Lt_min + Lt_max) + mu * (x_f - avg) / (x_gb - avg)

class AgingIndividual(Individual):
	def __init__(self, **kwargs):
		Individual.__init__(self, **kwargs)
		self.age = 0

class AgingNpDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Differential evolution algorithm with aging individuals.

	**Algorithm:** Differential evolution algorithm with dynamic population size that is defined by the quality of population

	**Date:** 2018

	**Author:** Klemen Berkovič

	**License:** MIT
	"""
	Name = ['AgingNpDifferentialEvolution', 'ANpSDE']

	@staticmethod
	def typeParameters():
		r = DifferentialEvolution.typeParameters()
		# TODO add other parameters to data check list
		return r

	def setParameters(self, Lt_min=1, Lt_max=10, age=bilinear, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
		Lt_min {integer} -- Minimu life time
		Lt_max {integer} -- Maximum life time
		age {function} -- Function for calculation of age for individual
		"""
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.Lt_min, self.Lt_max, self.age = Lt_min, Lt_max, age
		self.mu = abs(self.Lt_max - self.Lt_min) / 2
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def reduction(self, task, pop, npop, x_b, x_w):
		fpop, fnpop = asarray([x.f for x in pop]), asarray([x.f for x in npop])
		xn_b, xn_w = npop[argmin(fnpop)], npop[argmax(fnpop)]
		if xn_b.f < x_b.f: x_b = xn_b
		if xn_w.f > x_w.f: x_w = xn_w
		avg, nnpop = mean(concatenate((fpop, fnpop))), []
		for x in pop:
			Lt = self.age(self.Lt_min, self.Lt_max, self.mu, x.f, avg, x_w.f, x_b.f)
			if x.age <= round(Lt): nnpop.append(x)
		nnpop.extend(sorted(npop, key=lambda x: x.f)[:int(len(pop))])
		return nnpop, x_b, x_w

	def runTask(self, task):
		pop = [AgingIndividual(task=task, rand=self.Rand, e=True) for _i in range(self.Np)]
		x_b, x_w = pop[argmin([x.f for x in pop])], pop[argmax([x.f for x in pop])]
		while not task.stopCondI():
			for x in pop: x.age += 1
			npop = [AgingIndividual(x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand), task=task, rand=self.Rand, e=True) for i in range(len(pop))]
			pop, x_b, x_w = self.reduction(task, pop, npop, x_b, x_w)
		return x_b.x, x_b.f

class MultiStrategyDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Differential evolution algorithm with multiple mutation strateys.

	**Algorithm:** Implementation of Differential evolution algorithm with multiple mutation strateys

	**Date:** 2018

	**Author:** Klemen Berkovič

	**License:** MIT
	"""
	Name = ['MultiStrategyDifferentialEvolution', 'MsDE']

	@staticmethod
	def typeParameters():
		r = DifferentialEvolution.typeParameters()
		r.pop('CrossMutt', None)
		# TODO add constraint method for selection of stratgy methos
		return r

	def setParameters(self, strategys=[CrossRand1, CrossBest1, CrossCurr2Best1, CrossRand2], **ukwargs):
		r"""Set the arguments of the algorithm.

		Arguments:
		strategys {array} of {function} -- Mutation stratgeys to use

		See:
		DifferentialEvolution.setParameters
		"""
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.strategys = strategys

	def multiMutations(self, pop, i, x_b, task):
		L = [Individual(x=strategy(pop, i, x_b, self.F, self.CR, rnd=self.Rand), task=task, e=True, rand=self.Rand) for strategy in self.strategys]
		return L[argmin([x.f for x in L])]

	def runTask(self, task):
		pop = [Individual(task=task, e=True, rand=self.Rand) for _i in range(self.Np)]
		ib = argmin([x.f for x in pop])
		x_b, x_bf = pop[ib].x, pop[ib].f
		while not task.stopCondI():
			npop = [self.multiMutations(pop, i, x_b, task) for i in range(self.Np)]
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			ib = argmin([x.f for x in pop])
			if x_bf > pop[ib].f: x_b, x_bf = pop[ib].x, pop[ib].f
		return x_b, x_bf

class DynNpMultiStrategyDifferentialEvolution(MultiStrategyDifferentialEvolution):
	r"""Implementation of Dynamic population size Differential evolution algorithm with dynamic population size that is defined by the quality of population.

	**Algorithm:** Dynamic population size Differential evolution algorithm with dynamic population size that is defined by the quality of population

	**Date:** 2018

	**Author:** Klemen Berkovič

	**License:** MIT
	"""
	Name = ['DynNpMultiStrategyDifferentialEvolution', 'dynNpMsDE']

	@staticmethod
	def typeParameters():
		r = MultiStrategyDifferentialEvolution.typeParameters()
		r['rp'] = lambda x: isinstance(x, (float, int)) and x > 0
		r['pmax'] = lambda x: isinstance(x, int) and x > 0
		return r

	def setParameters(self, pmax=10, rp=3, **ukwargs):
		r"""Set the arguments of the algorithm.

		Arguments:
		strategys {array} of {function} -- Mutation stratgeys to use

		See:
		DifferentialEvolution.setParameters
		"""
		MultiStrategyDifferentialEvolution.setParameters(self, **ukwargs)
		self.pmax, self.rp = pmax, rp

	def runTask(self, task):
		Gr = task.nFES // (self.pmax * self.Np) + self.rp
		pop = [Individual(task=task, e=True, rand=self.Rand) for _i in range(self.Np)]
		ib = argmin([x.f for x in pop])
		x_b, x_bf = pop[ib].x, pop[ib].f
		while not task.stopCondI():
			npop = [self.multiMutations(pop, i, x_b, task) for i in range(len(pop))]
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			ib = argmin([x.f for x in pop])
			if x_bf > pop[ib].f: x_b, x_bf = pop[ib].x, pop[ib].f
			if task.Iters == Gr and len(pop) > 3:
				NP = int(len(pop) / 2)
				pop = [pop[i] if pop[i].f < pop[i + NP].f else pop[i + NP] for i in range(NP)]
				Gr += task.nFES // (self.pmax * NP) + self.rp
		return x_b, x_bf

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
