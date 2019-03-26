# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, unused-argument, no-self-use, no-self-use, attribute-defined-outside-init, logging-not-lazy, len-as-condition, singleton-comparison, arguments-differ, bad-continuation, dangerous-default-value, keyword-arg-before-vararg
import logging
from numpy import random as rand, argmin, argmax, mean, asarray, cos
from scipy.spatial.distance import euclidean

from NiaPy.algorithms.algorithm import Algorithm, Individual

__all__ = ['DifferentialEvolution', 'DynNpDifferentialEvolution', 'AgingNpDifferentialEvolution', 'CrowdingDifferentialEvolution', 'MultiStrategyDifferentialEvolution', 'DynNpMultiStrategyDifferentialEvolution', 'AgingNpMultiMutationDifferentialEvolution', 'AgingIndividual', 'CrossRand1', 'CrossBest2', 'CrossBest1', 'CrossBest2', 'CrossCurr2Rand1', 'CrossCurr2Best1']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

def CrossRand1(pop, ic, x_b, f, cr, rnd=rand, *args):
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 3 else None
	r = rnd.choice(len(pop), 3, replace=not len(pop) >= 3, p=p)
	x = [pop[r[0]][i] + f * (pop[r[1]][i] - pop[r[2]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossBest1(pop, ic, x_b, f, cr, rnd=rand, *args):
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 2 else None
	r = rnd.choice(len(pop), 2, replace=not len(pop) >= 2, p=p)
	x = [x_b[i] + f * (pop[r[0]][i] - pop[r[1]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossRand2(pop, ic, x_b, f, cr, rnd=rand, *args):
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 5 else None
	r = rnd.choice(len(pop), 5, replace=not len(pop) >= 5, p=p)
	x = [pop[r[0]][i] + f * (pop[r[1]][i] - pop[r[2]][i]) + f * (pop[r[3]][i] - pop[r[4]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossBest2(pop, ic, x_b, f, cr, rnd=rand, *args):
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 4 else None
	r = rnd.choice(len(pop), 4, replace=not len(pop) >= 4, p=p)
	x = [x_b[i] + f * (pop[r[0]][i] - pop[r[1]][i]) + f * (pop[r[2]][i] - pop[r[3]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossCurr2Rand1(pop, ic, x_b, f, cr, rnd=rand, *args):
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 4 else None
	r = rnd.choice(len(pop), 4, replace=not len(pop) >= 4, p=p)
	x = [pop[ic][i] + f * (pop[r[0]][i] - pop[r[1]][i]) + f * (pop[r[2]][i] - pop[r[3]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossCurr2Best1(pop, ic, x_b, f, cr, rnd=rand, **kwargs):
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 3 else None
	r = rnd.choice(len(pop), 3, replace=not len(pop) >= 3, p=p)
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
	NP, F, CR = 100, 0.5, 0.9

	@staticmethod
	def typeParameters(): return {
			'NP': lambda x: isinstance(x, int) and x > 0,
			'F': lambda x: isinstance(x, (float, int)) and 0 < x <= 2,
			'CR': lambda x: isinstance(x, float) and 0 <= x <= 1
			# TODO add constraint testing for mutation strategy method
	}

	def setParameters(self, NP=50, F=1, CR=0.8, CrossMutt=CrossRand1, **ukwargs):
		r"""Set the algorithm parameters.

		**Arguments:**
		NP {integer} -- population size
		F {decimal} -- scaling factor
		CR {decimal} -- crossover rate
		CrossMutt {function} -- crossover and mutation strategy
		"""
		self.NP, self.F, self.CR, self.CrossMutt = NP, F, CR, CrossMutt
		self.IndividualType = Individual
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def evolve(self, pop, xb, task, **kwargs):
		return [self.IndividualType(x=self.CrossMutt(pop, i, xb, self.F, self.CR, self.Rand), task=task, rand=self.Rand, e=True) for i in range(len(pop))]

	def selection(self, pop, npop, **kwargs):
		return [e if e.f < pop[i].f else pop[i] for i, e in enumerate(npop)]

	def postSelection(self, pop, task, **kwargs):
		return pop

	def initPopulation(self, task):
		pop = [self.IndividualType(task=task, e=True, rand=self.Rand) for _ in range(self.NP)]
		fpop = [x.f for x in pop]
		return pop, fpop, {}

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		npop = self.evolve(pop, xb, task)
		pop = self.selection(pop, npop)
		pop = self.postSelection(pop, task)
		return pop, [x.f for x in pop], {}

class CrowdingDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Differential evolution algorithm with multiple mutation strateys.

	**Algorithm:** Implementation of Differential evolution algorithm with multiple mutation strateys
	**Date:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	Name = ['CrowdingDifferentialEvolution', 'CDE']
	CrowPop = 0.1

	def __init__(self, **kwargs): DifferentialEvolution.__init__(self, **kwargs)

	def setParameters(self, CrowPop=0.1, **ukwargs):
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.CrowPop = CrowPop

	def selection(self, pop, npop):
		P = []
		for e in npop:
			i = argmin([euclidean(e, f) for f in pop])
			P.append(pop[i] if pop[i].f < e.f else e)
		return P

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

	def postSelection(self, pop, task):
		Gr = task.nFES // (self.pmax * len(pop)) + self.rp
		nNP = len(pop) // 2
		if task.Iters == Gr and len(pop) > 3: pop = [pop[i] if pop[i].f < pop[i + nNP].f else pop[i + nNP] for i in range(nNP)]
		return pop

def proportional(Lt_min, Lt_max, mu, x_f, avg, *args): return min(Lt_min + mu * avg / x_f, Lt_max)

def linear(Lt_min, Lt_max, mu, x_f, avg, x_gw, x_gb, *args): return Lt_min + 2 * mu * (x_f - x_gw) / (x_gb - x_gw)

def bilinear(Lt_min, Lt_max, mu, x_f, avg, x_gw, x_gb, *args):
	if avg < x_f: return Lt_min + mu * (x_f - x_gw) / (x_gb - x_gw)
	return 0.5 * (Lt_min + Lt_max) + mu * (x_f - avg) / (x_gb - avg)

class AgingIndividual(Individual):
	age = 0

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
	Name = ['AgingNpDifferentialEvolution', 'ANpDE']
	Lt_min, Lt_max, delta_np, omega = 1, 12, 0.3, 0.3

	@staticmethod
	def typeParameters():
		r = DifferentialEvolution.typeParameters()
		# TODO add other parameters to data check list
		return r

	def setParameters(self, Lt_min=0, Lt_max=12, delta_np=0.3, omega=0.3, age=proportional, CrossMutt=CrossBest1, **ukwargs):
		r"""Set the algorithm parameters.

		**Arguments**:
		Lt_min {integer} -- Minimu life time
		Lt_max {integer} -- Maximum life time
		age {function} -- Function for calculation of age for individual
		"""
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.IndividualType = AgingIndividual
		self.Lt_min, self.Lt_max, self.age, self.delta_np, self.omega = Lt_min, Lt_max, age, delta_np, omega
		self.mu = abs(self.Lt_max - self.Lt_min) / 2
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def deltaPopE(self, t): return self.delta_np * abs(cos(t))

	def deltaPopC(self, t): return self.delta_np * abs(cos(t + 78))

	def aging(self, task, pop):
		fpop = asarray([x.f for x in pop])
		x_b, x_w = pop[argmin(fpop)], pop[argmax(fpop)]
		avg, npop = mean(fpop), []
		for x in pop:
			x.age += 1
			Lt = round(self.age(self.Lt_min, self.Lt_max, self.mu, x.f, avg, x_w, x_b))
			if x.age <= Lt: npop.append(x)
		if len(npop) == 0: npop = [self.IndividualType(task=task, rand=self.Rand, e=True) for _i in range(len(pop))]
		return npop, x_b

	def popIncrement(self, pop, task, xb):
		deltapop = int(round(max(1, self.NP * self.deltaPopE(task.Iters))))
		ni = self.Rand.choice(len(pop), deltapop, replace=False)
		return [self.IndividualType(task=task, rand=self.Rand, e=True) for i in ni]

	def popDecrement(self, pop, task, xb):
		deltapop = int(round(max(1, self.NP * self.deltaPopC(task.Iters))))
		if len(pop) - deltapop <= 0: return pop
		ni = self.Rand.choice(len(pop), deltapop, replace=False)
		npop = []
		for i, e in enumerate(pop):
			if i not in ni: npop.append(e)
			elif self.rand() >= self.omega: npop.append(e)
		return npop

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		npop = self.evolve(pop, xb, task)
		npop = self.selection(pop, npop)
		npop.extend(self.popIncrement(pop, task, xb))
		pop, xbn = self.aging(task, npop)
		if len(pop) > self.NP: pop = self.popDecrement(pop, task, xbn)
		return pop, [x.f for x in pop], {}

def multiMutations(pop, i, xb, F, CR, rnd, task, IndividualType, strategys, **kwargs):
	L = [IndividualType(x=strategy(pop, i, xb, F, CR, rnd=rnd), task=task, e=True, rand=rnd) for strategy in strategys]
	return L[argmin([x.f for x in L])]

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
		self.CrossMutt, self.strategys = multiMutations, strategys

	def evolve(self, pop, xb, task, **kwargs):
		return [self.CrossMutt(pop, i, xb, self.F, self.CR, self.Rand, task, self.IndividualType, self.strategys) for i in range(len(pop))]

class DynNpMultiStrategyDifferentialEvolution(MultiStrategyDifferentialEvolution):
	r"""Implementation of Dynamic population size Differential evolution algorithm with dynamic population size that is defined by the quality of population.

	**Algorithm:** Dynamic population size Differential evolution algorithm with dynamic population size that is defined by the quality of population
	**Date:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	Name = ['DynNpMultiStrategyDifferentialEvolution', 'dynNpMsDE']
	pmax, rp = 10, 3

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

	def postSelection(self, pop, task, **kwargs):
		Gr = task.nFES // (self.pmax * len(pop)) + self.rp
		nNP = len(pop) // 2
		if task.Iters == Gr and len(pop) > 3: pop = [pop[i] if pop[i].f < pop[i + nNP].f else pop[i + nNP] for i in range(nNP)]
		return pop

class AgingNpMultiMutationDifferentialEvolution(AgingNpDifferentialEvolution, MultiStrategyDifferentialEvolution):
	r"""Implementation of Differential evolution algorithm with aging individuals.

	**Algorithm:** Differential evolution algorithm with dynamic population size that is defined by the quality of population
	**Date:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	Name = ['AgingNpMultiMutationDifferentialEvolution', 'ANpMSDE']

	@staticmethod
	def typeParameters():
		r = AgingNpDifferentialEvolution.typeParameters()
		# TODO add other parameters to data check list
		return r

	def setParameters(self, **ukwargs):
		AgingNpDifferentialEvolution.setParameters(self, **ukwargs)
		MultiStrategyDifferentialEvolution.setParameters(self, stratgeys=[CrossRand1, CrossBest1, CrossCurr2Rand1, CrossRand2], **ukwargs)
		self.IndividualType = AgingIndividual

	def evolve(self, pop, xb, task):
		return [self.CrossMutt(pop, i, xb, self.F, self.CR, self.Rand, task, self.IndividualType, self.strategys) for i in range(len(pop))]

	def popIncrement(self, pop, task, xb):
		deltapop = int(round(max(1, self.NP * self.deltaPopE(task.Iters))))
		ni = self.Rand.choice(len(pop), deltapop, replace=False)
		return [self.CrossMutt(pop, i, xb, self.F, self.CR, self.Rand, task, self.IndividualType, self.strategys) for i in ni]

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
