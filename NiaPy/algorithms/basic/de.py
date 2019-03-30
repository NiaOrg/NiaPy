# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, unused-argument, no-self-use, no-self-use, attribute-defined-outside-init, logging-not-lazy, len-as-condition, singleton-comparison, arguments-differ, bad-continuation, dangerous-default-value, keyword-arg-before-vararg
import logging
from numpy import random as rand, argmin, argmax, mean, cos, asarray, empty
from scipy.spatial.distance import euclidean

from NiaPy.algorithms.algorithm import Algorithm, Individual

__all__ = ['DifferentialEvolution', 'DynNpDifferentialEvolution', 'AgingNpDifferentialEvolution', 'CrowdingDifferentialEvolution', 'MultiStrategyDifferentialEvolution', 'DynNpMultiStrategyDifferentialEvolution', 'AgingNpMultiMutationDifferentialEvolution', 'AgingIndividual', 'CrossRand1', 'CrossBest2', 'CrossBest1', 'CrossBest2', 'CrossCurr2Rand1', 'CrossCurr2Best1']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

def CrossRand1(pop, ic, x_b, f, cr, rnd=rand, *args):
	r"""Mutation strategy with crossover.

	Mutation strategy uses three different random individuals from population to perform mutation.

	Mutation:
		Name: DE/rand/1

		:math:`\mathbf{x}_{r_1, G} + F \cdot (\mathbf{x}_{r_2, G} - \mathbf{x}_{r_3, G}`
		where :math:`r_1, r_2, r_3` are random indexes representing current population individuals.

	Crossover:
		Name: Binomial crossover

		:math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of individual being mutated.
		x_b (individual): Current global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		*args (List): Additional arguments.

	Returns:
		numpy.ndarray: Mutated and mixed individual
	"""
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 3 else None
	r = rnd.choice(len(pop), 3, replace=not len(pop) >= 3, p=p)
	x = [pop[r[0]][i] + f * (pop[r[1]][i] - pop[r[2]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossBest1(pop, ic, x_b, f, cr, rnd=rand, *args):
	r"""Mutation strategy with crossover.

	Mutation strategy uses two different random individuals from population and global best individual.

	Mutation:
		Name: de/best/1

		:math:`\mathbf{v}_{i, G} = \mathbf{x}_{best, G} + F \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G})`
		where :math:`r_1, r_2` are random indexes representing current population individuals.

	Crossover:
		Name: Binomial crossover

		:math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

	args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of individual being mutated.
		x_b (individual): Current global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		*args (List): Additional arguments.

	returns:
		numpy.ndarray: Mutated and mixed individual.
	"""
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 2 else None
	r = rnd.choice(len(pop), 2, replace=not len(pop) >= 2, p=p)
	x = [x_b[i] + f * (pop[r[0]][i] - pop[r[1]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossRand2(pop, ic, x_b, f, cr, rnd=rand, *args):
	r"""Mutation strategy with crossover.

	Mutation strategy uses five different random individuals from population.

	Mutation:
		Name: de/best/1

		:math:`\mathbf{v}_{i, G} = \mathbf{x}_{r_1, G} + F \cdot (\mathbf{x}_{r_2, G} - \mathbf{x}_{r_3, G}) + F \cdot (\mathbf{x}_{r_4, G} - \mathbf{x}_{r_5, G})`
		where :math:`r_1, r_2, r_3, r_4, r_5` are random indexes representing current population individuals.

	Crossover:
		Name: Binomial crossover

		:math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of individual being mutated.
		x_b (individual): Current global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		*args (List): Additional arguments.

	Returns:
		numpy.ndarray: mutated and mixed individual
	"""
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 5 else None
	r = rnd.choice(len(pop), 5, replace=not len(pop) >= 5, p=p)
	x = [pop[r[0]][i] + f * (pop[r[1]][i] - pop[r[2]][i]) + f * (pop[r[3]][i] - pop[r[4]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossBest2(pop, ic, x_b, f, cr, rnd=rand, *args):
	r"""Mutation strategy with crossover.

	Mutation:
		Name: de/best/2

		:math:`\mathbf{v}_{i, G} = \mathbf{x}_{best, G} + F \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G}) + F \cdot (\mathbf{x}_{r_3, G} - \mathbf{x}_{r_4, G})`
		where :math:`r_1, r_2, r_3, r_4` are random indexes representing current population individuals.

	Crossover:
		Name: Binomial crossover

		:math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of individual being mutated.
		x_b (individual): Current global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		*args (List): Additional arguments.

	Returns:
		numpy.ndarray: mutated and mixed individual
	"""
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 4 else None
	r = rnd.choice(len(pop), 4, replace=not len(pop) >= 4, p=p)
	x = [x_b[i] + f * (pop[r[0]][i] - pop[r[1]][i]) + f * (pop[r[2]][i] - pop[r[3]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossCurr2Rand1(pop, ic, x_b, f, cr, rnd=rand, *args):
	r"""Mutation strategy with crossover.

	Mutation:
		Name: de/curr2rand/1

		:math:`\mathbf{v}_{i, G} = \mathbf{x}_{i, G} + F \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G}) + F \cdot (\mathbf{x}_{r_3, G} - \mathbf{x}_{r_4, G})`
		where :math:`r_1, r_2, r_3, r_4` are random indexes representing current population individuals

	Crossover:
		Name: Binomial crossover

		:math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of individual being mutated.
		x_b (individual): Current global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		*args (List): Additional arguments.

	Returns:
		numpy.ndarray: mutated and mixed individual
	"""
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 4 else None
	r = rnd.choice(len(pop), 4, replace=not len(pop) >= 4, p=p)
	x = [pop[ic][i] + f * (pop[r[0]][i] - pop[r[1]][i]) + f * (pop[r[2]][i] - pop[r[3]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossCurr2Best1(pop, ic, x_b, f, cr, rnd=rand, **kwargs):
	r"""mutation strategy with crossover.

	Mutation:
		Name: de/curr-to-best/1

		:math:`\mathbf{v}_{i, G} = \mathbf{x}_{i, G} + F \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G}) + F \cdot (\mathbf{x}_{r_3, G} - \mathbf{x}_{r_4, G})`
		where :math:`r_1, r_2, r_3, r_4` are random indexes representing current population individuals

	Crossover:
		Name: Binomial crossover

		:math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of individual being mutated.
		x_b (individual): Current global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		*args (List): Additional arguments.

	Returns:
		numpy.ndarray: mutated and mixed individual
	"""
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 3 else None
	r = rnd.choice(len(pop), 3, replace=not len(pop) >= 3, p=p)
	x = [pop[ic][i] + f * (x_b[i] - pop[r[0]][i]) + f * (pop[r[1]][i] - pop[r[2]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

class DifferentialEvolution(Algorithm):
	r"""Implementation of Differential evolution algorithm.

	Algorithm:
	 	Differential evolution algorithm

	Date:
		2018

	Author:
		Uros Mlakar and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and efficient heuristic for global optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359.

	Attributes:
		Name (List[str]): List of string of names for algorithm.
		NP (int): Population size.
		F (float): Scale factor.
		CR (float): Crossover probability.
		CrossMutt (Callable[numpy.ndarray, int, numpy.ndarray, float, float, mtrand.RandomState, Dict[str, Any]]): crossover and mutation strategy.
		IndividualType (Individual): Type of individual used in algorithm.
	"""
	Name = ['DifferentialEvolution', 'DE']
	IndividualType = Individual
	NP, F, CR = 100, 0.5, 0.9

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable[[Any], bool]]:
				* F (Callable[[Union[float, int]], bool]): Check for correct value of parameter.
				* CR (Callable[[float], bool]): Check for correct value of parameter.
		"""
		d = Algorithm.typeParameters()
		d.update({
			'F': lambda x: isinstance(x, (float, int)) and 0 < x <= 2,
			'CR': lambda x: isinstance(x, float) and 0 <= x <= 1
			# TODO add constraint testing for mutation strategy method
		})
		return d

	def setParameters(self, NP=50, F=1, CR=0.8, CrossMutt=CrossRand1, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			NP (Optional[int]): Population size.
			F (Optional[float]): Scaling factor.
			CR (Optional[float]): Crossover rate.
			CrossMutt (Optional[Callable[numpy.ndarray, int, numpy.ndarray, float, float, mtrand.RandomState, Dict[str, Any]]]): Crossover and mutation strategy.
			ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			:func:`Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP, self.initPop)
		self.F, self.CR, self.CrossMutt = F, CR, CrossMutt
		self.IndividualType = IndividualType
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def evolve(self, pop, xb, task, **kwargs):
		r"""Evolve population.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			xb (Individual): Current best individual.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray[Individual]: New evolved populations.
		"""
		return asarray([self.IndividualType(x=self.CrossMutt(pop, i, xb, self.F, self.CR, self.Rand), task=task, rand=self.Rand, e=True) for i in range(len(pop))])

	def selection(self, pop, npop, **kwargs):
		r"""Selection operator.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			npop (numpy.ndarray[Individual]): New Population.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray[Individual]: New selected individuals.
		"""
		return asarray([e if e.f < pop[i].f else pop[i] for i, e in enumerate(npop)])

	def postSelection(self, pop, task, **kwargs):
		r"""Apply additional operation after selection.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray[Individual]: New population.
		"""
		return pop

	def initPop(self, NP, task, rnd):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray[Individual], numpy.ndarray[float]]:
				1. Initialized population.
				2. Initialized population fitness/function values.
		"""
		pop = empty(NP, dtype=object)
		for i in range(NP): pop[i] = self.IndividualType(task=task, e=True, rnd=rnd)
		fpop = asarray([x.f for x in pop])
		return pop, fpop

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core function of Differential Evolution algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray[Initialized]): Current population.
			fpop (numpy.ndarray[float]): Current populations fitness/function values.
			xb (Individual): Current best individual.
			fxb (float): Current best individual function/fitness value.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray[Individual], numpy.ndarray[float], Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments.
		"""
		npop = self.evolve(pop, xb, task)
		pop = self.selection(pop, npop)
		pop = self.postSelection(pop, task)
		return pop, asarray([x.f for x in pop]), {}

class CrowdingDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Differential evolution algorithm with multiple mutation strateys.

	Algorithm:
		Implementation of Differential evolution algorithm with multiple mutation strateys

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		CrowPop (float): Proportion of range for cowding.
	"""
	Name = ['CrowdingDifferentialEvolution', 'CDE']
	CrowPop = 0.1

	def __init__(self, **kwargs): DifferentialEvolution.__init__(self, **kwargs)

	def setParameters(self, CrowPop=0.1, **ukwargs):
		r"""Set core parameters of algorithm.

		Args:
			CrowPop (Optional[float]): Crowding distance.
			**ukwargs: Additional arguments.

		See Also:
			:func:`DifferentialEvolution.setParameters`
		"""
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.CrowPop = CrowPop

	def selection(self, pop, npop):
		r"""Operator for selection of individuals.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			npop (numpy.ndarray[Individual]): New population.

		Returns:
			numpy.ndarray[Individual]: New population.
		"""
		P = []
		for e in npop:
			i = argmin([euclidean(e, f) for f in pop])
			P.append(pop[i] if pop[i].f < e.f else e)
		return asarray(P)

class DynNpDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Dynamic poulation size Differential evolution algorithm.

	Algorithm:
		Dynamic poulation size Differential evolution algorithm

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		pmax (int): TODO
		rp (int): TODO
	"""
	Name = ['DynNpDifferentialEvolution', 'dynNpDE']
	pmax, rp = 50, 3

	@staticmethod
	def typeParameters():
		r"""

		Returns:
			Dict[str, Callable]:
				* rp (Callable[[Union[float, int]], bool]): TODo

		See Also:
			:func:`DifferentialEvolution.typeParameters`
		"""
		r = DifferentialEvolution.typeParameters()
		r['rp'] = lambda x: isinstance(x, (float, int)) and x > 0
		r['pmax'] = lambda x: isinstance(x, int) and x > 0
		return r

	def setParameters(self, pmax=50, rp=3, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			pmax (Optional[int]): TODO
			rp (Optional[int]): TODO

		See Also:
			:func:`DifferentialEvoluton.setParameters`
		"""
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.pmax, self.rp = pmax, rp
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def postSelection(self, pop, task):
		r"""Post selection operator.

		In this algorithm the post selection operator decrements the population at specific iterations/generations.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray[Individual]: Changed current population
		"""
		Gr = task.nFES // (self.pmax * len(pop)) + self.rp
		nNP = len(pop) // 2
		if task.Iters == Gr and len(pop) > 3: pop = [pop[i] if pop[i].f < pop[i + nNP].f else pop[i + nNP] for i in range(nNP)]
		return pop

def proportional(Lt_min, Lt_max, mu, x_f, avg, *args):
	r"""Proportional calculation of age of individual.

	Args:
		Lt_min (int): Minimal life time.
		Lt_max (int): Maximal life time.
		mu (float): TODO
		x_f (float): Individuals function/fitness value.
		avg (float): Average fitness/function value of current population.
		*args (List): Additional arguments.

	Returns:
		int: Age of individual.
	"""
	return min(Lt_min + mu * avg / x_f, Lt_max)

def linear(Lt_min, Lt_max, mu, x_f, avg, x_gw, x_gb, *args):
	r"""Linear calculation of age of individual.

	Args:
		Lt_min (int): Minimal life time.
		Lt_max (int): Maximal life time.
		mu (float): TODO
		x_f (float): Individual function/fitness value.
		avg (float): Average fitness/function value.
		x_gw (float): Global worst fitness/function value.
		x_gb (float): Global best fitness/function value.
		*args (List):

	Returns:
		int: Age of individual.
	"""
	return Lt_min + 2 * mu * (x_f - x_gw) / (x_gb - x_gw)

def bilinear(Lt_min, Lt_max, mu, x_f, avg, x_gw, x_gb, *args):
	r"""Bilinear calculation of age of individual.

	Args:
		Lt_min (int): Minimal life time.
		Lt_max (int): Maximal life time.
		mu (float): TODO
		x_f (float): Individual function/fitness value.
		avg (float): Average fitness/function value.
		x_gw (float): Global worst fitness/function value.
		x_gb (float): Global best fitness/function value.
		*args (List):

	Returns:
		int: Age of individual.
	"""
	if avg < x_f: return Lt_min + mu * (x_f - x_gw) / (x_gb - x_gw)
	return 0.5 * (Lt_min + Lt_max) + mu * (x_f - avg) / (x_gb - avg)

class AgingIndividual(Individual):
	r"""Individual with aging.

	Attributes:
		age (int): Age of individual.

	See Also:
		:class:`Individual`
	"""
	age = 0

	def __init__(self, **kwargs):
		r"""Initalize Aging Individual.

		Args:
			**kwargs Dict[str, Any]: Additional arguments sent to parent

		See Also:
			:func:`Individual.__init__`
		"""
		Individual.__init__(self, **kwargs)
		self.age = 0

class AgingNpDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Differential evolution algorithm with aging individuals.

	Algorithm:
		Differential evolution algorithm with dynamic population size that is defined by the quality of population

	Date:
		2018
	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): list of strings representing algorithm names
		Lt_min (int): minimal age of individual
		Lt_max (int): maximal age of individual
		delta_np (float): TODO
		omega (float): TODO
		age (Callable[[int, int, float, float, float, float, float], int]): Function for calculation of age for individual
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

		Arguments:
			Lt_min (Optional[int]): Minimu life time
			Lt_max (Optional[int]): Maximum life time
			age (Callable[[int, int, float, float, float, float, float], int]): Function for calculation of age for individual

		See Also:
			:func:`DifferentialEvolution.setParameters`
		"""
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.IndividualType = AgingIndividual
		self.Lt_min, self.Lt_max, self.age, self.delta_np, self.omega = Lt_min, Lt_max, age, delta_np, omega
		self.mu = abs(self.Lt_max - self.Lt_min) / 2
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def deltaPopE(self, t):
		r"""Calculate how many individuals are going to dye.

		Args:
			t (float): TODO

		Returns:
			float: Number of individuals to dye.
		"""
		return self.delta_np * abs(cos(t))

	def deltaPopC(self, t):
		r"""Calculate how many individuals are going to be created.

		Args:
			t (float): TODO

		Returns:
			float: TODO
		"""
		return self.delta_np * abs(cos(t + 78))

	def aging(self, task, pop):
		r"""Function applying aging to individuals.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray[Individual]): Current population.

		Returns:
			Tuple[array of Individual, Individual]:
				1. New population.
				2. New best individual.
		"""
		fpop = asarray([x.f for x in pop])
		x_b, x_w = pop[argmin(fpop)], pop[argmax(fpop)]
		avg, npop = mean(fpop), []
		for x in pop:
			x.age += 1
			Lt = round(self.age(self.Lt_min, self.Lt_max, self.mu, x.f, avg, x_w, x_b))
			if x.age <= Lt: npop.append(x)
		if len(npop) == 0: npop = [self.IndividualType(task=task, rand=self.Rand, e=True) for _i in range(len(pop))]
		return npop, x_b

	def popIncrement(self, pop, task):
		r"""Function increments population.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray[Individual]: Increased population.
		"""
		deltapop = int(round(max(1, self.NP * self.deltaPopE(task.Iters))))
		ni = self.Rand.choice(len(pop), deltapop, replace=False)
		return asarray([self.IndividualType(task=task, rand=self.Rand, e=True) for i in ni])

	def popDecrement(self, pop, task):
		r"""Function decrements population.

		Args:
			pop (numpy.ndarray): Current population.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray[Individual]: Decreased population.
		"""
		deltapop = int(round(max(1, self.NP * self.deltaPopC(task.Iters))))
		if len(pop) - deltapop <= 0: return pop
		ni = self.Rand.choice(len(pop), deltapop, replace=False)
		npop = []
		for i, e in enumerate(pop):
			if i not in ni: npop.append(e)
			elif self.rand() >= self.omega: npop.append(e)
		return npop

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core function of AgingNpDifferentialEvolution algorithm.

		Args:
			task (Task): Optimization task
			pop (numpy.ndarray[Individual]): Current population
			fpop (numpy.ndarray[float]): Current populations function/fitness values
			xb (Individual): Current best individual
			fxb (float): Current best individual function/fitness value
			**dparams (Dict[str, Any]): Additional parameters

		Returns:
			Tuple[numpy.ndarray[Individual], numpy.ndarray[float], Dict[str, Any]]:
				1. New population
				2. New population fitness/function values
				3. Additional parameters
		"""
		npop = self.evolve(pop, xb, task)
		npop = self.selection(pop, npop)
		npop.extend(self.popIncrement(pop, task, xb))
		pop, xbn = self.aging(task, npop)
		if len(pop) > self.NP: pop = self.popDecrement(pop, task, xbn)
		return pop, [x.f for x in pop], {}

def multiMutations(pop, i, xb, F, CR, rnd, task, IndividualType, strategies, **kwargs):
	r"""Mutation strategy that takes more than one strategy and applys them to individual.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		i (int): Index of current individual.
		xb (Individual): Current best individual.
		F (float): Scale factor.
		CR (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		task (Task): Optimization task.
		IndividualType (Individual): Individual type used in algorithm.
		strategies (Iterable[Callable[[numpy.ndarray[Individual], int, Individual, float, float, mtrand.RandomState], numpy.ndarray[Individual]]]): List of mutation strategies.
		**kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Individual: Best individual from applyed mutations strategies.
	"""
	L = [IndividualType(x=strategy(pop, i, xb, F, CR, rnd=rnd), task=task, e=True, rand=rnd) for strategy in strategies]
	return L[argmin([x.f for x in L])]

class MultiStrategyDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Differential evolution algorithm with multiple mutation strateys.

	Algorithm:
		Implementation of Differential evolution algorithm with multiple mutation strateys

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		strategies (Iterable[Callable[[numpy.ndarray[Individual], int, Individual, float, float, mtrand.RandomState], numpy.ndarray[Individual]]]): List of mutation strategies.
	"""
	Name = ['MultiStrategyDifferentialEvolution', 'MsDE']

	@staticmethod
	def typeParameters():
		r"""

		Returns:
			Dict[str, Callable]:
				* CrossMutt (Callable[[Callable, bool]): TODO

		See Also:
			:func:`DifferentialEvolution.typeParameters`
		"""
		r = DifferentialEvolution.typeParameters()
		r.pop('CrossMutt', None)
		# TODO add constraint method for selection of stratgy methos
		return r

	def setParameters(self, strategys=(CrossRand1, CrossBest1, CrossCurr2Best1, CrossRand2), **ukwargs):
		r"""Set the arguments of the algorithm.

		Arguments:
			strategies (Optional[Iterable[Callable[[numpy.ndarray[Individual], int, Individual, float, float, mtrand.RandomState], numpy.ndarray[Individual]]]]): List of mutation strategyis.

		See Also:
			:func:`DifferentialEvolution.setParameters`
		"""
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.CrossMutt, self.strategies = multiMutations, strategys

	def evolve(self, pop, xb, task, **kwargs):
		r"""Evolve population with the help multiple mutation strategies.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			xb (Individual): Current best individual.
			task (Task): Optimization task.
			**kwargs: Additional arguments.

		Returns:
			numpy.ndarray[Individual]: New population of individuals.
		"""
		return asarray([self.CrossMutt(pop, i, xb, self.F, self.CR, self.Rand, task, self.IndividualType, self.strategies) for i in range(len(pop))])

class DynNpMultiStrategyDifferentialEvolution(MultiStrategyDifferentialEvolution):
	r"""Implementation of Dynamic population size Differential evolution algorithm with dynamic population size that is defined by the quality of population.

	Algorithm:
		Dynamic population size Differential evolution algorithm with dynamic population size that is defined by the quality of population

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): list of strings representing algorithm name
		pmax (int): TODO
		rp (int): TODO
	"""
	Name = ['DynNpMultiStrategyDifferentialEvolution', 'dynNpMsDE']
	pmax, rp = 10, 3

	@staticmethod
	def typeParameters():
		r"""

		Returns:
			Dict[str, Callable]:
				* rp (Callable[[Union[float, int]], bool]): TODO
				* pmax (Callable[[int], bool]): TODO

		See Also:
			:func:`MultiStrategyDifferentialEvolution.typeParameters`
		"""
		r = MultiStrategyDifferentialEvolution.typeParameters()
		r['rp'] = lambda x: isinstance(x, (float, int)) and x > 0
		r['pmax'] = lambda x: isinstance(x, int) and x > 0
		return r

	def setParameters(self, pmax=10, rp=3, **ukwargs):
		r"""Set the arguments of the algorithm.

		Arguments:
			pmax (Optional[int]): TODO
			rp (Optional[int]): TODO

		See:
			:func:`DifferentialEvolution.setParameters`
		"""
		MultiStrategyDifferentialEvolution.setParameters(self, **ukwargs)
		self.pmax, self.rp = pmax, rp

	def postSelection(self, pop, task, **kwargs):
		r"""Function decrease population on specific iterations.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray[Individual]: New population.
		"""
		Gr = task.nFES // (self.pmax * len(pop)) + self.rp
		nNP = len(pop) // 2
		if task.Iters == Gr and len(pop) > 3: pop = asarray([pop[i] if pop[i].f < pop[i + nNP].f else pop[i + nNP] for i in range(nNP)])
		return pop

class AgingNpMultiMutationDifferentialEvolution(AgingNpDifferentialEvolution, MultiStrategyDifferentialEvolution):
	r"""Implementation of Differential evolution algorithm with aging individuals.

	Algorithm:
		Differential evolution algorithm with dynamic population size that is defined by the quality of population

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm names
	"""
	Name = ['AgingNpMultiMutationDifferentialEvolution', 'ANpMSDE']

	@staticmethod
	def typeParameters():
		r = AgingNpDifferentialEvolution.typeParameters()
		# TODO add other parameters to data check list
		return r

	def setParameters(self, **ukwargs):
		r"""Set core parameter arguments.

		Args:
			**ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`AgingNpDifferentialEvolution.setParameters`
			* :func:`MultiStrategyDifferentialEvolution.setParameters`
		"""
		AgingNpDifferentialEvolution.setParameters(self, **ukwargs)
		MultiStrategyDifferentialEvolution.setParameters(self, stratgeys=[CrossRand1, CrossBest1, CrossCurr2Rand1, CrossRand2], **ukwargs)
		self.IndividualType = AgingIndividual

	def evolve(self, pop, xb, task):
		r"""Evolve current population.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			xb (Individual): Current best individuals.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray[Individual]: New population.
		"""
		return asarray([self.CrossMutt(pop, i, xb, self.F, self.CR, self.Rand, task, self.IndividualType, self.strategys) for i in range(len(pop))])

	def popIncrement(self, pop, task, xb):
		r"""Increment population

		Args:
			pop (numpy.ndarray[Individual]): Current population
			task (Task): Optimization task
			xb (Individual): Current best individual

		Returns:
			numpy.ndarray[Individual]: New population.
		"""
		deltapop = int(round(max(1, self.NP * self.deltaPopE(task.Iters))))
		ni = self.Rand.choice(len(pop), deltapop, replace=False)
		return asarray([self.CrossMutt(pop, i, xb, self.F, self.CR, self.Rand, task, self.IndividualType, self.strategys) for i in ni])

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core function of AgingNpMultiMutationDifferentialEvolution algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray[Individual]): Current population.
			fpop (numpy.ndarray[float]): Current population fitness/function values.
			xb (Individual): Current best individual.
			fxb (float): Current best individual fitness/function value.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray[Individual], numpy.ndarray[float], Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments.

		See Also:
			:func:`AgingNpDifferentialEvolution.runIteration`
		"""
		return AgingNpDifferentialEvolution.runIteration(self, task, pop, fpop, xb, fxb, **dparams)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
