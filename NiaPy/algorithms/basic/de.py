# encoding=utf8
import logging

from numpy import random as rand, argmin, argmax, mean, cos, asarray, append, sin
from scipy.spatial.distance import euclidean

from NiaPy.algorithms.algorithm import Algorithm, Individual, defaultIndividualInit
from NiaPy.util.utility import objects2array

__all__ = ['DifferentialEvolution', 'DynNpDifferentialEvolution', 'AgingNpDifferentialEvolution', 'CrowdingDifferentialEvolution', 'MultiStrategyDifferentialEvolution', 'DynNpMultiStrategyDifferentialEvolution', 'AgingNpMultiMutationDifferentialEvolution', 'AgingIndividual', 'CrossRand1', 'CrossBest2', 'CrossBest1', 'CrossBest2', 'CrossCurr2Rand1', 'CrossCurr2Best1', 'multiMutations']

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
		x_b (Individual): Current global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		*args (list): Additional arguments.

	Returns:
		numpy.ndarray: Mutated and mixed individual.
	"""
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 3 else None
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
		x_b (Individual): Current global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		*args (list): Additional arguments.

	returns:
		numpy.ndarray: Mutated and mixed individual.
	"""
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 2 else None
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
		x_b (Individual): Current global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		*args (list): Additional arguments.

	Returns:
		numpy.ndarray: mutated and mixed individual.
	"""
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 5 else None
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
		x_b (Individual): Current global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		*args (list): Additional arguments.

	Returns:
		numpy.ndarray: mutated and mixed individual.
	"""
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 4 else None
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
		x_b (Individual): Current global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		*args (list): Additional arguments.

	Returns:
		numpy.ndarray: mutated and mixed individual.
	"""
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 4 else None
	r = rnd.choice(len(pop), 4, replace=not len(pop) >= 4, p=p)
	x = [pop[ic][i] + f * (pop[r[0]][i] - pop[r[1]][i]) + f * (pop[r[2]][i] - pop[r[3]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossCurr2Best1(pop, ic, x_b, f, cr, rnd=rand, **kwargs):
	r"""Mutation strategy with crossover.

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
		x_b (Individual): Current global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.
		*args (list): Additional arguments.

	Returns:
		numpy.ndarray: mutated and mixed individual.
	"""
	j = rnd.randint(len(pop[ic]))
	p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 3 else None
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
		F (float): Scale factor.
		CR (float): Crossover probability.
		CrossMutt (Callable[numpy.ndarray, int, numpy.ndarray, float, float, mtrand.RandomState, Dict[str, Any]]): crossover and mutation strategy.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['DifferentialEvolution', 'DE']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and efficient heuristic for global optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359."""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* F (Callable[[Union[float, int]], bool]): Check for correct value of parameter.
				* CR (Callable[[float], bool]): Check for correct value of parameter.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'F': lambda x: isinstance(x, (float, int)) and 0 < x <= 2,
			'CR': lambda x: isinstance(x, float) and 0 <= x <= 1
		})
		return d

	def setParameters(self, NP=50, F=1, CR=0.8, CrossMutt=CrossRand1, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			NP (Optional[int]): Population size.
			F (Optional[float]): Scaling factor.
			CR (Optional[float]): Crossover rate.
			CrossMutt (Optional[Callable[[numpy.ndarray, int, numpy.ndarray, float, float, mtrand.RandomState, list], numpy.ndarray]]): Crossover and mutation strategy.
			ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, InitPopFunc=ukwargs.pop('InitPopFunc', defaultIndividualInit), itype=ukwargs.pop('itype', Individual), **ukwargs)
		self.F, self.CR, self.CrossMutt = F, CR, CrossMutt

	def getParameters(self):
		r"""Get parameters values of the algorithm.

		Returns:
			Dict[str, Any]: TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.getParameters`
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'F': self.F,
			'CR': self.CR,
			'CrossMutt': self.CrossMutt
		})
		return d

	def evolve(self, pop, xb, task, **kwargs):
		r"""Evolve population.

		Args:
			pop (numpy.ndarray): Current population.
			xb (Individual): Current best individual.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray: New evolved populations.
		"""
		return objects2array([self.itype(x=self.CrossMutt(pop, i, xb, self.F, self.CR, self.Rand), task=task, rnd=self.Rand, e=True) for i in range(len(pop))])

	def selection(self, pop, npop, xb, fxb, task, **kwargs):
		r"""Operator for selection.

		Args:
			pop (numpy.ndarray): Current population.
			npop (numpy.ndarray): New Population.
			xb (numpy.ndarray): Current global best solution.
			fxb (float): Current global best solutions fitness/objective value.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, float]:
				1. New selected individuals.
				2. New global best solution.
				3. New global best solutions fitness/objective value.
		"""
		arr = objects2array([e if e.f < pop[i].f else pop[i] for i, e in enumerate(npop)])
		xb, fxb = self.getBest(arr, asarray([e.f for e in arr]), xb, fxb)
		return arr, xb, fxb

	def postSelection(self, pop, task, xb, fxb, **kwargs):
		r"""Apply additional operation after selection.

		Args:
			pop (numpy.ndarray): Current population.
			task (Task): Optimization task.
			xb (numpy.ndarray): Global best solution.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, float]:
				1. New population.
				2. New global best solution.
				3. New global best solutions fitness/objective value.
		"""
		return pop, xb, fxb

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core function of Differential Evolution algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current populations fitness/function values.
			xb (numpy.ndarray): Current best individual.
			fxb (float): Current best individual function/fitness value.
			**dparams (dict): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. New global best solution.
				4. New global best solutions fitness/objective value.
				5. Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.evolve`
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.selection`
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.postSelection`
		"""
		npop = self.evolve(pop, xb, task)
		pop, xb, fxb = self.selection(pop, npop, xb, fxb, task=task)
		pop, xb, fxb = self.postSelection(pop, task, xb, fxb)
		fpop = asarray([x.f for x in pop])
		xb, fxb = self.getBest(pop, fpop, xb, fxb)
		return pop, fpop, xb, fxb, {}

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

	See Also:
		* :class:`NiaPy.algorithms.basic.DifferentialEvolution`
	"""
	Name = ['CrowdingDifferentialEvolution', 'CDE']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""No New"""

	def setParameters(self, CrowPop=0.1, **ukwargs):
		r"""Set core parameters of algorithm.

		Args:
			CrowPop (Optional[float]): Crowding distance.
			**ukwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.setParameters`
		"""
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.CrowPop = CrowPop

	def selection(self, pop, npop, xb, fxb, task, **kwargs):
		r"""Operator for selection of individuals.

		Args:
			pop (numpy.ndarray): Current population.
			npop (numpy.ndarray): New population.
			xb (numpy.ndarray): Current global best solution.
			fxb (float): Current global best solutions fitness/objective value.
			task (Task): Optimization task.
			kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, float]:
				1. New population.
				2. New global best solution.
				3. New global best solutions fitness/objective value.
		"""
		P = []
		for e in npop:
			i = argmin([euclidean(e, f) for f in pop])
			P.append(pop[i] if pop[i].f < e.f else e)
		return asarray(P), xb, fxb

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
		pmax (int): Number of population reductions.
		rp (int): Small non-negative number which is added to value of generations.

	See Also:
		* :class:`NiaPy.algorithms.basic.DifferentialEvolution`
	"""
	Name = ['DynNpDifferentialEvolution', 'dynNpDE']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""No info"""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* rp (Callable[[Union[float, int]], bool])
				* pmax (Callable[[int], bool])

		See Also:
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.typeParameters`
		"""
		r = DifferentialEvolution.typeParameters()
		r['rp'] = lambda x: isinstance(x, (float, int)) and x > 0
		r['pmax'] = lambda x: isinstance(x, int) and x > 0
		return r

	def setParameters(self, pmax=50, rp=3, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			pmax (Optional[int]): umber of population reductions.
			rp (Optional[int]): Small non-negative number which is added to value of generations.

		See Also:
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.setParameters`
		"""
		DifferentialEvolution.setParameters(self, **ukwargs)
		self.pmax, self.rp = pmax, rp

	def postSelection(self, pop, task, xb, fxb, **kwargs):
		r"""Post selection operator.

		In this algorithm the post selection operator decrements the population at specific iterations/generations.

		Args:
			pop (numpy.ndarray): Current population.
			task (Task): Optimization task.
			kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, float]:
				1. Changed current population.
				2. New global best solution.
				3. New global best solutions fitness/objective value.
		"""
		Gr = task.nFES // (self.pmax * len(pop)) + self.rp
		nNP = len(pop) // 2
		if task.Iters == Gr and len(pop) > 3: pop = objects2array([pop[i] if pop[i].f < pop[i + nNP].f else pop[i + nNP] for i in range(nNP)])
		return pop, xb, fxb

def proportional(Lt_min, Lt_max, mu, x_f, avg, **args):
	r"""Proportional calculation of age of individual.

	Args:
		Lt_min (int): Minimal life time.
		Lt_max (int): Maximal life time.
		mu (float): Median of life time.
		x_f (float): Individuals function/fitness value.
		avg (float): Average fitness/function value of current population.
		*args (list): Additional arguments.

	Returns:
		int: Age of individual.
	"""
	return min(Lt_min + mu * avg / x_f, Lt_max)

def linear(Lt_min, mu, x_f, x_gw, x_gb, **args):
	r"""Linear calculation of age of individual.

	Args:
		Lt_min (int): Minimal life time.
		Lt_max (int): Maximal life time.
		mu (float): Median of life time.
		x_f (float): Individual function/fitness value.
		avg (float): Average fitness/function value.
		x_gw (float): Global worst fitness/function value.
		x_gb (float): Global best fitness/function value.
		*args (list): Additional arguments.

	Returns:
		int: Age of individual.
	"""
	return Lt_min + 2 * mu * (x_f - x_gw) / (x_gb - x_gw)

def bilinear(Lt_min, Lt_max, mu, x_f, avg, x_gw, x_gb, **args):
	r"""Bilinear calculation of age of individual.

	Args:
		Lt_min (int): Minimal life time.
		Lt_max (int): Maximal life time.
		mu (float): Median of life time.
		x_f (float): Individual function/fitness value.
		avg (float): Average fitness/function value.
		x_gw (float): Global worst fitness/function value.
		x_gb (float): Global best fitness/function value.
		*args (list): Additional arguments.

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
		* :class:`NiaPy.algorithms.Individual`
	"""
	age = 0

	def __init__(self, **kwargs):
		r"""Init Aging Individual.

		Args:
			**kwargs (Dict[str, Any]): Additional arguments sent to parent.

		See Also:
			* :func:`NiaPy.algorithms.Individual.__init__`
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
		Name (List[str]): list of strings representing algorithm names.
		Lt_min (int): Minimal age of individual.
		Lt_max (int): Maximal age of individual.
		delta_np (float): Proportion of how many individuals shall die.
		omega (float): Acceptance rate for individuals to die.
		mu (int): Mean of individual max and min age.
		age (Callable[[int, int, float, float, float, float, float], int]): Function for calculation of age for individual.

	See Also:
		* :class:`NiaPy.algorithms.basic.DifferentialEvolution`
	"""
	Name = ['AgingNpDifferentialEvolution', 'ANpDE']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""No info"""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* Lt_min (Callable[[int], bool])
				* Lt_max (Callable[[int], bool])
				* delta_np (Callable[[float], bool])
				* omega (Callable[[float], bool])

		See Also:
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.typeParameters`
		"""
		r = DifferentialEvolution.typeParameters()
		r.update({
			'Lt_min': lambda x: isinstance(x, int) and x >= 0,
			'Lt_max': lambda x: isinstance(x, int) and x >= 0,
			'delta_np': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'omega': lambda x: isinstance(x, float) and 1 >= x >= 0
		})
		return r

	def setParameters(self, Lt_min=0, Lt_max=12, delta_np=0.3, omega=0.3, age=proportional, CrossMutt=CrossBest1, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			Lt_min (Optional[int]): Minimum life time.
			Lt_max (Optional[int]): Maximum life time.
			age (Optional[Callable[[int, int, float, float, float, float, float], int]]): Function for calculation of age for individual.

		See Also:
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.setParameters`
		"""
		DifferentialEvolution.setParameters(self, itype=AgingIndividual, **ukwargs)
		self.Lt_min, self.Lt_max, self.age, self.delta_np, self.omega = Lt_min, Lt_max, age, delta_np, omega
		self.mu = abs(self.Lt_max - self.Lt_min) / 2

	def deltaPopE(self, t):
		r"""Calculate how many individuals are going to dye.

		Args:
			t (int): Number of generations made by the algorithm.

		Returns:
			int: Number of individuals to dye.
		"""
		return int(self.delta_np * abs(sin(t)))

	def deltaPopC(self, t):
		r"""Calculate how many individuals are going to be created.

		Args:
			t (int): Number of generations made by the algorithm.

		Returns:
			int: Number of individuals to be born.
		"""
		return int(self.delta_np * abs(cos(t)))

	def aging(self, task, pop):
		r"""Apply aging to individuals.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray[Individual]): Current population.

		Returns:
			numpy.ndarray[Individual]: New population.
		"""
		fpop = asarray([x.f for x in pop])
		x_b, x_w = pop[argmin(fpop)], pop[argmax(fpop)]
		avg, npop = mean(fpop), []
		for x in pop:
			x.age += 1
			Lt = round(self.age(Lt_min=self.Lt_min, Lt_max=self.Lt_max, mu=self.mu, x_f=x.f, avg=avg, x_gw=x_w.f, x_gb=x_b.f))
			if x.age <= Lt: npop.append(x)
		if len(npop) == 0: npop = objects2array([self.itype(task=task, rnd=self.Rand, e=True) for _ in range(self.NP)])
		return npop

	def popIncrement(self, pop, task):
		r"""Increment population.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray[Individual]: Increased population.
		"""
		deltapop = int(round(max(1, self.NP * self.deltaPopE(task.Iters))))
		return objects2array([self.itype(task=task, rnd=self.Rand, e=True) for _ in range(deltapop)])

	def popDecrement(self, pop, task):
		r"""Decrement population.

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
		return objects2array(npop)

	def selection(self, pop, npop, xb, fxb, task, **kwargs):
		r"""Select operator for individuals with aging.

		Args:
			pop (numpy.ndarray): Current population.
			npop (numpy.ndarray): New population.
			xb (numpy.ndarray): Current global best solution.
			fxb (float): Current global best solutions fitness/objective value.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, float]:
				1. New population of individuals.
				2. New global best solution.
				3. New global best solutions fitness/objective value.
		"""
		npop, xb, fxb = DifferentialEvolution.selection(self, pop, npop, xb, fxb, task)
		npop = append(npop, self.popIncrement(pop, task))
		xb, fxb = self.getBest(npop, asarray([e.f for e in npop]), xb, fxb)
		pop = self.aging(task, npop)
		return pop, xb, fxb

	def postSelection(self, pop, task, xb, fxb, **kwargs):
		r"""Post selection operator.

		Args:
			pop (numpy.ndarray): Current population.
			task (Task): Optimization task.
			xb (Individual): Global best individual.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, float]:
			1. New population.
			2. New global best solution
			3. New global best solutions fitness/objective value
		"""
		return self.popDecrement(pop, task) if len(pop) > self.NP else pop, xb, fxb

def multiMutations(pop, i, xb, F, CR, rnd, task, itype, strategies, **kwargs):
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
	L = [itype(x=strategy(pop, i, xb, F, CR, rnd=rnd), task=task, e=True, rnd=rnd) for strategy in strategies]
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
		CrossMutt (Callable[[numpy.ndarray[Individual], int, Individual, float, float, Task, Individual, Iterable[Callable[[numpy.ndarray, int, numpy.ndarray, float, float, mtrand.RandomState, Dict[str, Any]], Individual]]], Individual]): Multi crossover and mutation combiner function.

	See Also:
		* :class:`NiaPy.algorithms.basic.DifferentialEvolution`
	"""
	Name = ['MultiStrategyDifferentialEvolution', 'MsDE']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""No info"""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]: Testing functions for parameters.

		See Also:
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.typeParameters`
		"""
		r = DifferentialEvolution.typeParameters()
		r.pop('CrossMutt', None)
		r.update({'strategies': lambda x: callable(x)})
		return r

	def setParameters(self, strategies=(CrossRand1, CrossBest1, CrossCurr2Best1, CrossRand2), **ukwargs):
		r"""Set the arguments of the algorithm.

		Arguments:
			strategies (Optional[Iterable[Callable[[numpy.ndarray[Individual], int, Individual, float, float, mtrand.RandomState], numpy.ndarray[Individual]]]]): List of mutation strategyis.
			CrossMutt (Optional[Callable[[numpy.ndarray[Individual], int, Individual, float, float, Task, Individual, Iterable[Callable[[numpy.ndarray, int, numpy.ndarray, float, float, mtrand.RandomState, Dict[str, Any]], Individual]]], Individual]]): Multi crossover and mutation combiner function.

		See Also:
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.setParameters`
		"""
		DifferentialEvolution.setParameters(self, CrossMutt=multiMutations, **ukwargs)
		self.strategies = strategies

	def getParameters(self):
		r"""Get parameters values of the algorithm.

		Returns:
			Dict[str, Any]: TODO.

		See Also:
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.getParameters`
		"""
		d = DifferentialEvolution.getParameters(self)
		d.update({'strategies': self.strategies})
		return d

	def evolve(self, pop, xb, task, **kwargs):
		r"""Evolve population with the help multiple mutation strategies.

		Args:
			pop (numpy.ndarray): Current population.
			xb (numpy.ndarray): Current best individual.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray: New population of individuals.
		"""
		return objects2array([self.CrossMutt(pop, i, xb, self.F, self.CR, self.Rand, task, self.itype, self.strategies) for i in range(len(pop))])

class DynNpMultiStrategyDifferentialEvolution(MultiStrategyDifferentialEvolution, DynNpDifferentialEvolution):
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
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.basic.MultiStrategyDifferentialEvolution`
		* :class:`NiaPy.algorithms.basic.DynNpDifferentialEvolution`
	"""
	Name = ['DynNpMultiStrategyDifferentialEvolution', 'dynNpMsDE']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""No info"""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* rp (Callable[[Union[float, int]], bool]): TODO
				* pmax (Callable[[int], bool]): TODO

		See Also:
			* :func:`NiaPy.algorithms.basic.MultiStrategyDifferentialEvolution.typeParameters`
		"""
		r = MultiStrategyDifferentialEvolution.typeParameters()
		r['rp'] = lambda x: isinstance(x, (float, int)) and x > 0
		r['pmax'] = lambda x: isinstance(x, int) and x > 0
		return r

	def setParameters(self, **ukwargs):
		r"""Set the arguments of the algorithm.

		Args:
			ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.MultiStrategyDifferentialEvolution.setParameters`
			* :func:`NiaPy.algorithms.basic.DynNpDifferentialEvolution.setParameters`
		"""
		DynNpDifferentialEvolution.setParameters(self, **ukwargs)
		MultiStrategyDifferentialEvolution.setParameters(self, **ukwargs)

	def evolve(self, pop, xb, task, **kwargs):
		r"""Evolve the current population.

		Args:
			pop (numpy.ndarray): Current population.
			xb (numpy.ndarray): Global best solution.
			task (Task): Optimization task.
			**kwargs (dict): Additional arguments.

		Returns:
			numpy.ndarray: Evolved new population.
		"""
		return MultiStrategyDifferentialEvolution.evolve(self, pop, xb, task, **kwargs)

	def postSelection(self, pop, task, xb, fxb, **kwargs):
		r"""Post selection operator.

		Args:
			pop (numpy.ndarray): Current population.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, float]:
				1. New population.
				2. New global best solution.
				3. New global best solutions fitness/objective value.

		See Also:
			* :func:`NiaPy.algorithms.basic.DynNpDifferentialEvolution.postSelection`
		"""
		return DynNpDifferentialEvolution.postSelection(self, pop, task, xb, fxb)

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

	See Also:
		* :class:`NiaPy.algorithms.basic.AgingNpDifferentialEvolution`
		* :class:`NiaPy.algorithms.basic.MultiStrategyDifferentialEvolution`
	"""
	Name = ['AgingNpMultiMutationDifferentialEvolution', 'ANpMSDE']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""No info"""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]: Mappings form parameter names to test functions.

		See Also:
			* :func:`NiaPy.algorithms.basic.MultiStrategyDifferentialEvolution.typeParameters`
			* :func:`NiaPy.algorithms.basic.AgingNpDifferentialEvolution.typeParameters`
		"""
		d = AgingNpDifferentialEvolution.typeParameters()
		d.update(MultiStrategyDifferentialEvolution.typeParameters())
		return d

	def setParameters(self, **ukwargs):
		r"""Set core parameter arguments.

		Args:
			**ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.AgingNpDifferentialEvolution.setParameters`
			* :func:`NiaPy.algorithms.basic.MultiStrategyDifferentialEvolution.setParameters`
		"""
		AgingNpDifferentialEvolution.setParameters(self, **ukwargs)
		MultiStrategyDifferentialEvolution.setParameters(self, stratgeys=(CrossRand1, CrossBest1, CrossCurr2Rand1, CrossRand2), itype=AgingIndividual, **ukwargs)

	def evolve(self, pop, xb, task, **kwargs):
		r"""Evolve current population.

		Args:
			pop (numpy.ndarray): Current population.
			xb (numpy.ndarray): Global best individual.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray: New population of individuals.
		"""
		return MultiStrategyDifferentialEvolution.evolve(self, pop, xb, task, **kwargs)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
