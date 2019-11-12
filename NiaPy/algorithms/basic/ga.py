# encoding=utf8
import logging

from numpy import argmin, sort, random as rand, asarray, fmin, fmax, sum, empty

from NiaPy.algorithms.algorithm import Algorithm, Individual, defaultIndividualInit

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['GeneticAlgorithm', 'TournamentSelection', 'RouletteSelection', 'TwoPointCrossover', 'MultiPointCrossover', 'UniformCrossover', 'UniformMutation', 'CreepMutation', 'CrossoverUros', 'MutationUros']

def TournamentSelection(pop, ic, ts, x_b, rnd=rand):
	r"""Tournament selection method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual in population.
		ts (int): Tournament size.
		x_b (Individual): Global best individual.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		Individual: Winner of the tournament.
	"""
	comps = [pop[i] for i in rand.choice(len(pop), ts, replace=False)]
	return comps[argmin([c.f for c in comps])]

def RouletteSelection(pop, ic, ts, x_b, rnd=rand):
	r"""Roulette selection method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual in population.
		ts (int): Unused argument.
		x_b (Individual): Global best individual.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		Individual: selected individual.
	"""
	f = sum([x.f for x in pop])
	qi = sum([pop[i].f / f for i in range(ic + 1)])
	return pop[ic].x if rnd.rand() < qi else x_b

def TwoPointCrossover(pop, ic, cr, rnd=rand):
	r"""Two point crossover method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	r = sort(rnd.choice(len(pop[ic]), 2))
	x = pop[ic].x
	x[r[0]:r[1]] = pop[io].x[r[0]:r[1]]
	return asarray(x)

def MultiPointCrossover(pop, ic, n, rnd=rand):
	r"""Multi point crossover method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		n (flat): TODO.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	r, x = sort(rnd.choice(len(pop[ic]), 2 * n)), pop[ic].x
	for i in range(n): x[r[2 * i]:r[2 * i + 1]] = pop[io].x[r[2 * i]:r[2 * i + 1]]
	return asarray(x)

def UniformCrossover(pop, ic, cr, rnd=rand):
	r"""Uniform crossover method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	j = rnd.randint(len(pop[ic]))
	x = [pop[io][i] if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossoverUros(pop, ic, cr, rnd=rand):
	r"""Crossover made by Uros Mlakar.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		cr (float): Crossover probability.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	alpha = cr + (1 + 2 * cr) * rnd.rand(len(pop[ic]))
	x = alpha * pop[ic] + (1 - alpha) * pop[io]
	return x

def UniformMutation(pop, ic, mr, task, rnd=rand):
	r"""Uniform mutation method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		mr (float): Mutation probability.
		task (Task): Optimization task.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	j = rnd.randint(task.D)
	nx = [rnd.uniform(task.Lower[i], task.Upper[i]) if rnd.rand() < mr or i == j else pop[ic][i] for i in range(task.D)]
	return asarray(nx)

def MutationUros(pop, ic, mr, task, rnd=rand):
	r"""Mutation method made by Uros Mlakar.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of individual.
		mr (float): Mutation rate.
		task (Task): Optimization task.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	return fmin(fmax(rnd.normal(pop[ic], mr * task.bRange), task.Lower), task.Upper)

def CreepMutation(pop, ic, mr, task, rnd=rand):
	r"""Creep mutation method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		mr (float): Mutation probability.
		task (Task): Optimization task.
		rnd (mtrand.RandomState): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	ic, j = rnd.randint(len(pop)), rnd.randint(task.D)
	nx = [rnd.uniform(task.Lower[i], task.Upper[i]) if rnd.rand() < mr or i == j else pop[ic][i] for i in range(task.D)]
	return asarray(nx)

class GeneticAlgorithm(Algorithm):
	r"""Implementation of Genetic algorithm.

	Algorithm:
		Genetic algorithm

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		Ts (int): Tournament size.
		Mr (float): Mutation rate.
		Cr (float): Crossover rate.
		Selection (Callable[[numpy.ndarray[Individual], int, int, Individual, mtrand.RandomState], Individual]): Selection operator.
		Crossover (Callable[[numpy.ndarray[Individual], int, float, mtrand.RandomState], Individual]): Crossover operator.
		Mutation (Callable[[numpy.ndarray[Individual], int, float, Task, mtrand.RandomState], Individual]): Mutation operator.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['GeneticAlgorithm', 'GA']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.
		"""
		return r"""On info"""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* Ts (Callable[[int], bool]): Tournament size.
				* Mr (Callable[[float], bool]): Probability of mutation.
				* Cr (Callable[[float], bool]): Probability of crossover.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'Ts': lambda x: isinstance(x, int) and x > 1,
			'Mr': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'Cr': lambda x: isinstance(x, float) and 0 <= x <= 1
		})
		return d

	def setParameters(self, NP=25, Ts=5, Mr=0.25, Cr=0.25, Selection=TournamentSelection, Crossover=UniformCrossover, Mutation=UniformMutation, **ukwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
			NP (Optional[int]): Population size.
			Ts (Optional[int]): Tournament selection.
			Mr (Optional[int]): Mutation rate.
			Cr (Optional[float]): Crossover rate.
			Selection (Optional[Callable[[numpy.ndarray[Individual], int, int, Individual, mtrand.RandomState], Individual]]): Selection operator.
			Crossover (Optional[Callable[[numpy.ndarray[Individual], int, float, mtrand.RandomState], Individual]]): Crossover operator.
			Mutation (Optional[Callable[[numpy.ndarray[Individual], int, float, Task, mtrand.RandomState], Individual]]): Mutation operator.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
			* Selection:
				* :func:`NiaPy.algorithms.basic.TournamentSelection`
				* :func:`NiaPy.algorithms.basic.RouletteSelection`
			* Crossover:
				* :func:`NiaPy.algorithms.basic.UniformCrossover`
				* :func:`NiaPy.algorithms.basic.TwoPointCrossover`
				* :func:`NiaPy.algorithms.basic.MultiPointCrossover`
				* :func:`NiaPy.algorithms.basic.CrossoverUros`
			* Mutations:
				* :func:`NiaPy.algorithms.basic.UniformMutation`
				* :func:`NiaPy.algorithms.basic.CreepMutation`
				* :func:`NiaPy.algorithms.basic.MutationUros`
		"""
		Algorithm.setParameters(self, NP=NP, itype=ukwargs.pop('itype', Individual), InitPopFunc=ukwargs.pop('InitPopFunc', defaultIndividualInit), **ukwargs)
		self.Ts, self.Mr, self.Cr = Ts, Mr, Cr
		self.Selection, self.Crossover, self.Mutation = Selection, Crossover, Mutation

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core function of GeneticAlgorithm algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current populations fitness/function values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individuals function/fitness value.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New populations function/fitness values.
				3. New global best solution
				4. New global best solutions fitness/objective value
				5. Additional arguments.
		"""
		npop = empty(self.NP, dtype=object)
		for i in range(self.NP):
			ind = self.itype(x=self.Selection(pop, i, self.Ts, xb, self.Rand), e=False)
			ind.x = self.Crossover(pop, i, self.Cr, self.Rand)
			ind.x = self.Mutation(pop, i, self.Mr, task, self.Rand)
			ind.evaluate(task, rnd=self.Rand)
			npop[i] = ind
			if npop[i].f < fxb: xb, fxb = self.getBest(npop[i], npop[i].f, xb, fxb)
		return npop, asarray([i.f for i in npop]), xb, fxb, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
