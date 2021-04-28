# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm, Individual, defaultIndividualInit

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['GeneticAlgorithm', 'TournamentSelection', 'RouletteSelection', 'TwoPointCrossover', 'MultiPointCrossover', 'UniformCrossover', 'UniformMutation', 'CreepMutation', 'CrossoverUros', 'MutationUros']

def TournamentSelection(pop, ic, ts, x_b, rng):
	r"""Tournament selection method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual in population.
		ts (int): Tournament size.
		x_b (Individual): Global best individual.
		rng (numpy.random.Generator): Random generator.

	Returns:
		Individual: Winner of the tournament.
	"""
	comps = [pop[i] for i in rng.choice(len(pop), ts, replace=False)]
	return comps[np.argmin([c.f for c in comps])]

def RouletteSelection(pop, ic, ts, x_b, rng):
	r"""Roulette selection method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual in population.
		ts (int): Unused argument.
		x_b (Individual): Global best individual.
		rng (numpy.random.Generator): Random generator.

	Returns:
		Individual: selected individual.
	"""
	f = np.sum([x.f for x in pop])
	qi = np.sum([pop[i].f / f for i in range(ic + 1)])
	return pop[ic].x if rng.random() < qi else x_b

def TwoPointCrossover(pop, ic, cr, rng):
	r"""Two point crossover method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		cr (float): Crossover probability.
		rng (numpy.random.Generator): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	io = ic
	while io != ic: io = rng.integers(len(pop))
	r = np.sort(rng.choice(len(pop[ic]), 2))
	x = pop[ic].x
	x[r[0]:r[1]] = pop[io].x[r[0]:r[1]]
	return np.asarray(x)

def MultiPointCrossover(pop, ic, n, rng):
	r"""Multi point crossover method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		n (flat): TODO.
		rng (numpy.random.Generator): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	io = ic
	while io != ic: io = rng.integers(len(pop))
	r, x = np.sort(rng.choice(len(pop[ic]), 2 * n)), pop[ic].x
	for i in range(n): x[r[2 * i]:r[2 * i + 1]] = pop[io].x[r[2 * i]:r[2 * i + 1]]
	return np.asarray(x)

def UniformCrossover(pop, ic, cr, rng):
	r"""Uniform crossover method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		cr (float): Crossover probability.
		rng (numpy.random.Generator): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	io = ic
	while io != ic: io = rng.integers(len(pop))
	j = rng.integers(len(pop[ic]))
	x = [pop[io][i] if rng.random() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return np.asarray(x)

def CrossoverUros(pop, ic, cr, rng):
	r"""Crossover made by Uros Mlakar.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		cr (float): Crossover probability.
		rng (numpy.random.Generator): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	io = ic
	while io != ic: io = rng.integers(len(pop))
	alpha = cr + (1 + 2 * cr) * rng.random(len(pop[ic]))
	x = alpha * pop[ic] + (1 - alpha) * pop[io]
	return x

def UniformMutation(pop, ic, mr, task, rng):
	r"""Uniform mutation method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		mr (float): Mutation probability.
		task (Task): Optimization task.
		rng (numpy.random.Generator): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	j = rng.integers(task.D)
	nx = [rng.uniform(task.Lower[i], task.Upper[i]) if rng.random() < mr or i == j else pop[ic][i] for i in range(task.D)]
	return np.asarray(nx)

def MutationUros(pop, ic, mr, task, rng):
	r"""Mutation method made by Uros Mlakar.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of individual.
		mr (float): Mutation rate.
		task (Task): Optimization task.
		rng (numpy.random.Generator): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	return np.fmin(np.fmax(rng.normal(pop[ic], mr * task.bRange), task.Lower), task.Upper)

def CreepMutation(pop, ic, mr, task, rng):
	r"""Creep mutation method.

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		mr (float): Mutation probability.
		task (Task): Optimization task.
		rng (numpy.random.Generator): Random generator.

	Returns:
		numpy.ndarray: New genotype.
	"""
	ic, j = rng.integers(len(pop)), rng.integers(task.D)
	nx = [rng.uniform(task.Lower[i], task.Upper[i]) if rng.random() < mr or i == j else pop[ic][i] for i in range(task.D)]
	return np.asarray(nx)

class GeneticAlgorithm(Algorithm):
	r"""Implementation of Genetic Algorithm.

	Algorithm:
		Genetic algorithm

	Date:
		2018

	Author:
		Klemen Berkovič

	Reference paper:
		Goldberg, David (1989). Genetic Algorithms in Search, Optimization and Machine Learning. Reading, MA: Addison-Wesley Professional.

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		Ts (int): Tournament size.
		Mr (float): Mutation rate.
		Cr (float): Crossover rate.
		Selection (Callable[[numpy.ndarray[Individual], int, int, Individual, numpy.random.Generator], Individual]): Selection operator.
		Crossover (Callable[[numpy.ndarray[Individual], int, float, numpy.random.Generator], Individual]): Crossover operator.
		Mutation (Callable[[numpy.ndarray[Individual], int, float, Task, numpy.random.Generator], Individual]): Mutation operator.

	See Also:
		* :class:`niapy.algorithms.Algorithm`
	"""
	Name = ['GeneticAlgorithm', 'GA']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`niapy.algorithms.Algorithm.algorithmInfo`
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
			* :func:`niapy.algorithms.Algorithm.typeParameters`
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
			Selection (Optional[Callable[[numpy.ndarray[Individual], int, int, Individual, numpy.random.Generator], Individual]]): Selection operator.
			Crossover (Optional[Callable[[numpy.ndarray[Individual], int, float, numpy.random.Generator], Individual]]): Crossover operator.
			Mutation (Optional[Callable[[numpy.ndarray[Individual], int, float, Task, numpy.random.Generator], Individual]]): Mutation operator.

		See Also:
			* :func:`niapy.algorithms.Algorithm.setParameters`
			* Selection:
				* :func:`niapy.algorithms.basic.TournamentSelection`
				* :func:`niapy.algorithms.basic.RouletteSelection`
			* Crossover:
				* :func:`niapy.algorithms.basic.UniformCrossover`
				* :func:`niapy.algorithms.basic.TwoPointCrossover`
				* :func:`niapy.algorithms.basic.MultiPointCrossover`
				* :func:`niapy.algorithms.basic.CrossoverUros`
			* Mutations:
				* :func:`niapy.algorithms.basic.UniformMutation`
				* :func:`niapy.algorithms.basic.CreepMutation`
				* :func:`niapy.algorithms.basic.MutationUros`
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
		npop = np.empty(self.NP, dtype=object)
		for i in range(self.NP):
			ind = self.itype(x=self.Selection(pop, i, self.Ts, xb, self.rng), e=False)
			ind.x = self.Crossover(pop, i, self.Cr, self.rng)
			ind.x = self.Mutation(pop, i, self.Mr, task, self.rng)
			ind.evaluate(task, rng=self.rng)
			npop[i] = ind
			if npop[i].f < fxb: xb, fxb = self.getBest(npop[i], npop[i].f, xb, fxb)
		return npop, np.asarray([i.f for i in npop]), xb, fxb, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
