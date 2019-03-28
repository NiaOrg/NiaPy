# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, expression-not-assigned, len-as-condition, no-self-use, unused-argument, no-else-return, dangerous-default-value, unnecessary-pass
import logging
from numpy import random as rand, inf, ndarray, array, asarray, array_equal, argmin, apply_along_axis
from NiaPy.util import FesException, GenException, TimeException, RefException

logging.basicConfig()
logger = logging.getLogger('NiaPy.util.utility')
logger.setLevel('INFO')

__all__ = ['Algorithm', 'Individual']

class Algorithm:
	r"""Class for implementing algorithms.

	Date:
		2018

	Author
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (array or list): List of names for algorithm
		Rand (RandomState): 	Random generator
		NP (int): Number of inidividuals in populatin
		individualType (class or None): Type of individual used in algorithm
	"""
	Name = ['Algorithm', 'AAA']
	Rand = rand.RandomState(None)
	NP = 50
	individualType = None

	@staticmethod
	def typeParameters():
		r"""TODO documentation.

		Returns:
			dict: Dictionary where key represents the argument name and values represents a function for testing the correctnes of paramether with given key
		"""
		return {}

	def __init__(self, **kwargs):
		r"""Initialize algorithm and create name for an algorithm.

		Args:
			seed (int): Starting seed for random generator

		See Also:
			:func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		self.Rand = rand.RandomState(kwargs.pop('seed', None))
		self.setParameters(**kwargs)

	def setParameters(self, NP=50, individualType=None, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		Args:
			NP (Optional[int]): Number of individuals in population
			individualType (Optional[class]): Type of individuals used by algorithm
			**kwargs: Parameter values dictionary
		"""
		self.NP = NP

	def rand(self, D=1):
		r"""Get random distribution of shape D in range from 0 to 1.

		Args:
			D (array or int): Shape of returned random distribution

		Returns:
			array or float: TODO
		"""
		if isinstance(D, (ndarray, list)): return self.Rand.rand(*D)
		elif D > 1: return self.Rand.rand(D)
		else: return self.Rand.rand()

	def uniform(self, Lower, Upper, D=None):
		r"""Get uniform random distribution of shape D in range from "Lower" to "Upper".

		Args:
			Lower (array or float or int): Lower bound
			Upper (array or float or int): Upper bound
			D (array or int): Shape of returned uniform random distribution

		Returns:
			array: TODO
		"""
		return self.Rand.uniform(Lower, Upper, D) if D is not None else self.Rand.uniform(Lower, Upper)

	def normal(self, loc, scale, D=None):
		r"""Get normal random distribution of shape D with mean "loc" and standard deviation "scale".

		Args:
			loc (float): Mean of the normal random distribution
			scale (float): Standard deviation of the normal random distribution
			D (array or float): Shape of returned normal random distribution

		Returns:
			array or float: TODO
		"""
		return self.Rand.normal(loc, scale, D) if D is not None else self.Rand.normal(loc, scale)

	def randn(self, D=None):
		r"""Get standard normal distribution of shape D.

		Args:
			D (array or int): Shape of returned standard normal distribution

		Returns:
			array or float: Random generated numbers or one random generated number in rage [0, 1]
		"""
		if D is None: return self.Rand.randn()
		elif isinstance(D, int): return self.Rand.randn(D)
		return self.Rand.randn(*D)

	def randint(self, Nmax, D=1, Nmin=0, skip=None):
		r"""Get discrete uniform (integer) random distribution of D shape in range from "Nmin" to "Nmax".

		Args:
			Nmin (int): Lower integer bound
			Nmax (int): One above upper integer bound
			D (array of int or int): shape of returned discrete uniform random distribution
			skip (array): numbers to skip

		Returns:
			int: Random generated integer number
		"""
		r = None
		if isinstance(D, (list, tuple, ndarray, array)): r = self.Rand.randint(Nmin, Nmax, D)
		elif D > 1: r = self.Rand.randint(Nmin, Nmax, D)
		else: r = self.Rand.randint(Nmin, Nmax)
		return r if skip is None and r not in skip else self.randint(Nmax, D, Nmin, skip)

	def getBest(self, X, X_f, xb=None, xb_f=inf):
		r"""Get the best individual for population.

		Args:
			X (array of array of (float or int)): Population
			X_f (array of float): Fitness values of aligned individuals
			xb (array of (float or int)): Best individual
			xb_f (real): Fitness value of best individal

		Returns:
			Tuple[array of (float or int), float]:
				1. Coordinates of best solution
				2. beset fitnes value
		"""
		ib = argmin(X_f)
		if isinstance(X_f, (float, int)) and xb_f >= X_f: return X, X_f
		elif isinstance(X_f, (ndarray, list)) and xb_f >= X_f[ib]: return X[ib], X_f[ib]
		else: return xb, xb_f

	def initPopulation(self, task):
		r"""Initialization for starting population of optimization algorithm.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[array of (float or int), array of float, dict]:
				1. New population
				2. New population fitness values.
				3. dict:
					* Additional arguments.

		See Also:
			:class:`NiaPy.algorithms.algorithm.Individual`
		"""
		pop, fpop = None, None
		if issubclass(self.individualType, Individual):
			pop = ndarray([self.individualType(task=task, rnd=self.Rand) for _ in range(self.NP)])
			fpop = ndarray([x.f for x in pop])
		else:
			pop = task.Lower + self.rand([self.NP, task.D]) * task.bRange
			fpop = apply_along_axis(task.eval, 1, pop)
		return pop, fpop, {}

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core functionality of algorithm.

		This function is called on every algorithm iteration.

		Args:
			task (Task): Optimization task
			pop (array of array of (float or int)): Current population coordinates
			fpop (array of float): Current population fitness value
			xb (array of (float or int)): Current generation best individuals coordinates
			xb_f (float): current generation best individuals fitness value
			**dparams: Additional arguments for algorithms

		Returns:
			Tuple[array of (float or int), array of float, dict]:
				1. New populations coordinates
				2. New populations fitness values
				3. Additional arguments of the algorithm

		See Also:
			:func:`NiaPy.algorithms.algorithm.Algorithm.runYield`
		"""
		return pop, fpop, {}

	def runYield(self, task):
		r"""Run the algorithm for a single iteration and return the best solution.

		Args:
			task (Task): Task with bounds and objective function for optimization

		Yield:
			Tuple[array of array of (float or int), float]:
				1. New population best individuals coordinates
				2. Fitness value of the best solution

		See Also:
			:func:`NiaPy.algorithms.algorithm.Algorithm.runIteration`
			:func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		pop, fpop, dparams = self.initPopulation(task)
		xb, fxb = self.getBest(pop, fpop)
		yield xb, fxb
		while True:
			pop, fpop, dparams = self.runIteration(task, pop, fpop, xb, fxb, **dparams)
			xb, fxb = self.getBest(pop, fpop, xb, fxb)
			yield xb, fxb

	def runTask(self, task):
		r"""Start the optimization.

		Args:
			task (Task): Task with bounds and objective function for optimization

		Returns:
			Tuple[array of array of (float or int), float]:
				1. Best individuals components found in optimization process
				2. Best fitness value found in optimization process

		See Also:
			:func:`NiaPy.algorithms.algorithm.Algorithm.runYield`
		"""
		algo, xb, fxb = self.runYield(task), None, inf
		while not task.stopCond():
			xb, fxb = next(algo)
			task.nextIter()
		return xb, fxb

	def run(self, task):
		r"""Start the optimization.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple[array of array of (float or int), float]:
				1. Best individuals components found in optimization process
				2. Best fitness value found in optimization process

		See Also:
			:func:`NiaPy.algorithms.algorithm.Algorithm.runTask`
		"""
		try:
			task.start()
			r = self.runTask(task)
			return r[0], r[1] * task.optType.value
		except (FesException, GenException, TimeException, RefException): return task.x, task.x_f * task.optType.value

class Individual:
	r"""Class that represents one solution in population of solutions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		x (array of (float or int)): Coordinates of inidividual
		f (float): Function/fitness value of individual
	"""
	x = None
	f = inf

	def __init__(self, x=None, task=None, e=True, rnd=rand, **kwargs):
		r"""Initialize new individual.

		Parameters:
			task (Optional[Task]): Optimization task
			rand (Optional[RandomState]): Random generator
			x (Optional[array of (float or int)]): Individuals components
			e (bool): True to evaluate the individual on initialization. Default value is True.
			**kwargs: Additional arguments
		"""
		self.f = task.optType.value * inf if task is not None else inf
		if x is not None: self.x = x if isinstance(x, ndarray) else asarray(x)
		else: self.generateSolution(task, rnd)
		if e and task is not None: self.evaluate(task, rnd)

	def generateSolution(self, task, rnd=rand):
		r"""Generate new solution.

		Generate new solution for this individual and set it to ``self.x``.
		This method uses ``rnd`` for getting random numbers.
		For generating random components ``rnd`` and ``task`` is used.

		Args:
			task (Task): Optimization task
			rnd (Optional[RandomState]: Random numbers generator object
		"""
		if task is not None: self.x = task.Lower + task.bRange * rnd.rand(task.D)

	def evaluate(self, task, rnd=rand):
		r"""Evaluate the solution.

		Evaluate solution ``this.x`` with the help of task.
		Task is used for reparing the solution and then evaluating it.

		Args:
			task (Task): Objective function object
			rnd (Optional[RandomState]: Random generator

		See Also:
			:func:`NiaPy.util.utillity.Task.repair`
		"""
		task.repair(task, rnd=rnd)
		self.f = task.eval(self.x)

	def copy(self):
		r"""Return a copy of self.

		Method returns copy of ``this`` object so it is safe for editing.

		Returns:
			:class:`NiaPy.algorithms.algorithm.Individual`: Copy of self
		"""
		return Individual(x=self.x, f=self.f, e=False)

	def __eq__(self, other):
		r"""Compare the individuals for equalities.

		Args:
			other (object or :class:`NiaPy.algorithms.algorithm.Individual`): Object that we want to compare this object to

		Returns:
			bool: ``True`` if equal or ``False`` if no equal
		"""
		return array_equal(self.x, other.x) and self.f == other.f

	def __str__(self):
		r"""Print the individual with the solution and objective value.

		Returns:
			str: String representation of self
		"""
		return '%s -> %s' % (self.x, self.f)

	def __getitem__(self, i):
		r"""Get the value of i-th component of the solution.

		Args:
			i (int): Position of the solution component

		Returns:
			float or int: Value of ith component
		"""
		return self.x[i]

	def __setitem__(self, i, v):
		r"""Set the value of i-th component of the solution to v value.

		Args:
			i (int): Position of the solution component
			v (float, int): Value to set to i-th component
		"""
		self.x[i] = v

	def __len__(self):
		r"""Get the length of the solution or the number of components.

		Returns:
			int: Number of components
		"""
		return len(self.x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
