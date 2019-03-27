# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, expression-not-assigned, len-as-condition, no-self-use, unused-argument, no-else-return, dangerous-default-value, unnecessary-pass
import logging
from numpy import random as rand, inf, ndarray, array, asarray, array_equal, argmin
from NiaPy.util import FesException, GenException, TimeException, RefException

logging.basicConfig()
logger = logging.getLogger('NiaPy.util.utility')
logger.setLevel('INFO')

__all__ = ['Algorithm', 'Individual']

class Algorithm:
	r"""Class for implementing algorithms.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (array, list): List of names for algorithm
		Rand (:py:class:RandomState): Random generator
	"""
	Name = ['Algorithm', 'AAA']
	Rand = rand.RandomState(None)

	@staticmethod
	def typeParameters():
		r"""TODO documentation.

		Returns:
			(dict): Dictionary where key represents the argument name and values represents a function for testing the correctnes of paramether with given key
		"""
		return {}

	def __init__(self, **kwargs):
		r"""Initialize algorithm and create name for an algorithm.

		Args:
			name (str): Full name of algorithm
			shortName (str): Short name of algorithm

		Raises:
			TypeError -- raised when given benchmark function does not exist

		See Also:
			:py:meth:Algorithm.setParameters(self, **kwargs)
		"""
		self.Rand = rand.RandomState(kwargs.pop('seed', None))
		self.setParameters(**kwargs)

	def setParameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		Args:
			kwargs (dict): Parameter values dictionary
		"""
		pass

	def rand(self, D=1):
		r"""Get random distribution of shape D in range from 0 to 1.

		Args:
			D (array, int): Shape of returned random distribution

		Returns:
			(array, float): TODO
		"""
		if isinstance(D, (ndarray, list)): return self.Rand.rand(*D)
		elif D > 1: return self.Rand.rand(D)
		else: return self.Rand.rand()

	def uniform(self, Lower, Upper, D=None):
		r"""Get uniform random distribution of shape D in range from "Lower" to "Upper".

		Args:
			Lower (array, float, int): Lower bound
			Upper (array, float, int): Upper bound
			D (array, int): Shape of returned uniform random distribution

		Returns:
			(array): TODO
		"""
		return self.Rand.uniform(Lower, Upper, D) if D is not None else self.Rand.uniform(Lower, Upper)

	def normal(self, loc, scale, D=None):
		r"""Get normal random distribution of shape D with mean "loc" and standard deviation "scale".

		Args:
			loc (): Mean of the normal random distribution
			scale (): Standard deviation of the normal random distribution
			D (array, int): Shape of returned normal random distribution

		Returns:
			(array, float): TODO
		"""
		return self.Rand.normal(loc, scale, D) if D is not None else self.Rand.normal(loc, scale)

	def randn(self, D=None):
		r"""Get standard normal distribution of shape D.

		Args:
			D (array): Shape of returned standard normal distribution

		Returns:
			(array): Random generated numbers or one random generated number in rage [0, 1]
		"""
		if D is None: return self.Rand.randn()
		elif isinstance(D, int): return self.Rand.randn(D)
		return self.Rand.randn(*D)

	def randint(self, Nmax, D=1, Nmin=0, skip=None):
		r"""Get discrete uniform (integer) random distribution of D shape in range from "Nmin" to "Nmax".

		Args:
			Nmin (int): Lower integer bound
			Nmax (int): One above upper integer bound
			D (array, int) -- shape of returned discrete uniform random distribution
			skip {array} -- numbers to skip

		Returns:
			(int): Random generated integer number
		"""
		r = None
		if isinstance(D, (list, tuple, ndarray, array)): r = self.Rand.randint(Nmin, Nmax, D)
		elif D > 1: r = self.Rand.randint(Nmin, Nmax, D)
		else: r = self.Rand.randint(Nmin, Nmax)
		return r if skip is None and r not in skip else self.randint(Nmax, D, Nmin, skip)

	def getBest(self, X, X_f, xb=None, xb_f=inf):
		r"""Get the best individual for population.

		Args:
			X (array): Population
			X_f (array): Fitness values of aligned individuals
			xb (array): Best individual
			xb_f (real): Fitness value of best individal

		Returns:
			(tuple): tuple containing:
				xb (array): coordinates of best solution
				xb_f (float): beset fitnes value
		"""
		ib = argmin(X_f)
		if isinstance(X_f, (float, int)) and xb_f >= X_f: return X, X_f
		elif isinstance(X_f, (ndarray, list)) and xb_f >= X_f[ib]: return X[ib], X_f[ib]
		else: return xb, xb_f

	def initPopulation(self, task):
		r"""Initialization for starting population of optimization algorithm.

		Args:
			task (:class:Task): Optimization task.

		Returns:
			(tuple): tuple containing
				(array): New population
				(array): New population fitness values.
				(dict): Additional arguments.
		"""
		return [], [], {}

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core functionality of algorithm.

		This function is called on every algorithm iteration.

		Args:
			task (:class:Task): Optimization task
			pop (array): Current population coordinates
			fpop (array): Current population fitness value
			xb (array): Current generation best individuals coordinates
			xb_f (float): Current generation best individuals fitness value
			dparams (dict): Additional arguments for algorithms

		Returns:
			(tuple): tuple containing:
				pop (array): New populations coordinates
				fpop (array): New populations fitness values
				(dict): Additional arguments of the algorithm
		"""
		return pop, fpop, {}

	def runYield(self, task):
		r"""Run the algorithm for a single iteration and return the best solution.

		Args:
			task (:class:Task): Task with bounds and objective function for optimization

		Yields:
			xb (array): New population best individuals coordinates
			fxb (float): Fitness value of the best solution
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
			task (:class:Task); Task with bounds and objective function for optimization

		Returns:
			(tuple): Tuple containing:
				xb (array): Best individual found in optimization process
				fxb (real): Best fitness value found in optimization process
		"""
		algo, xb, fxb = self.runYield(task), None, inf
		while not task.stopCond():
			xb, fxb = next(algo)
			task.nextIter()
		return xb, fxb

	def run(self, task):
		r"""Start the optimization.

		Args:
			task (:py:class:Task): Optimization task

		Returns:
			(tuple): Tuple containing:
				(array): Best individuals components found in optimization process
				(float): Best fitness value found in optimization process

		See Also:
			:py:meth:Algorithm.runTask(self, taks)
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
		x (array): Coordinates of inidividual
		f (float): Function/fitness value of individual
	"""
	x = None
	f = inf

	def __init__(self, **kwargs):
		task, rnd, x = kwargs.pop('task', None), kwargs.pop('rand', rand), kwargs.pop('x', None)
		self.f = task.optType.value * inf if task is not None else inf
		if x is not None: self.x = x if isinstance(x, ndarray) else asarray(x)
		else: self.generateSolution(task, rnd)
		if kwargs.pop('e', True) and task is not None: self.evaluate(task, rnd)

	def generateSolution(self, task, rnd=rand):
		r"""Generate new solution.

		Args:
			task (:py:class:Task): Optimization task
			e (bool): Evaluate the solution
			rnd (:py:class:RandomState): Random numbers generator object
		"""
		if task is not None: self.x = task.Lower + task.bRange * rnd.rand(task.D)

	def evaluate(self, task, rnd=rand):
		r"""Evaluate the solution.

		Args:
			task (Task): Objective function object
		"""
		self.repair(task, rnd=rnd)
		self.f = task.eval(self.x)

	def repair(self, task, rnd=rand):
		r"""Repair solution and put the solution in the bounds of problem.

		Args:
			task (:py:class:Task): Optimization task
		"""
		self.x = task.repair(self.x, rnd=rnd)

	def copy(self):
		r"""Return a copy of self.

		Returns:
			(:py:class:Individual): Copy of self
		"""
		return Individual(x=self.x, f=self.f, e=False)

	def __eq__(self, other):
		r"""Compare the individuals for equalities.

		Args:
			other (object, :py:class:Individual): Object that we want to compare this object to

		Returns:
			(bool): ``True`` if equal or ``False`` if no equal
		"""
		return array_equal(self.x, other.x) and self.f == other.f

	def __str__(self):
		r"""Print the individual with the solution and objective value.

		Returns:
			(str): String representation of self
		"""
		return '%s -> %s' % (self.x, self.f)

	def __getitem__(self, i):
		r"""Get the value of i-th component of the solution.

		Args:
			i (int): Position of the solution component

		Returns:
			(float, int): Value of ith component
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
			(int): Number of components
		"""
		return len(self.x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
