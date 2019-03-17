# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, expression-not-assigned, len-as-condition, no-self-use, unused-argument, no-else-return, dangerous-default-value, unnecessary-pass
import logging
from numpy import random as rand, inf, ndarray, asarray, array_equal, argmin
from NiaPy.util import StopingTask, OptimizationType
from NiaPy.util import FesException, GenException, TimeException, RefException

logging.basicConfig()
logger = logging.getLogger('NiaPy.util.utility')
logger.setLevel('INFO')

__all__ = ['Algorithm', 'Individual']

class Algorithm:
	r"""Class for implementing algorithms.

	**Date:** 2018

	**Author:** Klemen Berkovič

	**License:** MIT

	**Fields**:
	Name {array} -- List of names for algorithm
	Rand {class) -- Random generator
	task {class} -- Optimization task
	"""
	Name = ['Algorithm', 'AAA']
	Rand = rand.RandomState(None)

	@staticmethod
	def typeParameters():
		r"""TODO documentation."""
		pass

	def __init__(self, **kwargs):
		r"""Initialize algorithm and create name for an algorithm.

		**Arguments:**
		name {string} -- full name of algorithm
		shortName {string} -- short name of algorithm

		**Raises:**
		TypeError -- raised when given benchmark function does not exist

		**See**:
		Algorithm.setParameters(self, **kwargs)
		"""
		self.Rand = rand.RandomState(kwargs.pop('seed', None))
		self.setParameters(**kwargs)

	def setParameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		**Arguments:**
		kwargs {dict} -- parameter values dictionary
		"""
		pass

	def setBenchmark(self, bech):
		r"""Set the benchmark for the algorithm.

		**Arguments**:
		bech {Task} -- optimization task to perform

		**See**:
		Algorithm.setTask
		"""
		return self.setTask(bech)

	def rand(self, D=1):
		r"""Get random distribution of shape D in range from 0 to 1.

		**Arguments:**
		D {array} or {int} -- shape of returned random distribution
		"""
		if isinstance(D, (ndarray, list)): return self.Rand.rand(*D)
		elif D > 1: return self.Rand.rand(D)
		else: return self.Rand.rand()

	def uniform(self, Lower, Upper, D=None):
		r"""Get uniform random distribution of shape D in range from "Lower" to "Upper".

		**Arguments:**
		Lower {array} or {real} or {int} -- lower bound
		Upper {array} or {real} or {int} -- upper bound
		D {array} or {int} -- shape of returned uniform random distribution
		"""
		return self.Rand.uniform(Lower, Upper, D) if D is not None else self.Rand.uniform(Lower, Upper)

	def normal(self, loc, scale, D=None):
		r"""Get normal random distribution of shape D with mean "loc" and standard deviation "scale".

		**Arguments:**
		loc {} -- mean of the normal random distribution
		scale {} -- standard deviation of the normal random distribution
		D {array} or {int} -- shape of returned normal random distribution
		"""
		return self.Rand.normal(loc, scale, D) if D is not None else self.Rand.normal(loc, scale)

	def randn(self, D=None):
		r"""Get standard normal distribution of shape D.

		**Arguments**:
		D {array} -- shape of returned standard normal distribution
		"""
		if D is None: return self.Rand.randn()
		elif isinstance(D, int): return self.Rand.randn(D)
		return self.Rand.randn(*D)

	def randint(self, Nmax, D=1, Nmin=0, skip=[]):
		r"""Get discrete uniform (integer) random distribution of D shape in range from "Nmin" to "Nmax".

		**Arguments:**
		Nmin {integer} -- lower integer bound
		Nmax {integer} -- one above upper integer bound
		D {array} or {int} -- shape of returned discrete uniform random distribution
		skip {array} -- numbers to skip
		"""
		r = None
		if isinstance(D, (list, tuple, ndarray)): r = self.Rand.randint(Nmin, Nmax, D)
		elif D > 1: r = self.Rand.randint(Nmin, Nmax, D)
		else: r = self.Rand.randint(Nmin, Nmax)
		return r if r not in skip else self.randint(Nmax, D, Nmin, skip)

	def getBest(self, X, X_f, xb=None, xb_f=inf):
		r"""Get the best individual for population.

		***Arguments:***
		X {array} -- Population
		X_f {array} -- Fitness values of aligned individuals
		xb {array} -- Best individual
		xb_f {real} -- Fitness value of best individal
		"""
		ib = argmin(X_f)
		if xb_f >= X_f[ib]: return X[ib], X_f[ib]
		else: return xb, xb_f

	def initPopulation(self, task):
		r"""Initialization for starting population of optimization algorithm.

		**Arguments:**
		task {class} -- Optimization task.

		**Return:**
		New population.
		New population fitness values.
		Additional arguments.
		"""
		return [], [], {}

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""

		:param task:
		:param pop:
		:param fpop:
		:param xb:
		:param fxb:
		:param kwargs:
		:return:
		"""
		return pop, fpop

	def runYield(self, task):
		r"""Run the algorithm for a single iteration and return the best solution.

		**Arguments:**
		task {Task} -- task with bounds and objective function for optimization

		**Return:**
		solution {array} -- point of the best solution
		fitness {real} -- fitness value of the best solution
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

		**Arguments:**
		task {Task} -- task with bounds and objective function for optimization

		**Return:**
		solution {array} -- point of the best solution
		fitness {real} -- fitness value of best solution
		"""
		algo = self.runYield(task)
		while not task.stopCond():
			xb, fxb = next(algo)
			task.nextIter()
		return xb, fxb

	def run(self, task):
		r"""Start the optimization.

		**See**:
		Algorithm.runTask(self, taks)
		"""
		try:
			task.start()
			r = self.runTask(task)
			return r[0], r[1] * task.optType.value
		except (FesException, GenException, TimeException, RefException): return task.x, task.x_f * task.optType.value
		return None, inf * task.optType.value

class Individual:
	r"""Class that represents one solution in population of solutions.

	**Date:** 2018

	**Author:** Klemen Berkovič

	**License:** MIT
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

		**Arguments:**
		task {Task}
		e {bool} -- evaluate the solution
		rnd {random} -- random numbers generator object
		"""
		if task is not None: self.x = task.Lower + task.bRange * rnd.rand(task.D)

	def evaluate(self, task, rnd=rand):
		r"""Evaluate the solution.

		**Arguments:**
		task {Task} -- objective function object
		"""
		self.repair(task, rnd=rnd)
		self.f = task.eval(self.x)

	def repair(self, task, rnd=rand):
		r"""Repair solution and put the solution in the bounds of problem.

		**Arguments:**
		task {Task}
		"""
		self.x = task.repair(self.x, rnd=rnd)

	def copy(self):
		r"""Return a copy of self."""
		return Individual(x=self.x, f=self.f, e=False)

	def __eq__(self, other):
		r"""Compare the individuals for equalities."""
		return array_equal(self.x, other.x) and self.f == other.f

	def __str__(self):
		r"""Print the individual with the solution and objective value."""
		return '%s -> %s' % (self.x, self.f)

	def __getitem__(self, i):
		r"""Get the value of i-th component of the solution.

		**Arguments:**
		i {integer} -- position of the solution component
		"""
		return self.x[i]

	def __setitem__(self, i, v):
		r"""Set the value of i-th component of the solution to v value.

		**Arguments:**
		i {integer} -- position of the solution component
		v {dynamic} -- value to set to i-th component
		"""
		self.x[i] = v

	def __len__(self):
		r"""Get the length of the solution or the number of components."""
		return len(self.x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
