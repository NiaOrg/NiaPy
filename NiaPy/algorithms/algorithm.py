# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, expression-not-assigned, len-as-condition, no-self-use, unused-argument, no-else-return, old-style-class, dangerous-default-value
import logging
from numpy import random as rand, inf, ndarray, asarray, array_equal
from NiaPy.util import Task, OptimizationType
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
	"""
	Name = ['Algorithm', 'AAA']

	def __init__(self, **kwargs):
		r"""Initialize algorithm and create name for an algorithm.

		**Arguments:**

		name {string} -- full name of algorithm

		shortName {string} -- short name of algorithm

		NP {integer} -- population size

		D {integer} -- dimension of the problem

		nGEN {integer} -- number of generations/iterations

		nFES {integer} -- number of function evaluations

		benchmark {object} -- benchmark implementation object

		task {Task} -- optimization task to perform

		**Raises:**

		TypeError -- raised when given benchmark function does not exist

		**See**:
		Algorithm.setParameters(self, **kwargs)
		"""
		task, self.Rand = kwargs.pop('task', None), rand.RandomState(kwargs.pop('seed', None))
		self.task = task if task is not None else Task(kwargs.pop('D', 10), nFES=kwargs.pop('nFES', inf), nGEN=kwargs.pop('nGEN', inf), benchmark=kwargs.pop('benchmark', 'ackley'), optType=kwargs.pop('optType', OptimizationType.MINIMIZATION))
		self.setParameters(**kwargs)

	def setParameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		**Arguments:**

		kwargs {dict} -- parameter values dictionary
		"""
		pass

	def setTask(self, task):
		r"""Set the benchmark function for the algorithm.

		**Arguments**:
		bech {Task} -- optimization task to perform
		"""
		self.task = task
		return self

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

	def run(self):
		r"""Start the optimization.

		**See**:
		Algorithm.runTask(self, taks)
		"""
		try:
			self.task.start()
			r = self.runTask(self.task)
			return r[0], r[1] * self.task.optType.value
		except (FesException, GenException, TimeException, RefException): return self.task.x, self.task.x_f * self.task.optType.value
		return None, inf * self.task.optType.value

	def runYield(self, task):
		r"""Run the algorithm for a single iteration and return the best solution.

		**Arguments:**

		task {Task} -- task with bounds and objective function for optimization

		Return:

		solution {array} -- point of the best solution

		fitness {real} -- fitness value of the best solution
		"""
		yield None, None

	def runTask(self, task):
		r"""Start the optimization.

		**Arguments:**

		task {Task} -- task with bounds and objective function for optimization

		**Return:**

		solution {array} -- point of the best solution

		fitness {real} -- fitness value of best solution
		"""
		return None, None

class Individual:
	r"""Class that represents one solution in population of solutions.

	**Date:** 2018

	**Author:** Klemen Berkovič

	**License:** MIT
	"""
	def __init__(self, **kwargs):
		task, rnd, x = kwargs.pop('task', None), kwargs.pop('rand', rand), kwargs.pop('x', [])
		self.f = task.optType.value * inf if task is not None else inf
		if len(x) > 0: self.x = x if isinstance(x, ndarray) else asarray(x)
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

	def __len__(self):
		r"""Get the length of the solution or the number of components."""
		return len(self.x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
