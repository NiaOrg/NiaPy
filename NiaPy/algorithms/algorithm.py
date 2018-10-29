# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, expression-not-assigned, len-as-condition, no-self-use, unused-argument, no-else-return, old-style-class, dangerous-default-value
from numpy import random as rand, inf, ndarray, asarray, array_equal
from NiaPy.util import Task, OptimizationType

__all__ = ['Algorithm', 'Individual']

class Algorithm:
	r"""Class for implementing algorithms.

	**Data:** 2018

	**Author:** Klemen Berkovič

	**License:** MIT
	"""
	Name = ['Algorithm', 'AAA']

	def __init__(self, **kwargs):
		r"""Initialize algorithm and create name for an algorithm.

		**Arguments:**

		name {string} -- Full name of algorithm

		shortName {string} -- Short name of algorithm

		NP {integer} -- population size

		D {integer} -- dimension of problem

		nGEN {integer} -- nuber of generation/iterations

		nFES {integer} -- number of function evaluations

		benchmark {object} -- benchmark implementation object

		task {Task} -- task to perform optimization on

		**Raises:**

		TypeError -- Raised when given benchmark function does not exists.

		**See**:
		Algorithm.setParameters(self, **kwargs)
		"""
		task, self.Rand = kwargs.pop('task', None), rand.RandomState(kwargs.pop('seed', None))
		self.task = task if task is not None else Task(kwargs.pop('D', 10), kwargs.pop('nFES', inf), kwargs.pop('nGEN', inf), kwargs.pop('benchmark', 'ackley'), optType=kwargs.pop('optType', OptimizationType.MINIMIZATION))
		self.setParameters(**kwargs)

	def setParameters(self, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		**Arguments:**

		kwargs {dict} -- Dictionary with values of the parametres
		"""
		pass

	def setTask(self, task):
		r"""Set the benchmark for the algorithm.

		**Arguments**:
		bech {Task} -- Optimization task to perform
		"""
		self.task = task
		return self

	def setBenchmark(self, bech):
		r"""Set the benchmark for the algorithm.

		**Arguments**:
		bech {Task} -- Optimization task to perform

		**See**:
		Algorithm.setTask
		"""
		return self.setTask(bech)

	def rand(self, D=1):
		r"""Get random numbers of shape D in range from 0 to 1.

		**Arguments:**

		D {array} or {int} -- Shape of return random numbers
		"""
		if isinstance(D, (ndarray, list)): return self.Rand.rand(*D)
		elif D > 1: return self.Rand.rand(D)
		else: return self.Rand.rand()

	def uniform(self, Lower, Upper, D=None):
		r"""Get D shape random uniform numbers in range from Lower to Upper.

		**Arguments:**

		Lower {array} or {real} or {int} -- Lower bound

		Upper {array} or {real} or {int} -- Upper bound

		D {array} or {int} -- Shape of returnd random uniform numbers
		"""
		return self.Rand.uniform(Lower, Upper, D) if D is not None else self.Rand.uniform(Lower, Upper)

	def normal(self, loc, scale, D=None):
		r"""Get D shape random normal distributed numbers.

		**Arguments:**

		loc {} --

		scale {} --

		D {array} or {int} -- Shape of returnd random uniform numbers
		"""
		return self.Rand.normal(loc, scale, D) if D is not None else self.Rand.normal(loc, scale)

	def randn(self, D=None):
		r"""Get D shape random normal distributed numbers.

		**Arguments**:
		D {array} -- Shape of returnd random numbers
		"""
		if D is None: return self.Rand.randn()
		elif isinstance(D, int): return self.Rand.randn(D)
		return self.Rand.randn(*D)

	def randint(self, Nmax, D=1, Nmin=0, skip=[]):
		r"""Get D shape random full numbers in range Nmin to Nmax.

		**Arguments:**

		Nmin {integer} --

		Nmax {integer} --

		D {array} or {int} -- Shape of returned random uniform numbers

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
		return self.runTask(self.task)

	def runYield(self, task):
		r"""Run the algorithm for only one iteration and return the best solution.

		**Arguments:**

		task {Task} -- Task with bounds and objective function for optimization

		Return:

		solution {array} -- point of the best solution

		fitness {real} -- fitness value of the best solution
		"""
		yield None, None

	def runTask(self, task):
		r"""Start the optimization.

		**Arguments:**

		task {Task} -- Task with bounds and objective function for optimization

		**Return:**

		solution {array} -- point of the best solution

		fitness {real} -- fitness value of best solution
		"""
		return None, None

class Individual:
	r"""Class that represent one solution in population of solutions.

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

		e {bool} -- Eval the solution

		rnd {random} -- Object for generating random numbers
		"""
		if task is not None: self.x = task.bcLower() + task.bcRange() * rnd.rand(task.dim())

	def evaluate(self, task, rnd=rand):
		r"""Evaluate the solution.

		**Arguments:**

		task {Task} -- Object with objective function for optimization
		"""
		self.repair(task, rnd=rnd)
		self.f = task.eval(self.x)

	def repair(self, task, rnd=rand):
		r"""Reper solution and put the solution in the bounds of problem.

		**Arguments:**

		task {Task}
		"""
		self.x = task.repair(self.x, rnd=rnd)

	def __eq__(self, other):
		r"""Compare the individuals if they are one of the same."""
		return array_equal(self.x, other.x) and self.f == other.f

	def __str__(self):
		r"""Print the individula with the solution and objective value."""
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
