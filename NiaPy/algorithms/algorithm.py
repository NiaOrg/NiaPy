# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, expression-not-assigned
from numpy import random as rand, inf, ndarray, asarray, where
from NiaPy.benchmarks.utility import Task

__all__ = ['Algorithm', 'Individual']

class Algorithm:
	r"""Class for implementing algorithms.

	**Data:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	def __init__(self, **kwargs):
		r"""Initialize algorithm and create name for an algorithm.

		**Arguments**:
		name {string} -- Full name of algorithm
		shortName {string} -- Short name of algorithm
		NP {integer} -- population size
		D {integer} -- dimension of problem
		nGEN {integer} -- nuber of generation/iterations
		nFES {integer} -- number of function evaluations
		benchmark {object} -- benchmark implementation object
		task {Task} -- task to perform optimization on

		**Raises**:
		TypeError -- Raised when given benchmark function which does not exists.

		**See**:
		Algorithm.setParameters(self, **kwargs)
		"""
		task = kwargs.get('task', None)
		self.name, self.sName, self.rand = kwargs.get('name', 'Algorith'), kwargs.get('sName', 'algo'), rand.RandomState(kwargs.get('seed', 1))
		self.task = task if task != None else Task(kwargs.get('D', 10), kwargs.get('nFES', 100000), kwargs.get('nGEN', None), kwargs.get('benchmark', 'ackley'))
		kwargs.pop('name', None), kwargs.pop('sName', None), kwargs.pop('D', None), kwargs.pop('nFES', None), kwargs.pop('nGEN', None), kwargs.pop('benchmark', None), kwargs.pop('task', None)
		self.setParameters(**kwargs)

	def setParameters(self, **kwargs): pass

	def run(self):
		r"""Start the optimization.

		**See**:
		Algorithm.runTask(self, taks)
		"""
		return self.runTask(self.task)

	def runTask(self, task):
		r"""Start the optimization.

		**Arguments**
		task {Task} -- Task with bounds and objective function for optimization

		**Return**
		solution {array} -- point of best solution
		fitnes {real} -- fitnes value of best solution
		"""
		pass

class Individual:
	r"""
	"""
	def __init__(self, **kwargs):
		self.f = inf
		task, rnd, x = kwargs.get('task', None), kwargs.get('rand', rand), kwargs.get('x', [])
		if len(x) > 0: self.x = x if isinstance(x, ndarray) else asarray(x)
		else: self.generateSolution(task, kwargs.get('e', True), rnd)

	def generateSolution(self, task, e=True, rnd=rand):
		r"""Generate new solution.

		Arguments:
		task {Task}
		e {bool}
		rnd {random} -- Object for generating random numbers
		"""
		self.x = task.Lower + task.bRange * rnd.rand(task.D)
		if e: self.evaluate(task)

	def evaluate(self, task):
		r"""Evalue the solution.

		Arguments:
		task {Task} -- Object with objective function for optimization
		"""
		self.f = task.eval(self.x)

	def repair(self, task):
		r"""Reper solution and put the solution in the bounds of problem.

		Arguments:
		task {Task}
		"""
		ir = where(self.x > task.Upper)
		self.x[ir] = task.Upper[ir]
		ir = where(self.x < task.Lower)
		self.x[ir] = task.Lower[ir]

	def __eq__(self, other):
		r"""
		"""
		return array_equal(self.x, other.x) and self.f == other.f

	def __str__(self):
		r"""
		"""
		return '%s -> %s' % (self.x, self.f)

	def __getitem__(self, i):
		r"""
		"""
		return self.x[i]

	def __len__(self):
		r"""
		"""
		return len(self.x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
