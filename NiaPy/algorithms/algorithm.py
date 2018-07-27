# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long
from numpy.random import RandomState
from NiaPy.benchmarks.utility import Task

__all__ = ['Algorithm']

class Algorithm(object):
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
		self.name, self.sName, self.rand = kwargs.get('name', 'Algorith'), kwargs.get('sName', 'algo'), RandomState(kwargs.get('seed', 1))
		self.task = task if task != None else Task(kwargs.get('D', 10), kwargs.get('nFES', 100000), kwargs.get('nGEN', None), kwargs.get('benchmark', 'ackley'))
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

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
