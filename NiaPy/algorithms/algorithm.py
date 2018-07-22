# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from NiaPy.benchmarks.utility import Task

__all__ = ['Algorithm']

class Algorithm(object):
	r"""Class for implementing algorithms.

	**Data:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	def __init__(self, **kwargs):
		r"""Initialize algorithm and create name for an algorithm

		**Arguments**:
		name {string} -- Full name of algorithm
		shortName {string} -- Short name of algorithm
		"""
		task = kwargs.get('task', None)
		self.name, self.sName, = kwargs.get('name', 'Algorith'), kwargs.get('sName', 'algo')
		self.task = task if task != None else Task(kwargs.get('D', 10), kwargs.get('nFES', 100000), None, kwargs.get('benchmark', 'ackley'))

	def setParameters(self, **kwargs): pass

	def run(self):
		r"""Method that start the optimization

		**See**:
		Algorithm.runTask(self, taks)
		"""
		return self.runTask(self.task)

	def runTask(self, task):
		r"""Method that start the optimization.

		**Arguments**
		task {Task} -- Task with bounds and objective function for optimization

		**Return**
		solution {array} -- point of best solution
		fitnes {real} -- fitnes value of best solution
		"""
		pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
