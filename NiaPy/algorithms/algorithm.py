# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements

__all__ = ['Algorithm']

class Algorithm(object):
	r"""Class for implementing algorithms.

	**Data:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	def __init__(self, **kwargs):
		r"""Initialization of algorithm, that creates name for an algorithm

		**Arguments**:
		name {string} -- Full name of algorithm
		shortName {string} -- Short name of algorithm
		"""
		self.name, self.sName, self.task = kwargs.get('name', 'Algorith'), kwargs.get('sName', 'algo'), None

	def setParameters(self, **kwargs): pass

	def run(self): self.runTask(self.task)

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
