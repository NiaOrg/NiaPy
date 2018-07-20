# encoding=utf8
# pylint: disable=mixed-indentation

__all__ = ['Algorithm']

class Algorithm(object):
	r"""
	**Data:** 2018
	**Author:** Klemen Berkovič
	**License:** MIT
	"""
	def __init__(self, **kwargs):
		r"""__init__(self, name, shortName)
		**Arguments**:
		name {string} -- Full name of algorithm
		shortName {string} -- Short name of algorithm
		"""
		self.name, self.sName = kwargs.get('name', 'Algorith'), kwargs.get('sName', 'algo')

	def setParameters(self, **kwargs): pass

	def run(self): pass

	def runTask(self, task): 
		r"""runTask(self, task)
		**Arguments**
			task {Task} -- Task with bounds and objective function for optimization

		**Return**
			solution {array} -- point of best solution
			fitnes {real} -- fitnes value of best solution
		"""
		pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
