# encoding=utf8
import logging

import numpy as np

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['RandomSearch']

class RandomSearch(Algorithm):
	r"""Implementation of a simple Random Algorithm.

	Algorithm:
		Random Search

	Date:
		11.10.2020

	Authors:
		Iztok Fister Jr., Grega Vrbančič

	License:
		MIT

	Reference URL: https://en.wikipedia.org/wiki/Random_search

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['RandomSearch', 'RS']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""None"""

	def setParameters(self, **ukwargs):
		r"""Set the algorithm parameters/arguments.

		Arguments:
		See Also
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""

		ukwargs.pop('NP', None)
		Algorithm.setParameters(self, NP=1)

	def getParameters(self):
		r"""Get algorithms parametes values.

		Returns:
			Dict[str, Any]:
		See Also
			* :func:`NiaPy.algorithms.Algorithm.getParameters`
		"""
		d = Algorithm.getParameters(self)
		return d

	def initPopulation(self, task):
		r"""Initialize the starting population.

		Args:
			task (Task): Optimization task.
		Returns:
			Tuple[numpy.ndarray, float, dict]:
			1. Initial solution
			2. Initial solutions fitness/objective value
			3. Additional arguments
		"""
		total_candidates = 0
		if task.nGEN or task.nFES:
			total_candidates = task.nGEN if task.nGEN else task.nFES
		self.candidates = []
		for i in range(total_candidates):
			while True:
				x = task.Lower + task.bcRange() * self.rand(task.D)
				if not np.any([np.all(a == x) for a in self.candidates]):
					self.candidates.append(x)
					break

		xfit = task.eval(self.candidates[0])
		return x, xfit, {}

	def runIteration(self, task, x, xfit, xb, fxb, **dparams):
		r"""Core function of the algorithm.

		Args:
			task (Task):
			x (numpy.ndarray):
			xfit (float):
			xb (numpy.ndarray):
			fxb (float):
			**dparams (dict): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, float, numpy.ndarray, float, dict]:
			1. New solution
			2. New solutions fitness/objective value
			3. New global best solution
			4. New global best solutions fitness/objective value
			5. Additional arguments
		"""
		current_candidate = task.Evals if task.Evals else task.Iters
		x = self.candidates[current_candidate]
		xfit = task.eval(x)
		xb, fxb = self.getBest(x, xfit, xb, fxb)
		return x, xfit, xb, fxb, {}
