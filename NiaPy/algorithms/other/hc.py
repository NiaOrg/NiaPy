# encoding=utf8
import logging

from numpy import random as rand

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['HillClimbAlgorithm']

def Neighborhood(x, delta, task, rnd=rand):
	r"""Get neighbours of point.

	Args:
		x numpy.ndarray: Point.
		delta (float): Standard deviation.
		task (Task): Optimization task.
		rnd (Optional[mtrand.RandomState]): Random generator.

	Returns:
		Tuple[numpy.ndarray, float]:
			1. New solution.
			2. New solutions function/fitness value.
	"""
	X = x + rnd.normal(0, delta, task.D)
	X = task.repair(X, rnd)
	Xfit = task.eval(X)
	return X, Xfit

class HillClimbAlgorithm(Algorithm):
	r"""Implementation of iterative hill climbing algorithm.

	Algorithm:
		Hill Climbing Algorithm

	Date:
		2018

	Authors:
		Jan Popič

	License:
		MIT

	Reference URL:

	Reference paper:

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`

	Attributes:
		delta (float): Change for searching in neighborhood.
		Neighborhood (Callable[numpy.ndarray, float, Task], Tuple[numpy.ndarray, float]]): Function for getting neighbours.
	"""
	Name = ['HillClimbAlgorithm', 'BBFA']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information.
		"""
		return r"""TODO"""

	@staticmethod
	def typeParameters():
		r"""TODO.

		Returns:
			Dict[str, Callable]:
				* delta (Callable[[Union[int, float]], bool]): TODO
		"""
		return {'delta': lambda x: isinstance(x, (int, float)) and x > 0}

	def setParameters(self, delta=0.5, Neighborhood=Neighborhood, **ukwargs):
		r"""Set the algorithm parameters/arguments.

		Args:
			* delta (Optional[float]): Change for searching in neighborhood.
			* Neighborhood (Optional[Callable[numpy.ndarray, float, Task], Tuple[numpy.ndarray, float]]]): Function for getting neighbours.
		"""
		Algorithm.setParameters(self, NP=1, **ukwargs)
		self.delta, self.Neighborhood = delta, Neighborhood

	def getParameters(self):
		d = Algorithm.getParameters(self)
		d.update({
			'delta': self.delta,
			'Neighborhood': self.Neighborhood
		})
		return d

	def initPopulation(self, task):
		r"""Initialize stating point.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, float, Dict[str, Any]]:
				1. New individual.
				2. New individual function/fitness value.
				3. Additional arguments.
		"""
		x = task.Lower + self.rand(task.D) * task.bRange
		return x, task.eval(x), {}

	def runIteration(self, task, x, fx, xb, fxb, **dparams):
		r"""Core function of HillClimbAlgorithm algorithm.

		Args:
			task (Task): Optimization task.
			x (numpy.ndarray): Current solution.
			fx (float): Current solutions fitness/function value.
			xb (numpy.ndarray): Global best solution.
			fxb (float): Global best solutions function/fitness value.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, float, numpy.ndarray, float, Dict[str, Any]]:
				1. New solution.
				2. New solutions function/fitness value.
				3. Additional arguments.
		"""
		lo, xn = False, task.bcLower() + task.bcRange() * self.rand(task.D)
		xn_f = task.eval(xn)
		while not lo:
			yn, yn_f = self.Neighborhood(x, self.delta, task, rnd=self.Rand)
			if yn_f < xn_f: xn, xn_f = yn, yn_f
			else: lo = True or task.stopCond()
		xb, fxb = self.getBest(xn, xn_f, xb, fxb)
		return xn, xn_f, xb, fxb, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
