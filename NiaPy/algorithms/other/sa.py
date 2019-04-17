# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, unused-argument, arguments-differ, bad-continuation
import logging

from numpy import exp

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['SimulatedAnnealing', 'coolDelta', 'coolLinear']

def coolDelta(currentT, T, deltaT, nFES, **kwargs):
	r"""Calculate new temperature by differences.

	Args:
		currentT (float):
		T (float):
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		float: New temperature.
	"""
	return currentT - deltaT

def coolLinear(currentT, T, deltaT, nFES, **kwargs):
	r"""Calculate temperature with linear function.

	Args:
		currentT (float): Current temperature.
		T (float):
		deltaT (float):
		nFES (int): Number of evaluations done.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		float: New temperature.
	"""
	return currentT - T / nFES

class SimulatedAnnealing(Algorithm):
	r"""Implementation of Simulated Annealing Algorithm.

	Algorithm:
		Simulated Annealing Algorithm

	Date:
		2018

	Authors:
		Jan Popič and Klemen Berkovič

	License:
		MIT

	Reference URL:

	Reference paper:

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		delta (float): Movement for neighbour search.
		T (float); Starting temperature.
		deltaT (float): Change in temperature.
		coolingMethod (Callable): Neighbourhood function.
		epsilon (float): Error value.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['SimulatedAnnealing', 'SA']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""None"""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* delta (Callable[[Union[float, int], bool]): TODO
		"""
		return {
			'delta': lambda x: isinstance(x, (int, float)) and x > 0,
			'T': lambda x: isinstance(x, (int, float)) and x > 0,
			'deltaT': lambda x: isinstance(x, (int, float)) and x > 0,
			'epsilon': lambda x: isinstance(x, float) and 0 < x < 1
		}

	def setParameters(self, delta=0.5, T=2000, deltaT=0.8, coolingMethod=coolDelta, epsilon=1e-23, **ukwargs):
		r"""Set the algorithm parameters/arguments.

		Arguments:
			delta (Optional[float]): Movement for neighbour search.
			T (Optional[float]); Starting temperature.
			deltaT (Optional[float]): Change in temperature.
			coolingMethod (Optional[Callable]): Neighbourhood function.
			epsilon (Optional[float]): Error value.

		See Also
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		ukwargs.pop('NP', None)
		Algorithm.setParameters(self, NP=1, **ukwargs)
		self.delta, self.T, self.deltaT, self.cool, self.epsilon = delta, T, deltaT, coolingMethod, epsilon
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def initPopulation(self, task):
		x = task.Lower + task.bcRange() * self.rand(task.D)
		curT, xfit = self.T, task.eval(x)
		return x, xfit, {'curT': curT}

	def runIteration(self, task, x, xfit, xb, fxb, curT, **dparams):
		c = task.repair(x - self.delta / 2 + self.rand(task.D) * self.delta, rnd=self.Rand)
		cfit = task.eval(c)
		deltaFit, r = cfit - xfit, self.rand()
		if deltaFit < 0 or r < exp(deltaFit / curT): x, xfit = c, cfit
		curT = self.cool(curT, self.T, deltaT=self.deltaT, nFES=task.nFES)
		return x, xfit, {'curT': curT}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
