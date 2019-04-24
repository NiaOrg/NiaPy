# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, redefined-builtin, line-too-long, no-self-use, arguments-differ, no-else-return, bad-continuation
import logging

from numpy import argsort, sum, exp, apply_along_axis, asarray, where

from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['FireflyAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

class FireflyAlgorithm(Algorithm):
	r"""Implementation of Firefly algorithm.

	Algorithm:
		Firefly algorithm

	Date:
		2016

	Authors:
		Iztok Fister Jr, Iztok Fister and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Fister, I., Fister Jr, I., Yang, X. S., & Brest, J. (2013). A comprehensive review of firefly algorithms. Swarm and Evolutionary Computation, 13, 34-46.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		alpha (float): Alpha parameter.
		betamin (float): Betamin parameter.
		gamma (flaot): Gamma parameter.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['FireflyAlgorithm', 'FA']

	@staticmethod
	def algorithmInfo():
		return r"""
        	Description: Firefly algorithm is inspired by the flashing behavior of fireflies.
        	Author: Xin-She Yang
        	Year: 2008
                Main reference: Yang, Xin-She. Nature-Inspired Metaheuristic Algorithms,  Luniver Press, 2008.
        """

	@staticmethod
	def typeParameters():
		r"""TODO.

		Returns:
			Dict[str, Callable]:
				* alpha (Callable[[Union[float, int]], bool]): TODO.
				* betamin (Callable[[Union[float, int]], bool]): TODO.
				* gamma (Callable[[Union[float, int]], bool]): TODO.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'alpha': lambda x: isinstance(x, (float, int)) and x > 0,
			'betamin': lambda x: isinstance(x, (float, int)) and x > 0,
			'gamma': lambda x: isinstance(x, (float, int)) and x > 0,
		})
		return d

	def setParameters(self, NP=20, alpha=1, betamin=1, gamma=2, **ukwargs):
		r"""Set the parameters of the algorithm.

		Args:
			NP (Optional[int]): Population size.
			alpha (Optional[float]): Alpha parameter.
			betamin (Optional[float]): Betamin parameter.
			gamma (Optional[flaot]): Gamma parameter.
			ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.alpha, self.betamin, self.gamma = alpha, betamin, gamma
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def alpha_new(self, a, alpha):
		r"""Optionally recalculate the new alpha value.

		Args:
			a (float):
			alpha (float):

		Returns:
			float: New value of parameter alpha
		"""
		delta = 1.0 - pow(pow(10.0, -4.0) / 0.9, 1.0 / float(a))
		return (1 - delta) * alpha

	def move_ffa(self, i, Fireflies, Intensity, oFireflies, alpha, task):
		r"""Move fireflies.

		Args:
			i (int): Index of current individual.
			Fireflies (numpy.ndarray):
			Intensity (numpy.ndarray):
			oFireflies (numpy.ndarray):
			alpha (float):
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, bool]:
				1. New individual
				2. ``True`` if individual was moved, ``False`` if individual was not moved
		"""
		moved = False
		for j in range(self.NP):
			r = sum((Fireflies[i] - Fireflies[j]) ** 2) ** (1 / 2)
			if Intensity[i] <= Intensity[j]: continue
			beta = (1.0 - self.betamin) * exp(-self.gamma * r ** 2.0) + self.betamin
			tmpf = alpha * (self.uniform(0, 1, task.D) - 0.5) * task.bRange
			Fireflies[i] = task.repair(Fireflies[i] * (1.0 - beta) + oFireflies[j] * beta + tmpf, rnd=self.Rand)
			moved = True
		return Fireflies[i], moved

	def initPopulation(self, task):
		r"""Initialize the starting population.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments:
					* alpha (float): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		Fireflies, Intensity, _ = Algorithm.initPopulation(self, task)
		return Fireflies, Intensity, {'alpha': self.alpha}

	def runIteration(self, task, Fireflies, Intensity, xb, fxb, alpha, **dparams):
		r"""Core function of Firefly Algorithm.

		Args:
			task (Task): Optimization task.
			Fireflies (numpy.ndarray): Current population.
			Intensity (numpy.ndarray[float]): Current population function/fitness values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individual fitness/function value.
			alpha (float): TODO.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments:
					* alpha (float): TODO

		See Also:
			* :func:`NiaPy.algorithms.basic.FireflyAlgorithm.move_ffa`
		"""
		alpha = self.alpha_new(task.nFES / self.NP, alpha)
		Index = argsort(Intensity)
		tmp = asarray([self.move_ffa(i, Fireflies[Index], Intensity[Index], Fireflies, alpha, task) for i in range(self.NP)])
		Fireflies, evalF = asarray([tmp[i][0] for i in range(len(tmp))]), asarray([tmp[i][1] for i in range(len(tmp))])
		Intensity[where(evalF)] = apply_along_axis(task.eval, 1, Fireflies[where(evalF)])
		return Fireflies, Intensity, {'alpha': alpha}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
