# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, logging-not-lazy, attribute-defined-outside-init, arguments-differ, bad-continuation, unused-argument
import logging

from numpy import full

from NiaPy.algorithms.basic import BatAlgorithm
from NiaPy.algorithms.basic.de import CrossBest1

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['HybridBatAlgorithm']

class HybridBatAlgorithm(BatAlgorithm):
	r"""Implementation of Hybrid bat algorithm.

	Algorithm:
		Hybrid bat algorithm

	Date:
		2018

	Author:
		Grega Vrbancic and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She. "A Hybrid Bat Algorithm". Elektrotehniski vestnik, 2013. 1-7.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		F (float): Scaling factor.
		CR (float): Crossover.

	See Also:
		* :class:`NiaPy.algorithms.basic.BatAlgorithm`
	"""
	Name = ['HybridBatAlgorithm', 'HBA']

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* F (Callable[[Union[int, float]], bool]): TODO
				* CR (Callable[[float], bool]): TODO
		"""
		d = BatAlgorithm.typeParameters()
		d['F'] = lambda x: isinstance(x, (int, float)) and x > 0
		d['CR'] = lambda x: isinstance(x, float) and 0 <= x <= 1
		return d

	def setParameters(self, NP=40, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, F=0.78, CR=0.35, CrossMutt=CrossBest1, **ukwargs):
		r"""Set core parameters of HybridBatAlgorithm algorithm.

		Arguments:
			F (Optional[float]): Scaling factor.
			CR (Optional[float]): Crossover.

		See Also:
			* :func:`NiaPy.algorithms.basic.BatAlgorithm.setParameters`
		"""
		BatAlgorithm.setParameters(self, **ukwargs)
		self.A, self.r, self.Qmin, self.Qmax, self.F, self.CR, self.CrossMutt = A, r, Qmin, Qmax, F, CR, CrossMutt
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def initPopulation(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population.
				2. Initialized populations fitness/function values.
				3. Additional arguments:
					* v (numpy.ndarray[float]): TODO
		"""
		Sol, Fitness, _ = BatAlgorithm.initPopulation(self, task)
		v = full([self.NP, task.D], 0.0)
		return Sol, Fitness, {'v': v}

	def runIteration(self, task, Sol, Fitness, best, f_min, v, **dparams):
		r"""Core function of HybridBatAlgorithm algorithm.

		Args:
			task (Task): Optimization task.
			Sol (numpy.ndarray): Current population.
			Fitness (numpy.ndarray[float]: Current populations function/fitness values.
			best:
			f_min (float):
			v:
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population.
				2. New populations function/fitness values.
				3. Additional arguments:
					* v (numpy.ndarray[float]): TODO
		"""
		Q = self.Qmin + (self.Qmax - self.Qmin) * self.uniform(0, 1, self.NP)
		for i in range(self.NP):
			v[i] = v[i] + (Sol[i] - best) * Q[i]
			S = task.repair(Sol[i] + v[i], rnd=self.Rand)
			if self.rand() > self.r: S = task.repair(self.CrossMutt(Sol, i, best, self.F, self.CR, rnd=self.Rand), rnd=self.Rand)
			f_new = task.eval(S)
			if f_new <= Fitness[i] and self.rand() < self.A: Sol[i], Fitness[i] = S, f_new
			if f_new <= f_min: best, f_min = S, f_new
		return Sol, Fitness, {'S': S, 'Q': Q, 'v': v}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
