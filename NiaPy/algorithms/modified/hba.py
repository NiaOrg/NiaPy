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
				* F (Callable[[Union[int, float]], bool]): Scaling factor.
				* CR (Callable[[float], bool]): Crossover probability.
		"""
		d = BatAlgorithm.typeParameters()
		d['F'] = lambda x: isinstance(x, (int, float)) and x > 0
		d['CR'] = lambda x: isinstance(x, float) and 0 <= x <= 1
		return d

	def setParameters(self, F=0.78, CR=0.35, CrossMutt=CrossBest1, **ukwargs):
		r"""Set core parameters of HybridBatAlgorithm algorithm.

		Arguments:
			F (Optional[float]): Scaling factor.
			CR (Optional[float]): Crossover.

		See Also:
			* :func:`NiaPy.algorithms.basic.BatAlgorithm.setParameters`
		"""
		BatAlgorithm.setParameters(self, **ukwargs)
		self.F, self.CR, self.CrossMutt = F, CR, CrossMutt
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def generateBest(self, best, task, i, Sol, **kwargs):
		r"""Generate new solution based on global best known solution.

		Args:
			best (numpy.ndarray): Global best individual.
			task (Task): Optimization task.
			i (int): Index of current individual.
			Sol (numpy.ndarray): Current best population.
			**kwargs (Dict[str, Any]):

		Returns:
			numpy.ndarray: New solution based on global best individual.
		"""
		return task.repair(self.CrossMutt(Sol, i, best, self.F, self.CR, rnd=self.Rand), rnd=self.Rand)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
