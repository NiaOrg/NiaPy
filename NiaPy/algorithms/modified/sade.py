# encoding=utf8
import logging

# from numpy import random as rand, argmin, argmax, mean, asarray, cos

# from NiaPy.algorithms.algorithm import Individual
from NiaPy.algorithms.basic.de import DifferentialEvolution  # , CrossBest1, CrossRand1, CrossCurr2Best1, CrossBest2, CrossCurr2Rand1, proportional

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = [
	'StrategyAdaptationDifferentialEvolution',
	'StrategyAdaptationDifferentialEvolutionV1'
]

class StrategyAdaptationDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Differential Evolution Algorithm With Strategy Adaptation algorihtm.

	Algorithm:
		Differential Evolution Algorithm With StrategyAdaptation

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://ieeexplore.ieee.org/document/1554904

	Reference paper:
		Qin, A. Kai, and Ponnuthurai N. Suganthan. "Self-adaptive differential evolution algorithm for numerical optimization." 2005 IEEE congress on evolutionary computation. Vol. 2. IEEE, 2005.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		:class:`NiaPy.algorithms.basic.DifferentialEvolution`
	"""
	Name = ['StrategyAdaptationDifferentialEvolution', 'SADE', 'SaDE']

	@staticmethod
	def algorithmInfo():
		r"""Geg basic algorithm information.

		Returns:
			str: Basic algorithm information.

		See Also:
			:func:`NiaPy.algorithms.algorithm.Algorithm.algorithmInfo`
		"""
		return r"""Qin, A. Kai, and Ponnuthurai N. Suganthan. "Self-adaptive differential evolution algorithm for numerical optimization." 2005 IEEE congress on evolutionary computation. Vol. 2. IEEE, 2005."""

	def setParameters(self, **kwargs):
		DifferentialEvolution.setParameters(self, **kwargs)
		# TODO add parameters of the algorithm

	def getParameters(self):
		d = DifferentialEvolution.getParameters(self)
		# TODO add paramters values
		return d

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		# TODO implemnt algorithm
		return pop, fpop, xb, fxb, dparams

class StrategyAdaptationDifferentialEvolutionV1(DifferentialEvolution):
	r"""Implementation of Differential Evolution Algorithm With Strategy Adaptation algorihtm.

	Algorithm:
		Differential Evolution Algorithm With StrategyAdaptation

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://ieeexplore.ieee.org/document/4632146

	Reference paper:
		Qin, A. Kai, Vicky Ling Huang, and Ponnuthurai N. Suganthan. "Differential evolution algorithm with strategy adaptation for global numerical optimization." IEEE transactions on Evolutionary Computation 13.2 (2009): 398-417.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		:class:`NiaPy.algorithms.basic.DifferentialEvolution`
	"""
	Name = ['StrategyAdaptationDifferentialEvolutionV1', 'SADEV1', 'SaDEV1']

	@staticmethod
	def algorithmInfo():
		r"""Get algorithm information.

		Returns:
			str: Get algorithm information.

		See Also:
			:func:`NiaPy.algorithms.algorithm.Algorithm.algorithmInfo`
		"""
		return r"""Qin, A. Kai, Vicky Ling Huang, and Ponnuthurai N. Suganthan. "Differential evolution algorithm with strategy adaptation for global numerical optimization." IEEE transactions on Evolutionary Computation 13.2 (2009): 398-417."""

	def setParameters(self, **kwargs):
		DifferentialEvolution.setParameters(self, **kwargs)
		# TODO add parameters of the algorithm

	def getParameters(self):
		d = DifferentialEvolution.getParameters(self)
		# TODO add paramters values
		return d

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		# TODO implement algorithm
		return pop, fpop, xb, fxb, dparams

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
