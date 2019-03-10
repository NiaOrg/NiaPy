# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, logging-not-lazy, attribute-defined-outside-init, line-too-long, arguments-differ, singleton-comparison, bad-continuation, dangerous-default-value, consider-using-enumerate
import logging
from numpy import random as rand, argmin, argmax, mean, asarray, cos
from NiaPy.algorithms.algorithm import Individual
from NiaPy.algorithms.basic.de import DifferentialEvolution, CrossBest1, CrossRand1, CrossCurr2Best1, CrossBest2, CrossCurr2Rand1, proportional

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = [
	'StrategyAdaptationDifferentialEvolution',
	'StrategyAdaptationDifferentialEvolutionV1'
]

class StrategyAdaptationDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Differential Evolution Algorithm With Strategy Adaptation algorihtm.

	**Algorithm:** Differential Evolution Algorithm With StrategyAdaptation

	**Date:** 2018

	**Author:** Klemen Berkovič

	**License:** MIT

	**Reference URL:** https://ieeexplore.ieee.org/document/1554904

	**Reference paper:** Qin, A. Kai, and Ponnuthurai N. Suganthan. "Self-adaptive differential evolution algorithm for numerical optimization." 2005 IEEE congress on evolutionary computation. Vol. 2. IEEE, 2005.
	"""
	Name = ['StrategyAdaptationDifferentialEvolution', 'SADE', 'SaDE']

	def setParameters(self, **kwargs):
		pass

class StrategyAdaptationDifferentialEvolutionV1(DifferentialEvolution):
	r"""Implementation of Differential Evolution Algorithm With Strategy Adaptation algorihtm.

	**Algorithm:** Differential Evolution Algorithm With StrategyAdaptation

	**Date:** 2018

	**Author:** Klemen Berkovič

	**License:** MIT

	**Reference URL:** https://ieeexplore.ieee.org/document/4632146

	**Reference paper:** Qin, A. Kai, Vicky Ling Huang, and Ponnuthurai N. Suganthan. "Differential evolution algorithm with strategy adaptation for global numerical optimization." IEEE transactions on Evolutionary Computation 13.2 (2009): 398-417.
	"""
	Name = ['StrategyAdaptationDifferentialEvolutionV1', 'SADEV1', 'SaDEV1']

	def setParameters(self, **kwargs):
		pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
