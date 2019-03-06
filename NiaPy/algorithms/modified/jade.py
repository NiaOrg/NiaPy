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
	'AdaptiveArchiveDifferentialEvolution'
]

def CrossRandCurr2Pbest(pop, ic, x_b, f, cr, rnd=rand, *args):
	pass

class AdaptiveArchiveDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Dynamic population size with aging self-adaptive differential evolution algorithm.

	**Algorithm:** Adaptive Differential Evolution With Optional External Archive

	**Date:** 2018

	**Author:** Klemen Berkovič

	**License:** MIT

	**Reference URL:** https://ieeexplore.ieee.org/document/5208221

	**Reference paper:** Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: adaptive differential evolution with optional external archive." IEEE Transactions on evolutionary computation 13.5 (2009): 945-958.
	"""
	Name = ['AdaptiveArchiveDifferentialEvolution', 'JADE']

	def setParameters(self, **kwargs):
		pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
