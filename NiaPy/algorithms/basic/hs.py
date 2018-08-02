# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy
import logging
from numpy import apply_along_axis, argmin
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['HarmonySearch']

class HarmonySearch(Algorithm):
	r"""Implementation of harmony search algorithm.

	**Algorithm:** Harmony Search Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:**
	https://www.sciencedirect.com/science/article/pii/S1568494617306609
	**Reference paper:**
	Junzhi Li, Ying Tan, The bare bones fireworks algorithm: A minimalist global optimizer, Applied Soft Computing, Volume 62, 2018, Pages 454-462, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2017.10.046.
	"""
	def __init__(self, **kwargs): Algorithm.__init__(self, name='HarmonySearch', sName='HS', **kwargs)

	def setParameters(self, **kwargs):
		pass

	def runTask(self, task):
		pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
