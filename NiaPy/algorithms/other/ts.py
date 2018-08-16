# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, unused-argument, arguments-differ
import logging
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['TabuSearch']

class TabuSearch(Algorithm):
	r"""Implementation of Tabu Search Algorithm.

	**Algorithm:** Tabu Search Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs): Algorithm.__init__(self, name='TabuSearch', sName='TS', **kwargs)

	def setParameters(self, **ukwargs):
		r"""Set the algorithm parameters/arguments."""
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		return None, None

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
