# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, unused-argument, arguments-differ
import logging
from numpy import random as rand, full
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['TabuSearch']

def TabuSearchF(TL_size=25, task, rnd=rand):
	x, TL  = rnd.uniform(task.Lower, task.Upper), list()
	x_f = task.eval(x)
	while not task.stopCond():
		# Generate neigours
		# get best neignour
		pass
	return None, None

class TabuSearch(Algorithm):
	r"""Implementation of Tabu Search Algorithm.

	**Algorithm:** Tabu Search Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** http://www.cleveralgorithms.com/nature-inspired/stochastic/tabu_search.html
	**Reference paper:**
	"""
	def __init__(self, **kwargs): Algorithm.__init__(self, name='TabuSearch', sName='TS', **kwargs)

	def setParameters(self, **ukwargs):
		r"""Set the algorithm parameters/arguments."""
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def move(self): return list()

	def runTask(self, task):
		return None, None

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
