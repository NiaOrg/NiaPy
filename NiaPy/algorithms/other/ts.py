# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, unused-argument, arguments-differ, bad-continuation, singleton-comparison, no-self-use, unused-variable
import logging
from numpy import random as rand
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['TabuSearch']

def TabuSearchF(task, SR=None, TL_size=25, rnd=rand):
	if SR == None: SR = task.bRange
	x, TL = rnd.uniform(task.Lower, task.Upper), list()
	x_f = task.eval(x)
	# while not task.stopCondI():
	# Generate neigours
	# evaluate x not in ts
	# get best of of evaluated
	# compare new best with best
	return x, x_f

class TabuSearch(Algorithm):
	r"""Implementation of Tabu Search Algorithm.

	**Algorithm:** Tabu Search Algorithm

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:** http://www.cleveralgorithms.com/nature-inspired/stochastic/tabu_search.html

	**Reference paper:**
	"""
	Name = ['TabuSearch', 'TS']

	@staticmethod
	def typeParameters(): return {
			'NP': lambda x: isinstance(x, int) and x > 0
	}

	def setParameters(self, **ukwargs):
		r"""Set the algorithm parameters/arguments."""
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def move(self): return list()

	def runTask(self, task): return TabuSearchF(task, rnd=self.Rand)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
