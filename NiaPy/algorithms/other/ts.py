# encoding=utf8
import logging
from numpy import random as rand
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['TabuSearch']

# TODO implement algorithm

def TabuSearchF(task, SR=None, TL_size=25, rnd=rand):
	if SR == None: SR = task.bRange
	x = rnd.uniform(task.Lower, task.Upper)
	x_f = task.eval(x)
	# while not task.stopCondI():
	# Generate neigours
	# evaluate x not in ts
	# get best of of evaluated
	# compare new best with best
	return x, x_f

class TabuSearch(Algorithm):
	r"""Implementation of Tabu Search Algorithm.

	Algorithm:
		Tabu Search Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		http://www.cleveralgorithms.com/nature-inspired/stochastic/tabu_search.html

	Reference paper:

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
	"""
	Name = ['TabuSearch', 'TS']

	@staticmethod
	def typeParameters(): return {
			'NP': lambda x: isinstance(x, int) and x > 0
	}

	def setParameters(self, **ukwargs):
		r"""Set the algorithm parameters/arguments."""
		Algorithm.setParameters(self, **ukwargs)

	def move(self): return list()

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core function of the algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Individuals fitness/objective values.
			xb (numpy.ndarray): Global best solution.
			fxb (float): Global best solutions fitness/objective value.
			**dparams (dict):

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, dict]:
		"""
		return pop, fpop, xb, fxb, dparams

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
