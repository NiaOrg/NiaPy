# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, unused-argument, no-self-use, no-self-use, attribute-defined-outside-init, logging-not-lazy, len-as-condition, singleton-comparison, arguments-differ
import logging
from numpy import argmin, asarray
from NiaPy.algorithms.algorithm import Algorithm, Individual

__all__ = ['DifferentialEvolutionAlgorithm', 'CrossRand1', 'CrossBest2', 'CrossBest1', 'CrossBest2', 'CrossCurr2Rand1', 'CrossCurr2Best1']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

def CrossRand1(pop, ic, x_b, f, cr, rnd):
	j = rnd.randint(len(pop[0]))
	r = rnd.choice(len(pop), 3, replace=False)
	x = [pop[r[0]][i] + f * (pop[r[1]][i] - pop[r[2]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossBest1(pop, ic, x_b, f, cr, rnd):
	j = rnd.randint(len(pop[0]))
	r = rnd.choice(len(pop), 2, replace=False)
	x = [x_b[i] + f * (pop[r[0]][i] - pop[r[1]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossRand2(pop, ic, x_b, f, cr, rnd):
	j = rnd.randint(len(pop[0]))
	r = rnd.choice(len(pop), 5, replace=False)
	x = [pop[r[0]][i] + f * (pop[r[1]][i] - pop[r[2]][i]) + f * (pop[r[3]][i] - pop[r[4]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossBest2(pop, ic, x_b, f, cr, rnd):
	j = rnd.randint(len(pop[0]))
	r = rnd.choice(len(pop), 4, replace=False)
	x = [x_b[i] + f * (pop[r[0]][i] - pop[r[1]][i]) + f * (pop[r[2]][i] - pop[r[3]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossCurr2Rand1(pop, ic, x_b, f, cr, rnd):
	j = rnd.randint(len(pop[0]))
	r = rnd.choice(len(pop), 4, replace=False)
	x = [pop[ic][i] + f * (pop[r[0]][i] - pop[r[1]][i]) + f * (pop[r[2]][i] - pop[r[3]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossCurr2Best1(pop, ic, x_b, f, cr, rnd):
	j = rnd.randint(len(pop[0]))
	r = rnd.choice(len(pop), 3, replace=False)
	x = [pop[ic][i] + f * (x_b[i] - pop[r[0]][i]) + f * (pop[r[1]][i] - pop[r[2]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

class DifferentialEvolutionAlgorithm(Algorithm):
	r"""Implementation of Differential evolution algorithm.

	**Algorithm:** Differential evolution algorithm

	**Date:** 2018

	**Author:** Uros Mlakar and Klemen Berkovič

	**License:** MIT

	**Reference paper:**
	Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and efficient heuristic for global optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359.
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: Algorithm.__init__(self, name=kwargs.get('name', 'DifferentialEvolutionAlgorithm'), sName=kwargs.get('sName', 'DE'), **kwargs)
		else: Algorithm.__init__(self, **kwargs)

	def setParameters(self, NP=25, F=2, CR=0.2, CrossMutt=CrossRand1, **ukwargs):
		r"""Set the algorithm parameters.

		**Arguments:**

		NP {integer} -- population size

		F {decimal} -- scaling factor

		CR {decimal} -- crossover rate

		CrossMutt {function} -- crossover and mutation strategy
		"""
		self.Np, self.F, self.CR, self.CrossMutt = NP, F, CR, CrossMutt
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def selectBetter(self, x, y): return x if x.f < y.f else y

	def evalPopulation(self, x, x_old, task):
		"""Evaluate element."""
		x.evaluate(task)
		return self.selectBetter(x, x_old)

	def runTask(self, task):
		"""Run."""
		pop = [Individual(task=task) for _i in range(self.Np)]
		pop = [self.evalPopulation(pop[i], pop[i], task) for i in range(self.Np)]
		x_b = pop[argmin([x.f for x in pop])]
		while not task.stopCond():
			npop = [Individual(x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand)) for i in range(self.Np)]
			pop = [self.evalPopulation(npop[i], pop[i], task) for i in range(self.Np)]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
