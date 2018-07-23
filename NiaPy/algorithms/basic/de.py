# encoding=utf8
import copy
import logging
from numpy import vectorize, argmin
from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['DifferentialEvolutionAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

def Rand1(pop, ic, f, cr):
	# TODO
	return None

def Best1(pop, ic, f, cr):
	# TODO
	return None

def Rand2(pop, ic, f, cr):
	# TODO
	return None

def Best2(pop, ic, f, cr):
	# TODO
	return None

def Curr2Rand1(pop, ic, f, cr):
	# TODO
	return None

def Curr2Best1(pop, ic, f, cr):
	# TODO
	return None

class SolutionDE(object):
	def __init__(self, task):
		self.Solution = []
		self.Fitness = float('inf')
		self.generateSolution(task)

	def generateSolution(self, task): self.Solution = task.Lower +  task.bRange * rnd.rand(task.D)

	def evaluate(self, task): self.Fitness = task.eval(self.Solution)

	def repair(self, task):
		ir = where(self.Solution > task.Upper)
		self.Solution[ir] = task.Upper[ir]
		ir = where(self.Solution < task.Lower)
		self.Solution[ir] = task.Lower[ir]

	def __eq__(self, other): return self.Solution == other.Solution and self.Fitness == other.Fitness

class DifferentialEvolutionAlgorithm(Algorithm):
	r"""Implementation of Differential evolution algorithm.

	**Algorithm:** Differential evolution algorithm

	**Date:** 2018

	**Author:** Uros Mlakar and Klemen Berkoivč

	**License:** MIT

	**Reference paper:**
	Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and
	efficient heuristic for global optimization over continuous spaces."
	Journal of global optimization 11.4 (1997): 341-359.
	"""

	def __init__(self, **kwargs):
		r"""**__init__(self, D, NP, nFES, F, CR, benchmark)**.

				Raises:
		TypeError -- Raised when given benchmark function which does not exists.
		"""
		super(DifferentialEvolutionAlgorithm, self).__init__(name='DifferentialEvolutionAlgorithm', sName='DE', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP=25, F=2, CR=0.2, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
		NP {integer} -- population size
		F {decimal} -- scaling factor
		CR {decimal} -- crossover rate
		Mutation {function} -- mutation stratgy
		"""
		self.Np = NP  # population size
		self.F = F  # scaling factor
		self.CR = CR  # crossover rate
		self.Mutation
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def evalPopulation(self, x, x_old, task):
		"""Evaluate element."""
		x.evaluate(task)
		return x if x.Fitness < x_old.Fitness else x_old

	def runTask(self, taks):
		"""Run."""
		pop = [SolutionDE(task) for _i in range(self.Np)]
		pop = vectorize(self.evalPopulation)(pop, pop, task)
		ix_b = argmin([x.Fitness for x in pop])
		x_best = pop[ix_best]
		while not task.stopCond():
			# FIXME popravi mutacijo
			npop = [self.Mutation(pop, i, self.F, self.CR) for i in range(self.Np)]
			npop = [x.repair() for x in npop]
			pop = vectorize(self.evalPopulation)(npop, pop, task)
			ix_b = argmin([x.Fitness for x in pop])
			if x_best.Fitness < pop[ix_b].Fitness: x_best = pop[ix_best]
		return x_best.Solution, x_best.Fitness

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
