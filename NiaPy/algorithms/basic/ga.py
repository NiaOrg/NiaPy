# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, unused-argument, line-too-long, len-as-condition, useless-super-delegation
import logging
from numpy import argmin, sort, random as rand, asarray, fmin, fmax
from NiaPy.algorithms.algorithm import Algorithm, Individual

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['GeneticAlgorithm', 'TurnamentSelection', 'BackerSelection', 'LinearSelection', 'NonlinearSelection', 'TwoPointCrossover', 'MultiPointCrossover', 'UniformCrossover', 'UniformMutation', 'CreepMutation']

def TurnamentSelection(pop, ts, rnd=rand):
	comps = [pop[i] for i in rand.choice(len(pop), ts, replace=False)]
	return comps[argmin([c.f for c in comps])]

def BackerSelection(pop, p, rnd=rand):
	pass

def LinearSelection(pop, p, rnd=rand):
	pass

def NonlinearSelection(pop, p, rnd=rand):
	pass

def TwoPointCrossover(pop, ic, cr, rnd=rand):
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	r = sort(rnd.choice(len(pop[ic]), 2))
	x = pop[ic].x
	x[r[0]:r[1]] = pop[io].x[r[0]:r[1]]
	return asarray(x)

def MultiPointCrossover(pop, ic, n, rnd=rand):
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	r, x = sort(rnd.choice(len(pop[ic]), 2 * n)), pop[ic].x
	for i in range(n): x[r[2 * i]:r[2 * i + 1]] = pop[io].x[r[2 * i]:r[2 * i + 1]]
	return asarray(x)

def UniformCrossover(pop, ic, cr, rnd=rand):
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	j = rnd.randint(len(pop[ic]))
	x = [pop[io][i] if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

def CrossoverUros(pop, ic, cr, rnd=rand):
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	alpha = cr + (1 + 2 * cr) * rnd.rand(len(pop[ic]))
	x = alpha * pop[ic] + (1 - alpha) * pop[io] 
	return x

def UniformMutation(pop, ic, mr, task, rnd=rand):
	j = rnd.randint(task.D)
	nx = [rnd.uniform(task.Upper[i], task.Lower[i]) if rnd.rand() < mr or i == j else pop[ic][i] for i in range(task.D)]
	return asarray(nx)

def MutationUros(pop, ic, mr, task, rnd=rand):
	return fmin(fmax(rnd.normal(pop[ic], mr * task.bRange), task.Lower), task.Upper)

def CreepMutation(pop, ic, mr, task, rnd=rand):
	ic, j = rnd.randint(len(pop)), rnd.randint(task.D)
	nx = [rnd.uniform(task.Upper[i], task.Lower[i]) if rnd.rand() < mr or i == j else pop[ic][i] for i in range(task.D)]
	return asarray(nx)

class GeneticAlgorithm(Algorithm):
	r"""Implementation of Genetic algorithm.

	**Algorithm:** Genetic algorithm
	**Date:** 2018
	**Author:** Uros Mlakar and Klemen Berkovič
	**License:** MIT
	"""
	def __init__(self, **kwargs): super(GeneticAlgorithm, self).__init__(name='GeneticAlgorithm', sName='GA', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP=25, Ts=5, Mr=0.25, Cr=0.25, Selection=TurnamentSelection, Crossover=UniformCrossover, Mutation=UniformMutation, **ukwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
		NP {integer} -- population size
		Ts {integer} -- tournament selection
		Mr {decimal} -- mutation rate
		Cr {decimal} -- crossover rate
		"""
		self.NP, self.Ts, self.Mr, self.Cr = NP, Ts, Mr, Cr
		self.Selection, self.Crossover, self.Mutation = Selection, Crossover, Mutation
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def evolve(self, pop, x_b, task):
		npop = list()
		for i in range(self.NP): 
			ind = Individual(x=self.Selection(pop, self.Ts, self.rand), e=False)
			ind.x = self.Crossover(pop, i, self.Cr, self.rand)
			ind.x = self.Mutation(pop, i, self.Mr, task, self.rand)
			ind.evaluate(task)
			npop.append(ind)
			if x_b.f > ind.f: x_b = ind
		return npop, x_b

	def runTask(self, task):
		pop = [Individual(task=task, rand=self.rand) for _i in range(self.NP)]
		x_b = pop[argmin([c.f for c in pop])]
		while not task.stopCond(): pop, x_b = self.evolve(pop, x_b, task)
		return x_b.x, x_b.f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
