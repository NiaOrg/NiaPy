# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, unused-argument, line-too-long, len-as-condition
import logging
from numpy import random as rand, inf, where, argmin, ndarray, asarray, sort
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['GeneticAlgorithm', 'TurnamentSelection', 'BackerSelection', 'LinearSelection', 'NonlinearSelection', 'TwoPointCrossover', 'MultiPointCrossover', 'UniformCrossover', 'UniformMutation', 'CreepMutation']

def TurnamentSelection(pop, ts, rnd=rand):
	comps = [pop[i] for i in rand.choice(len(pop), ts)]
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
	return Chromosome(x=x)

def MultiPointCrossover(pop, ic, n, rnd=rand):
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	r = sort(rnd.choice(len(pop[ic]), 2 * n))
	x = pop[ic].x
	for i in range(n): x[r[2 * i]:r[2 * i + 1]] = pop[io].x[r[2 * i]:r[2 * i + 1]]
	return Chromosome(x=x)

def UniformCrossover(pop, ic, cr, rnd=rand):
	io = ic
	while io != ic: io = rnd.randint(len(pop))
	j = rnd.randint(len(pop[ic]))
	x = [pop[io][i] if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return Chromosome(x=x)

def UniformMutation(pop, ic, cr, task, rnd=rand):
	j = rnd.randint(task.D)
	nx = [rnd.uniform(task.Upper[i], task.Lower[i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(task.D)]
	return Chromosome(x=nx)

def CreepMutation(pop, ic, cr, task, rnd=rand):
	ic, j = rnd.randint(len(pop)), rnd.randint(task.D)
	nx = [rnd.uniform(task.Upper[i], task.Lower[i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(task.D)]
	return Chromosome(x=nx)

class Chromosome(object):
	def __init__(self, **kwargs):
		self.f = inf
		task, rnd, x = kwargs.get('task', None), kwargs.get('rand', rand), kwargs.get('x', [])
		if len(x) > 0: self.x = x if isinstance(x, ndarray) else asarray(x)
		else: self.generateSolution(task, rnd)

	def generateSolution(self, task, rnd):
		self.x = task.Lower + task.bRange * rnd.rand(task.D)
		self.evaluate(task)

	def evaluate(self, task): self.f = task.eval(self.x)

	def repair(self, task):
		ir = where(self.x > task.Upper)
		self.x[ir] = task.Upper[ir]
		ir = where(self.x < task.Lower)
		self.x[ir] = task.Lower[ir]

	def __eq__(self, other): return self.x == other.x and self.f == other.f

	def toString(self): return '%s -> %s' % (self.x, self.f)

	def __getitem__(self, i): return self.x[i]

	def __len__(self): return len(self.x)

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

	def runTask(self, task):
		pop = [Chromosome(task=task, rand=self.rand) for _i in range(self.NP)]
		x_b = pop[0]
		while not task.stopCond():
			npop = [self.Selection(pop, self.Ts, self.rand) for _i in range(self.NP)]
			npop = [self.Crossover(pop, i, self.Cr, self.rand) for i in range(self.NP)]
			pop = [self.Mutation(npop, i, self.Cr, task, self.rand) for i in range(self.NP)]
			for c in pop: c.evaluate(task)
			ix_b = argmin([c.f for c in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
