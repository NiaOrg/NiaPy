# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, singleton-comparison, multiple-statements, attribute-defined-outside-init, no-self-use, logging-not-lazy, unused-variable, arguments-differ, bad-continuation
import logging
from numpy import vectorize, argmin, exp
from NiaPy.algorithms.algorithm import Algorithm, Individual

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CamelAlgorithm']

class Camel(Individual):
	r"""Implementation of population individual that is a camel for Camel algorithm.

	**Algorithm:** Camel algorithm

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT
	"""
	E_init, S_init = 1, 1
	T_min, T_max = -1, 1

	def __init__(self, rand, E_init=None, S_init=None, **kwargs):
		Individual.__init__(self, rand=rand, **kwargs)
		self.E, self.E_past = Camel.E_init if E_init is None else E_init, Camel.E_init if E_init is None else E_init
		self.S, self.S_past = Camel.S_init if S_init is None else S_init, Camel.S_init if S_init is None else S_init
		self.x_past, self.f_past = self.x, self.f
		self.rand = rand
		self.steps = 0

	def nextT(self): self.T = (self.T_max - self.T_min) * self.rand.rand() + self.T_min

	def nextS(self, omega, n_gens): self.S = self.S_past * (1 - omega * self.steps / n_gens)

	def nextE(self, n_gens): self.E = self.E_past * (1 - self.T / Camel.T_max) * (1 - self.steps / n_gens)

	def nextX(self, xb, E_init, S_init, task):
		delta = -1 + self.rand.rand() * 2
		self.x = self.x_past + delta * (1 - (self.E / E_init)) * exp(1 - self.S / S_init) * (xb - self.x_past)
		if not task.isFeasible(self.x): self.x = self.x_past
		else: self.f = task.eval(self.x)

	def next(self):
		self.x_past, self.f_past, self.E_past, self.S_past = self.x, self.f, self.E, self.S
		self.steps += 1

	def refill(self, S=None, E=None):
		self.S = Camel.S_init if S == None else S
		self.E = Camel.E_init if E == None else E

class CamelAlgorithm(Algorithm):
	r"""Implementation of Camel traveling behavior.

	**Algorithm:** Camel algorithm

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:**
	https://www.iasj.net/iasj?func=fulltext&aId=118375

	**Reference paper:**
	Ali, Ramzy. (2016). Novel Optimization Algorithm Inspired by Camel Traveling Behavior. Iraq J. Electrical and Electronic Engineering. 12. 167-177.
	"""
	Name = ['CamelAlgorithm', 'CA']

	@staticmethod
	def typeParameters(): return {
			'NP': lambda x: isinstance(x, int) and x > 0,
			'omega': lambda x: isinstance(x, (float, int)),
			'mu': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'alpha': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'S_init': lambda x: isinstance(x, (float, int)) and x > 0,
			'E_init': lambda x: isinstance(x, (float, int)) and x > 0,
			'T_min': lambda x: isinstance(x, (float, int)) and x > 0,
			'T_max': lambda x: isinstance(x, (float, int)) and x > 0
	}

	def setParameters(self, NP=50, omega=0.25, mu=0.5, alpha=0.5, S_init=10, E_init=10, T_min=-10, T_max=10, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments:**

		NP {integer} -- population size $\in [1, \infty)$

		T_min {real} -- minimum temperature, must be true $T_{min} < T_{max}$

		T_max {real} -- maximum temperature, must be true $T_{min} < T_{max}$

		omega {real} -- burden factor $\in [0, 1]$

		mu {real} -- dying rate $\in [0, 1]$

		S_init {real} -- initial supply $\in (0, \infty)$

		E_init {real} -- initial endurance $\in (0, \infty)$
		"""
		self.NP, self.omega, self.mu, self.alpha, self.S_init, self.E_init, self.T_min, self.T_max = NP, omega, mu, alpha, S_init, E_init, T_min, T_max
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def walk(self, c, task, omega, cb):
		c.nextT()
		c.nextS(omega, task.nGEN)
		c.nextE(task.nGEN)
		c.nextX(cb.x, self.E_init, self.S_init, task)
		return c

	def oasis(self, c, rn, alpha):
		if rn > 1 - alpha and c.f < c.f_past: c.refill(self.S_init, self.E_init)
		return c

	def lifeCycle(self, c, mu, task):
		if c.f_past < mu * c.f: return Camel(c.rand, self.E_init, self.S_init, task=task)
		c.next()
		return c

	def initPopulation(self, task):
		pop = [Camel(self.Rand, self.E_init, self.S_init, task=task) for i in range(self.NP)]
		return pop, [x.f for x in pop], {}

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		npop = [self.walk(c, task, self.omega, xb) for c in pop]
		npop = [self.oasis(c, self.rand(), self.alpha) for c in npop]
		npop = [self.lifeCycle(c, self.mu, task) for c in npop]
		return npop, [x.f for x in npop], {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
