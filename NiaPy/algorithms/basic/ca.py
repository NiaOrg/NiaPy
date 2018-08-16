# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, singleton-comparison, multiple-statements, attribute-defined-outside-init, no-self-use, logging-not-lazy, unused-variable, arguments-differ, old-style-class
import logging
from numpy import vectorize, argmin, exp
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CamelAlgorithm']

class Camel:
	r"""Implementation of population individual that is a camel for Camel algorithm.

	**Algorithm:** Camel algorithm

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT
	"""
	E_init, S_init = 1, 1
	T_min, T_max = -1, 1

	def __init__(self, x, rand):
		self.E, self.E_past = Camel.E_init, Camel.E_init
		self.S, self.S_past = Camel.S_init, Camel.S_init
		self.x, self.x_past = x, x
		self.rand = rand
		self.steps = 0

	def nextT(self): self.T = (Camel.T_max - Camel.T_min) * self.rand() + Camel.T_min

	def nextS(self, omega, n_gens): self.S = self.S_past * (1 - omega * self.steps / n_gens)

	def nextE(self, n_gens): self.E = self.E_past * (1 - self.T / Camel.T_max) * (1 - self.steps / n_gens)

	def nextX(self, x_best, task):
		delta = -1 + self.rand() * 2
		self.x = self.x_past + delta * (1 - (self.E / Camel.E_init)) * exp(1 - self.S / Camel.S_init) * (x_best - self.x_past)
		if not task.isFeasible(self.x) and task.stopCond():
			self.x = self.x_past
			return False
		return True

	def next(self):
		self.x_past, self.E_past, self.S_past = self.x, self.E, self.S
		self.steps += 1

	def refill(self, S=None, E=None):
		self.S = Camel.S_init if S == None else S
		self.E = Camel.E_init if E == None else E

	def __getitem__(self, i): return self.x[i]

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
	def __init__(self, **kwargs): Algorithm.__init__(self, name='CamelAlgorithm', sName='CA', **kwargs)

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

	def walk(self, c, fit, task, omega, c_best):
		c.nextT()
		c.nextS(omega, task.nGEN)
		c.nextE(task.nGEN)
		if c.nextX(c_best.x, task):
			c.next()
			return c, task.eval(c.x)
		return c, fit

	def oasis(self, c, rn, fit, fitn, alpha):
		if rn > 1 - alpha and fit < fitn: c.refill(Camel.S_init, Camel.E_init)
		return c

	def lifeCycle(self, c, fit, fitn, mu, task):
		if fit < mu * fitn:
			cn = Camel(c.rand(task.D) * task.bRange, c.rand)
			return cn, task.eval(cn.x)
		return c, fitn

	def runTask(self, task):
		Camel.E_init, Camel.S_init = self.E_init, self.S_init
		ccaravan = [Camel(self.uniform(task.Lower, task.Upper, [task.D]), self.Rand.rand) for i in range(self.NP)]
		c_fits = [task.eval(c.x) for c in ccaravan]
		ic_b = argmin(c_fits)
		c_best, c_best_fit = ccaravan[ic_b], c_fits[ic_b]
		while not task.stopCondI():
			ccaravan, c_fitsn = vectorize(self.walk)(ccaravan, c_fits, task, self.omega, c_best)
			ccaravan = vectorize(self.oasis)(ccaravan, self.rand(self.NP), c_fits, c_fitsn, self.alpha)
			ci_b = argmin(c_fitsn)
			if c_fitsn[ci_b] < c_best_fit: c_best, c_best_fit = ccaravan[ci_b], c_fits[ci_b]
			ccaravan, c_fits = vectorize(self.lifeCycle)(ccaravan, c_fits, c_fitsn, self.mu, task)
		return c_best.x, c_best_fit

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
