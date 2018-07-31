# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, unused-argument, singleton-comparison, no-else-return
import logging
from numpy import argsort, argmin, random as rand, full
from NiaPy.algorithms.algorithm import Algorithm, Individual

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['EvolutionStrategy1p1', 'EvolutionStrategyMp1', 'EvolutionStrategyML', 'EvolutionStrategyMpL']

def PlusStrategy(pop_1, pop_2): 
	pop_1.extend(pop_2)
	return pop_1

def MinusStrategy(pop_1, pop_2): return pop_2

class IndividualES(Individual):
	def __init__(self, **kwargs):
		task, x, rho = kwargs.get('task', None), kwargs.get('x', None), kwargs.pop('rho', [])
		if len(rho) > 0 != None: self.rho = rho
		elif task != None: self.rho = full(task.D, 1.0)
		elif x != None: self.rho = full(len(x), 1.0)
		super(IndividualES, self).__init__(**kwargs)

class EvolutionStrategy(Algorithm):
	r"""Implementation of evolution strategy algorithm.

	**Algorithm:** (1 + 1) Evolution Strategy Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: super(EvolutionStrategy, self).__init__(name='EvolutionStrategy', sName='ES', **kwargs)
		else: super(EvolutionStrategy, self).__init__(**kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, mu=40, lam=40, k=10, c_a=1.1, c_r=0.5, Strategy=PlusStrategy, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		mu {integer} --
		lam {integer} --
		k {integer} --
		c_a {real} --
		c_r {real} --
		Strategy {function} -- 
		"""
		self.mu, self.lam, self.k, self.c_a, self.c_r, self.Strategy = mu, lam, k, c_a, c_r, Strategy
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def mutate(self, x, rho): return x + self.rand.normal(0, rho)

	def updateRho(self, rho, k):
		phi = k / self.k
		if phi < 0.2: return self.c_r * rho
		elif phi > 0.2: return self.c_a * rho
		else: return rho

	def makeNewPop(self, pop):
		print (pop)
		npop = list()
		for _i in range(self.lam):
			i = self.rand.randint(self.mu)
			npop.append(IndividualES(x=self.mutate(pop[i].x, pop[i].rho), rho=pop[i].rho))
		return npop

	def runTask(self, task):
		c, ki = [IndividualES(task=task, rand=self.rand) for _i in range(self.mu)], 0
		while not task.stopCondI():
			if task.Iters % self.k == 0:
				for i in self.mu: c[i].rho = self.updateRho(c[i].rho, ki)
				ki = 0
			cn = self.makeNewPop(c)
			cn = self.Strategy(c, cn)
			print ('lol', cn)
			cn_s = argsort([i.f for i in cn])
			if len(cn) < self.mu:
				# FIXME
				print (cn)
			else: c = [cn[i] for i in cn_s]
			# TODO prestej izboljsave
		return c[0].x, c[0].f

class EvolutionStrategy1p1(EvolutionStrategy):
	r"""Implementation of evolution strategy algorithm.

	**Algorithm:** (1 + 1) Evolution Strategy Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs): super(EvolutionStrategy1p1, self).__init__(name='(mu+1)-EvolutionStrategy', sName='(mu+1)-ES', **kwargs)

	def setParameters(self, **kwargs):
		_, _ = kwargs.pop('mu', None), kwargs.pop('lam', None)
		super(EvolutionStrategy1p1, self).setParameters(mu=1, lam=1, **kwargs)

	def __setParams(self, mu=40, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		mu {integer} -- number of parent population
		"""
		self.mu, self.lam = 1, mu
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

class EvolutionStrategyMp1(EvolutionStrategy):
	r"""Implementation of evolution strategy algorithm.

	**Algorithm:** ($\mu$ + 1) Evolution Strategy Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs): super(EvolutionStrategyMp1, self).__init__(name='(mu+1)-EvolutionStrategy', sName='(mu+1)-ES', **kwargs)

	def setParameters(self, **kwargs):
		lam, _ = kwargs.pop('mu', 40), kwargs.pop('lam', 1)
		super(EvolutionStrategyMp1, self).setParameters(mu=1, lam=lam, **kwargs)

class EvolutionStrategyML(EvolutionStrategy):
	r"""Implementation of evolution strategy algorithm.

	**Algorithm:** ($\mu$ + 1) Evolution Strategy Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs): super(EvolutionStrategyML, self).__init__(name='(mu,lambda)-EvolutionStrategy', sName='(mu,lambda)-ES', **kwargs)

	def setParameters(self, **kwargs):
		_ = kwargs.pop('Strategy', None)
		super(EvolutionStrategyML, self).setParameters(Strategy=MinusStrategy, **kwargs)

class EvolutionStrategyMpL(EvolutionStrategy):
	r"""Implementation of evolution strategy algorithm.

	**Algorithm:** ($\mu$ + 1) Evolution Strategy Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs): super(EvolutionStrategyMpL, self).__init__(name='(mu+lambda)-EvolutionStrategy', sName='(mu+lambda)-ES', **kwargs)

	def setParameters(self, **kwargs):
		_ = kwargs.pop('Strategy', None)
		super(EvolutionStrategyMpL, self).setParameters(Strategy=PlusStrategy, **kwargs)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
