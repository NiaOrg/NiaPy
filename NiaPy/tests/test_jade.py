# encoding=utf8
from unittest import TestCase

import numpy as np

from NiaPy.util import fullArray
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.modified import AdaptiveArchiveDifferentialEvolution, CrossRandCurr2Pbest

class CrossRandCurr2pbestTestCase(TestCase):
	def setUp(self):
		self.D, self.NP, self.F, self.CR, self.p = 10, 100, 0.5, 0.5, 0.25
		self.Upper, self.Lower = fullArray(100, self.D), fullArray(-100, self.D)
		self.evalFun = MyBenchmark().function()

	def init_pop(self):
		pop = self.Lower + np.random.rand(self.NP, self.D) * (self.Upper - self.Lower)
		return pop, np.asarray([self.evalFun(self.D, x) for x in pop])

	def test_function_fine(self):
		pop, fpop = self.init_pop()
		apop, _ = self.init_pop()
		ib = np.argmin(fpop)
		xb, fxb = pop[ib].copy(), fpop[ib]
		for i, x in enumerate(pop):
			xn = CrossRandCurr2Pbest(pop, i, xb, self.F, self.CR, self.p, apop)
			self.assertFalse(np.array_equal(x, xn))

class JADETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = AdaptiveArchiveDifferentialEvolution

	def test_custom_works_fine(self):
		jade_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		jade_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jade_custom, jade_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		jade_griewank = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		jade_griewankc = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jade_griewank, jade_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
