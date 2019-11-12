# encoding=utf8
from unittest import TestCase, skip

from numpy import random as rnd

from NiaPy.task.task import Task
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolution, DynNpSelfAdaptiveDifferentialEvolutionAlgorithm, MultiStrategySelfAdaptiveDifferentialEvolution, DynNpMultiStrategySelfAdaptiveDifferentialEvolution
from NiaPy.algorithms.modified.jde import SolutionjDE
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class SolutionjDETestCase(TestCase):
	def setUp(self):
		self.D, self.F, self.CR = 10, 0.9, 0.3
		self.x, self.task = rnd.uniform(10, 50, self.D), Task(self.D, nFES=230, nGEN=None, benchmark=MyBenchmark())
		self.s1, self.s2 = SolutionjDE(task=self.task, e=False), SolutionjDE(x=self.x, CR=self.CR, F=self.F)

	def test_F_fine(self):
		self.assertAlmostEqual(self.s1.F, 2)
		self.assertAlmostEqual(self.s2.F, self.F)

	def test_cr_fine(self):
		self.assertAlmostEqual(self.s1.CR, 0.5)
		self.assertAlmostEqual(self.s2.CR, self.CR)

class jDETestCase(AlgorithmTestCase):
	def test_typeParameters(self):
		d = SelfAdaptiveDifferentialEvolution.typeParameters()
		self.assertTrue(d['F_l'](10))
		self.assertFalse(d['F_l'](-10))
		self.assertFalse(d['F_l'](-0))
		self.assertTrue(d['F_u'](10))
		self.assertFalse(d['F_u'](-10))
		self.assertFalse(d['F_u'](-0))
		self.assertTrue(d['Tao1'](0.32))
		self.assertFalse(d['Tao1'](-1.123))
		self.assertFalse(d['Tao1'](1.123))
		self.assertTrue(d['Tao2'](0.32))
		self.assertFalse(d['Tao2'](-1.123))
		self.assertFalse(d['Tao2'](1.123))

	def test_custom_works_fine(self):
		jde_custom = SelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		jde_customc = SelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jde_custom, jde_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		jde_griewank = SelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		jde_griewankc = SelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jde_griewank, jde_griewankc)

class dyNPjDETestCase(AlgorithmTestCase):
	def test_typeParameters(self):
		d = DynNpSelfAdaptiveDifferentialEvolutionAlgorithm.typeParameters()
		self.assertTrue(d['rp'](10))
		self.assertTrue(d['rp'](10.10))
		self.assertFalse(d['rp'](0))
		self.assertFalse(d['rp'](-10))
		self.assertTrue(d['pmax'](10))
		self.assertFalse(d['pmax'](0))
		self.assertFalse(d['pmax'](-10))
		self.assertFalse(d['pmax'](10.12))

	@skip("Not working")
	def test_custom_works_fine(self):
		dynnpjde_custom = DynNpSelfAdaptiveDifferentialEvolutionAlgorithm(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		dynnpjde_customc = DynNpSelfAdaptiveDifferentialEvolutionAlgorithm(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, dynnpjde_custom, dynnpjde_customc, MyBenchmark())

	@skip("Not working")
	def test_griewank_works_fine(self):
		dynnpjde_griewank = DynNpSelfAdaptiveDifferentialEvolutionAlgorithm(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		dynnpjde_griewankc = DynNpSelfAdaptiveDifferentialEvolutionAlgorithm(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, dynnpjde_griewank, dynnpjde_griewankc)

class MsjDETestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		jde_custom = MultiStrategySelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		jde_customc = MultiStrategySelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jde_custom, jde_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		jde_griewank = MultiStrategySelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		jde_griewankc = MultiStrategySelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jde_griewank, jde_griewankc)

class dynNpMsjDETestCase(AlgorithmTestCase):
	@skip("Not working")
	def test_custom_works_fine(self):
		jde_custom = DynNpMultiStrategySelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		jde_customc = DynNpMultiStrategySelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jde_custom, jde_customc, MyBenchmark())

	@skip("Not working")
	def test_griewank_works_fine(self):
		jde_griewank = DynNpMultiStrategySelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		jde_griewankc = MultiStrategySelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jde_griewank, jde_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
