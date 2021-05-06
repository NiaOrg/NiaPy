# encoding=utf8
# from unittest import skip
#
# from niapy.algorithms.modified import StrategyAdaptationDifferentialEvolution, StrategyAdaptationDifferentialEvolutionV1
# from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
#
#
# class SADETestCase(AlgorithmTestCase):
#     def setUp(self):
#         AlgorithmTestCase.setUp(self)
#         self.algo = StrategyAdaptationDifferentialEvolution
#
#     @skip('Not implemented yet!')
#     def test_custom(self):
#         sade_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
#         sade_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
#         AlgorithmTestCase.test_algorithm_run(self, sade_custom, sade_customc, MyBenchmark())
#
#     @skip('Not implemented yet!')
#     def test_griewank(self):
#         sade_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
#         sade_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
#         AlgorithmTestCase.test_algorithm_run(self, sade_griewank, sade_griewankc)
#
#
# class SADEv1TestCase(AlgorithmTestCase):
#     def setUp(self):
#         AlgorithmTestCase.setUp(self)
#         self.algo = StrategyAdaptationDifferentialEvolutionV1
#
#     @skip('Not implemented yet!')
#     def test_custom(self):
#         sadev1_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
#         sadev1_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
#         AlgorithmTestCase.test_algorithm_run(self, sadev1_custom, sadev1_customc, MyBenchmark())
#
#     @skip('Not implemented yet!')
#     def test_griewank(self):
#         sadev1_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
#         sadev1_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
#         AlgorithmTestCase.test_algorithm_run(self, sadev1_griewank, sadev1_griewankc)
