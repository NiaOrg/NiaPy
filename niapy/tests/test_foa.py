# encoding=utf8
from niapy.algorithms.basic import ForestOptimizationAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class FOATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = ForestOptimizationAlgorithm

    def test_type_parameters(self):
        tp = self.algo.type_parameters()
        self.assertTrue(tp['population_size'](1))
        self.assertFalse(tp['population_size'](0))
        self.assertFalse(tp['population_size'](-1))
        self.assertFalse(tp['population_size'](1.0))
        self.assertTrue(tp['lifetime'](1))
        self.assertFalse(tp['lifetime'](0))
        self.assertFalse(tp['lifetime'](-1))
        self.assertFalse(tp['lifetime'](1.0))
        self.assertTrue(tp['area_limit'](1))
        self.assertFalse(tp['area_limit'](0))
        self.assertFalse(tp['area_limit'](-1))
        self.assertFalse(tp['area_limit'](1.0))
        self.assertTrue(tp['local_seeding_changes'](1))
        self.assertFalse(tp['local_seeding_changes'](0))
        self.assertFalse(tp['local_seeding_changes'](-1))
        self.assertFalse(tp['local_seeding_changes'](1.0))
        self.assertTrue(tp['global_seeding_changes'](1))
        self.assertFalse(tp['global_seeding_changes'](0))
        self.assertFalse(tp['global_seeding_changes'](-1))
        self.assertFalse(tp['global_seeding_changes'](1.0))
        self.assertTrue(tp['transfer_rate'](1.0))
        self.assertTrue(tp['transfer_rate'](0.5))
        self.assertTrue(tp['transfer_rate'](0.0))
        self.assertFalse(tp['transfer_rate'](-1))
        self.assertFalse(tp['transfer_rate'](1.1))

    def test(self):
        foa = self.algo(population_size=20, lifetime=5, local_seeding_changes=1, global_seeding_changes=1, area_limit=20, transfer_rate=0.35, seed=self.seed)
        foac = self.algo(population_size=20, lifetime=5, local_seeding_changes=1, global_seeding_changes=1, area_limit=20, transfer_rate=0.35, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, foa, foac, MyBenchmark())

    def test_griewank(self):
        foa_griewank = self.algo(population_size=20, lifetime=5, local_seeding_changes=1, global_seeding_changes=1, area_limit=20, transfer_rate=0.35, seed=self.seed)
        foa_griewankc = self.algo(population_size=20, lifetime=5, local_seeding_changes=1, global_seeding_changes=1, area_limit=20, transfer_rate=0.35, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, foa_griewank, foa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
