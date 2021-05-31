# encoding=utf8
from niapy.algorithms.basic import ForestOptimizationAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class FOATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = ForestOptimizationAlgorithm

    def test(self):
        foa = self.algo(population_size=10, lifetime=5, local_seeding_changes=1, global_seeding_changes=1, area_limit=20, transfer_rate=0.35, seed=self.seed)
        foac = self.algo(population_size=10, lifetime=5, local_seeding_changes=1, global_seeding_changes=1, area_limit=20, transfer_rate=0.35, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, foa, foac, MyProblem())

    def test_griewank(self):
        foa_griewank = self.algo(population_size=10, lifetime=5, local_seeding_changes=1, global_seeding_changes=1, area_limit=20, transfer_rate=0.35, seed=self.seed)
        foa_griewankc = self.algo(population_size=10, lifetime=5, local_seeding_changes=1, global_seeding_changes=1, area_limit=20, transfer_rate=0.35, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, foa_griewank, foa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
