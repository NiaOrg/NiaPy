# encoding=utf8
from niapy.algorithms.basic import CamelAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class CATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = CamelAlgorithm

    def test_type_parameters(self):
        d = self.algo.type_parameters()
        self.assertTrue(d['population_size'](1))
        self.assertFalse(d['population_size'](0))
        self.assertFalse(d['population_size'](-1))
        self.assertTrue(d['burden_factor'](.1))
        self.assertTrue(d['burden_factor'](10))
        self.assertFalse(d['burden_factor'](None))
        self.assertTrue(d['visibility'](.342))
        self.assertTrue(d['death_rate'](.342))
        self.assertTrue(d['burden_factor'](3))
        self.assertTrue(d['burden_factor'](-3))
        self.assertFalse(d['death_rate'](3))
        self.assertFalse(d['death_rate'](-3))
        self.assertFalse(d['supply_init'](-1))
        self.assertFalse(d['endurance_init'](-1))
        self.assertFalse(d['min_temperature'](-1))
        self.assertFalse(d['max_temperature'](-1))
        self.assertTrue(d['supply_init'](10))
        self.assertTrue(d['endurance_init'](10))
        self.assertTrue(d['min_temperature'](10))
        self.assertTrue(d['max_temperature'](10))

    def test_custom(self):
        ca_custom = self.algo(population_size=40, seed=self.seed)
        ca_customc = self.algo(population_size=40, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyBenchmark())

    def test_griewank(self):
        ca_griewank = self.algo(population_size=40, seed=self.seed)
        ca_griewankc = self.algo(population_size=40, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
