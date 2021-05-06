# encoding=utf8

from niapy.algorithms.basic import BatAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class BATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = BatAlgorithm

    def test_parameter_type(self):
        d = self.algo.type_parameters()
        self.assertTrue(d['max_frequency'](10))
        self.assertTrue(d['min_frequency'](10))
        self.assertTrue(d['pulse_rate'](10))
        self.assertFalse(d['pulse_rate'](-10))
        self.assertFalse(d['pulse_rate'](0))
        self.assertFalse(d['loudness'](0))
        self.assertFalse(d['loudness'](-19))
        self.assertTrue(d['population_size'](10))
        self.assertFalse(d['population_size'](-10))
        self.assertFalse(d['population_size'](0))
        self.assertTrue(d['loudness'](10))
        self.assertFalse(d['min_frequency'](None))
        self.assertFalse(d['max_frequency'](None))

    def test_custom(self):
        ba_custom = self.algo(population_size=20, loudness=0.5, pulse_rate=0.5, min_frequency=0.0, max_frequency=2.0, seed=self.seed)
        ba_customc = self.algo(population_size=20, loudness=0.5, pulse_rate=0.5, min_frequency=0.0, max_frequency=2.0, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ba_custom, ba_customc, MyBenchmark())

    def test_griewank(self):
        ba_griewank = self.algo(population_size=10, loudness=0.5, pulse_rate=0.5, min_frequency=0.0, max_frequency=2.0, seed=self.seed)
        ba_griewankc = self.algo(population_size=10, loudness=0.5, pulse_rate=0.5, min_frequency=0.0, max_frequency=2.0, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ba_griewank, ba_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
