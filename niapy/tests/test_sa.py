# encoding=utf8

from niapy.algorithms.other import SimulatedAnnealing
from niapy.algorithms.other.sa import cool_linear
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class SATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = SimulatedAnnealing

    def test_type_parameters(self):
        d = self.algo.type_parameters()
        self.assertTrue(d['delta'](1))
        self.assertFalse(d['delta'](0))
        self.assertFalse(d['delta'](-1))
        self.assertTrue(d['starting_temperature'](1))
        self.assertFalse(d['starting_temperature'](0))
        self.assertFalse(d['starting_temperature'](-1))
        self.assertTrue(d['delta_temperature'](1))
        self.assertFalse(d['delta_temperature'](0))
        self.assertFalse(d['delta_temperature'](-1))
        self.assertTrue(d['epsilon'](0.1))
        self.assertFalse(d['epsilon'](-0.1))
        self.assertFalse(d['epsilon'](10))

    def test_custom(self):
        ca_custom = self.algo(NP=40, seed=self.seed)
        ca_customc = self.algo(NP=40, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyBenchmark())

    def test_griewank(self):
        ca_griewank = self.algo(NP=40, seed=self.seed)
        ca_griewankc = self.algo(NP=40, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc)

    def test_custom1(self):
        ca_custom = self.algo(NP=40, seed=self.seed, coolingMethod=cool_linear)
        ca_customc = self.algo(NP=40, seed=self.seed, coolingMethod=cool_linear)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyBenchmark())

    def test_griewank1(self):
        ca_griewank = self.algo(NP=40, seed=self.seed, coolingMethod=cool_linear)
        ca_griewankc = self.algo(NP=40, seed=self.seed, coolingMethod=cool_linear)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
