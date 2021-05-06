# encoding=utf8
from niapy.algorithms.modified import HybridSelfAdaptiveBatAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class HSABATestCase(AlgorithmTestCase):
    r"""Test case for HybridSelfAdaptiveBatAlgorithm algorithm.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.algorithms.modified.HybridSelfAdaptiveBatAlgorithm`

    """

    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = HybridSelfAdaptiveBatAlgorithm

    def test_algorithm_info(self):
        """Test case for algorithm info."""
        i = self.algo.info()
        self.assertIsNotNone(i)

    def test_type_parameters(self):
        """Test case for type parameters."""
        d = self.algo.type_parameters()
        # Test differential_weight parameter check
        self.assertIsNotNone(d.get('differential_weight', None))
        self.assertFalse(d['differential_weight'](-30))
        self.assertFalse(d['differential_weight'](-.3))
        self.assertTrue(d['differential_weight'](.3))
        self.assertTrue(d['differential_weight'](.39))
        # Test CR parameter check
        self.assertIsNotNone(d.get('crossover_probability', None))
        self.assertFalse(d['crossover_probability'](10))
        self.assertFalse(d['crossover_probability'](-10))
        self.assertFalse(d['crossover_probability'](-1))
        self.assertTrue(d['crossover_probability'](.3))
        self.assertTrue(d['crossover_probability'](.0))
        self.assertTrue(d['crossover_probability'](1.))

    def test_custom(self):
        """Test case for running algorithm on costume benchmarks."""
        hsaba_custom = self.algo(population_size=10, Limit=2, seed=self.seed)
        hsaba_customc = self.algo(population_size=10, Limit=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hsaba_custom, hsaba_customc, MyBenchmark())

    def test_griewank(self):
        """Test case for running algorithm on benchmark."""
        hsaba_griewank = self.algo(population_size=10, seed=self.seed)
        hsaba_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hsaba_griewank, hsaba_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
