# pylint: disable=line-too-long
from niapy.algorithms.modified import AdaptiveBatAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class ABATestCase(AlgorithmTestCase):
    r"""Test case for AdaptiveBatAlgorithm algorithm.

    Date:
        April 2019

    Author:
        Klemen Berkovic

    See Also:
        * :class:`niapy.algorithms.modified.AdaptiveBatAlgorithm`
    """

    def test_algorithm_info(self):
        """Test algorithm info method of class AdaptiveBatAlgorithm."""
        self.assertIsNotNone(AdaptiveBatAlgorithm.info())

    def test_type_parameters(self):
        """Test type parameters method of class AdaptiveBatAlgorithm."""
        d = AdaptiveBatAlgorithm.type_parameters()
        # Test epsilon parameter check
        self.assertIsNotNone(d.get('epsilon', None))
        self.assertFalse(d['epsilon'](-100))
        self.assertFalse(d['epsilon'](-.3))
        self.assertTrue(d['epsilon'](3))
        self.assertTrue(d['epsilon'](.3))
        self.assertTrue(d['epsilon'](300))
        # Test alpha parameter check
        self.assertIsNotNone(d.get('alpha', None))
        self.assertFalse(d['alpha'](-100))
        self.assertFalse(d['alpha'](-.3))
        self.assertTrue(d['alpha'](3))
        self.assertTrue(d['alpha'](.3))
        self.assertTrue(d['alpha'](300))
        # Test r parameter check
        self.assertIsNotNone(d.get('pulse_rate', None))
        self.assertFalse(d['pulse_rate'](-100))
        self.assertFalse(d['pulse_rate'](-.3))
        self.assertTrue(d['pulse_rate'](3))
        self.assertTrue(d['pulse_rate'](.3))
        self.assertTrue(d['pulse_rate'](300))
        # Test Qmin parameter check
        self.assertIsNotNone(d.get('min_frequency', None))
        self.assertTrue(d['min_frequency'](3))
        # Test Qmax parameter check
        self.assertIsNotNone(d.get('max_frequency', None))
        self.assertTrue(d['max_frequency'](300))

    def test_custom(self):
        aba_custom = AdaptiveBatAlgorithm(NP=40, A=.75, epsilon=2, alpha=0.65, r=0.7, Qmin=0.0, Qmax=2.0,
                                          seed=self.seed)
        aba_customc = AdaptiveBatAlgorithm(NP=40, A=.75, epsilon=2, alpha=0.65, r=0.7, Qmin=0.0, Qmax=2.0,
                                           seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aba_custom, aba_customc, MyBenchmark())

    def test_griewank(self):
        aba_griewank = AdaptiveBatAlgorithm(NP=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
        aba_griewankc = AdaptiveBatAlgorithm(NP=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aba_griewank, aba_griewankc)
