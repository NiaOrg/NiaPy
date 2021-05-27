# encoding=utf8
from niapy.algorithms.modified import HybridSelfAdaptiveBatAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


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

    def test_custom(self):
        """Test case for running algorithm on custom problem."""
        hsaba_custom = self.algo(population_size=10, Limit=2, seed=self.seed)
        hsaba_customc = self.algo(population_size=10, Limit=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hsaba_custom, hsaba_customc, MyProblem())

    def test_griewank(self):
        """Test case for running algorithm on griewank problem."""
        hsaba_griewank = self.algo(population_size=10, seed=self.seed)
        hsaba_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hsaba_griewank, hsaba_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
