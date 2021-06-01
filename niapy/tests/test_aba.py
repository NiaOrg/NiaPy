# pylint: disable=line-too-long
from niapy.algorithms.modified import AdaptiveBatAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class ABATestCase(AlgorithmTestCase):
    r"""Test case for AdaptiveBatAlgorithm algorithm.

    Date:
        April 2019

    Author:
        Klemen Berkovic

    See Also:
        * :class:`niapy.algorithms.modified.AdaptiveBatAlgorithm`
    """

    def test_custom(self):
        aba_custom = AdaptiveBatAlgorithm(population_size=10, A=.75, epsilon=2, alpha=0.65, r=0.7, Qmin=0.0, Qmax=2.0,
                                          seed=self.seed)
        aba_customc = AdaptiveBatAlgorithm(population_size=10, A=.75, epsilon=2, alpha=0.65, r=0.7, Qmin=0.0, Qmax=2.0,
                                           seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aba_custom, aba_customc, MyProblem())

    def test_griewank(self):
        aba_griewank = AdaptiveBatAlgorithm(population_size=10, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
        aba_griewankc = AdaptiveBatAlgorithm(population_size=10, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aba_griewank, aba_griewankc)
