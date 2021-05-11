# encoding=utf8

from niapy.algorithms.basic import BacterialForagingOptimizationAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class BFOATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = BacterialForagingOptimizationAlgorithm

    def test_custom(self):
        bfoa_custom = self.algo(population_size=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1,
                                seed=self.seed)
        bfoa_customc = self.algo(population_size=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1,
                                 seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, bfoa_custom, bfoa_customc, MyBenchmark())

    def test_griewank(self):
        bfoa_custom = self.algo(population_size=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1,
                                seed=self.seed)
        bfoa_customc = self.algo(population_size=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1,
                                 seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, bfoa_custom, bfoa_customc)

    def test_griewank_nfes(self):
        bfoa_custom = self.algo(population_size=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1,
                                seed=self.seed)
        bfoa_customc = self.algo(population_size=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1,
                                 seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, bfoa_custom, bfoa_customc, max_evals=10000)

    def test_griewank_ngen(self):
        bfoa_custom = self.algo(population_size=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1,
                                seed=self.seed)
        bfoa_customc = self.algo(population_size=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1,
                                 seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, bfoa_custom, bfoa_customc, max_iters=10000)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
