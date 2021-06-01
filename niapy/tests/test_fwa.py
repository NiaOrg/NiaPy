# encoding=utf8

from niapy.algorithms.basic import BareBonesFireworksAlgorithm, FireworksAlgorithm, EnhancedFireworksAlgorithm, \
    DynamicFireworksAlgorithm, DynamicFireworksAlgorithmGauss
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class BBFWATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = BareBonesFireworksAlgorithm

    def test_custom(self):
        bbfwa_custom = self.algo(num_sparks=10, amplification_coefficient=2, reduction_coefficient=0.5, seed=self.seed)
        bbfwa_customc = self.algo(num_sparks=10, amplification_coefficient=2, reduction_coefficient=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, bbfwa_custom, bbfwa_customc, MyProblem())

    def test_griewank(self):
        bbfwa_griewank = self.algo(num_sparks=10, amplification_coefficient=5, reduction_coefficient=0.5, seed=self.seed)
        bbfwa_griewankc = self.algo(num_sparks=10, amplification_coefficient=5, reduction_coefficient=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, bbfwa_griewank, bbfwa_griewankc)


class FWATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = FireworksAlgorithm

    def test_custom(self):
        fwa_custom = self.algo(population_size=10, seed=self.seed)
        fwa_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fwa_custom, fwa_customc, MyProblem())

    def test_griewank(self):
        fwa_griewank = self.algo(population_size=10, seed=self.seed)
        fwa_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fwa_griewank, fwa_griewankc)


class EFWATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = EnhancedFireworksAlgorithm

    def test_custom(self):
        fwa_custom = self.algo(population_size=10, seed=self.seed)
        fwa_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fwa_custom, fwa_customc, MyProblem(), max_evals=12345, max_iters=17)

    def test_griewank(self):
        fwa_griewank = self.algo(population_size=10, seed=self.seed)
        fwa_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fwa_griewank, fwa_griewankc)


class DFWATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = DynamicFireworksAlgorithm

    def test_custom(self):
        fwa_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        fwa_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fwa_custom, fwa_customc, MyProblem())

    def test_griewank(self):
        fwa_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        fwa_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fwa_griewank, fwa_griewankc)


class DFWAGTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = DynamicFireworksAlgorithmGauss

    def test_custom(self):
        fwa_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        fwa_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fwa_custom, fwa_customc, MyProblem())

    def test_griewank(self):
        fwa_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        fwa_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fwa_griewank, fwa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
