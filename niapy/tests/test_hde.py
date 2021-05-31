# encoding=utf8
from niapy.algorithms.modified import DifferentialEvolutionMTS, DifferentialEvolutionMTSv1, \
    MultiStrategyDifferentialEvolutionMTS, MultiStrategyDifferentialEvolutionMTSv1, \
    DynNpMultiStrategyDifferentialEvolutionMTS, DynNpMultiStrategyDifferentialEvolutionMTSv1, \
    DynNpDifferentialEvolutionMTS, DynNpDifferentialEvolutionMTSv1
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class DEMTSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = DifferentialEvolutionMTS

    def test_custom(self):
        ca_custom = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        ca_customc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyProblem(), max_iters=100)

    def test_griewank(self):
        ca_griewank = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        ca_griewankc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc, max_iters=100)


class DEMTSv1TestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = DifferentialEvolutionMTSv1

    def test_custom(self):
        ca_custom = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        ca_customc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyProblem(), max_iters=100)

    def test_griewank(self):
        ca_griewank = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        ca_griewankc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc, max_iters=100)


class DynNpDEMTSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = DynNpDifferentialEvolutionMTS

    def test_custom(self):
        ca_custom = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, p_max=10, rp=3, seed=self.seed)
        ca_customc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, p_max=10, rp=3, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyProblem())

    def test_griewank(self):
        ca_griewank = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, p_max=10, rp=3, seed=self.seed)
        ca_griewankc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, p_max=10, rp=3, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc, max_iters=100)


class DynNpDEMTSv1TestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = DynNpDifferentialEvolutionMTSv1

    def test_custom(self):
        ca_custom = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, p_max=10, rp=3, seed=self.seed)
        ca_customc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, p_max=10, rp=3, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyProblem(), max_iters=100)

    def test_griewank(self):
        ca_griewank = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, p_max=10, rp=3, seed=self.seed)
        ca_griewankc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, p_max=10, rp=3, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc, max_iters=100)


class MSDEMTSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MultiStrategyDifferentialEvolutionMTS

    def test_custom(self):
        ca_custom = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        ca_customc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyProblem())

    def test_griewank(self):
        ca_griewank = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        ca_griewankc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc, max_iters=100)


class MSDEMTSv1STestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MultiStrategyDifferentialEvolutionMTSv1

    def test_custom(self):
        ca_custom = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        ca_customc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyProblem(), max_iters=100)

    def test_griewank(self):
        ca_griewank = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        ca_griewankc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc, max_iters=100)


class DynNpMSDEMTSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = DynNpMultiStrategyDifferentialEvolutionMTS

    def test_custom(self):
        ca_custom = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        ca_customc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyProblem(), max_iters=100)

    def test_griewank(self):
        ca_griewank = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        ca_griewankc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc, max_iters=100)


class DynNpMSDEMTSv1TestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = DynNpMultiStrategyDifferentialEvolutionMTSv1

    def test_custom(self):
        ca_custom = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        ca_customc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyProblem(), max_iters=100)

    def test_griewank(self):
        ca_griewank = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        ca_griewankc = self.algo(population_size=10, num_tests=1, num_searches=2, num_enabled=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc, max_iters=100)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
