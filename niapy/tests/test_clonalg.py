# encoding=utf8
from niapy.algorithms.basic import ClonalSelectionAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class ClonalgTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = ClonalSelectionAlgorithm

    def test_custom(self):
        clonalg_custom = self.algo(population_size=10, seed=self.seed)
        clonalg_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, clonalg_custom, clonalg_customc, MyProblem())

    def test_griewank(self):
        clonalg_griewank = self.algo(population_size=10, seed=self.seed)
        clonalg_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, clonalg_griewank, clonalg_griewankc)
