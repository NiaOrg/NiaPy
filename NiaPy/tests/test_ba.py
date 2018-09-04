# pylint: disable=old-style-class, line-too-long
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import BatAlgorithm


class BATestCase(AlgorithmTestCase):

    def test_custom_works_fine(self):
        ba_custom = BatAlgorithm(D=self.D, NP=20, nFES=self.nFES, nGEN=self.nGEN, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, benchmark=MyBenchmark(), seed=self.seed)
        ba_customc = BatAlgorithm(D=self.D, NP=20, nFES=self.nFES, nGEN=self.nGEN, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, benchmark=MyBenchmark(), seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, ba_custom, ba_customc)

    def test_griewank_works_fine(self):
        ba_griewank = BatAlgorithm(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, benchmark='griewank', seed=self.seed)
        ba_griewankc = BatAlgorithm(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, benchmark='griewank', seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, ba_griewank, ba_griewankc)
