# pylint: disable=old-style-class, line-too-long
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.modified import HybridBatAlgorithm


class HBATestCase(AlgorithmTestCase):

    def test_custom_works_fine(self):
        hba_custom = HybridBatAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, benchmark=MyBenchmark(), seed=self.seed)
        hba_customc = HybridBatAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, benchmark=MyBenchmark(), seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, hba_custom, hba_customc)

    def test_griewank_works_fine(self):
        hba_griewank = HybridBatAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, benchmark='griewank', seed=self.seed)
        hba_griewankc = HybridBatAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, benchmark='griewank', seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, hba_griewank, hba_griewankc)
