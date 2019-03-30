# pylint: disable=line-too-long
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class PSOTestCase(AlgorithmTestCase):

    def test_parameter_type(self):
        d = ParticleSwarmAlgorithm.typeParameters()
        self.assertTrue(d['C1'](10))
        self.assertTrue(d['C2'](10))
        self.assertTrue(d['C1'](0))
        self.assertTrue(d['C2'](0))
        self.assertFalse(d['C1'](-10))
        self.assertFalse(d['C2'](-10))
        self.assertTrue(d['vMax'](10))
        self.assertTrue(d['vMin'](10))
        self.assertTrue(d['NP'](10))
        self.assertFalse(d['NP'](-10))
        self.assertFalse(d['NP'](0))
        self.assertFalse(d['vMin'](None))
        self.assertFalse(d['vMax'](None))

    def test_custom_works_fine(self):
        pso_custom = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
        pso_customc = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, pso_custom, pso_customc, MyBenchmark())

    def test_griewank_works_fine(self):
        pso_griewank = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
        pso_griewankc = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, pso_griewank, pso_griewankc)
