# encoding=utf8
from niapy.algorithms.basic import ParticleSwarmOptimization, ParticleSwarmAlgorithm, \
    OppositionVelocityClampingParticleSwarmOptimization, CenterParticleSwarmOptimization, \
    MutatedParticleSwarmOptimization, MutatedCenterParticleSwarmOptimization, \
    ComprehensiveLearningParticleSwarmOptimizer, MutatedCenterUnifiedParticleSwarmOptimization
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class PSOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = ParticleSwarmOptimization

    def test_algorithm_info(self):
        al = self.algo.info()
        self.assertIsNotNone(al)

    def test_custom(self):
        pso_custom = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        pso_customc = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, pso_custom, pso_customc, MyProblem())

    def test_griewank(self):
        pso_griewank = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        pso_griewankc = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, pso_griewank, pso_griewankc)


class PSATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = ParticleSwarmAlgorithm

    def test_algorithm_info(self):
        al = ParticleSwarmAlgorithm.info()
        self.assertIsNotNone(al)

    def test_custom(self):
        wvcpso_custom = ParticleSwarmAlgorithm(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        wvcpso_customc = ParticleSwarmAlgorithm(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, wvcpso_custom, wvcpso_customc, MyProblem())

    def test_griewank(self):
        wvcpso_griewank = ParticleSwarmAlgorithm(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        wvcpso_griewankc = ParticleSwarmAlgorithm(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, wvcpso_griewank, wvcpso_griewankc)


class OVCPSOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = OppositionVelocityClampingParticleSwarmOptimization

    def test_algorithm_info(self):
        al = self.algo.info()
        self.assertIsNotNone(al)

    def test_custom(self):
        wvcpso_custom = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        wvcpso_customc = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, wvcpso_custom, wvcpso_customc, MyProblem())

    def test_griewank(self):
        wvcpso_griewank = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        wvcpso_griewankc = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, wvcpso_griewank, wvcpso_griewankc)


class CPSOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = CenterParticleSwarmOptimization

    def test_algorithm_info(self):
        al = self.algo.info()
        self.assertIsNotNone(al)

    def test_custom(self):
        cpso_custom = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        cpso_customc = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, cpso_custom, cpso_customc, MyProblem())

    def test_griewank(self):
        cpso_griewank = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        cpso_griewankc = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, cpso_griewank, cpso_griewankc)


class MPSOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MutatedParticleSwarmOptimization

    def test_algorithm_info(self):
        al = MutatedParticleSwarmOptimization.info()
        self.assertIsNotNone(al)

    def test_custom(self):
        mpso_custom = MutatedParticleSwarmOptimization(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        mpso_customc = MutatedParticleSwarmOptimization(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mpso_custom, mpso_customc, MyProblem())

    def test_griewank(self):
        mpso_griewank = MutatedParticleSwarmOptimization(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        mpso_griewankc = MutatedParticleSwarmOptimization(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mpso_griewank, mpso_griewankc)


class MCPSOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MutatedCenterParticleSwarmOptimization

    def test_algorithm_info(self):
        al = self.algo.info()
        self.assertIsNotNone(al)

    def test_custom(self):
        mcpso_custom = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        mcpso_customc = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mcpso_custom, mcpso_customc, MyProblem())

    def test_griewank(self):
        mcpso_griewank = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        mcpso_griewankc = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mcpso_griewank, mcpso_griewankc)


class MCUPSOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MutatedCenterUnifiedParticleSwarmOptimization

    def test_algorithm_info(self):
        al = self.algo.info()
        self.assertIsNotNone(al)

    def test_custom(self):
        mcupso_custom = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        mcupso_customc = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mcupso_custom, mcupso_customc, MyProblem())

    def test_griewank(self):
        mcupso_griewank = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        mcupso_griewankc = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mcupso_griewank, mcupso_griewankc)


class CLPSOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = ComprehensiveLearningParticleSwarmOptimizer

    def test_algorithm_info(self):
        al = self.algo.info()
        self.assertIsNotNone(al)

    def test_custom(self):
        clpso_custom = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        clpso_customc = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, clpso_custom, clpso_customc, MyProblem())

    def test_griewank(self):
        clpso_griewank = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        clpso_griewankc = self.algo(population_size=10, c1=2.0, c2=2.0, w=0.7, min_velocity=-4, max_velocity=4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, clpso_griewank, clpso_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
