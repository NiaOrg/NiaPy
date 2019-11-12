# encoding=utf8
from NiaPy.algorithms.basic import ParticleSwarmOptimization, ParticleSwarmAlgorithm, OppositionVelocityClampingParticleSwarmOptimization, CenterParticleSwarmOptimization, MutatedParticleSwarmOptimization, MutatedCenterParticleSwarmOptimization, ComprehensiveLearningParticleSwarmOptimizer, MutatedCenterUnifiedParticleSwarmOptimization
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class PSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = ParticleSwarmOptimization

	def test_algorithm_info(self):
		al = self.algo.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = self.algo.typeParameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](-10))
		self.assertFalse(d['NP'](0))

	def test_custom_works_fine(self):
		pso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		pso_customc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, pso_custom, pso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		pso_griewank = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		pso_griewankc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, pso_griewank, pso_griewankc)

class PSATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = ParticleSwarmAlgorithm

	def test_algorithm_info(self):
		al = ParticleSwarmAlgorithm.algorithmInfo()
		self.assertIsNotNone(al)

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
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		wvcpso_custom = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		wvcpso_customc = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, wvcpso_custom, wvcpso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		wvcpso_griewank = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		wvcpso_griewankc = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, wvcpso_griewank, wvcpso_griewankc)

class OVCPSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = OppositionVelocityClampingParticleSwarmOptimization

	def test_algorithm_info(self):
		al = self.algo.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = self.algo.typeParameters()
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
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		wvcpso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		wvcpso_customc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, wvcpso_custom, wvcpso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		wvcpso_griewank = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		wvcpso_griewankc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, wvcpso_griewank, wvcpso_griewankc)

class CPSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = CenterParticleSwarmOptimization

	def test_algorithm_info(self):
		al = self.algo.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = self.algo.typeParameters()
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
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		cpso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		cpso_customc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, cpso_custom, cpso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		cpso_griewank = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		cpso_griewankc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, cpso_griewank, cpso_griewankc)

class MPSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MutatedParticleSwarmOptimization

	def test_algorithm_info(self):
		al = MutatedParticleSwarmOptimization.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = MutatedParticleSwarmOptimization.typeParameters()
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
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		mpso_custom = MutatedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mpso_customc = MutatedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mpso_custom, mpso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		mpso_griewank = MutatedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mpso_griewankc = MutatedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mpso_griewank, mpso_griewankc)

class MCPSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MutatedCenterParticleSwarmOptimization

	def test_algorithm_info(self):
		al = self.algo.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = self.algo.typeParameters()
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
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		mcpso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mcpso_customc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mcpso_custom, mcpso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		mcpso_griewank = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mcpso_griewankc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mcpso_griewank, mcpso_griewankc)

class MCUPSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MutatedCenterUnifiedParticleSwarmOptimization

	def test_algorithm_info(self):
		al = self.algo.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = self.algo.typeParameters()
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
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		mcupso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mcupso_customc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mcupso_custom, mcupso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		mcupso_griewank = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mcupso_griewankc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mcupso_griewank, mcupso_griewankc)

class CLPSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = ComprehensiveLearningParticleSwarmOptimizer

	def test_algorithm_info(self):
		al = self.algo.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = self.algo.typeParameters()
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

	def test_custom_works_fine(self):
		clpso_custom = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		clpso_customc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, clpso_custom, clpso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		clpso_griewank = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		clpso_griewankc = self.algo(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, clpso_griewank, clpso_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
