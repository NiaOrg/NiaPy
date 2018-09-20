# encoding=utf8
# pylint: disable=mixed-indentation, function-redefined, multiple-statements, old-style-class, line-too-long
from NiaPy.algorithms.basic import GeneticAlgorithm
from NiaPy.algorithms.basic.ga import TwoPointCrossover, MultiPointCrossover, CreepMutation, RouletteSelection, CrossoverUros, MutationUros
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class GATestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ga_custom = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, benchmark=MyBenchmark(), seed=self.seed)
		ga_customc = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_custom, ga_customc)

	def test_griewank_works_fine(self):
		ga_griewank = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, benchmark='griewank', seed=self.seed)
		ga_griewankc = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_griewank, ga_griewankc)

	def test_two_point_crossover_fine(self):
		ga_tpcr = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, Crossover=TwoPointCrossover, benchmark='griewank', seed=self.seed)
		ga_tpcrc = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, Crossover=TwoPointCrossover, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_tpcr, ga_tpcrc)

	def test_multi_point_crossover_fine(self):
		ga_mpcr = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=4, Crossover=MultiPointCrossover, benchmark='griewank', seed=self.seed)
		ga_mpcrc = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=4, Crossover=MultiPointCrossover, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_mpcr, ga_mpcrc)

	def test_creep_mutation_fine(self):
		ga_crmt = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, Mutation=CreepMutation, benchmark='griewank', seed=self.seed)
		ga_crmtc = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, Mutation=CreepMutation, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_crmt, ga_crmtc)

	def test_reulete_selection(self):
		ga_crmt = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, Selection=RouletteSelection, benchmark='griewank', seed=self.seed)
		ga_crmtc = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, Selection=RouletteSelection, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_crmt, ga_crmtc)

	def test_crossover_urso(self):
		ga_crmt = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, Crossover=CrossoverUros, benchmark='griewank', seed=self.seed)
		ga_crmtc = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, Crossover=CrossoverUros, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_crmt, ga_crmtc)

	def test_mutation_urso(self):
		ga_crmt = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, Mutation=MutationUros, benchmark='griewank', seed=self.seed)
		ga_crmtc = GeneticAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, Ts=4, Mr=0.05, Cr=0.4, Mutation=MutationUros, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_crmt, ga_crmtc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
