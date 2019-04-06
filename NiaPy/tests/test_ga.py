# encoding=utf8
# pylint: disable=mixed-indentation, function-redefined, multiple-statements, line-too-long
from NiaPy.algorithms.basic import GeneticAlgorithm
from NiaPy.algorithms.basic.ga import TwoPointCrossover, MultiPointCrossover, CreepMutation, RouletteSelection, CrossoverUros, MutationUros
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class GATestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ga_custom = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, seed=self.seed)
		ga_customc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_custom, ga_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ga_griewank = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, seed=self.seed)
		ga_griewankc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_griewank, ga_griewankc)

	def test_two_point_crossover_fine_c(self):
		ga_tpcr = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=TwoPointCrossover, seed=self.seed)
		ga_tpcrc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=TwoPointCrossover, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_tpcr, ga_tpcrc, MyBenchmark())

	def test_two_point_crossover_fine(self):
		ga_tpcr = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=TwoPointCrossover, seed=self.seed)
		ga_tpcrc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=TwoPointCrossover, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_tpcr, ga_tpcrc)

	def test_multi_point_crossover_fine_c(self):
		ga_mpcr = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=4, Crossover=MultiPointCrossover, seed=self.seed)
		ga_mpcrc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=4, Crossover=MultiPointCrossover, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_mpcr, ga_mpcrc, MyBenchmark())

	def test_multi_point_crossover_fine(self):
		ga_mpcr = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=4, Crossover=MultiPointCrossover, seed=self.seed)
		ga_mpcrc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=4, Crossover=MultiPointCrossover, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_mpcr, ga_mpcrc)

	def test_creep_mutation_fine_c(self):
		ga_crmt = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=CreepMutation, seed=self.seed)
		ga_crmtc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=CreepMutation, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_crmt, ga_crmtc, MyBenchmark())

	def test_creep_mutation_fine(self):
		ga_crmt = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=CreepMutation, seed=self.seed)
		ga_crmtc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=CreepMutation, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_crmt, ga_crmtc)

	def test_reulete_selection_c(self):
		ga_crmt = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Selection=RouletteSelection, seed=self.seed)
		ga_crmtc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Selection=RouletteSelection, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_crmt, ga_crmtc, MyBenchmark())

	def test_reulete_selection(self):
		ga_crmt = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Selection=RouletteSelection, seed=self.seed)
		ga_crmtc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Selection=RouletteSelection, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_crmt, ga_crmtc)

	def test_crossover_urso_c(self):
		ga_crmt = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=CrossoverUros, seed=self.seed)
		ga_crmtc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=CrossoverUros, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_crmt, ga_crmtc, MyBenchmark())

	def test_crossover_urso(self):
		ga_crmt = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=CrossoverUros, seed=self.seed)
		ga_crmtc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Crossover=CrossoverUros, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_crmt, ga_crmtc)

	def test_mutation_urso_c(self):
		ga_crmt = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=MutationUros, seed=self.seed)
		ga_crmtc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=MutationUros, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_crmt, ga_crmtc, MyBenchmark())

	def test_mutation_urso(self):
		ga_crmt = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=MutationUros, seed=self.seed)
		ga_crmtc = GeneticAlgorithm(NP=40, Ts=4, Mr=0.05, Cr=0.4, Mutation=MutationUros, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ga_crmt, ga_crmtc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
