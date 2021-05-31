# encoding=utf8
from niapy.algorithms.basic import GeneticAlgorithm
from niapy.algorithms.basic.ga import two_point_crossover, multi_point_crossover, creep_mutation, roulette_selection, \
    crossover_uros, mutation_uros
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class GATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = GeneticAlgorithm

    def test_custom(self):
        ga_custom = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, seed=self.seed)
        ga_customc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_custom, ga_customc, MyProblem())

    def test_griewank(self):
        ga_griewank = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, seed=self.seed)
        ga_griewankc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_griewank, ga_griewankc)

    def test_two_point_crossover_fine_c(self):
        ga_tpcr = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, crossover=two_point_crossover, seed=self.seed)
        ga_tpcrc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, crossover=two_point_crossover, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_tpcr, ga_tpcrc, MyProblem())

    def test_two_point_crossover(self):
        ga_tpcr = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, crossover=two_point_crossover, seed=self.seed)
        ga_tpcrc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, crossover=two_point_crossover, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_tpcr, ga_tpcrc)

    def test_multi_point_crossover_fine_c(self):
        ga_mpcr = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=4, crossover=multi_point_crossover, seed=self.seed)
        ga_mpcrc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=4, crossover=multi_point_crossover, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_mpcr, ga_mpcrc, MyProblem())

    def test_multi_point_crossover(self):
        ga_mpcr = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=4, crossover=multi_point_crossover, seed=self.seed)
        ga_mpcrc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=4, crossover=multi_point_crossover, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_mpcr, ga_mpcrc)

    def test_creep_mutation_fine_c(self):
        ga_crmt = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, mutation=creep_mutation, seed=self.seed)
        ga_crmtc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, mutation=creep_mutation, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_crmt, ga_crmtc, MyProblem())

    def test_creep_mutation(self):
        ga_crmt = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, mutation=creep_mutation, seed=self.seed)
        ga_crmtc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, mutation=creep_mutation, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_crmt, ga_crmtc)

    def test_roulette_selection_c(self):
        ga_crmt = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, selection=roulette_selection, seed=self.seed)
        ga_crmtc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, selection=roulette_selection, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_crmt, ga_crmtc, MyProblem())

    def test_roulette_selection(self):
        ga_crmt = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, selection=roulette_selection, seed=self.seed)
        ga_crmtc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, selection=roulette_selection, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_crmt, ga_crmtc)

    def test_crossover_uros_c(self):
        ga_crmt = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, crossover=crossover_uros, seed=self.seed)
        ga_crmtc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, crossover=crossover_uros, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_crmt, ga_crmtc, MyProblem())

    def test_crossover_uros(self):
        ga_crmt = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, crossover=crossover_uros, seed=self.seed)
        ga_crmtc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, crossover=crossover_uros, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_crmt, ga_crmtc)

    def test_mutation_uros_c(self):
        ga_crmt = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, mutation=mutation_uros, seed=self.seed)
        ga_crmtc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, mutation=mutation_uros, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_crmt, ga_crmtc, MyProblem())

    def test_mutation_uros(self):
        ga_crmt = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, mutation=mutation_uros, seed=self.seed)
        ga_crmtc = self.algo(population_size=10, tournament_size=4, mutation_rate=0.05, crossover_rate=0.4, mutation=mutation_uros, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ga_crmt, ga_crmtc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
