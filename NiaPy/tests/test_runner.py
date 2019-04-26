# encoding=utf8

from unittest import TestCase

import NiaPy


class MyBenchmark(NiaPy.benchmarks.Benchmark):
    def __init__(self):
        self.Lower = -11
        self.Upper = 11

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate


class RunnerTestCase(TestCase):
    def setUp(self):
        self.algorithms = ['DifferentialEvolution', 'GreyWolfOptimizer', 'GeneticAlgorithm', 'ParticleSwarmAlgorithm', 'HybridBatAlgorithm', 'SelfAdaptiveDifferentialEvolution', 'CamelAlgorithm', 'BareBonesFireworksAlgorithm', 'MonkeyKingEvolutionV1', 'MonkeyKingEvolutionV2', 'MonkeyKingEvolutionV3', 'EvolutionStrategy1p1', 'EvolutionStrategyMp1', 'SineCosineAlgorithm', 'GlowwormSwarmOptimization', 'GlowwormSwarmOptimizationV1', 'GlowwormSwarmOptimizationV2', 'GlowwormSwarmOptimizationV3', 'KrillHerdV1', 'KrillHerdV2', 'KrillHerdV3', 'KrillHerdV4', 'KrillHerdV11', 'HarmonySearch', 'HarmonySearchV1', 'FireworksAlgorithm', 'EnhancedFireworksAlgorithm', 'DynamicFireworksAlgorithm', 'MultipleTrajectorySearch', 'MultipleTrajectorySearchV1', 'NelderMeadMethod', 'HillClimbAlgorithm', 'SimulatedAnnealing', 'GravitationalSearchAlgorithm', 'AnarchicSocietyOptimization']
        self.benchmarks = ['griewank', MyBenchmark()]

    def test_runner_works_fine(self):
        runner = NiaPy.Runner(4, 10, 2, useAlgorithms=self.algorithms, useBenchmarks=self.benchmarks)
        runner.run()
        self.assertTrue(runner.results)

    def test_runner_bad_algorithm_throws_fine(self):
        self.assertRaises(TypeError, lambda: NiaPy.Runner(4, 10, 2, 'EvolutionStrategy', self.benchmarks).run())

    def test_runner_bad_benchmark_throws_fine(self):
        self.assertRaises(TypeError, lambda: NiaPy.Runner(4, 10, 2, 'EvolutionStrategy1p1', 'TesterMan').run())
