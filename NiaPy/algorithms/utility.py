# encoding=utf8
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.algorithms import basic as basic_algorithms
from NiaPy.algorithms import modified as modified_algorithms
from NiaPy.algorithms import other as other_algorithms


class AlgorithmUtility:
    r"""Base class with string mappings to algorithms.

    Attributes:
        classes (Dict[str, Algorithm]): Mapping from stings to algorithms.

    """

    def __init__(self):
        r"""Initialize the algorithms."""

        self.algorithm_classes = {
            "BatAlgorithm": basic_algorithms.BatAlgorithm,
            "FireflyAlgorithm": basic_algorithms.FireflyAlgorithm,
            "DifferentialEvolution": basic_algorithms.DifferentialEvolution,
            "CrowdingDifferentialEvolution": basic_algorithms.CrowdingDifferentialEvolution,
            "AgingNpDifferentialEvolution": basic_algorithms.AgingNpDifferentialEvolution,
            "DynNpDifferentialEvolution": basic_algorithms.DynNpDifferentialEvolution,
            "MultiStrategyDifferentialEvolution": basic_algorithms.MultiStrategyDifferentialEvolution,
            "DynNpMultiStrategyDifferentialEvolution": basic_algorithms.DynNpMultiStrategyDifferentialEvolution,
            "AgingNpMultiMutationDifferentialEvolution": basic_algorithms.AgingNpMultiMutationDifferentialEvolution,
            "FlowerPollinationAlgorithm": basic_algorithms.FlowerPollinationAlgorithm,
            "GreyWolfOptimizer": basic_algorithms.GreyWolfOptimizer,
            "GeneticAlgorithm": basic_algorithms.GeneticAlgorithm,
            "ArtificialBeeColonyAlgorithm": basic_algorithms.ArtificialBeeColonyAlgorithm,
            "ParticleSwarmAlgorithm": basic_algorithms.ParticleSwarmAlgorithm,
            "BareBonesFireworksAlgorithm": basic_algorithms.BareBonesFireworksAlgorithm,
            "CamelAlgorithm": basic_algorithms.CamelAlgorithm,
            "MonkeyKingEvolutionV1": basic_algorithms.MonkeyKingEvolutionV1,
            "MonkeyKingEvolutionV2": basic_algorithms.MonkeyKingEvolutionV2,
            "MonkeyKingEvolutionV3": basic_algorithms.MonkeyKingEvolutionV3,
            "EvolutionStrategy1p1": basic_algorithms.EvolutionStrategy1p1,
            "EvolutionStrategyMp1": basic_algorithms.EvolutionStrategyMp1,
            "EvolutionStrategyMpL": basic_algorithms.EvolutionStrategyMpL,
            "EvolutionStrategyML": basic_algorithms.EvolutionStrategyML,
            "CovarianceMatrixAdaptionEvolutionStrategy": basic_algorithms.CovarianceMatrixAdaptionEvolutionStrategy,
            "SineCosineAlgorithm": basic_algorithms.SineCosineAlgorithm,
            "GlowwormSwarmOptimization": basic_algorithms.GlowwormSwarmOptimization,
            "GlowwormSwarmOptimizationV1": basic_algorithms.GlowwormSwarmOptimizationV1,
            "GlowwormSwarmOptimizationV2": basic_algorithms.GlowwormSwarmOptimizationV2,
            "GlowwormSwarmOptimizationV3": basic_algorithms.GlowwormSwarmOptimizationV3,
            "HarmonySearch": basic_algorithms.HarmonySearch,
            "HarmonySearchV1": basic_algorithms.HarmonySearchV1,
            "KrillHerdV1": basic_algorithms.KrillHerdV1,
            "KrillHerdV2": basic_algorithms.KrillHerdV2,
            "KrillHerdV3": basic_algorithms.KrillHerdV3,
            "KrillHerdV4": basic_algorithms.KrillHerdV4,
            "KrillHerdV11": basic_algorithms.KrillHerdV11,
            "FireworksAlgorithm": basic_algorithms.FireworksAlgorithm,
            "EnhancedFireworksAlgorithm": basic_algorithms.EnhancedFireworksAlgorithm,
            "DynamicFireworksAlgorithm": basic_algorithms.DynamicFireworksAlgorithm,
            "DynamicFireworksAlgorithmGauss": basic_algorithms.DynamicFireworksAlgorithmGauss,
            "GravitationalSearchAlgorithm": basic_algorithms.GravitationalSearchAlgorithm,
            "MothFlameOptimizer": basic_algorithms.MothFlameOptimizer,
            "FishSchoolSearch": basic_algorithms.FishSchoolSearch,
            "CuckooSearch": basic_algorithms.CuckooSearch,
            "CoralReefsOptimization": basic_algorithms.CoralReefsOptimization,
            "ForestOptimizationAlgorithm": basic_algorithms.ForestOptimizationAlgorithm,
            "MonarchButterflyOptimization": basic_algorithms.MonarchButterflyOptimization,
            "HybridBatAlgorithm": modified_algorithms.HybridBatAlgorithm,
            "DifferentialEvolutionMTS": modified_algorithms.DifferentialEvolutionMTS,
            "DifferentialEvolutionMTSv1": modified_algorithms.DifferentialEvolutionMTSv1,
            "DynNpDifferentialEvolutionMTS": modified_algorithms.DynNpDifferentialEvolutionMTS,
            "DynNpDifferentialEvolutionMTSv1": modified_algorithms.DynNpDifferentialEvolutionMTSv1,
            "MultiStrategyDifferentialEvolutionMTS": modified_algorithms.MultiStrategyDifferentialEvolutionMTS,
            "MultiStrategyDifferentialEvolutionMTSv1": modified_algorithms.MultiStrategyDifferentialEvolutionMTSv1,
            "SelfAdaptiveDifferentialEvolution": modified_algorithms.SelfAdaptiveDifferentialEvolution,
            "DynNpSelfAdaptiveDifferentialEvolutionAlgorithm": modified_algorithms.DynNpSelfAdaptiveDifferentialEvolutionAlgorithm,
            "MultiStrategySelfAdaptiveDifferentialEvolution": modified_algorithms.MultiStrategySelfAdaptiveDifferentialEvolution,
            "AgingSelfAdaptiveDifferentialEvolution": modified_algorithms.AgingSelfAdaptiveDifferentialEvolution,
            "NelderMeadMethod": other_algorithms.NelderMeadMethod,
            "HillClimbAlgorithm": other_algorithms.HillClimbAlgorithm,
            "SimulatedAnnealing": other_algorithms.SimulatedAnnealing,
            "MultipleTrajectorySearch": other_algorithms.MultipleTrajectorySearch,
            "MultipleTrajectorySearchV1": other_algorithms.MultipleTrajectorySearchV1,
            "AnarchicSocietyOptimization": other_algorithms.AnarchicSocietyOptimization
        }

    def get_algorithm(self, algorithm):
        r"""Get the algorithm.

        Arguments:
            algorithm (Union[str, Algorithm]): String or class that represents the algorithm.

        Returns:
            Algorithm: Instance of an Algorithm.
        """

        if issubclass(type(algorithm), Algorithm) or isinstance(algorithm, Algorithm):
            return algorithm
        elif algorithm in self.algorithm_classes:
            return self.algorithm_classes[algorithm]()
        else:
            raise TypeError("Passed algorithm is not defined! --> %s" % algorithm)
