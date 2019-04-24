from NiaPy.algorithms import Algorithm
from NiaPy.algorithms.basic import (
    BatAlgorithm,
    FireflyAlgorithm,
    DifferentialEvolution,
    CrowdingDifferentialEvolution,
    AgingNpDifferentialEvolution,
    DynNpDifferentialEvolution,
    MultiStrategyDifferentialEvolution,
    DynNpMultiStrategyDifferentialEvolution,
    multiMutations,
    AgingNpMultiMutationDifferentialEvolution,
    FlowerPollinationAlgorithm,
    GreyWolfOptimizer,
    GeneticAlgorithm,
    ArtificialBeeColonyAlgorithm,
    ParticleSwarmAlgorithm,
    BareBonesFireworksAlgorithm,
    CamelAlgorithm,
    MonkeyKingEvolutionV1,
    MonkeyKingEvolutionV2,
    MonkeyKingEvolutionV3,
    EvolutionStrategy1p1,
    EvolutionStrategyMp1,
    EvolutionStrategyMpL,
    EvolutionStrategyML,
    CovarianceMatrixAdaptionEvolutionStrategy,
    SineCosineAlgorithm,
    GlowwormSwarmOptimization,
    GlowwormSwarmOptimizationV1,
    GlowwormSwarmOptimizationV2,
    GlowwormSwarmOptimizationV3,
    HarmonySearch,
    HarmonySearchV1,
    KrillHerdV1,
    KrillHerdV2,
    KrillHerdV3,
    KrillHerdV4,
    KrillHerdV11,
    FireworksAlgorithm,
    EnhancedFireworksAlgorithm,
    DynamicFireworksAlgorithm,
    DynamicFireworksAlgorithmGauss,
    GravitationalSearchAlgorithm,
    MothFlameOptimizer,
    FishSchoolSearch,
    CuckooSearch,
    CoralReefsOptimization,
    ForestOptimizationAlgorithm
)
from NiaPy.algorithms.modified import (
    HybridBatAlgorithm,
    DifferentialEvolutionMTS,
    DifferentialEvolutionMTSv1,
    DynNpDifferentialEvolutionMTS,
    DynNpDifferentialEvolutionMTSv1,
    MultiStrategyDifferentialEvolutionMTS,
    MultiStrategyDifferentialEvolutionMTSv1,
    SelfAdaptiveDifferentialEvolution,
    DynNpSelfAdaptiveDifferentialEvolutionAlgorithm,
    MultiStrategySelfAdaptiveDifferentialEvolution,
    AgingSelfAdaptiveDifferentialEvolution
)
from NiaPy.algorithms.other import (
    NelderMeadMethod,
    HillClimbAlgorithm,
    SimulatedAnnealing,
    MultipleTrajectorySearch,
    MultipleTrajectorySearchV1
)


class AlgorithmUtility:
    r"""Base class with string mappings to algorithms.

    Attributes:
        classes (Dict[str, Algorithm]): Mapping from stings to algorithms.

    """

    def __init__(self):
        r"""Initializing the algorithms."""

        self.algorithm_classes = {
            "BatAlgorithm": BatAlgorithm,
            "FireflyAlgorithm": FireflyAlgorithm,
            "DifferentialEvolution": DifferentialEvolution,
            "CrowdingDifferentialEvolution": CrowdingDifferentialEvolution,
            "AgingNpDifferentialEvolution": AgingNpDifferentialEvolution,
            "DynNpDifferentialEvolution": DynNpDifferentialEvolution,
            "MultiStrategyDifferentialEvolution": MultiStrategyDifferentialEvolution,
            "DynNpMultiStrategyDifferentialEvolution": DynNpMultiStrategyDifferentialEvolution,
            "multiMutations": multiMutations,
            "AgingNpMultiMutationDifferentialEvolution": AgingNpMultiMutationDifferentialEvolution,
            "FlowerPollinationAlgorithm": FlowerPollinationAlgorithm,
            "GreyWolfOptimizer": GreyWolfOptimizer,
            "GeneticAlgorithm": GeneticAlgorithm,
            "ArtificialBeeColonyAlgorithm": ArtificialBeeColonyAlgorithm,
            "ParticleSwarmAlgorithm": ParticleSwarmAlgorithm,
            "BareBonesFireworksAlgorithm": BareBonesFireworksAlgorithm,
            "CamelAlgorithm": CamelAlgorithm,
            "MonkeyKingEvolutionV1": MonkeyKingEvolutionV1,
            "MonkeyKingEvolutionV2": MonkeyKingEvolutionV2,
            "MonkeyKingEvolutionV3": MonkeyKingEvolutionV3,
            "EvolutionStrategy1p1": EvolutionStrategy1p1,
            "EvolutionStrategyMp1": EvolutionStrategyMp1,
            "EvolutionStrategyMpL": EvolutionStrategyMpL,
            "EvolutionStrategyML": EvolutionStrategyML,
            "CovarianceMatrixAdaptionEvolutionStrategy": CovarianceMatrixAdaptionEvolutionStrategy,
            "SineCosineAlgorithm": SineCosineAlgorithm,
            "GlowwormSwarmOptimization": GlowwormSwarmOptimization,
            "GlowwormSwarmOptimizationV1": GlowwormSwarmOptimizationV1,
            "GlowwormSwarmOptimizationV2": GlowwormSwarmOptimizationV2,
            "GlowwormSwarmOptimizationV3": GlowwormSwarmOptimizationV3,
            "HarmonySearch": HarmonySearch,
            "HarmonySearchV1": HarmonySearchV1,
            "KrillHerdV1": KrillHerdV1,
            "KrillHerdV2": KrillHerdV2,
            "KrillHerdV3": KrillHerdV3,
            "KrillHerdV4": KrillHerdV4,
            "KrillHerdV11": KrillHerdV11,
            "FireworksAlgorithm": FireworksAlgorithm,
            "EnhancedFireworksAlgorithm": EnhancedFireworksAlgorithm,
            "DynamicFireworksAlgorithm": DynamicFireworksAlgorithm,
            "DynamicFireworksAlgorithmGauss": DynamicFireworksAlgorithmGauss,
            "GravitationalSearchAlgorithm": GravitationalSearchAlgorithm,
            "MothFlameOptimizer": MothFlameOptimizer,
            "FishSchoolSearch": FishSchoolSearch,
            "CuckooSearch": CuckooSearch,
            "CoralReefsOptimization": CoralReefsOptimization,
            "ForestOptimizationAlgorithm": ForestOptimizationAlgorithm,
            "HybridBatAlgorithm": HybridBatAlgorithm,
            "DifferentialEvolutionMTS": DifferentialEvolutionMTS,
            "DifferentialEvolutionMTSv1": DifferentialEvolutionMTSv1,
            "DynNpDifferentialEvolutionMTS": DynNpDifferentialEvolutionMTS,
            "DynNpDifferentialEvolutionMTSv1": DynNpDifferentialEvolutionMTSv1,
            "MultiStrategyDifferentialEvolutionMTS": MultiStrategyDifferentialEvolutionMTS,
            "MultiStrategyDifferentialEvolutionMTSv1": MultiStrategyDifferentialEvolutionMTSv1,
            "SelfAdaptiveDifferentialEvolution": SelfAdaptiveDifferentialEvolution,
            "DynNpSelfAdaptiveDifferentialEvolutionAlgorithm": DynNpSelfAdaptiveDifferentialEvolutionAlgorithm,
            "MultiStrategySelfAdaptiveDifferentialEvolution": MultiStrategySelfAdaptiveDifferentialEvolution,
            "AgingSelfAdaptiveDifferentialEvolution": AgingSelfAdaptiveDifferentialEvolution,
            "NelderMeadMethod": NelderMeadMethod,
            "HillClimbAlgorithm": HillClimbAlgorithm,
            "SimulatedAnnealing": SimulatedAnnealing,
            "MultipleTrajectorySearch": MultipleTrajectorySearch,
            "MultipleTrajectorySearchV1": MultipleTrajectorySearchV1
        }

    def get_algorithm(self, algorithm):
        r"""Get the algorithm.

        Arguments:
            algorithm (Union[str, Algorithm]): String or class that represents the algorithm.

        Returns:
            Algorithm: Instance of an Algorithm.
        """

        if issubclass(type(algorithm), Algorithm):
            return algorithm
        elif algorithm in self.algorithm_classes:
            return self.algorithm_classes[algorithm]()
        else:
            raise TypeError("Passed algorithm is not defined!")
