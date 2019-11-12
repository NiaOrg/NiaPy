"""Implementation of basic nature-inspired algorithms."""

from NiaPy.algorithms.basic.ba import BatAlgorithm
from NiaPy.algorithms.basic.fa import FireflyAlgorithm
from NiaPy.algorithms.basic.de import DifferentialEvolution, MultiStrategyDifferentialEvolution, DynNpDifferentialEvolution, AgingNpDifferentialEvolution, DynNpMultiStrategyDifferentialEvolution, AgingNpMultiMutationDifferentialEvolution, CrowdingDifferentialEvolution, multiMutations
from NiaPy.algorithms.basic.fpa import FlowerPollinationAlgorithm
from NiaPy.algorithms.basic.gwo import GreyWolfOptimizer
from NiaPy.algorithms.basic.cso import CatSwarmOptimization
from NiaPy.algorithms.basic.ga import GeneticAlgorithm
from NiaPy.algorithms.basic.abc import ArtificialBeeColonyAlgorithm
from NiaPy.algorithms.basic.pso import ParticleSwarmAlgorithm, ParticleSwarmOptimization, CenterParticleSwarmOptimization, ComprehensiveLearningParticleSwarmOptimizer, OppositionVelocityClampingParticleSwarmOptimization, MutatedCenterParticleSwarmOptimization, MutatedCenterUnifiedParticleSwarmOptimization, MutatedParticleSwarmOptimization
from NiaPy.algorithms.basic.ca import CamelAlgorithm
from NiaPy.algorithms.basic.mke import MonkeyKingEvolutionV1, MonkeyKingEvolutionV2, MonkeyKingEvolutionV3
from NiaPy.algorithms.basic.es import EvolutionStrategy1p1, EvolutionStrategyMp1, EvolutionStrategyMpL, EvolutionStrategyML, CovarianceMatrixAdaptionEvolutionStrategy
from NiaPy.algorithms.basic.sca import SineCosineAlgorithm
from NiaPy.algorithms.basic.gso import GlowwormSwarmOptimization, GlowwormSwarmOptimizationV1, GlowwormSwarmOptimizationV2, GlowwormSwarmOptimizationV3
from NiaPy.algorithms.basic.hs import HarmonySearch, HarmonySearchV1
from NiaPy.algorithms.basic.kh import KrillHerdV1, KrillHerdV2, KrillHerdV3, KrillHerdV4, KrillHerdV11
from NiaPy.algorithms.basic.fwa import FireworksAlgorithm, EnhancedFireworksAlgorithm, DynamicFireworksAlgorithm, DynamicFireworksAlgorithmGauss, BareBonesFireworksAlgorithm
from NiaPy.algorithms.basic.gsa import GravitationalSearchAlgorithm
from NiaPy.algorithms.basic.mfo import MothFlameOptimizer
from NiaPy.algorithms.basic.fss import FishSchoolSearch
from NiaPy.algorithms.basic.cs import CuckooSearch
from NiaPy.algorithms.basic.cro import CoralReefsOptimization
from NiaPy.algorithms.basic.foa import ForestOptimizationAlgorithm
from NiaPy.algorithms.basic.mbo import MonarchButterflyOptimization
from NiaPy.algorithms.basic.bea import BeesAlgorithm
__all__ = [
    'BatAlgorithm',
    'FireflyAlgorithm',
    'DifferentialEvolution',
    'CrowdingDifferentialEvolution',
    'AgingNpDifferentialEvolution',
    'DynNpDifferentialEvolution',
    'MultiStrategyDifferentialEvolution',
    'DynNpMultiStrategyDifferentialEvolution',
    'multiMutations',
    'AgingNpMultiMutationDifferentialEvolution',
    'FlowerPollinationAlgorithm',
    'GreyWolfOptimizer',
    'CatSwarmOptimization',
    'GeneticAlgorithm',
    'ArtificialBeeColonyAlgorithm',
    'ParticleSwarmAlgorithm',
    'BareBonesFireworksAlgorithm',
    'CamelAlgorithm',
    'MonkeyKingEvolutionV1',
    'MonkeyKingEvolutionV2',
    'MonkeyKingEvolutionV3',
    'EvolutionStrategy1p1',
    'EvolutionStrategyMp1',
    'EvolutionStrategyMpL',
    'EvolutionStrategyML',
    'CovarianceMatrixAdaptionEvolutionStrategy',
    'SineCosineAlgorithm',
    'GlowwormSwarmOptimization',
    'GlowwormSwarmOptimizationV1',
    'GlowwormSwarmOptimizationV2',
    'GlowwormSwarmOptimizationV3',
    'HarmonySearch',
    'HarmonySearchV1',
    'KrillHerdV1',
    'KrillHerdV2',
    'KrillHerdV3',
    'KrillHerdV4',
    'KrillHerdV11',
    'FireworksAlgorithm',
    'EnhancedFireworksAlgorithm',
    'DynamicFireworksAlgorithm',
    'DynamicFireworksAlgorithmGauss',
    'GravitationalSearchAlgorithm',
    'MothFlameOptimizer',
    'FishSchoolSearch',
    'CuckooSearch',
    'CoralReefsOptimization',
    'ForestOptimizationAlgorithm',
    'MonarchButterflyOptimization',
    'BeesAlgorithm',
    'ParticleSwarmOptimization',
    'MutatedParticleSwarmOptimization',
    'MutatedCenterUnifiedParticleSwarmOptimization',
    'MutatedCenterParticleSwarmOptimization',
    'OppositionVelocityClampingParticleSwarmOptimization',
    'ComprehensiveLearningParticleSwarmOptimizer',
    'CenterParticleSwarmOptimization'
]
