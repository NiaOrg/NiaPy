"""Implementation of basic nature-inspired algorithms."""

from niapy.algorithms.basic.abc import ArtificialBeeColonyAlgorithm
from niapy.algorithms.basic.ba import BatAlgorithm
from niapy.algorithms.basic.bea import BeesAlgorithm
from niapy.algorithms.basic.bfoa import BacterialForagingOptimizationAlgorithm
from niapy.algorithms.basic.ca import CamelAlgorithm
from niapy.algorithms.basic.cro import CoralReefsOptimization
from niapy.algorithms.basic.cs import CuckooSearch
from niapy.algorithms.basic.cso import CatSwarmOptimization
from niapy.algorithms.basic.de import DifferentialEvolution, MultiStrategyDifferentialEvolution, \
    DynNpDifferentialEvolution, AgingNpDifferentialEvolution, DynNpMultiStrategyDifferentialEvolution, \
    CrowdingDifferentialEvolution, multi_mutations
from niapy.algorithms.basic.es import EvolutionStrategy1p1, EvolutionStrategyMp1, EvolutionStrategyMpL, \
    EvolutionStrategyML, CovarianceMatrixAdaptionEvolutionStrategy
from niapy.algorithms.basic.fa import FireflyAlgorithm
from niapy.algorithms.basic.foa import ForestOptimizationAlgorithm
from niapy.algorithms.basic.fpa import FlowerPollinationAlgorithm
from niapy.algorithms.basic.fss import FishSchoolSearch
from niapy.algorithms.basic.fwa import FireworksAlgorithm, EnhancedFireworksAlgorithm, DynamicFireworksAlgorithm, \
    DynamicFireworksAlgorithmGauss, BareBonesFireworksAlgorithm
from niapy.algorithms.basic.ga import GeneticAlgorithm
from niapy.algorithms.basic.gsa import GravitationalSearchAlgorithm
from niapy.algorithms.basic.gso import GlowwormSwarmOptimization, GlowwormSwarmOptimizationV1, \
    GlowwormSwarmOptimizationV2, GlowwormSwarmOptimizationV3
from niapy.algorithms.basic.gwo import GreyWolfOptimizer
from niapy.algorithms.basic.hho import HarrisHawksOptimization
from niapy.algorithms.basic.hs import HarmonySearch, HarmonySearchV1
from niapy.algorithms.basic.kh import KrillHerdV1, KrillHerdV2, KrillHerdV3, KrillHerdV4, KrillHerdV11
from niapy.algorithms.basic.mbo import MonarchButterflyOptimization
from niapy.algorithms.basic.mfo import MothFlameOptimizer
from niapy.algorithms.basic.mke import MonkeyKingEvolutionV1, MonkeyKingEvolutionV2, MonkeyKingEvolutionV3
from niapy.algorithms.basic.pso import ParticleSwarmAlgorithm, ParticleSwarmOptimization, \
    CenterParticleSwarmOptimization, ComprehensiveLearningParticleSwarmOptimizer, \
    OppositionVelocityClampingParticleSwarmOptimization, MutatedCenterParticleSwarmOptimization, \
    MutatedCenterUnifiedParticleSwarmOptimization, MutatedParticleSwarmOptimization
from niapy.algorithms.basic.sca import SineCosineAlgorithm

__all__ = [
    'BatAlgorithm',
    'FireflyAlgorithm',
    'DifferentialEvolution',
    'CrowdingDifferentialEvolution',
    'AgingNpDifferentialEvolution',
    'DynNpDifferentialEvolution',
    'MultiStrategyDifferentialEvolution',
    'DynNpMultiStrategyDifferentialEvolution',
    'multi_mutations',
    # 'AgingNpMultiMutationDifferentialEvolution',
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
    'CenterParticleSwarmOptimization',
    'HarrisHawksOptimization',
    'BacterialForagingOptimizationAlgorithm'
]
