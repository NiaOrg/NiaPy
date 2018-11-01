"""Implementation of basic nature-inspired algorithms."""
# pylint: disable=line-too-long

from NiaPy.algorithms.basic.ba import BatAlgorithm
from NiaPy.algorithms.basic.fa import FireflyAlgorithm
from NiaPy.algorithms.basic.de import DifferentialEvolution, MultiStrategyDifferentialEvolution, DynNpDifferentialEvolution, AgingNpDifferentialEvolution, DynNpMultiStrategyDifferentialEvolution
from NiaPy.algorithms.basic.fpa import FlowerPollinationAlgorithm
from NiaPy.algorithms.basic.gwo import GreyWolfOptimizer
from NiaPy.algorithms.basic.ga import GeneticAlgorithm
from NiaPy.algorithms.basic.abc import ArtificialBeeColonyAlgorithm
from NiaPy.algorithms.basic.pso import ParticleSwarmAlgorithm
from NiaPy.algorithms.basic.ca import CamelAlgorithm
from NiaPy.algorithms.basic.mke import MonkeyKingEvolutionV1, MonkeyKingEvolutionV2, MonkeyKingEvolutionV3
from NiaPy.algorithms.basic.es import EvolutionStrategy1p1, EvolutionStrategyMp1, EvolutionStrategyMpL, EvolutionStrategyML, CovarianceMaatrixAdaptionEvolutionStrategy
from NiaPy.algorithms.basic.sca import SineCosineAlgorithm
from NiaPy.algorithms.basic.gso import GlowwormSwarmOptimization, GlowwormSwarmOptimizationV1, GlowwormSwarmOptimizationV2, GlowwormSwarmOptimizationV3
from NiaPy.algorithms.basic.hs import HarmonySearch, HarmonySearchV1
from NiaPy.algorithms.basic.kh import KrillHerdV1, KrillHerdV2, KrillHerdV3, KrillHerdV4, KrillHerdV11
from NiaPy.algorithms.basic.fwa import FireworksAlgorithm, EnhancedFireworksAlgorithm, DynamicFireworksAlgorithm, DynamicFireworksAlgorithmGauss, BareBonesFireworksAlgorithm
from NiaPy.algorithms.basic.gsa import GravitationalSearchAlgorithm
from NiaPy.algorithms.basic.mfo import MothFlameOptimizer
from NiaPy.algorithms.basic.fss import FishSchoolSearch

__all__ = [
    'BatAlgorithm',
    'FireflyAlgorithm',
    'DifferentialEvolution',
    'AgingNpDifferentialEvolution',
    'DynNpDifferentialEvolution',
    'MultiStrategyDifferentialEvolution',
    'DynNpMultiStrategyDifferentialEvolution',
    'FlowerPollinationAlgorithm',
    'GreyWolfOptimizer',
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
    'CovarianceMaatrixAdaptionEvolutionStrategy',
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
]
