"""Implementation of basic nature-inspired algorithms."""
# pylint: disable=line-too-long

from NiaPy.algorithms.basic.ba import BatAlgorithm
from NiaPy.algorithms.basic.fa import FireflyAlgorithm
from NiaPy.algorithms.basic.de import DifferentialEvolutionAlgorithm
from NiaPy.algorithms.basic.fpa import FlowerPollinationAlgorithm
from NiaPy.algorithms.basic.gwo import GreyWolfOptimizer
from NiaPy.algorithms.basic.ga import GeneticAlgorithm
from NiaPy.algorithms.basic.abc import ArtificialBeeColonyAlgorithm
from NiaPy.algorithms.basic.pso import ParticleSwarmAlgorithm
from NiaPy.algorithms.basic.ca import CamelAlgorithm
from NiaPy.algorithms.basic.bbfwa import BareBonesFireworksAlgorithm
from NiaPy.algorithms.basic.mke import MonkeyKingEvolutionV1, MonkeyKingEvolutionV2, MonkeyKingEvolutionV3
from NiaPy.algorithms.basic.es import EvolutionStrategy1p1, EvolutionStrategyMp1, EvolutionStrategyMpL, EvolutionStrategyML
from NiaPy.algorithms.basic.sca import SineCosineAlgorithm
from NiaPy.algorithms.basic.gso import GlowwormSwarmOptimization, GlowwormSwarmOptimizationV1, GlowwormSwarmOptimizationV2, GlowwormSwarmOptimizationV3

__all__ = [
    'BatAlgorithm',
    'FireflyAlgorithm',
    'DifferentialEvolutionAlgorithm',
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
    'SineCosineAlgorithm',
    'GlowwormSwarmOptimization',
    'GlowwormSwarmOptimizationV1',
    'GlowwormSwarmOptimizationV2',
    'GlowwormSwarmOptimizationV3'
]
