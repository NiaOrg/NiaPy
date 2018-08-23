# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, logging-not-lazy, attribute-defined-outside-init, line-too-long, arguments-differ, singleton-comparison, bad-continuation
import logging
from numpy import argmin
from NiaPy.algorithms.algorithm import Individual, Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['DifferentialEvolutionBestSimulatedAneling', 'DifferentialEvolutionBestHarmonySearch', 'DifferentialEvolutionPBestHarmonySearch', 'DifferentialEvolutionBestMTS1', 'DifferentialEvolutionBestMTS2', 'DifferentialEvolutionBestMTS3']

class DifferentialEvolutionBestSimulatedAneling(Algorithm):
	Name = ['DifferentialEvolutionBestSimulatedAneling', 'DEbSA']

	def setParameters(self, **ukwargs): pass

	def runTask(self, task): pass

class DifferentialEvolutionBestHarmonySearch(Algorithm):
	Name = ['DifferentialEvolutionBestHarmonySearch', 'DEbHS']

	def setParameters(self, **ukwargs): pass

	def runTask(self, task): pass

class DifferentialEvolutionPBestHarmonySearch(Algorithm):
	Name = ['DifferentialEvolutionBestHarmonySearch', 'DEpbHS']

	def setParameters(self, **ukwargs): pass

	def runTask(self, task): pass

class DifferentialEvolutionBestMTS1(Algorithm):
	Name = ['DifferentialEvolutionBestMTS1', 'DEbMTS1']

	def setParameters(self, **ukwargs): pass

	def runTask(self, task): pass

class DifferentialEvolutionBestMTS2(Algorithm):
	Name = ['DifferentialEvolutionBestMTS2', 'DEbMTS2']

	def setParameters(self, **ukwargs): pass

	def runTask(self, task): pass

class DifferentialEvolutionBestMTS3(Algorithm):
	Name = ['DifferentialEvolutionBestMTS3', 'DEbMTS3']

	def setParameters(self, **ukwargs): pass

	def runTask(self, task): pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
