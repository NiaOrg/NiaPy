# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, logging-not-lazy, attribute-defined-outside-init, line-too-long, arguments-differ, singleton-comparison, bad-continuation
import logging
from math import log
from numpy import argmin, argmax, asarray, mean, concatenate
from NiaPy.algorithms.algorithm import Individual, Algorithm
from NiaPy.algorithms.basic.de import DifferentialEvolutionAlgorithm, CrossRand1, CrossBest1
from NiaPy.algorithms.other.sa import SimulatedAnnealingBF
from NiaPy.algorithms.other.mts import MTS_LS1, MTS_LS1v1, MTS_LS2, MTS_LS3, MTS_LS3v1

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = []

class MtsIndividual(Individual):
	def __init__(self, SR=0.1, grade=0, enable=True, improved=False, **kwargs):
		Individual.__init__(self, **kwargs)
		self.SR, self.grade, self.enable, self.improved = SR, grade, enable, improved

class DifferentialEvolutionMTS(DifferentialEvolutionAlgorithm):
	Name = ['DifferentialEvolutionMTS', 'DEMTS']

	@staticmethod
	def typeParameters(): return DifferentialEvolutionAlgorithm.typeParameters()

	def setParameters(self, SR=0.2, NoLSRuns=25, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
		SR {real} -- Normalized search range
		"""
		self.SR, self.NoLSRuns = SR, NoLSRuns
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		# FIXME add MTS_LS1 algorithm to the mix
		pop = [MtsIndividual(task=task, rand=self.Rand, e=True) for _i in range(self.Np)]
		x_b, x_w = pop[argmin([x.f for x in pop])], pop[argmax([x.f for x in pop])]
		while not task.stopCondI():
			npop = [MtsIndividual(x=self.CrossMutt(pop, i, x_b, self.F, self.CR, self.Rand), task=task, rand=self.Rand, e=True) for i in range(len(pop))]
			pop = [np if np.f < pop[i].f else pop[i] for i, np in enumerate(npop)]
			ix_b = argmin([x.f for x in pop])
			if x_b.f > pop[ix_b].f: x_b = pop[ix_b]
		return x_b.x, x_b.f

class DifferentialEvolutionMTSv1(DifferentialEvolutionMTS):
	Name = ['DifferentialEvolutionMTSv1', 'DEMTSv1']

	def runTask(self, task):
		# TODO main run method
		pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
