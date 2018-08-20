# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
sys.path.append('cec2015')
# End of fix

import random
import logging
from numpy import asarray
from margparser import getDictArgs
from NiaPy.algorithms.basic import ArtificialBeeColonyAlgorithm
from NiaPy.algorithms.basic import BatAlgorithm
from NiaPy.algorithms.basic import CamelAlgorithm
from NiaPy.algorithms.basic import DifferentialEvolutionAlgorithm
from NiaPy.algorithms.basic import FireflyAlgorithm
from NiaPy.algorithms.basic import FlowerPollinationAlgorithm
from NiaPy.algorithms.basic import GeneticAlgorithm
from NiaPy.algorithms.basic import GravitationalSearchAlgorithm
from NiaPy.algorithms.basic import GlowwormSwarmOptimization
from NiaPy.algorithms.basic import HarmonySearch, HarmonySearchV1
from NiaPy.algorithms.basic import KrillHerdV11
from NiaPy.algorithms.basic import MonkeyKingEvolutionV1, MonkeyKingEvolutionV2, MonkeyKingEvolutionV3
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm
from NiaPy.algorithms.basic import SineCosineAlgorithm
from NiaPy.algorithms.basic import CovarianceMaatrixAdaptionEvolutionStrategy
from NiaPy.algorithms.basic import BareBonesFireworksAlgorithm, EnhancedFireworksAlgorithm, DynamicFireworksAlgorithm, DynamicFireworksAlgorithmGauss, FireworksAlgorithm
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolutionAlgorithm, DynNPSelfAdaptiveDifferentialEvolutionAlgorithm
from NiaPy.algorithms.other import MultipleTrajectorySearch, MultipleTrajectorySearchV1
from NiaPy.algorithms.other import AnarchicSocietyOptimization
from NiaPy.benchmarks.utility import Task, TaskConvPrint, TaskConvPlot, OptimizationType
from cec2015 import run_fun

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

# For reproducive results
random.seed(1234)

class MinMB(object):
	def __init__(self, fnum=1):
		self.Lower = -100
		self.Upper = 100
		self.fnum = fnum

	def function(self):
		def evaluate(D, sol): return run_fun(asarray(sol), self.fnum)
		return evaluate

class MaxMB(MinMB):
	def function(self):
		f = MinMB.function(self)
		def e(D, sol): return -f(D, sol)
		return e

def simple_example(alg, fnum=1, runs=10, D=10, nFES=50000, nGEN=5000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB, **kwu):
	for i in range(runs):
		task = Task(D=D, nFES=nFES, nGEN=nGEN, optType=optType, benchmark=optFunc(fnum))
		algo = alg(seed=seed, task=task)
		Best = algo.run()
		logger.info('%s %s' % (Best[0], Best[1]))

def logging_example(alg, fnum=1, D=10, nFES=50000, nGEN=5000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB, **ukw):
	task = TaskConvPrint(D=D, nFES=nFES, nGEN=nGEN, optType=optType, benchmark=optFunc(fnum))
	algo = alg(seed=seed, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))

def plot_example(alg, fnum=1, D=10, nFES=50000, nGEN=5000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB, **kwy):
	task = TaskConvPlot(D=D, nFES=nFES, nGEN=nGEN, optType=optType, benchmark=optFunc(fnum))
	algo = alg(seed=seed, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))
	input('Press [enter] to continue')

def getOptType(strtype):
	if strtype == 'min': return OptimizationType.MINIMIZATION, MinMB
	elif strtype == 'max': return OptimizationType.MAXIMIZATION, MaxMB
	else: return None

if __name__ == '__main__':
	# algo = ArtificialBeeColonyAlgorithm
	# algo = BatAlgorithm
	# algo = CamelAlgorithm
	# algo = DifferentialEvolutionAlgorithm
	# algo = FireflyAlgorithm
	# algo = FlowerPollinationAlgorithm
	# algo = GeneticAlgorithm
	# algo = GravitationalSearchAlgorithm
	# algo = GlowwormSwarmOptimization
	# algo = HarmonySearch
	# algo = HarmonySearchV1
	# algo = KrillHerdV11
	# algo = MonkeyKingEvolutionV1
	# algo = MonkeyKingEvolutionV2
	# algo = MonkeyKingEvolutionV3
	# algo = ParticleSwarmAlgorithm
	# algo = SineCosineAlgorithm
	# algo = MultipleTrajectorySearch
	# algo = MultipleTrajectorySearchV1
	# algo = CovarianceMaatrixAdaptionEvolutionStrategy
	# algo = BareBonesFireworksAlgorithm
	# algo = FireworksAlgorithm
	# algo = EnhancedFireworksAlgorithm
	# algo = DynamicFireworksAlgorithm
	algo = DynamicFireworksAlgorithmGauss
	# algo = SelfAdaptiveDifferentialEvolutionAlgorithm
	# algo = DynNPSelfAdaptiveDifferentialEvolutionAlgorithm
	# algo = AnarchicSocietyOptimization
	pargs = getDictArgs(sys.argv[1:])
	optType, optFunc = getOptType(pargs.pop('optType', 'min'))
	if not pargs['runType']: simple_example(algo, optType=optType, optFunc=optFunc, **pargs)
	elif pargs['runType'] == 'log': logging_example(algo, optType=optType, optFunc=optFunc, **pargs)
	elif pargs['runType'] == 'plot': plot_example(algo, optType=optType, optFunc=optFunc, **pargs)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
