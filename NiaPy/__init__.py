# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, multiple-statements, too-many-function-args, singleton-comparison, bad-continuation
"""Python micro framework for building nature-inspired algorithms."""

from __future__ import print_function  # for backward compatibility purpose

import os
import logging
import json
import datetime
import xlsxwriter
from numpy import amin, amax, median, mean, std
from NiaPy import benchmarks, util, algorithms
from NiaPy.algorithms import basic as balgos, modified as malgos, other as oalgos

__all__ = ['algorithms', 'benchmarks', 'util']
__project__ = 'NiaPy'
__version__ = '2.0.0rc4'

VERSION = "{0} v{1}".format(__project__, __version__)

logging.basicConfig()
logger = logging.getLogger('NiaPy')
logger.setLevel('INFO')

NiaPyAlgos = [
	balgos.BatAlgorithm,
	balgos.DifferentialEvolution,
	balgos.CrowdingDifferentialEvolution,
	balgos.DynNpDifferentialEvolution,
	balgos.AgingNpDifferentialEvolution,
	balgos.MultiStrategyDifferentialEvolution,
	balgos.DynNpMultiStrategyDifferentialEvolution,
	balgos.AgingNpMultiMutationDifferentialEvolution,
	balgos.FireflyAlgorithm,
	balgos.FlowerPollinationAlgorithm,
	balgos.GreyWolfOptimizer,
	balgos.ArtificialBeeColonyAlgorithm,
	balgos.GeneticAlgorithm,
	balgos.ParticleSwarmAlgorithm,
	balgos.CamelAlgorithm,
	balgos.BareBonesFireworksAlgorithm,
	balgos.MonkeyKingEvolutionV1,
	balgos.MonkeyKingEvolutionV2,
	balgos.MonkeyKingEvolutionV3,
	balgos.EvolutionStrategy1p1,
	balgos.EvolutionStrategyMp1,
	balgos.EvolutionStrategyMpL,
	balgos.SineCosineAlgorithm,
	balgos.HarmonySearch,
	balgos.HarmonySearchV1,
	balgos.GlowwormSwarmOptimization,
	balgos.GlowwormSwarmOptimizationV1,
	balgos.GlowwormSwarmOptimizationV2,
	balgos.GlowwormSwarmOptimizationV3,
	balgos.KrillHerdV1,
	balgos.KrillHerdV2,
	balgos.KrillHerdV3,
	balgos.KrillHerdV4,
	balgos.KrillHerdV11,
	balgos.FireworksAlgorithm,
	balgos.EnhancedFireworksAlgorithm,
	balgos.DynamicFireworksAlgorithm,
	balgos.DynamicFireworksAlgorithmGauss,
	balgos.GravitationalSearchAlgorithm,
	balgos.FishSchoolSearch,
	balgos.MothFlameOptimizer,
	balgos.CuckooSearch,
	balgos.CovarianceMatrixAdaptionEvolutionStrategy,
	balgos.CoralReefsOptimization,
	balgos.ForestOptimizationAlgorithm
]

NiaPyAlgos += [
	malgos.HybridBatAlgorithm,
	malgos.DifferentialEvolutionMTS,
	malgos.DifferentialEvolutionMTSv1,
	malgos.DynNpDifferentialEvolutionMTS,
	malgos.DynNpDifferentialEvolutionMTSv1,
	malgos.MultiStrategyDifferentialEvolutionMTS,
	malgos.MultiStrategyDifferentialEvolutionMTSv1,
	malgos.DynNpMultiStrategyDifferentialEvolutionMTS,
	malgos.DynNpMultiStrategyDifferentialEvolutionMTSv1,
	malgos.SelfAdaptiveDifferentialEvolution,
	malgos.DynNpSelfAdaptiveDifferentialEvolutionAlgorithm,
	malgos.MultiStrategySelfAdaptiveDifferentialEvolution,
	malgos.DynNpMultiStrategySelfAdaptiveDifferentialEvolution
]

NiaPyAlgos += [
	oalgos.MultipleTrajectorySearch,
	oalgos.MultipleTrajectorySearchV1,
	oalgos.NelderMeadMethod,
	oalgos.HillClimbAlgorithm,
	oalgos.SimulatedAnnealing,
	oalgos.AnarchicSocietyOptimization,
	# oalgos.TabuSearch
]

class Runner:
	r"""Runner utility feature.

	Feature which enables running multiple algorithms with multiple benchmarks.
	It also support exporting results in various formats (e.g. LaTeX, Excel, JSON)

	Attributes:
		D (int): Dimension of problem
		NP (int): Population size
		nFES (int): Number of function evaluations
		nRuns (int): Number of repetitions
		useAlgorithms (list of Algorithm): List of algorithms to run
		useBenchmarks (list of Benchmarks): List of benchmarks to run
		results (List[str]): Results of runs.
	"""
	def __init__(self, D=10, nFES=1000000, nGEN=100000, nRuns=1, useAlgorithms='ArtificialBeeColonyAlgorithm', useBenchmarks='Ackley', **kwargs):
		r"""Initialize Runner.

		**__init__(self, D, NP, nFES, nRuns, useAlgorithms, useBenchmarks, ...)**

		Arguments:
			D (int): Dimension of problem
			NP (int): Population size
			nFES (int): Number of function evaluations
			nRuns (int): Number of repetitions
			useAlgorithms (Union[str, Algorithm, List[Union[Algorithm, str]]): List of algorithms to run
			useBenchmarks (Union[str, Benchmarks, List[Union[str, Benchmarks]]): List of benchmarks to run

		Keyword Args:
			A (float): Laudness
			r (float): Pulse rate
			Qmin (float): Minimum frequency
			Qmax (float): Maximum frequency
			Pa (float): Probability
			F (float): Scalling factor
			F_l (float): Lower limit of scalling factor
			F_u (float): Upper limit of scalling factor
			CR (float): Crossover rate
			alpha (float): Alpha parameter
			betamin (float): Betamin parameter
			gamma (float): Gamma parameter
			p (float): Probability switch
			Ts (float): Tournament selection
			Mr (float): Mutation rate
			C1 (float): Cognitive component
			C2 (float): Social component
			w (float): Inertia weight
			vMin (float): Minimal velocity
			vMax (float): Maximal velocity
			Tao1 (float): Probability
			Tao2 (float): Probability
			n (int): Number of sparks
			mu (float): Mu parameter
			omega (float): TODO
			S_init (float): Initial supply for camel
			E_init (float): Initial endurance for camel
			T_min (float): Minimal temperature
			T_max (float): Maximal temperature
			C_a (float): Amplification factor
			C_r (float): Reduction factor
			Limit (int): Limit
			k (int): Number of runs before adaptive
		"""
		self.D = D
		self.nFES = nFES
		self.nGEN = nGEN
		self.useAlgorithms = useAlgorithms
		self.useBenchmarks = useBenchmarks
		self.args = kwargs
		self.results = {}

	@staticmethod
	def getAlgorithm(name):
		r"""Get algorithm for optimization.

		Args:
			name (str): Name of the algorithm

		Returns:
			Algorithm: TODO
		"""
		algorithm = None
		for alg in NiaPyAlgos:
			if name in alg.Name: algorithm = alg; break
		if algorithm == None: raise TypeError('Passed algorithm is not defined!')
		return algorithm

	def benchmarkFactory(self, name):
		r"""Create optimization task.

		Args:
			name (str): Benchmark name.

		Returns:
			Task: Optimization task to use.
		"""
		return util.StoppingTask(D=self.D, nFES=self.nFES, nGEN=self.nGEN, optType=util.OptimizationType.MINIMIZATION, benchmark=name)

	def algorithmFactory(self, name):
		r"""TODO.

		Args:
			name (str): Name of algorithm.

		Returns:
			Algorithm: Initialized algorithm with parameters.
		"""
		algorithm, params = Runner.getAlgorithm(name), dict()
		for k, v in algorithm.typeParameters().items():
			val = self.args.get(k, None)
			if val != None and v(val): params[k] = val
		return algorithm(**params)

	@classmethod
	def __createExportDir(cls):
		r"""TODO."""
		if not os.path.exists('export'): os.makedirs('export')

	@classmethod
	def __generateExportName(cls, extension):
		r"""TODO.

		Args:
			extension:

		Returns:

		"""
		return 'export/' + str(datetime.datetime.now()).replace(':', '.') + '.' + extension

	def __exportToLog(self):
		r"""TODO."""
		print(self.results)

	def __exportToJson(self):
		r"""TODO.

		See Also:
			* :func:`NiaPy.Runner.__createExportDir`
		"""
		self.__createExportDir()
		with open(self.__generateExportName('json'), 'w') as outFile:
			json.dump(self.results, outFile)
			logger.info('Export to JSON completed!')

	def __exportToXls(self):
		r"""TODO.

		See Also:
			:func:`NiaPy.Runner.__generateExportName`
		"""
		self.__createExportDir()
		workbook = xlsxwriter.Workbook(self.__generateExportName('xlsx'))
		worksheet = workbook.add_worksheet()
		row, col, nRuns = 0, 0, 0
		for alg in self.results:
			_, col = worksheet.write(row, col, alg), col + 1
			for bench in self.results[alg]:
				worksheet.write(row, col, bench)
				nRuns = len(self.results[alg][bench])
				for i in range(len(self.results[alg][bench])): _, row = worksheet.write(row, col, self.results[alg][bench][i]), row + 1
				row, col = row - len(self.results[alg][bench]), col + 1
			row, col = row + 1 + nRuns, col - 1 + len(self.results[alg])
		workbook.close()
		logger.info('Export to XLSX completed!')

	def __exportToLatex(self):
		r"""TODO.

		See Also:
			:func:`NiaPy.Runner.__createExportDir`
			:func:`NiaPy.Runner.__generateExportName`
		"""
		self.__createExportDir()
		metrics = ['Best', 'Median', 'Worst', 'Mean', 'Std.']
		def only_upper(s): return "".join(c for c in s if c.isupper())
		with open(self.__generateExportName('tex'), 'a') as outFile:
			outFile.write('\\documentclass{article}\n')
			outFile.write('\\usepackage[utf8]{inputenc}\n')
			outFile.write('\\usepackage{siunitx}\n')
			outFile.write('\\sisetup{\n')
			outFile.write('round-mode=places,round-precision=3}\n')
			outFile.write('\\begin{document}\n')
			outFile.write('\\begin{table}[h]\n')
			outFile.write('\\centering\n')
			begin_tabular = '\\begin{tabular}{cc'
			for alg in self.results:
				for _i in range(len(self.results[alg])): begin_tabular += 'S'
				firstLine = '   &'
				for benchmark in self.results[alg].keys(): firstLine += '  &   \\multicolumn{1}{c}{\\textbf{' + benchmark + '}}'
				firstLine += ' \\\\'
				break
			begin_tabular += '}\n'
			outFile.write(begin_tabular)
			outFile.write('\\hline\n')
			outFile.write(firstLine + '\n')
			outFile.write('\\hline\n')
			for alg in self.results:
				for metric in metrics:
					line = ''
					if metric != 'Worst': line += '   &   ' + metric
					else:
						shortAlg = ''
						if alg.endswith('Algorithm'):	shortAlg = only_upper(alg[:-9])
						else: shortAlg = only_upper(alg)
						line += '\\textbf{' + shortAlg + '} &   ' + metric
						for benchmark in self.results[alg]:
							if metric == 'Best':	line += '   &   ' + str(amin(self.results[alg][benchmark]))
							elif metric == 'Median': line += '   &   ' + str(median(self.results[alg][benchmark]))
							elif metric == 'Worst': line += '   &   ' + str(amax(self.results[alg][benchmark]))
							elif metric == 'Mean': line += '   &   ' + str(mean(self.results[alg][benchmark]))
							else: line += '   &   ' + str(std(self.results[alg][benchmark]))
						line += '   \\\\'
						outFile.write(line + '\n')
					outFile.write('\\hline\n')
				outFile.write('\\end{tabular}\n')
				outFile.write('\\end{table}\n')
				outFile.write('\\end{document}')
		logger.info('Export to Latex completed!')

	def run(self, export='log', verbose=False):
		"""Execute runner.

		Arguments:
			export (str): Takes export type (e.g. log, json, xlsx, latex) (default: 'log')
			verbose (bool: Switch for verbose logging (default: {False})

		Raises:
			TypeError: Raises TypeError if export type is not supported

		Returns:
			dict: Returns dictionary of results

		See Also:
			* :func:`NiaPy.Runner.useAlgorithms`
			* :func:`NiaPy.Runner.useBenchmarks`
			* :func:`NiaPy.Runner.__algorithmFactory`
		"""
		for alg in self.useAlgorithms:
			self.results[alg] = {}
			if verbose:	logger.info('Running %s...', alg)
			for bench in self.useBenchmarks:
				benchName = ''
				if not isinstance(bench, ''.__class__): benchName = str(type(bench).__name__)
				else: benchName = bench
				if verbose: logger.info('Running %s algorithm on %s benchmark...', alg, benchName)
				bm = self.benchmarkFactory(bench)
				self.results[alg][benchName] = []
				for _ in range(self.nGEN):
					algorithm = self.algorithmFactory(alg)
					self.results[alg][benchName].append(algorithm.run(bm))
			if verbose: logger.info('---------------------------------------------------')
		if export == 'log': self.__exportToLog()
		elif export == 'json': self.__exportToJson()
		elif export == 'xlsx': self.__exportToXls()
		elif export == 'latex': self.__exportToLatex()
		else: raise TypeError('Passed export type is not supported!')
		return self.results

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
