# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, multiple-statements, too-many-function-args, old-style-class
"""Python micro framework for building nature-inspired algorithms."""

from __future__ import print_function  # for backward compatibility purpose

import os
import logging
import json
import datetime
import xlsxwriter
from numpy import amin, amax, median, mean, std
from NiaPy import algorithms, benchmarks, util

__all__ = ['algorithms', 'benchmarks', 'util']
__project__ = 'NiaPy'
__version__ = '1.0.1'

VERSION = "{0} v{1}".format(__project__, __version__)

logging.basicConfig()
logger = logging.getLogger('NiaPy')
logger.setLevel('INFO')

class Runner:
	r"""Runner utility feature.

	Feature which enables running multiple algorithms with multiple benchmarks.
	It also support exporting results in various formats (e.g. LaTeX, Excel, JSON)
	"""
	def __init__(self, D=10, nFES=1000000, nRuns=100000, useAlgorithms='ArtificialBeeColonyAlgorithm', useBenchmarks='Ackley', **kwargs):
		r"""Initialize Runner.

		**__init__(self, D, NP, nFES, nRuns, useAlgorithms, useBenchmarks, ...)**

		Arguments:
		D {integer} -- dimension of problem

		NP {integer} -- population size

		nFES {integer} -- number of function evaluations

		nRuns {integer} -- number of repetitions

		useAlgorithms [] -- array of algorithms to run

		useBenchmarks [] -- array of benchmarks to run

		A {decimal} -- laudness

		r {decimal} -- pulse rate

		Qmin {decimal} -- minimum frequency

		Qmax {decimal} -- maximum frequency

		Pa {decimal} -- probability

		F {decimal} -- scalling factor

		F_l {decimal} -- lower limit of scalling factor

		F_u {decimal} -- upper limit of scalling factor

		CR {decimal} -- crossover rate

		alpha {decimal} -- alpha parameter

		betamin {decimal} -- betamin parameter

		gamma {decimal} -- gamma parameter

		p {decimal} -- probability switch

		Ts {decimal} -- tournament selection

		Mr {decimal} -- mutation rate

		C1 {decimal} -- cognitive component

		C2 {decimal} -- social component

		w {decimal} -- inertia weight

		vMin {decimal} -- minimal velocity

		vMax {decimal} -- maximal velocity

		Tao1 {decimal} --

		Tao2 {decimal} --

		n {integer} -- number of sparks

		mu {decimal} -- mu parameter

		omega {decimal} -- TODO

		S_init {decimal} -- initial supply for camel

		E_init {decimal} -- initial endurance for camel

		T_min {decimal} -- minimal temperature

		T_max {decimal} -- maximal temperature

		C_a {decimal} -- Amplification factor

		C_r {decimal} -- Reduction factor

		Limit {integer} -- Limit

		k {integer} -- Number of runs before adaptive
		"""
		self.D = D
		self.nFES = nFES
		self.nGEN = nRuns
		self.useAlgorithms = useAlgorithms
		self.useBenchmarks = useBenchmarks
		self.args = kwargs
		self.results = {}

	@staticmethod
	def getAlgorithm(name):
		algorithm = None
		if name in ['BatAlgorithm', 'BT']: algorithm = algorithms.basic.BatAlgorithm
		elif name in ['DifferentialEvolutionAlgorithm', 'DE']: algorithm = algorithms.basic.DifferentialEvolutionAlgorithm
		elif name in ['FireflyAlgorithm', 'FA']: algorithm = algorithms.basic.FireflyAlgorithm
		elif name in ['FlowerPollinationAlgorithm', 'FPA']: algorithm = algorithms.basic.FlowerPollinationAlgorithm
		elif name in ['GreyWolfOptimizer', 'GWO']: algorithm = algorithms.basic.GreyWolfOptimizer
		elif name in ['ArtificialBeeColonyAlgorithm', 'ABC']: algorithm = algorithms.basic.ArtificialBeeColonyAlgorithm
		elif name in ['GeneticAlgorithm', 'GA']: algorithm = algorithms.basic.GeneticAlgorithm
		elif name in ['ParticleSwarmAlgorithm', 'PSO']: algorithm = algorithms.basic.ParticleSwarmAlgorithm
		elif name in ['HybridBatAlgorithm', 'HBA']: algorithm = algorithms.modified.HybridBatAlgorithm
		elif name in ['SelfAdaptiveDifferentialEvolutionAlgorithm', 'jDE']: algorithm = algorithms.modified.SelfAdaptiveDifferentialEvolutionAlgorithm
		elif name in ['CamelAlgorithm', 'CA']: algorithm = algorithms.basic.CamelAlgorithm
		elif name in ['BareBonesFireworksAlgorithm', 'BBFWA']: algorithm = algorithms.basic.BareBonesFireworksAlgorithm
		elif name in ['MonkeyKingEvolutionV1', 'MKE', 'MKEv1']: algorithm = algorithms.basic.MonkeyKingEvolutionV1
		elif name in ['MonkeyKingEvolutionV2', 'MKEv2']: algorithm = algorithms.basic.MonkeyKingEvolutionV2
		elif name in ['MonkeyKingEvolutionV3', 'MKEv3']: algorithm = algorithms.basic.MonkeyKingEvolutionV3
		elif name in ['EvolutionStrategy1p1', 'ES 1+1']: algorithm = algorithms.basic.EvolutionStrategy1p1
		elif name in ['EvolutionStrategyMp1', 'ES M+1']: algorithm = algorithms.basic.EvolutionStrategyMp1
		elif name in ['SineCosineAlgorithm', 'SCA']: algorithm = algorithms.basic.SineCosineAlgorithm
		elif name in ['HarmonySearch', 'HS']: algorithm = algorithms.basic.HarmonySearch
		elif name in ['HarmonySearchV1', 'HSv1']: algorithm = algorithms.basic.HarmonySearchV1
		elif name in ['GlowwormSwarmOptimization', 'GSO']: algorithm = algorithms.basic.GlowwormSwarmOptimization
		elif name in ['GlowwormSwarmOptimizationV1', 'GSOv1']: algorithm = algorithms.basic.GlowwormSwarmOptimizationV1
		elif name in ['GlowwormSwarmOptimizationV2', 'GSOv2']: algorithm = algorithms.basic.GlowwormSwarmOptimizationV2
		elif name in ['GlowwormSwarmOptimizationV3', 'GSOv3']: algorithm = algorithms.basic.GlowwormSwarmOptimizationV3
		elif name in ['KrillHerdV1', 'KHv1']: algorithm = algorithms.basic.KrillHerdV1
		elif name in ['KrillHerdV2', 'KHv2']: algorithm = algorithms.basic.KrillHerdV2
		elif name in ['KrillHerdV3', 'KHv3']: algorithm = algorithms.basic.KrillHerdV3
		elif name in ['KrillHerdV4', 'KHv4']: algorithm = algorithms.basic.KrillHerdV4
		elif name in ['KrillHerdV11', 'KHv11']: algorithm = algorithms.basic.KrillHerdV11
		elif name in ['FireworksAlgorithm', 'FWA']: algorithm = algorithms.basic.FireworksAlgorithm
		elif name in ['EnhancedFireworksAlgorithm', 'EFWA']: algorithm = algorithms.basic.EnhancedFireworksAlgorithm
		elif name in ['DynamicFireworksAlgorithm', 'dynFWA']: algorithm = algorithms.basic.DynamicFireworksAlgorithm
		elif name in ['DynamicFireworksAlgorithmGauss', 'dynFWAG']: algorithm = algorithms.basic.DynamicFireworksAlgorithmGauss
		elif name in ['MultipleTrajectorySearch', 'MTS', 'MTS_lsgo']: algorithm = algorithms.other.MultipleTrajectorySearch
		elif name in ['MultipleTrajectorySearchV1', 'MTSv1', 'MTS_moo']: algorithm = algorithms.other.MultipleTrajectorySearchV1
		elif name in ['NelderMeadMethod', 'NMM']: algorithm = algorithms.other.NelderMeadMethod
		elif name in ['HillClimbAlgorithm', 'HCA']: algorithm = algorithms.other.HillClimbAlgorithm
		elif name in ['SimulatedAnnealing', 'SA']: algorithm = algorithms.other.SimulatedAnnealing
		elif name in ['GravitationalSearchAlgorithm', 'GSA']: algorithm = algorithms.basic.GravitationalSearchAlgorithm
		elif name in ['AnarchicSocietyOptimization', 'ASO']: algorithm = algorithms.other.AnarchicSocietyOptimization
		elif name in ['CovarianceMaatrixAdaptionEvolutionStrategy', 'CMA ES']: algorithm = algorithms.basic.CovarianceMaatrixAdaptionEvolutionStrategy
		elif name in ['TabuSearch', 'TS']: algorithm = algorithms.other.TabuSearch
		else: raise TypeError('Passed algorithm is not defined!')
		return algorithm

	def benchmarkFactory(self, name): return util.Task(D=self.D, nFES=self.nFES, nGEN=self.nGEN, optType=util.OptimizationType.MINIMIZATION, benchmark=name)

	def algorithmFactory(self, name):
		algorithm, params = Runner.getAlgorithm(name), dict()
		for k, v in algorithm.typeParameters().items():
			val = self.args.get(k, None)
			if val != None and v(val): params[k] = val
		return algorithm(**params)

	def __algorithmFactory(self, aname, bname): return self.algorithmFactory(aname).setTask(self.benchmarkFactory(bname))

	@classmethod
	def __createExportDir(cls):
		if not os.path.exists('export'): os.makedirs('export')

	@classmethod
	def __generateExportName(cls, extension): return 'export/' + str(datetime.datetime.now()).replace(':', '.') + '.' + extension

	def __exportToLog(self): print(self.results)

	def __exportToJson(self):
		self.__createExportDir()
		with open(self.__generateExportName('json'), 'w') as outFile:
			json.dump(self.results, outFile)
			logger.info('Export to JSON completed!')

	def __exportToXls(self):
		self.__createExportDir()
		workbook = xlsxwriter.Workbook(self.__generateExportName('xlsx'))
		worksheet = workbook.add_worksheet()
		row, col, nRuns = 0, 0, 0
		for alg in self.results:
			_, col = worksheet.write(row, col, alg),  col + 1
			for bench in self.results[alg]:
				worksheet.write(row, col, bench)
				nRuns = len(self.results[alg][bench])
				for i in range(len(self.results[alg][bench])): _, row = worksheet.write(row, col, self.results[alg][bench][i]), row + 1
				row, col = row - len(self.results[alg][bench]), col + 1
			row, col = row + 1 + nRuns, col - 1 + len(self.results[alg])
		workbook.close()
		logger.info('Export to XLSX completed!')

	def __exportToLatex(self):
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

		Keyword Arguments:
		export  {string}  -- takes export type (e.g. log, json, xlsx, latex) (default: 'log')
		verbose {boolean} -- switch for verbose logging (default: {False})

		Raises:
		TypeError -- Raises TypeError if export type is not supported

		Returns:
		Dictionary -- Returns dictionary of results
		"""
		for alg in self.useAlgorithms:
			self.results[alg] = {}
			if verbose:	logger.info('Running %s...', alg)
			for bench in self.useBenchmarks:
				benchName = ''
				if not isinstance(bench, ''.__class__): benchName = str(type(bench).__name__)
				else: benchName = bench
				if verbose: logger.info('Running %s algorithm on %s benchmark...', alg, benchName)
				self.results[alg][benchName] = []
				for _ in range(self.nGEN):
					algorithm = self.__algorithmFactory(alg, bench)
					self.results[alg][benchName].append(algorithm.run())
			if verbose: logger.info('---------------------------------------------------')
		if export == 'log': self.__exportToLog()
		elif export == 'json': self.__exportToJson()
		elif export == 'xlsx': self.__exportToXls()
		elif export == 'latex': self.__exportToLatex()
		else: raise TypeError('Passed export type is not supported!')
		return self.results

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
