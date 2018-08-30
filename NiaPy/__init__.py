# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, multiple-statements, too-many-function-args, old-style-class
"""Python micro framework for building nature-inspired algorithms."""

from __future__ import print_function  # for backward compatibility purpose

import os
import logging
import json
import datetime
import xlsxwriter
import numpy as np
from NiaPy import algorithms, benchmarks

__all__ = ['algorithms', 'benchmarks']
__project__ = 'NiaPy'
__version__ = '2.0.0rc2'

VERSION = "{0} v{1}".format(__project__, __version__)

logging.basicConfig()
logger = logging.getLogger('NiaPy')
logger.setLevel('INFO')

class Runner:
	r"""Runner utility feature.

	Feature which enables running multiple algorithms with multiple benchmarks.
	It also support exporting results in various formats (e.g. LaTeX, Excel, JSON)
	"""
	def __init__(self, D, NP, nFES, nRuns, useAlgorithms, useBenchmarks, **kwargs):
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
		self.NP = NP
		self.nFES = nFES
		self.nRuns = nRuns
		self.useAlgorithms = useAlgorithms
		self.useBenchmarks = useBenchmarks
		self.A = kwargs.pop('A', 0.5)
		self.a = kwargs.pop('a', 5)
		self.r = kwargs.pop('r', 0.5)
		self.Qmin = kwargs.pop('Qmin', 0.0)
		self.Qmax = kwargs.pop('Qmax', 2.0)
		self.Pa = kwargs.pop('Pa', 0.25)
		self.F = kwargs.pop('F', 0.5)
		self.F_l = kwargs.pop('F_l', 0.0)
		self.F_u = kwargs.pop('F_u', 2.0)
		self.CR = kwargs.pop('CR', 0.9)
		self.alpha = kwargs.pop('alpha', 0.5)
		self.beta = kwargs.pop('beta', 2)
		self.betamin = kwargs.pop('betamin', 0.2)
		self.gamma = kwargs.pop('gamma', 1.0)
		self.p = kwargs.pop('p', 0.5)
		self.Ts = kwargs.pop('Ts', 4)
		self.Mr = kwargs.pop('Mr', 0.05)
		self.C1 = kwargs.pop('C1', 2.0)
		self.C2 = kwargs.pop('C2', 2.0)
		self.w = kwargs.pop('w', 0.7)
		self.vMin = kwargs.pop('vMin', -4)
		self.vMax = kwargs.pop('vMax', 4)
		self.Tao1 = kwargs.pop('Tao1', 0.43)
		self.Tao2 = kwargs.pop('Tao2', 0.1)
		self.n = kwargs.pop('n', 10)
		self.omega = kwargs.pop('omega', 0.25)
		self.mu = kwargs.pop('mu', 0.5)
		self.muES = kwargs.pop('muES', 35)
		self.E_init = kwargs.pop('E_init', 10)
		self.S_init = kwargs.pop('S_init', 10)
		self.T_min = kwargs.pop('T_min', -10)
		self.T_max = kwargs.pop('T_max', 10)
		self.C_a = kwargs.pop('C_a', 2)
		self.C_r = kwargs.pop('C_r', 0.5)
		self.Limit = kwargs.pop('Limit', 100)
		self.Rmin = kwargs.pop('Rmin', .0)
		self.Rmax = kwargs.pop('Rmax', 2)
		self.lam = kwargs.pop('lam', 40)
		self.C = kwargs.pop('C', 2)
		self.FC = kwargs.pop('FC', 0.7)
		self.R = kwargs.pop('R', 0.3)
		self.k = kwargs.pop('k', 15)
		self.results = {}

	def __algorithmFactory(self, name, benchmark):
		bench = benchmarks.utility.Utility().get_benchmark(benchmark)
		algorithm = None
		if name == 'BatAlgorithm':
			algorithm = algorithms.basic.BatAlgorithm(D=self.D, NP=self.NP, nFES=self.nFES, A=self.A, r=self.r, Qmin=self.Qmin, Qmax=self.Qmax, benchmark=bench)
		elif name == 'DifferentialEvolutionAlgorithm':
			algorithm = algorithms.basic.DifferentialEvolutionAlgorithm(D=self.D, NP=self.NP, nFES=self.nFES, F=self.F, CR=self.CR, benchmark=bench)
		elif name == 'FireflyAlgorithm':
			algorithm = algorithms.basic.FireflyAlgorithm(D=self.D, NP=self.NP, nFES=self.nFES, alpha=self.alpha, betamin=self.betamin, gamma=self.gamma, benchmark=bench)
		elif name == 'FlowerPollinationAlgorithm':
			algorithm = algorithms.basic.FlowerPollinationAlgorithm(D=self.D, NP=self.NP, nFES=self.nFES, p=self.p, beta=self.beta, benchmark=bench)
		elif name == 'GreyWolfOptimizer':
			algorithm = algorithms.basic.GreyWolfOptimizer(D=self.D, NP=self.NP, nFES=self.nFES, benchmark=bench)
		elif name == 'ArtificialBeeColonyAlgorithm':
			algorithm = algorithms.basic.ArtificialBeeColonyAlgorithm(D=self.D, NP=self.NP, nFES=self.nFES, Limit=self.Limit, benchmark=bench)
		elif name == 'GeneticAlgorithm':
			algorithm = algorithms.basic.GeneticAlgorithm(D=self.D, NP=self.NP, nFES=self.nFES, Ts=self.Ts, Mr=self.Mr, Cr=self.CR, benchmark=bench)
		elif name == 'ParticleSwarmAlgorithm':
			algorithm = algorithms.basic.ParticleSwarmAlgorithm(D=self.D, NP=self.NP, nFES=self.nFES, C1=self.C1, C2=self.C2, w=self.w, vMin=self.vMin, vMax=self.vMax, benchmark=bench)
		elif name == 'HybridBatAlgorithm':
			algorithm = algorithms.modified.HybridBatAlgorithm(D=self.D, NP=self.NP, nFES=self.nFES, A=self.A, r=self.r, F=self.F, CR=self.CR, Qmin=self.Qmin, Qmax=self.Qmax, benchmark=bench)
		elif name == 'SelfAdaptiveDifferentialEvolutionAlgorithm':
			algorithm = algorithms.modified.SelfAdaptiveDifferentialEvolutionAlgorithm(D=self.D, NP=self.NP, nFES=self.nFES, F=self.F, F_l=self.F_l, F_u=self.F_u, Tao1=self.Tao1, CR=self.CR, Tao2=self.Tao2, benchmark=bench)
		elif name == 'CamelAlgorithm':
			algorithm = algorithms.basic.CamelAlgorithm(NP=self.NP, D=self.D, nGEN=self.nRuns, nFES=self.nFES, omega=self.omega, mu=self.mu, alpha=self.alpha, S_init=self.S_init, E_init=self.E_init, T_min=self.T_min, T_max=self.T_max, benchmark=bench)
		elif name == 'BareBonesFireworksAlgorithm':
			algorithm = algorithms.basic.BareBonesFireworksAlgorithm(D=self.D, nFES=self.nFES, n=self.n, C_a=self.C_a, C_r=self.C_r, benchmark=bench)
		elif name == 'MonkeyKingEvolutionV1':
			algorithm = algorithms.basic.MonkeyKingEvolutionV1(D=self.D, nFES=self.nFES, NP=self.NP, C=self.C, R=self.R, FC=self.FC, benchmark=bench)
		elif name == 'MonkeyKingEvolutionV2':
			algorithm = algorithms.basic.MonkeyKingEvolutionV2(D=self.D, nFES=self.nFES, C=self.C, R=self.R, FC=self.FC, benchmark=bench)
		elif name == 'MonkeyKingEvolutionV3':
			algorithm = algorithms.basic.MonkeyKingEvolutionV3(D=self.D, nFES=self.nFES, C=self.C, R=self.R, FC=self.FC, benchmark=bench)
		elif name == 'EvolutionStrategy1p1':
			algorithm = algorithms.basic.EvolutionStrategy1p1(D=self.D, nFES=self.nFES, k=self.k, c_a=self.C_a, c_r=self.C_r, benchmark=bench)
		elif name == 'EvolutionStrategyMp1':
			algorithm = algorithms.basic.EvolutionStrategyMp1(D=self.D, nFES=self.nFES, mu=self.muES, k=self.k, c_a=self.C_a, c_r=self.C_r, benchmark=bench)
		elif name == 'SineCosineAlgorithm':
			algorithm = algorithms.basic.SineCosineAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nRuns, NP=self.NP, a=self.a, Rmin=self.Rmin, Rmax=self.Rmax, benchmark=bench)
		elif name == 'HarmonySearch':
			algorithm = algorithms.basic.HarmonySearch(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'HarmonySearchV1':
			algorithm = algorithms.basic.HarmonySearchV1(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'GlowwormSwarmOptimization':
			algorithm = algorithms.basic.GlowwormSwarmOptimization(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'GlowwormSwarmOptimizationV1':
			algorithm = algorithms.basic.GlowwormSwarmOptimizationV1(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'GlowwormSwarmOptimizationV2':
			algorithm = algorithms.basic.GlowwormSwarmOptimizationV2(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'GlowwormSwarmOptimizationV3':
			algorithm = algorithms.basic.GlowwormSwarmOptimizationV3(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'KrillHerdV1':
			algorithm = algorithms.basic.KrillHerdV1(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'KrillHerdV2':
			algorithm = algorithms.basic.KrillHerdV2(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'KrillHerdV3':
			algorithm = algorithms.basic.KrillHerdV3(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'KrillHerdV4':
			algorithm = algorithms.basic.KrillHerdV4(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'KrillHerdV11':
			algorithm = algorithms.basic.KrillHerdV11(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'FireworksAlgorithm':
			algorithm = algorithms.basic.FireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'EnhancedFireworksAlgorithm':
			algorithm = algorithms.basic.EnhancedFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'DynamicFireworksAlgorithm':
			algorithm = algorithms.basic.DynamicFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nRuns, benchmark=bench)
		elif name == 'MultipleTrajectorySearch':
			algorithm = algorithms.other.MultipleTrajectorySearch(D=self.D, nFES=self.nFES, benchmark=bench)
		elif name == 'MultipleTrajectorySearchV1':
			algorithm = algorithms.other.MultipleTrajectorySearchV1(D=self.D, nFES=self.nFES, benchmark=bench)
		elif name == 'NelderMeadMethod':
			algorithm = algorithms.other.NelderMeadMethod(D=self.D, nFES=self.nFES, benchmark=bench)
		elif name == 'HillClimbAlgorithm':
			algorithm = algorithms.other.HillClimbAlgorithm(D=self.D, nFES=self.nFES, benchmark=bench)
		elif name == 'SimulatedAnnealing':
			algorithm = algorithms.other.SimulatedAnnealing(D=self.D, nFES=self.nFES, benchmark=bench)
		elif name == 'GravitationalSearchAlgorithm':
			algorithm = algorithms.basic.GravitationalSearchAlgorithm(D=self.D, nFES=self.nFES, benchmark=bench)
		elif name == 'AnarchicSocietyOptimization':
			algorithm = algorithms.other.AnarchicSocietyOptimization(D=self.D, nFES=self.nFES, benchmark=bench)
		else:
			raise TypeError('Passed benchmark is not defined!')
		return algorithm

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
		row = 0
		col = 0
		nRuns = 0
		for alg in self.results:
			worksheet.write(row, col, alg)
			col += 1
			for bench in self.results[alg]:
				worksheet.write(row, col, bench)
				nRuns = len(self.results[alg][bench])
				for i in range(len(self.results[alg][bench])):
					row += 1
					worksheet.write(row, col, self.results[alg][bench][i])
				row -= len(self.results[alg][bench])  # jump back up
				col += 1
			row += 1 + nRuns  # jump down to row after previous results
			col -= 1 + len(self.results[alg])
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
				for benchmark in self.results[alg].keys():
					firstLine += '  &   \\multicolumn{1}{c}{\\textbf{' + benchmark + '}}'
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
							if metric == 'Best':	line += '   &   ' + str(np.amin(self.results[alg][benchmark]))
							elif metric == 'Median': line += '   &   ' + str(np.median(self.results[alg][benchmark]))
							elif metric == 'Worst': line += '   &   ' + str(np.amax(self.results[alg][benchmark]))
							elif metric == 'Mean': line += '   &   ' + str(np.mean(self.results[alg][benchmark]))
							else: line += '   &   ' + str(np.std(self.results[alg][benchmark]))
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
				# check if passed benchmark is class
				if not isinstance(bench, ''.__class__):
					# set class name as benchmark name
					benchName = str(type(bench).__name__)
				else: benchName = bench
				if verbose: logger.info('Running %s algorithm on %s benchmark...', alg, benchName)
				self.results[alg][benchName] = []
				for _i in range(self.nRuns):
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
