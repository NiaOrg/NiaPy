# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, multiple-statements, too-many-function-args
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
__version__ = '1.0.1'

VERSION = "{0} v{1}".format(__project__, __version__)

logging.basicConfig()
logger = logging.getLogger('NiaPy')
logger.setLevel('INFO')


class Runner(object):
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

		Tao {decimal}

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

		"""
		self.D = D
		self.NP = NP
		self.nFES = nFES
		self.nRuns = nRuns
		self.useAlgorithms = useAlgorithms
		self.useBenchmarks = useBenchmarks
		self.A = kwargs.get('A', 0.5)
		self.r = kwargs.get('r', 0.5)
		self.Qmin = kwargs.get('Qmin', 0.0)
		self.Qmax = kwargs.get('Qmax', 2.0)
		self.Pa = kwargs.get('Pa', 0.25)
		self.F = kwargs.get('F', 0.5)
		self.CR = kwargs.get('CR', 0.9)
		self.alpha = kwargs.get('alpha', 0.5)
		self.betamin = kwargs.get('betamin', 0.2)
		self.gamma = kwargs.get('gamma', 1.0)
		self.p = kwargs.get('p', 0.5)
		self.Ts = kwargs.get('Ts', 4)
		self.Mr = kwargs.get('Mr', 0.05)
		self.C1 = kwargs.get('C1', 2.0)
		self.C2 = kwargs.get('C2', 2.0)
		self.w = kwargs.get('w', 0.7)
		self.vMin = kwargs.get('vMin', -4)
		self.vMax = kwargs.get('vMax', 4)
		self.Tao = kwargs.get('Tao', 0.1)
		self.n = kwargs.get('n', 10)
		self.omega = kwargs.get('omega', 0.25)
		self.mu = kwargs.get('mu', 0.5)
		self.E_init = kwargs.get('E_init', 10)
		self.S_init = kwargs.get('S_init', 10)
		self.T_min = kwargs.get('T_min', -10)
		self.T_max = kwargs.get('T_max', 10)
		self.C_a = kwargs.get('C_a', 2)
		self.C_r = kwargs.get('C_r', 0.5)
		self.Limit = kwargs.get('Limit', 100)
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
			algorithm = algorithms.basic.FlowerPollinationAlgorithm(self.D, self.NP, self.nFES, self.p, bench)
		elif name == 'GreyWolfOptimizer':
			algorithm = algorithms.basic.GreyWolfOptimizer(D=self.D, NP=self.NP, nFES=self.nFES, benchmark=bench)
		elif name == 'ArtificialBeeColonyAlgorithm':
			algorithm = algorithms.basic.ArtificialBeeColonyAlgorithm(D=self.D, NP=self.NP, nFES=self.nFES, Limit=self.Limit, benchmark=bench)
		elif name == 'GeneticAlgorithm':
			algorithm = algorithms.basic.GeneticAlgorithm(D=self.D, NP=self.NP, nFES=self.nFES, Ts=self.Ts, Mr=self.Mr, Cr=self.CR, benchmark=bench)
		elif name == 'ParticleSwarmAlgorithm':
			algorithm = algorithms.basic.ParticleSwarmAlgorithm(D=self.D, NP=self.NP, nFES=self.nFES, C1=self.C1, C2=self.C2, w=self.w, vMin=self.vMin, vMax=self.vMax, benchmark=bench)
		elif name == 'HybridBatAlgorithm':
			algorithm = algorithms.modified.HybridBatAlgorithm(self.D, self.NP, self.nFES, self.A, self.r, self.F, self.CR, self.Qmin, self.Qmax, bench)
		elif name == 'SelfAdaptiveDifferentialEvolutionAlgorithm':
			algorithm = algorithms.modified.SelfAdaptiveDifferentialEvolutionAlgorithm(self.D, self.NP, self.nFES, self.F, self.CR, self.Tao, bench)
		elif name == 'CamelAlgorithm':
			algorithm = algorithms.basic.CamelAlgorithm(NP=self.NP, D=self.D, nGEN=self.nRuns, nFES=self.nFES, omega=self.omega, mu=self.mu, alpha=self.alpha, S_init=self.S_init, E_init=self.E_init, T_min=self.T_min, T_max=self.T_max, benchmark=bench)
		elif name == 'BareBonesFireworksAlgorithm':
			algorithm = algorithm.basic.BareBonesFireworksAlgorithm(D=self.D, nFES=self.nFES, n=self.n, C_a=self.C_a, C_r=self.C_r, benchmark=bench)
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
