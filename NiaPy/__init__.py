from __future__ import print_function  # for backward compatibility purpose

import os
import logging
import json
import datetime
import xlsxwriter
from tabulate import tabulate
from NiaPy import algorithms, benchmarks

__all__ = ['algorithms', 'benchmarks']
__project__ = 'NiaPy'
__version__ = '0.0.0'

VERSION = "{0} v{1}".format(__project__, __version__)

logging.basicConfig()
logger = logging.getLogger('NiaPy')
logger.setLevel('INFO')


class Runner(object):
    # pylint: disable=too-many-instance-attributes, too-many-locals
    def __init__(self, D, NP, nFES, nRuns, useAlgorithms, useBenchmarks,
                 A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, F=0.5, CR=0.9, alpha=0.5,
                 betamin=0.2, gamma=1.0, p=0.5):
        self.D = D
        self.NP = NP
        self.nFES = nFES
        self.nRuns = nRuns
        self.useAlgorithms = useAlgorithms
        self.useBenchmarks = useBenchmarks
        self.A = A
        self.r = r
        self.Qmin = Qmin
        self.Qmax = Qmax
        self.F = F
        self.CR = CR
        self.alpha = alpha
        self.betamin = betamin
        self.gamma = gamma
        self.p = p
        self.results = {}

    def __algorithmFactory(self, name, benchmark):
        bench = benchmarks.utility.Utility().get_benchmark(benchmark)
        algorithm = None

        if name == 'BatAlgorithm':
            algorithm = algorithms.basic.BatAlgorithm(
                self.D, self.NP, self.nFES, self.A, self.r, self.Qmin, self.Qmax, bench)
        elif name == 'DifferentialEvolutionAlgorithm':
            algorithm = algorithms.basic.DifferentialEvolutionAlgorithm(
                self.D, self.NP, self.nFES, self.F, self.CR, bench)
        elif name == 'FireflyAlgorithm':
            algorithm = algorithms.basic.FireflyAlgorithm(
                self.D, self.NP, self.nFES, self.alpha, self.betamin, self.gamma, bench)
        elif name == 'FlowerPollinationAlgorithm':
            algorithm = algorithms.basic.FlowerPollinationAlgorithm(
                self.D, self.NP, self.nFES, self.p, bench)
        elif name == 'GreyWolfOptimizer':
            algorithm = algorithms.basic.GreyWolfOptimizer(
                self.D, self.NP, self.nFES, bench)
        elif name == 'ArtificialBeeColonyAlgorithm':
            algorithm = algorithms.basic.ArtificialBeeColonyAlgorithm(self.D, self.NP, self.nFES, bench)
        elif name == 'HybridBatAlgorithm':
            algorithm = algorithms.modified.HybridBatAlgorithm(
                self.D, self.NP, self.nFES, self.A, self.r, self.F, self.CR, self.Qmin, self.Qmax, bench)
        else:
            raise TypeError('Passed benchmark is not defined!')

        return algorithm

    @classmethod
    def __createExportDir(cls):
        if not os.path.exists('export'):
            os.makedirs('export')

    @classmethod
    def __generateExportName(cls, extension):
        return 'export/' + str(datetime.datetime.now()) + '.' + extension

    def __exportToLog(self):
        print(self.results)

    def __exportToJson(self):
        self.__createExportDir()
        with open(self.__generateExportName('json'), 'w') as outFile:
            json.dump(self.results, outFile)
            logger.info('Export to JSON completed!')

    def __exportToXls(self):
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

        logger.info('Export to XLSX completed!')

    def __exportToLatex(self):
        table = [["spam", 42], ["eggs", 451], ["bacon", 0]]
        headers = ["item", "qty"]
        print(tabulate(table, headers, tablefmt="latex"))
        # TODO: implement export to Latex
        pass

    def run(self, export='log', verbose=False):
        for alg in self.useAlgorithms:
            self.results[alg] = {}
            if verbose:
                logger.info('Running %s...', alg)
            for bench in self.useBenchmarks:
                benchName = ''
                # check if passed benchmark is class
                if not isinstance(bench, ''.__class__):
                    # set class name as benchmark name
                    benchName = str(type(bench).__name__)
                else:
                    benchName = bench

                if verbose:
                    logger.info('Running %s algorithm on %s benchmark...', alg, benchName)

                self.results[alg][benchName] = []

                for _i in range(self.nRuns):
                    algorithm = self.__algorithmFactory(alg, bench)
                    self.results[alg][benchName].append(algorithm.run())

            if verbose:
                logger.info('---------------------------------------------------')

        if export == 'log':
            self.__exportToLog()
        elif export == 'json':
            self.__exportToJson()
        elif export == 'xlsx':
            self.__exportToXls()
        elif export == 'latex':
            self.__exportToLatex()
        else:
            raise TypeError('Passed export type is not supported!')

        return self.results
