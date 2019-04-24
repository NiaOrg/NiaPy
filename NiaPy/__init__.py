# encoding=utf8

"""Python micro framework for building nature-inspired algorithms."""

from __future__ import print_function

import os
import logging
import json
import datetime
import xlsxwriter
from numpy import amin, amax, median, mean, std

from NiaPy import util, algorithms, benchmarks, task
from NiaPy.algorithms import AlgorithmUtility

__all__ = ["algorithms", "benchmarks", "util", "task"]
__project__ = "NiaPy"
__version__ = "2.0.0rc4"

VERSION = "{0} v{1}".format(__project__, __version__)

logging.basicConfig()
logger = logging.getLogger("NiaPy")
logger.setLevel("INFO")


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
                nFES (int): Number of function evaluations
                nGEN (int): Number of generations
                nRuns (int): Number of repetitions
                useAlgorithms (list of Algorithm): List of algorithms to run
                useBenchmarks (list of Benchmarks): List of benchmarks to run

        """

        self.D = D
        self.nFES = nFES
        self.nRuns = nRuns
        self.useAlgorithms = useAlgorithms
        self.useBenchmarks = useBenchmarks
        self.results = {}

    def benchmarkFactory(self, name):
        r"""Create optimization task.

        Args:
                name (str): Benchmark name.

        Returns:
                Task: Optimization task to use.

        """

        from NiaPy.task import StoppingTask, OptimizationType
        return StoppingTask(D=self.D, nFES=self.nFES, optType=OptimizationType.MINIMIZATION, benchmark=name)

    @classmethod
    def __createExportDir(cls):
        r"""Create export directory if not already createed."""
        if not os.path.exists("export"):
            os.makedirs("export")

    @classmethod
    def __generateExportName(cls, extension):
        r"""Generate export file name.

        Args:
                extension (str): File format.

        Returns:

        """

        return "export/" + str(datetime.datetime.now()).replace(":", ".") + "." + extension

    def __exportToLog(self):
        r"""Print the results to terminal."""

        print(self.results)

    def __exportToJson(self):
        r"""Export the results in the JSON form.

        See Also:
                * :func:`NiaPy.Runner.__createExportDir`

        """

        self.__createExportDir()
        with open(self.__generateExportName("json"), "w") as outFile:
            json.dump(self.results, outFile)
            logger.info("Export to JSON completed!")

    def __exportToXls(self):
        r"""Export the results in the xlsx form.

        See Also:
                :func:`NiaPy.Runner.__generateExportName`

        """

        self.__createExportDir()
        workbook = xlsxwriter.Workbook(self.__generateExportName("xlsx"))
        worksheet = workbook.add_worksheet()
        row, col, nRuns = 0, 0, 0

        for alg in self.results:
            _, col = worksheet.write(row, col, alg), col + 1
            for bench in self.results[alg]:
                worksheet.write(row, col, bench)
                nRuns = len(self.results[alg][bench])
                for i in range(len(self.results[alg][bench])):
                    _, row = worksheet.write(row, col, self.results[alg][bench][i]), row + 1
                row, col = row - len(self.results[alg][bench]), col + 1
            row, col = row + 1 + nRuns, col - 1 + len(self.results[alg])

        workbook.close()
        logger.info("Export to XLSX completed!")

    def __exportToLatex(self):
        r"""Export the results in the form of latex table.

        See Also:
                :func:`NiaPy.Runner.__createExportDir`
                :func:`NiaPy.Runner.__generateExportName`

        """

        self.__createExportDir()

        metrics = ["Best", "Median", "Worst", "Mean", "Std."]

        def only_upper(s):
            return "".join(c for c in s if c.isupper())

        with open(self.__generateExportName("tex"), "a") as outFile:
            outFile.write("\\documentclass{article}\n")
            outFile.write("\\usepackage[utf8]{inputenc}\n")
            outFile.write("\\usepackage{siunitx}\n")
            outFile.write("\\sisetup{\n")
            outFile.write("round-mode=places,round-precision=3}\n")
            outFile.write("\\begin{document}\n")
            outFile.write("\\begin{table}[h]\n")
            outFile.write("\\centering\n")
            begin_tabular = "\\begin{tabular}{cc"
            for alg in self.results:
                for _i in range(len(self.results[alg])):
                    begin_tabular += "S"
                firstLine = "   &"
                for benchmark in self.results[alg].keys():
                    firstLine += "  &   \\multicolumn{1}{c}{\\textbf{" + benchmark + "}}"
                firstLine += " \\\\"
                break
            begin_tabular += "}\n"
            outFile.write(begin_tabular)
            outFile.write("\\hline\n")
            outFile.write(firstLine + "\n")
            outFile.write("\\hline\n")
            for alg in self.results:
                for metric in metrics:
                    line = ""
                    if metric != "Worst":
                        line += "   &   " + metric
                    else:
                        shortAlg = ""
                        if alg.endswith("Algorithm"):
                            shortAlg = only_upper(alg[:-9])
                        else:
                            shortAlg = only_upper(alg)
                        line += "\\textbf{" + shortAlg + "} &   " + metric
                        for benchmark in self.results[alg]:
                            if metric == "Best":
                                line += "   &   " + str(amin(self.results[alg][benchmark]))
                            elif metric == "Median":
                                line += "   &   " + str(median(self.results[alg][benchmark]))
                            elif metric == "Worst":
                                line += "   &   " + str(amax(self.results[alg][benchmark]))
                            elif metric == "Mean":
                                line += "   &   " + str(mean(self.results[alg][benchmark]))
                            else:
                                line += "   &   " + str(std(self.results[alg][benchmark]))
                        line += "   \\\\"
                        outFile.write(line + "\n")
                    outFile.write("\\hline\n")
                outFile.write("\\end{tabular}\n")
                outFile.write("\\end{table}\n")
                outFile.write("\\end{document}")
        logger.info("Export to Latex completed!")

    def run(self, export="log", verbose=False):
        """Execute runner.

        Arguments:
                export (str): Takes export type (e.g. log, json, xlsx, latex) (default: "log")
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
            alg_name = ""
            if not isinstance(alg, "".__class__):
                alg_name = str(type(alg).__name__)
            else:
                alg_name = alg

            self.results[alg_name] = {}
            if verbose:
                logger.info("Running %s...", alg_name)

            for bench in self.useBenchmarks:
                bench_name = ""
                if not isinstance(bench, "".__class__):
                    bench_name = str(type(bench).__name__)
                else:
                    bench_name = bench
                if verbose:
                    logger.info("Running %s algorithm on %s benchmark...", alg_name, bench_name)

                benchmark_stopping_task = self.benchmarkFactory(bench)
                self.results[alg_name][bench_name] = []
                for _ in range(self.nRuns):
                    algorithm = AlgorithmUtility().get_algorithm(alg)
                    self.results[alg_name][bench_name].append(algorithm.run(benchmark_stopping_task))
            if verbose:
                logger.info("---------------------------------------------------")
        if export == "log":
            self.__exportToLog()
        elif export == "json":
            self.__exportToJson()
        elif export == "xlsx":
            self.__exportToXls()
        elif export == "latex":
            self.__exportToLatex()
        else:
            raise TypeError("Passed export type is not supported!")
