# encoding=utf8

"""Implementation of Runner utility class."""

import datetime
import json
import os
import logging

import xlsxwriter
from numpy import (
    amin,
    median,
    amax,
    mean,
    std
)

from NiaPy.task import StoppingTask, OptimizationType
from NiaPy.algorithms import AlgorithmUtility

logging.basicConfig()
logger = logging.getLogger('NiaPy.runner.Runner')
logger.setLevel('INFO')

__all__ = ["Runner"]


class Runner:
    r"""Runner utility feature.

    Feature which enables running multiple algorithms with multiple benchmarks.
    It also support exporting results in various formats (e.g. LaTeX, Excel, JSON)

    Attributes:
            D (int): Dimension of problem
            NP (int): Population size
            nFES (int): Number of function evaluations
            nRuns (int): Number of repetitions
            useAlgorithms (Union[List[str], List[Algorithm]]): List of algorithms to run
            useBenchmarks (Union[List[str], List[Benchmark]]): List of benchmarks to run

    Returns:
            results (Dict[str, Dict]): Returns the results.

    """

    def __init__(self, D=10, nFES=1000000, nRuns=1, useAlgorithms='ArtificialBeeColonyAlgorithm', useBenchmarks='Ackley', **kwargs):
        r"""Initialize Runner.

        Args:
                D (int): Dimension of problem
                nFES (int): Number of function evaluations
                nRuns (int): Number of repetitions
                useAlgorithms (List[Algorithm]): List of algorithms to run
                useBenchmarks (List[Benchmarks]): List of benchmarks to run

        """

        self.D = D
        self.nFES = nFES
        self.nRuns = nRuns
        self.useAlgorithms = useAlgorithms
        self.useBenchmarks = useBenchmarks
        self.results = {}

    def benchmark_factory(self, name):
        r"""Create optimization task.

        Args:
                name (str): Benchmark name.

        Returns:
                Task: Optimization task to use.

        """
        return StoppingTask(D=self.D, nFES=self.nFES, optType=OptimizationType.MINIMIZATION, benchmark=name)

    @classmethod
    def __create_export_dir(cls):
        r"""Create export directory if not already createed."""
        if not os.path.exists("export"):
            os.makedirs("export")

    @classmethod
    def __generate_export_name(cls, extension):
        r"""Generate export file name.

        Args:
                extension (str): File format.

        Returns:

        """

        return "export/" + str(datetime.datetime.now()).replace(":", ".") + "." + extension

    def __export_to_log(self):
        r"""Print the results to terminal."""

        print(self.results)

    def __export_to_json(self):
        r"""Export the results in the JSON form.

        See Also:
                * :func:`NiaPy.Runner.__createExportDir`

        """

        self.__create_export_dir()
        with open(self.__generate_export_name("json"), "w") as outFile:
            json.dump(self.results, outFile)
            logger.info("Export to JSON completed!")

    def __export_to_xlsx(self):
        r"""Export the results in the xlsx form.

        See Also:
                :func:`NiaPy.Runner.__generateExportName`

        """

        self.__create_export_dir()
        workbook = xlsxwriter.Workbook(self.__generate_export_name("xlsx"))
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

    def __export_to_latex(self):
        r"""Export the results in the form of latex table.

        See Also:
                :func:`NiaPy.Runner.__createExportDir`
                :func:`NiaPy.Runner.__generateExportName`

        """

        self.__create_export_dir()

        metrics = ["Best", "Median", "Worst", "Mean", "Std."]

        def only_upper(s):
            return "".join(c for c in s if c.isupper())

        with open(self.__generate_export_name("tex"), "a") as outFile:
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
                verbose (bool): Switch for verbose logging (default: {False})

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
            if not isinstance(alg, "".__class__):
                alg_name = str(type(alg).__name__)
            else:
                alg_name = alg

            self.results[alg_name] = {}

            if verbose:
                logger.info("Running %s...", alg_name)

            for bench in self.useBenchmarks:
                if not isinstance(bench, "".__class__):
                    bench_name = str(type(bench).__name__)
                else:
                    bench_name = bench

                if verbose:
                    logger.info("Running %s algorithm on %s benchmark...", alg_name, bench_name)

                self.results[alg_name][bench_name] = []
                for _ in range(self.nRuns):
                    algorithm = AlgorithmUtility().get_algorithm(alg)
                    benchmark_stopping_task = self.benchmark_factory(bench)
                    self.results[alg_name][bench_name].append(algorithm.run(benchmark_stopping_task))
            if verbose:
                logger.info("---------------------------------------------------")
        if export == "log":
            self.__export_to_log()
        elif export == "json":
            self.__export_to_json()
        elif export == "xlsx":
            self.__export_to_xlsx()
        elif export == "latex":
            self.__export_to_latex()
        else:
            raise TypeError("Passed export type is not supported!")
        return self.results
