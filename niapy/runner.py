# encoding=utf8

"""Implementation of Runner utility class."""

import datetime
import logging
import os

import pandas as pd

from niapy.task import StoppingTask, OptimizationType
from niapy.util.factory import get_algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.runner.Runner')
logger.setLevel('INFO')

__all__ = ["Runner"]


class Runner:
    r"""Runner utility feature.

    Feature which enables running multiple algorithms with multiple benchmarks.
    It also support exporting results in various formats (e.g. Pandas DataFrame, JSON, Excel)

    Attributes:
        dimension (int): Dimension of problem
        max_evals (int): Number of function evaluations
        runs (int): Number of repetitions
        algorithms (Union[List[str], List[Algorithm]]): List of algorithms to run
        benchmarks (Union[List[str], List[Benchmark]]): List of benchmarks to run

    """

    def __init__(self, dimension=10, max_evals=1000000, runs=1, algorithms='ArtificialBeeColonyAlgorithm',
                 benchmarks='Ackley'):
        r"""Initialize Runner.

        Args:
            dimension (int): Dimension of problem
            max_evals (int): Number of function evaluations
            runs (int): Number of repetitions
            algorithms (List[Algorithm]): List of algorithms to run
            benchmarks (List[Benchmark]): List of benchmarks to run

        """
        self.dimension = dimension
        self.max_evals = max_evals
        self.runs = runs
        self.algorithms = algorithms
        self.benchmarks = benchmarks
        self.results = {}

    def task_factory(self, name):
        r"""Create optimization task.

        Args:
            name (str): Benchmark name.

        Returns:
            Task: Optimization task to use.

        """
        return StoppingTask(max_evals=self.max_evals, dimension=self.dimension,
                            optimization_type=OptimizationType.MINIMIZATION, benchmark=name)

    @classmethod
    def __create_export_dir(cls):
        if not os.path.exists("export"):
            os.makedirs("export")

    @classmethod
    def __generate_export_name(cls, extension):
        Runner.__create_export_dir()
        return "export/" + str(datetime.datetime.now()).replace(":", ".") + "." + extension

    def __export_to_dataframe_pickle(self):
        dataframe = pd.DataFrame.from_dict(self.results)
        dataframe.to_pickle(self.__generate_export_name("pkl"))
        logger.info("Export to Pandas DataFrame pickle (pkl) completed!")

    def __export_to_json(self):
        dataframe = pd.DataFrame.from_dict(self.results)
        dataframe.to_json(self.__generate_export_name("json"))
        logger.info("Export to JSON file completed!")

    def _export_to_xls(self):
        dataframe = pd.DataFrame.from_dict(self.results)
        dataframe.to_excel(self.__generate_export_name("xls"))
        logger.info("Export to XLS completed!")

    def __export_to_xlsx(self):
        dataframe = pd.DataFrame.from_dict(self.results)
        dataframe.to_excel(self.__generate_export_name("xslx"))
        logger.info("Export to XLSX file completed!")

    def run(self, export="dataframe", verbose=False):
        """Execute runner.

        Args:
            export (str): Takes export type (e.g. dataframe, json, xls, xlsx) (default: "dataframe")
            verbose (bool): Switch for verbose logging (default: {False})

        Returns:
            dict: Returns dictionary of results

        Raises:
            TypeError: Raises TypeError if export type is not supported

        """
        for alg in self.algorithms:
            if not isinstance(alg, "".__class__):
                alg_name = str(type(alg).__name__)
            else:
                alg_name = alg

            self.results[alg_name] = {}

            if verbose:
                logger.info("Running %s...", alg_name)

            for bench in self.benchmarks:
                if not isinstance(bench, "".__class__):
                    bench_name = str(type(bench).__name__)
                else:
                    bench_name = bench

                if verbose:
                    logger.info("Running %s algorithm on %s benchmark...", alg_name, bench_name)

                self.results[alg_name][bench_name] = []
                for _ in range(self.runs):
                    algorithm = get_algorithm(alg)
                    benchmark_stopping_task = self.task_factory(bench)
                    self.results[alg_name][bench_name].append(algorithm.run(benchmark_stopping_task))
            if verbose:
                logger.info("---------------------------------------------------")
        if export == "dataframe":
            self.__export_to_dataframe_pickle()
        elif export == "json":
            self.__export_to_json()
        elif export == "xsl":
            self._export_to_xls()
        elif export == "xlsx":
            self.__export_to_xlsx()
        else:
            raise TypeError("Passed export type %s is not supported!", export)
        return self.results
