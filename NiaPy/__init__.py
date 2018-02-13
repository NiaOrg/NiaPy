import logging
import copy
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
                 betamin=0.2, gamma=1.0, p=0.5, Lower=-5, Upper=5):
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
        self.Lower = Lower
        self.Upper = Upper
        self.results = {}

    def __algorithmFactory(self, name, benchmark):
        bench = benchmarks.utility.Utility.get_benchmark(
            benchmark, self.Lower, self.Upper)

        if name == 'BatAlgorithm':
            return algorithms.basic.BatAlgorithm(
                self.D, self.NP, self.nFES, self.A, self.r, self.Qmin, self.Qmax, bench)
        elif name == 'DifferentialEvolutionAlgorithm':
            return algorithms.basic.DifferentialEvolutionAlgorithm(
                self.D, self.NP, self.nFES, self.F, self.CR, bench)
        elif name == 'FireflyAlgorithm':
            return algorithms.basic.FireflyAlgorithm(
                self.D, self.NP, self.nFES, self.alpha, self.betamin, self.gamma, bench)
        elif name == 'FlowerPollinationAlgorithm':
            return algorithms.basic.FlowerPollinationAlgorithm(
                self.D, self.NP, self.nFES, self.p, bench)
        elif name == 'GreyWolfOptimizer':
            return algorithms.basic.GreyWolfOptimizer(
                self.D, self.NP, self.nFES, bench)
        elif name == 'HybridBatAlgorithm':
            return algorithms.modified.HybridBatAlgorithm(
                self.D, self.NP, self.nFES, self.A, self.r, self.Qmin, self.Qmax, bench)
        else:
            raise TypeError('Passed benchmark is not defined!')

    def __exportToLog(self):
        print(self.results)

    def run(self, export='log'):
        for alg in self.useAlgorithms:
            self.results[alg] = {}
            for bench in self.useBenchmarks:
                benchName = ''
                # check if passed benchmark is class
                if not isinstance(bench, ''.__class__):
                    # set class name as benchmark name
                    benchName = str(type(bench).__name__)
                else:
                    benchName = bench

                self.results[alg][benchName] = []

                for _i in range(self.nRuns):
                    algorithm = self.__algorithmFactory(alg, bench)
                    self.results[alg][benchName].append(algorithm.run())

        if export == 'log':
            self.__exportToLog()
        elif export == 'xls':
            # TODO: implement export to xls
            pass
        elif export == 'latex':
            # TODO: implement export to latex
            pass
        else:
            raise TypeError('Passed export type is not supported!')

        return self.results
