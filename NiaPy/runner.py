
__all__ = ['Runner']


class Runner(object):

    def __init__(self, algorithms, benchmarks, nRuns):
        self.algorithms = algorithms
        self.benchmarks = benchmarks
        self.nRuns = nRuns

    @staticmethod
    def run(export):
        pass
