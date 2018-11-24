# encoding=utf8
# pylint: disable=line-too-long, mixed-indentation, bad-continuation,multiple-statements, unused-argument, no-self-use, trailing-comma-tuple, logging-not-lazy, no-else-return, old-style-class, dangerous-default-value

"""Implementation of benchmarks utility function."""
import logging
from datetime import datetime
from enum import Enum
from numpy import ndarray, asarray, full, inf, dot, where, random as rand, fabs, ceil, array_equal
from matplotlib import pyplot as plt, animation as anim
from NiaPy.benchmarks import Rastrigin, Rosenbrock, Griewank, Sphere, Ackley, Schwefel, Schwefel221, Schwefel222, Whitley, Alpine1, Alpine2, HappyCat, Ridge, ChungReynolds, Csendes, Pinter, Qing, Quintic, Salomon, SchumerSteiglitz, Step, Step2, Step3, Stepint, SumSquares, StyblinskiTang, BentCigar, Discus, Elliptic, ExpandedGriewankPlusRosenbrock, HGBat, Katsuura, ExpandedSchaffer, ModifiedSchwefel, Weierstrass, Michalewichz, Levy, Sphere2, Sphere3, Trid, Perm, Zakharov, DixonPrice, Powell, CosineMixture, Infinity, SchafferN2, SchafferN4
from NiaPy.util.exception import FesException, GenException, TimeException, RefException

logging.basicConfig()
logger = logging.getLogger('NiaPy.util.utility')
logger.setLevel('INFO')

__all__ = [
    'Utility',
    'Task',
    'TaskConvPrint',
    'TaskConvPlot',
    'TaskConvSave',
    'fullArray',
    'TaskComposition',
    'OptimizationType',
    'ScaledTask']


def fullArray(a, D):
    r"""Fill or create array of length D, from value or value form a.

    Arguments:
    a {integer} or {real} or {list} or {ndarray} -- Input values for fill
    D {integer} -- Length of new array
    """
    A = list()
    if isinstance(a, (int, float)):
        A = full(D, a)
    elif isinstance(a, (ndarray, list)):
        if len(a) == D:
            A = a if isinstance(a, ndarray) else asarray(a)
        elif len(a) > D:
            A = a[:D] if isinstance(a, ndarray) else asarray(a[:D])
        else:
            for i in range(int(ceil(float(D) / len(a)))):
                A.extend(a[:D if (D - i * len(a)) >=
                           len(a) else D - i * len(a)])
            A = asarray(A)
    return A


class Utility:
    def __init__(self):
        self.classes = {
            'ackley': Ackley,
            'alpine1': Alpine1,
            'alpine2': Alpine2,
            'bentcigar': BentCigar,
            'chungReynolds': ChungReynolds,
            'cosinemixture': CosineMixture,
            'csendes': Csendes,
            'discus': Discus,
            'dixonprice': DixonPrice,
            'conditionedellptic': Elliptic,
            'elliptic': Elliptic,
            'expandedgriewankplusrosenbrock': ExpandedGriewankPlusRosenbrock,
            'expandedschaffer': ExpandedSchaffer,
            'griewank': Griewank,
            'happyCat': HappyCat,
            'hgbat': HGBat,
            'infinity': Infinity,
            'katsuura': Katsuura,
            'levy': Levy,
            'michalewicz': Michalewichz,
            'modifiedscwefel': ModifiedSchwefel,
            'perm': Perm,
            'pinter': Pinter,
            'powell': Powell,
            'qing': Qing,
            'quintic': Quintic,
            'rastrigin': Rastrigin,
            'ridge': Ridge,
            'rosenbrock': Rosenbrock,
            'salomon': Salomon,
            'schaffer2': SchafferN2,
            'schaffer4': SchafferN4,
            'schumerSteiglitz': SchumerSteiglitz,
            'schwefel': Schwefel,
            'schwefel221': Schwefel221,
            'schwefel222': Schwefel222,
            'sphere': Sphere,
            'sphere2': Sphere2,
            'sphere3': Sphere3,
            'step': Step,
            'step2': Step2,
            'step3': Step3,
            'stepint': Stepint,
            'styblinskiTang': StyblinskiTang,
            'sumSquares': SumSquares,
            'trid': Trid,
            'weierstrass': Weierstrass,
            'whitley': Whitley,
            'zakharov': Zakharov
        }

    def get_benchmark(self, benchmark):
        r"""Get the optimization problem.

        Arguments:
        benchmark {string} or {class} -- String or class that represents the optimization problem
        """
        if not isinstance(benchmark, str) and not callable(benchmark):
            return benchmark
        elif benchmark in self.classes:
            return self.classes[benchmark]()
        raise TypeError('Passed benchmark is not defined!')

    @classmethod
    def __raiseLowerAndUpperNotDefined(cls): raise TypeError(
        'Upper and Lower value must be defined!')


class OptimizationType(Enum):
    MINIMIZATION = 1.0
    MAXIMIZATION = -1.0


class ATask(Utility):
    def __init__(
            self,
            D=0,
            optType=OptimizationType.MINIMIZATION,
            benchmark=None,
            **kwargs):
        r"""Set the default felds of a task class."""
        Utility.__init__(self)
        self.D = D  # dimension of the problem
        self.benchmark = self.get_benchmark(
            benchmark) if benchmark is not None else None
        if self.benchmark is not None:
            self.Lower, self.Upper = fullArray(
                self.benchmark.Lower, self.D), fullArray(
                self.benchmark.Upper, self.D)
            self.bRange = fabs(self.Upper - self.Lower)
            self.Fun = self.benchmark.function() if self.benchmark is not None else None
        else:
            self.Lower, self.Upper = fullArray(0, self.D), fullArray(0, self.D)
            self.bRange = fullArray(0, 0)
        self.Iters, self.Evals = 0, 0
        self.nGEN, self.nFES = inf, inf
        self.optType = optType
        self.x, self.x_f = None, self.optType.value * inf
        self.startTime = datetime.now()

    def nGENs(self): return 100000 if self.nGEN == inf else self.nGEN

    def nFESs(self): return 100000 if self.nFES == inf else self.nFES

    def dim(self):
        r"""Get the number of dimensions."""
        return self.D

    def bcLower(self):
        r"""Get the array of lower bound constraint."""
        return self.Lower

    def bcUpper(self):
        r"""Get the array of upper bound constraint."""
        return self.Upper

    def bcRange(self):
        r"""Get the range of bound constraint."""
        return fabs(self.Upper - self.Lower)

    def stopCond(self):
        r"""Check if stopping condition reached."""
        return False

    def stopCondI(self):
        r"""Check if stopping condition reached and increase number of iterations."""
        r = self.stopCond()
        self.Iters += 1
        return r

    def stopCondE(self):
        r"""Throw exception for the given stopping condition."""
        pass

    def start(self):
        r"""Set the start time of the optimization run."""
        self.startTime = datetime.now()

    def eval(self, A):
        r"""Evaluate the solution A.

        Arguments:
        A {array} -- Solution to evaluate
        """
        pass

    def evals(self):
        r"""Get the number of evaluations made."""
        pass

    def iters(self):
        r"""Get the number of algorithm iteratins made."""
        pass

    def nextIter(self):
        r"""Increases the number of algorithm iterations made."""
        pass

    def isFeasible(self, A):
        r"""Check if the solution is feasible.

        Arguments:
        A {array} -- Solution to check for feasibility
        """
        return False

    def repair(self, x, rnd=rand):
        r"""Repair solution and put the solution in the random position inside of the bounds of problem.

        Arguments:
        x {array} -- solution to check and repair if needed
        """
        ir = where(x < self.Lower)
        x[ir] = rnd.uniform(self.Lower[ir], self.Upper[ir])
        ir = where(x > self.Upper)
        x[ir] = rnd.uniform(self.Lower[ir], self.Upper[ir])
        return x


class Task(ATask):
    def __init__(
            self,
            D,
            nFES=inf,
            nGEN=inf,
            runTime=None,
            refValue=inf,
            benchmark=None,
            o=None,
            fo=None,
            M=None,
            fM=None,
            optF=None,
            optType=OptimizationType.MINIMIZATION,
            **kwargs):
        r"""Initialize task class for optimization.

        Arguments:
        D {integer} -- Number of dimensions
        nFES {integer} -- Number of function evaluations
        nGEN {integer} -- Number of generations or iterations
        benchmark {class} or {string} -- Problem to solve
        o {array} -- Array for shifting
        of {function} -- Function applied on shifted input
        M {matrix} -- Matrix for rotating
        fM {function} -- Function applied after rotating
        optF {real} -- Value added to benchmark function return
        """
        ATask.__init__(
            self,
            D=D,
            optType=optType,
            benchmark=benchmark,
            **kwargs)
        self.nGEN, self.nFES, self.runTime, self.refValue = nGEN, nFES, runTime, refValue
        self.o = o if isinstance(o, ndarray) or o is None else asarray(o)
        self.M = M if isinstance(M, ndarray) or M is None else asarray(M)
        self.fo, self.fM, self.optF = fo, fM, optF

    def stopCond(self):
        return (self.Evals >= self.nFES) or (self.Iters >= self.nGEN)

    def stopCondE(self):
        # TODO add throwing exceptions for time and reference value
        dtime = datetime.now() - self.startTime
        if self.Evals >= self.nFES:
            raise FesException()
        elif self.Iters >= self.nGEN:
            raise GenException()
        elif self.runTime is not None and self.runTime >= dtime:
            raise TimeException()
        elif self.refValue != inf and self.refValue * self.optType.value >= self.x_f:
            raise RefException()

    def eval(self, A):
        self.stopCondE()
        self.Evals += 1
        X = A - self.o if self.o is not None else A
        X = self.fo(X) if self.fo is not None else X
        X = dot(X, self.M) if self.M is not None else X
        X = self.fM(X) if self.fM is not None else X
        r = self.optType.value * \
            self.Fun(self.D, X) + (self.optF if self.optF is not None else 0)
        if r <= self.x_f:
            self.x, self.x_f = A, r
        return r

    def evals(self): return self.Evals

    def iters(self): return self.Iters

    def nextIter(self): self.Iters += 1

    def isFeasible(self, A):
        return (
            False if True in (
                A < self.Lower) else True) and (
            False if True in (
                A > self.Upper) else True)


class ScaledTask(ATask):
    def __init__(self, task, Lower, Upper, **kwargs):
        ATask.__init__(self)
        self._task = task
        self.D = self._task.D
        self.Lower, self.Upper = fullArray(
            Lower, self.D), fullArray(
            Upper, self.D)
        self.bRange = fabs(Upper - Lower)

    def stopCond(self): return self._task.stopCond()

    def stopCondI(self): return self._task.stopCondI()

    def eval(self, A): return self._task.eval(A)

    def evals(self): return self._task.evals()

    def iters(self): return self._task.iters()

    def nextIter(self): self._task.nextIter()

    def isFeasible(self, A): return self._task.isFeasible(A)


class TaskConvPrint(Task):
    def __init__(self, **kwargs): Task.__init__(self, **kwargs)

    def eval(self, A):
        x_f = Task.eval(self, A)
        if x_f <= self.x_f and not array_equal(self.x, A):
            self.x, self.x_f = A, x_f
            logger.info(
                'nFES:%d nGEN:%d => %s -> %s' %
                (self.Evals, self.Iters, self.x, self.x_f))
        return x_f


class TaskConvSave(Task):
    def __init__(self, **kwargs):
        Task.__init__(self, **kwargs)
        self.evals = []
        self.x_f_vals = []

    def eval(self, A):
        x_f = Task.eval(self, A)
        if x_f <= self.x_f:
            self.evals.append(self.Evals)
            self.x_f_vals.append(x_f)
        return x_f

    def return_conv(self):
        return self.evals, self.x_f_vals


class TaskConvPlot(Task):
    def __init__(self, **kwargs):
        Task.__init__(self, **kwargs)
        self.x_fs, self.iters = list(), list()
        self.fig = plt.figure()
        self.ax = self.fig.subplots(nrows=1, ncols=1)
        self.ax.set_xlim(0, self.nFES)
        self.line, = self.ax.plot(self.iters, self.x_fs, animated=True)
        self.ani = anim.FuncAnimation(self.fig, self.updatePlot, blit=True)
        self.showPlot()

    def eval(self, A):
        x_f = Task.eval(self, A)
        if not self.x_fs:
            self.x_fs.append(x_f)
        elif x_f < self.x_fs[-1]:
            self.x_fs.append(x_f)
        else:
            self.x_fs.append(self.x_fs[-1])
        self.iters.append(self.Evals)
        return x_f

    def showPlot(self):
        plt.show(block=False)
        plt.pause(0.001)

    def updatePlot(self, frame):
        if self.x_fs:
            maxx_fs, minx_fs = self.x_fs[0], self.x_fs[-1]
            self.ax.set_ylim(minx_fs + 1, maxx_fs + 1)
        self.line.set_data(self.iters, self.x_fs)
        return self.line,


class TaskComposition(Task):
    def __init__(
            self,
            benchmarks=None,
            rho=None,
            lamb=None,
            bias=None,
            **kwargs):
        r"""Initialize of composite function problem.

        Arguments:
        benchmarks {array} of {problems} -- optimization function to use in composition
        delta {array} of {real} --
        lamb {array} of {real} --
        bias {array} of {real} --
        """
        Task.__init__(self, **kwargs)

    def eval(self, A):
        # TODO Usage of multiple functions on the same time
        return inf

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
