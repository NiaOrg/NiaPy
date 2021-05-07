# encoding=utf8
import logging
import operator

import numpy as np

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['MultipleTrajectorySearch', 'MultipleTrajectorySearchV1', 'mts_ls1', 'mts_ls1v1', 'mts_ls2', 'mts_ls3',
           'mts_ls3v1']


def mts_ls1(current_x, current_fitness, best_x, best_fitness, improve, search_range, task, rng, bonus1=10, bonus2=1,
            sr_fix=0.4, **_kwargs):
    r"""Multiple trajectory local search one.

    Args:
        current_x (numpy.ndarray): Current solution.
        current_fitness (float): Current solutions fitness/function value.
        best_x (numpy.ndarray): Global best solution.
        best_fitness (float): Global best solutions fitness/function value.
        improve (bool): Has the solution been improved.
        search_range (numpy.ndarray): Search range.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random number generator.
        bonus1 (int): Bonus reward for improving global best solution.
        bonus2 (int): Bonus reward for improving solution.
        sr_fix (numpy.ndarray): Fix when search range is to small.

    Returns:
        Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray]:
            1. New solution.
            2. New solutions fitness/function value.
            3. Global best if found else old global best.
            4. Global bests function/fitness value.
            5. If solution has improved.
            6. Search range.

    """
    if not improve:
        search_range /= 2
        i_fix = np.argwhere(search_range < 1e-15)
        search_range[i_fix] = task.range[i_fix] * sr_fix
    improve = False
    grade = 0.0
    for i in range(len(current_x)):
        x_old = current_x[i]
        current_x[i] = x_old - search_range[i]
        current_x = task.repair(current_x, rng)
        new_fitness = task.eval(current_x)
        if new_fitness < best_fitness:
            grade = grade + bonus1
            best_x = current_x.copy()
            best_fitness = new_fitness
        if new_fitness == current_fitness:
            current_x[i] = x_old
        elif new_fitness > current_fitness:
            current_x[i] = x_old + 0.5 * search_range[i]
            current_x = task.repair(current_x, rng)
            new_fitness = task.eval(current_x)
            if new_fitness < best_fitness:
                grade = grade + bonus1
                best_x = current_x.copy()
                best_fitness = new_fitness
            if new_fitness >= current_fitness:
                current_x[i] = x_old
            else:
                grade = grade + bonus2
                improve = True
                current_fitness = new_fitness
        else:
            grade = grade + bonus2
            improve = True
            current_fitness = new_fitness
    return current_x, current_fitness, best_x, best_fitness, improve, grade, search_range


def mts_ls1v1(current_x, current_fitness, best_x, best_fitness, improve, search_range, task, rng, bonus1=10, bonus2=1,
              sr_fix=0.4, **_kwargs):
    r"""Multiple trajectory local search one version two.

    Args:
        current_x (numpy.ndarray): Current solution.
        current_fitness (float): Current solutions fitness/function value.
        best_x (numpy.ndarray): Global best solution.
        best_fitness (float): Global best solutions fitness/function value.
        improve (bool): Has the solution been improved.
        search_range (numpy.ndarray): Search range.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random number generator.
        bonus1 (int): Bonus reward for improving global best solution.
        bonus2 (int): Bonus reward for improving solution.
        sr_fix (numpy.ndarray): Fix when search range is to small.

    Returns:
        Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray]:
            1. New solution.
            2. New solutions fitness/function value.
            3. Global best if found else old global best.
            4. Global bests function/fitness value.
            5. If solution has improved.
            6. Search range.

    """
    if not improve:
        search_range /= 2
        i_fix = np.argwhere(search_range < 1e-15)
        search_range[i_fix] = task.range[i_fix] * sr_fix
    improve, d, grade = False, rng.uniform(-1, 1, task.dimension), 0.0
    for i in range(len(current_x)):
        x_old = current_x[i]
        current_x[i] = x_old - search_range[i] * d[i]
        current_x = task.repair(current_x, rng)
        new_fitness = task.eval(current_x)
        if new_fitness < best_fitness:
            grade, best_x, best_fitness = grade + bonus1, current_x.copy(), new_fitness
        elif new_fitness == current_fitness:
            current_x[i] = x_old
        elif new_fitness > current_fitness:
            current_x[i] = x_old + 0.5 * search_range[i]
            current_x = task.repair(current_x, rng)
            new_fitness = task.eval(current_x)
            if new_fitness < best_fitness:
                grade, best_x, best_fitness = grade + bonus1, current_x.copy(), new_fitness
            elif new_fitness >= current_fitness:
                current_x[i] = x_old
            else:
                grade, improve, current_fitness = grade + bonus2, True, new_fitness
        else:
            grade, improve, current_fitness = grade + bonus2, True, new_fitness
    return current_x, current_fitness, best_x, best_fitness, improve, grade, search_range


def move_x(x, r, d, search_range, op):
    r"""Move solution to other position based on operator.

    Args:
        x (numpy.ndarray): Solution to move.
        r (int): Random number.
        d (float): Scale factor.
        search_range (numpy.ndarray): Search range.
        op (Callable): Operator to use.

    Returns:
        numpy.ndarray: Moved solution based on operator.

    """
    return op(x, search_range * d) if r == 0 else x


def mts_ls2(current_x, current_fitness, best_x, best_fitness, improve, search_range, task, rng, bonus1=10, bonus2=1,
            sr_fix=0.4, **_kwargs):
    r"""Multiple trajectory local search two.

    Args:
        current_x (numpy.ndarray): Current solution.
        current_fitness (float): Current solutions fitness/function value.
        best_x (numpy.ndarray): Global best solution.
        best_fitness (float): Global best solutions fitness/function value.
        improve (bool): Has the solution been improved.
        search_range (numpy.ndarray): Search range.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random number generator.
        bonus1 (int): Bonus reward for improving global best solution.
        bonus2 (int): Bonus reward for improving solution.
        sr_fix (numpy.ndarray): Fix when search range is to small.

    Returns:
        Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray]:
            1. New solution.
            2. New solutions fitness/function value.
            3. Global best if found else old global best.
            4. Global bests function/fitness value.
            5. If solution has improved.
            6. Search range.

    See Also:
        * :func:`niapy.algorithms.other.move_x`

    """
    if not improve:
        search_range /= 2
        i_fix = np.argwhere(search_range < 1e-15)
        search_range[i_fix] = task.range[i_fix] * sr_fix
    improve, grade = False, 0.0
    for _ in range(len(current_x)):
        d = -1 + rng.random(len(current_x)) * 2
        r = rng.choice([0, 1, 2, 3], len(current_x))
        new_x = task.repair(np.vectorize(move_x)(current_x, r, d, search_range, operator.sub), rng)
        new_fitness = task.eval(new_x)
        if new_fitness < best_fitness:
            grade, best_x, best_fitness = grade + bonus1, new_x.copy(), new_fitness
        elif new_fitness != current_fitness:
            if new_fitness > current_fitness:
                new_x = task.repair(np.vectorize(move_x)(current_x, r, d, search_range, operator.add), rng)
                new_fitness = task.eval(new_x)
                if new_fitness < best_fitness:
                    grade, best_x, best_fitness = grade + bonus1, new_x.copy(), new_fitness
                elif new_fitness < current_fitness:
                    grade, current_x, current_fitness, improve = grade + bonus2, new_x.copy(), new_fitness, True
            else:
                grade, current_x, current_fitness, improve = grade + bonus2, new_x.copy(), new_fitness, True
    return current_x, current_fitness, best_x, best_fitness, improve, grade, search_range


def mts_ls3(current_x, current_fitness, best_x, best_fitness, improve, search_range, task, rng, bonus1=10, bonus2=1,
            **_kwargs):
    r"""Multiple trajectory local search three.

    Args:
        current_x (numpy.ndarray): Current solution.
        current_fitness (float): Current solutions fitness/function value.
        best_x (numpy.ndarray): Global best solution.
        best_fitness (float): Global best solutions fitness/function value.
        improve (bool): Has the solution been improved.
        search_range (numpy.ndarray): Search range.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random number generator.
        bonus1 (int): Bonus reward for improving global best solution.
        bonus2 (int): Bonus reward for improving solution.

    Returns:
        Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray]:
            1. New solution.
            2. New solutions fitness/function value.
            3. Global best if found else old global best.
            4. Global bests function/fitness value.
            5. If solution has improved.
            6. Search range.

    """
    x_new, grade = np.copy(current_x), 0.0
    for i in range(len(current_x)):
        x1, x2, x3 = np.copy(x_new), np.copy(x_new), np.copy(x_new)
        x1[i], x2[i], x3[i] = x1[i] + 0.1, x2[i] - 0.1, x3[i] + 0.2
        x1, x2, x3 = task.repair(x1, rng), task.repair(x2, rng), task.repair(x3, rng)
        x1_fit, x2_fit, x3_fit = task.eval(x1), task.eval(x2), task.eval(x3)
        if x1_fit < best_fitness:
            grade, best_x, best_fitness, improve = grade + bonus1, x1.copy(), x1_fit, True
        if x2_fit < best_fitness:
            grade, best_x, best_fitness, improve = grade + bonus1, x2.copy(), x2_fit, True
        if x3_fit < best_fitness:
            grade, best_x, best_fitness, improve = grade + bonus1, x3.copy(), x3_fit, True
        d1, d2, d3 = current_fitness - x1_fit if np.abs(x1_fit) != np.inf else 0, current_fitness - x2_fit if np.abs(
            x2_fit) != np.inf else 0, current_fitness - x3_fit if np.abs(x3_fit) != np.inf else 0
        if d1 > 0:
            grade, improve = grade + bonus2, True
        if d2 > 0:
            grade, improve = grade + bonus2, True
        if d3 > 0:
            grade, improve = grade + bonus2, True
        a, b, c = 0.4 + rng.random() * 0.1, 0.1 + rng.random() * 0.2, rng.random()
        x_new[i] += a * (d1 - d2) + b * (d3 - 2 * d1) + c
        x_new = task.repair(x_new, rng)
        x_new_fitness = task.eval(x_new)
        if x_new_fitness < current_fitness:
            if x_new_fitness < best_fitness:
                best_x, best_fitness, grade = x_new.copy(), x_new_fitness, grade + bonus1
            else:
                grade += bonus2
            current_x, current_fitness, improve = x_new, x_new_fitness, True
    return current_x, current_fitness, best_x, best_fitness, improve, grade, search_range


def mts_ls3v1(current_x, current_fitness, best_x, best_fitness, improve, search_range, task, rng, bonus1=10, bonus2=1,
              phi=3, **_kwargs):
    r"""Multiple trajectory local search three version one.

    Args:
        current_x (numpy.ndarray): Current solution.
        current_fitness (float): Current solutions fitness/function value.
        best_x (numpy.ndarray): Global best solution.
        best_fitness (float): Global best solutions fitness/function value.
        improve (bool): Has the solution been improved.
        search_range (numpy.ndarray): Search range.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random number generator.
        phi (int): Number of new generated positions.
        bonus1 (int): Bonus reward for improving global best solution.
        bonus2 (int): Bonus reward for improving solution.

    Returns:
        Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray]:
            1. New solution.
            2. New solutions fitness/function value.
            3. Global best if found else old global best.
            4. Global bests function/fitness value.
            5. If solution has improved.
            6. Search range.

    """
    grade, disp = 0.0, task.range / 10
    while True in (disp > 1e-3):
        new_x = np.apply_along_axis(task.repair, 1, np.asarray(
            [rng.permutation(current_x) + disp * rng.uniform(-1, 1, len(current_x)) for _ in range(phi)]), rng)
        new_fitness = np.apply_along_axis(task.eval, 1, new_x)
        i_better, i_better_best = np.argwhere(new_fitness < current_fitness), np.argwhere(new_fitness < best_fitness)
        grade += len(i_better_best) * bonus1 + (len(i_better) - len(i_better_best)) * bonus2
        if len(new_fitness[i_better_best]) > 0:
            ib, improve = np.argmin(new_fitness[i_better_best]), True
            best_x, best_fitness, current_x, current_fitness = new_x[i_better_best][ib][0].copy(), new_fitness[i_better_best][ib][0], new_x[i_better_best][ib][
                0].copy(), new_fitness[i_better_best][ib][0]
        elif len(new_fitness[i_better]) > 0:
            ib, improve = np.argmin(new_fitness[i_better]), True
            current_x, current_fitness = new_x[i_better][ib][0].copy(), new_fitness[i_better][ib][0]
        su, sl = np.fmin(task.upper, current_x + 2 * disp), np.fmax(task.lower, current_x - 2 * disp)
        disp = (su - sl) / 10
    return current_x, current_fitness, best_x, best_fitness, improve, grade, search_range


class MultipleTrajectorySearch(Algorithm):
    r"""Implementation of Multiple trajectory search.

    Algorithm:
        Multiple trajectory search

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://ieeexplore.ieee.org/document/4631210/

    Reference paper:
        Lin-Yu Tseng and Chun Chen, "Multiple trajectory search for Large Scale Global Optimization," 2008 IEEE Congress on Evolutionary Computation (IEEE World Congress on Computational Intelligence), Hong Kong, 2008, pp. 3052-3059. doi: 10.1109/CEC.2008.4631210

    Attributes:
        Name (List[Str]): List of strings representing algorithm name.
        local_searches (Iterable[Callable[[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray, Task, Dict[str, Any]], Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, int, numpy.ndarray]]]): Local searches to use.
        bonus1 (int): Bonus for improving global best solution.
        bonus2 (int): Bonus for improving solution.
        num_tests (int): Number of test runs on local search algorithms.
        num_searches (int): Number of local search algorithm runs.
        num_searches_best (int): Number of locals search algorithm runs on best solution.
        num_enabled (int): Number of best solution for testing.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['MultipleTrajectorySearch', 'MTS']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Lin-Yu Tseng and Chun Chen, "Multiple trajectory search for Large Scale Global Optimization," 2008 IEEE Congress on Evolutionary Computation (IEEE World Congress on Computational Intelligence), Hong Kong, 2008, pp. 3052-3059. doi: 10.1109/CEC.2008.4631210"""

    def __init__(self, population_size=40, num_tests=5, num_searches=5, num_searches_best=5, num_enabled=17, bonus1=10,
                 bonus2=1, local_searches=(mts_ls1, mts_ls2, mts_ls3), *args, **kwargs):
        """Initialize MultipleTrajectorySearch.

        Args:
            population_size (int): Number of individuals in population.
            num_tests (int): Number of test runs on local search algorithms.
            num_searches (int): Number of local search algorithm runs.
            num_searches_best (int): Number of locals search algorithm runs on best solution.
            num_enabled (int): Number of best solution for testing.
            bonus1 (int): Bonus for improving global best solution.
            bonus2 (int): Bonus for improving self.
            local_searches (Iterable[Callable[[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray, Task, Dict[str, Any]], Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, int, numpy.ndarray]]]): Local searches to use.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.num_tests = num_tests
        self.num_searches = num_searches
        self.num_searches_best = num_searches_best
        self.num_enabled = num_enabled
        self.bonus1 = bonus1
        self.bonus2 = bonus2
        self.local_searches = local_searches

    def set_parameters(self, population_size=40, num_tests=5, num_searches=5, num_searches_best=5, num_enabled=17,
                       bonus1=10, bonus2=1, local_searches=(mts_ls1, mts_ls2, mts_ls3), **kwargs):
        r"""Set the arguments of the algorithm.

        Args:
            population_size (int): Number of individuals in population.
            num_tests (int): Number of test runs on local search algorithms.
            num_searches (int): Number of local search algorithm runs.
            num_searches_best (int): Number of locals search algorithm runs on best solution.
            num_enabled (int): Number of best solution for testing.
            bonus1 (int): Bonus for improving global best solution.
            bonus2 (int): Bonus for improving self.
            local_searches (Iterable[Callable[[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray, Task, Dict[str, Any]], Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, int, numpy.ndarray]]]): Local searches to use.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=kwargs.pop('population_size', population_size), **kwargs)
        self.num_tests = num_tests
        self.num_searches = num_searches
        self.num_searches_best = num_searches_best
        self.num_enabled = num_enabled
        self.bonus1 = bonus1
        self.bonus2 = bonus2
        self.local_searches = local_searches

    def get_parameters(self):
        r"""Get parameters values for the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        d = Algorithm.get_parameters(self)
        d.update({
            'M': d.pop('population_size', self.population_size),
            'num_tests': self.num_tests,
            'num_searches': self.num_searches,
            'num_searches_best': self.num_searches_best,
            'bonus1': self.bonus1,
            'bonus2': self.bonus2,
            'num_enabled': self.num_enabled,
            'local_searches': self.local_searches
        })
        return d

    def grading_run(self, x, x_f, xb, fxb, improve, search_range, task):
        r"""Run local search for getting scores of local searches.

        Args:
            x (numpy.ndarray): Solution for grading.
            x_f (float): Solutions fitness/function value.
            xb (numpy.ndarray): Global best solution.
            fxb (float): Global best solutions function/fitness value.
            improve (bool): Info if solution has improved.
            search_range (numpy.ndarray): Search range.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, float, numpy.ndarray, float]:
                1. New solution.
                2. New solutions function/fitness value.
                3. Global best solution.
                4. Global best solutions fitness/function value.

        """
        ls_grades, new_x = np.zeros(3), [[x, x_f]] * len(self.local_searches)
        k = None
        for k in range(len(self.local_searches)):
            for _ in range(self.num_tests):
                new_x[k][0], new_x[k][1], xb, fxb, improve, g, search_range = self.local_searches[k](new_x[k][0], new_x[k][1], xb, fxb, improve, search_range,
                                                                                                     task, BONUS1=self.bonus1, BONUS2=self.bonus2,
                                                                                                     rng=self.rng)
                ls_grades[k] += g
        xn, xn_f = min(new_x, key=lambda val: val[1])
        return xn, xn_f, xb, fxb, k

    def run_local_search(self, k, x, x_f, xb, fxb, improve, search_range, g, task):
        r"""Run a selected local search.

        Args:
            k (int): Index of local search.
            x (numpy.ndarray): Current solution.
            x_f (float): Current solutions function/fitness value.
            xb (numpy.ndarray): Global best solution.
            fxb (float): Global best solutions fitness/function value.
            improve (bool): If the solution has improved.
            search_range (numpy.ndarray): Search range.
            g (int): Grade.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray, int]:
                1. New best solution found.
                2. New best solutions found function/fitness value.
                3. Global best solution.
                4. Global best solutions function/fitness value.
                5. If the solution has improved.
                6. Grade of local search run.

        """
        for _ in range(self.num_searches):
            x, x_f, xb, fxb, improve, grade, search_range = self.local_searches[k](x, x_f, xb, fxb, improve, search_range, task, bonus1=self.bonus1,
                                                                                   bonus2=self.bonus2, rng=self.rng)
            g += grade
        return x, x_f, xb, fxb, improve, search_range, g

    def init_population(self, task):
        r"""Initialize starting population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations function/fitness value.
                3. Additional arguments:
                    * enable (numpy.ndarray): If solution/individual is enabled.
                    * improve (numpy.ndarray): If solution/individual is improved.
                    * search_range (numpy.ndarray): Search range.
                    * grades (numpy.ndarray): Grade of solution/individual.

        """
        population, fitness, d = Algorithm.init_population(self, task)
        enable = np.full(self.population_size, True)
        improve = np.full(self.population_size, True)
        search_range = np.full((self.population_size, task.dimension), task.range / 2)
        grades = np.zeros(self.population_size)
        d.update({
            'enable': enable,
            'improve': improve,
            'search_range': search_range,
            'grades': grades
        })
        return population, fitness, d

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of MultipleTrajectorySearch algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population of individuals.
            population_fitness (numpy.ndarray): Current individuals function/fitness values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individual function/fitness value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations function/fitness value.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * enable (numpy.ndarray): If solution/individual is enabled.
                    * improve (numpy.ndarray): If solution/individual is improved.
                    * search_range (numpy.ndarray): Search range.
                    * grades (numpy.ndarray): Grade of solution/individual.

        """
        enable = params.pop('enable')
        improve = params.pop('improve')
        search_range = params.pop('search_range')
        grades = params.pop('grades')

        for i in range(len(population)):
            if not enable[i]:
                continue
            enable[i], grades[i] = False, 0
            population[i], population_fitness[i], best_x, best_fitness, k = self.grading_run(population[i], population_fitness[i], best_x, best_fitness, improve[i], search_range[i], task)
            population[i], population_fitness[i], best_x, best_fitness, improve[i], search_range[i], grades[i] = self.run_local_search(k, population[i], population_fitness[i], best_x, best_fitness, improve[i],
                                                                                                                                       search_range[i], grades[i], task)
        for _ in range(self.num_searches_best):
            _, _, best_x, best_fitness, _, _, _ = mts_ls1(best_x, best_fitness, best_x, best_fitness, False, task.range.copy() / 10, task,
                                                          rng=self.rng)
        enable[np.argsort(grades)[:self.num_enabled]] = True
        return population, population_fitness, best_x, best_fitness, {'enable': enable, 'improve': improve, 'search_range': search_range, 'grades': grades}


class MultipleTrajectorySearchV1(MultipleTrajectorySearch):
    r"""Implementation of Multiple trajectory search.

    Algorithm:
        Multiple trajectory search

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://ieeexplore.ieee.org/document/4983179/

    Reference paper:
        Tseng, Lin-Yu, and Chun Chen. "Multiple trajectory search for unconstrained/constrained multi-objective optimization." Evolutionary Computation, 2009. CEC'09. IEEE Congress on. IEEE, 2009.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.

    See Also:
        * :class:`niapy.algorithms.other.MultipleTrajectorySearch``

    """

    Name = ['MultipleTrajectorySearchV1', 'MTSv1']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Tseng, Lin-Yu, and Chun Chen. "Multiple trajectory search for unconstrained/constrained multi-objective optimization." Evolutionary Computation, 2009. CEC'09. IEEE Congress on. IEEE, 2009."""

    def __init__(self, population_size=40, num_tests=5, num_searches=5, num_enabled=17, bonus1=10, bonus2=1, *args,
                 **kwargs):
        """Initialize MultipleTrajectorySearchV1.

        Args:
            population_size (int): Number of individuals in population.
            num_tests (int): Number of test runs on local search algorithms.
            num_searches (int): Number of local search algorithm runs.
            num_enabled (int): Number of best solution for testing.
            bonus1 (int): Bonus for improving global best solution.
            bonus2 (int): Bonus for improving self.

        See Also:
            * :func:`niapy.algorithms.other.MultipleTrajectorySearch.__init__`

        """
        kwargs.pop('num_searches_best', None)
        kwargs.pop('local_searches', None)
        super().__init__(population_size, num_tests, num_searches, 0, num_enabled, bonus1, bonus2,
                         local_searches=(mts_ls1v1, mts_ls2), *args, **kwargs)

    def set_parameters(self, population_size=40, num_tests=5, num_searches=5, num_enabled=17, bonus1=10, bonus2=1,
                       **kwargs):
        r"""Set core parameters of MultipleTrajectorySearchV1 algorithm.

        Args:
            population_size (int): Number of individuals in population.
            num_tests (int): Number of test runs on local search algorithms.
            num_searches (int): Number of local search algorithm runs.
            num_enabled (int): Number of best solution for testing.
            bonus1 (int): Bonus for improving global best solution.
            bonus2 (int): Bonus for improving self.

        See Also:
            * :func:`niapy.algorithms.other.MultipleTrajectorySearch.set_parameters`

        """
        kwargs.pop('num_searches_best', None)
        super().set_parameters(num_searches_best=0, local_searches=(mts_ls1v1, mts_ls2), **kwargs)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
