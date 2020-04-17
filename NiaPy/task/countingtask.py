# encoding=utf8

"""The implementation of tasks."""

from NiaPy.task.task import Task


class CountingTask(Task):
    r"""Optimization task with added counting of function evaluations and algorithm iterations/generations.

    Attributes:
            Iters (int): Number of algorithm iterations/generations.
            Evals (int): Number of function evaluations.

    See Also:
            * :class:`NiaPy.util.Task`

    """

    def __init__(self, **kwargs):
        r"""Initialize counting task.

        Args:
                **kwargs (Dict[str, Any]): Additional arguments.

        See Also:
                * :func:`NiaPy.util.Task.__init__`

        """

        Task.__init__(self, **kwargs)
        self.Iters, self.Evals = 0, 0

    def eval(self, A):
        r"""Evaluate the solution A.

        This function increments function evaluation counter `self.Evals`.

        Arguments:
                A (numpy.ndarray): Solutions to evaluate.

        Returns:
                float: Fitness/function values of solution.

        See Also:
                * :func:`NiaPy.util.Task.eval`

        """

        r = Task.eval(self, A)
        self.Evals += 1
        return r

    def evals(self):
        r"""Get the number of evaluations made.

        Returns:
                int: Number of evaluations made.

        """

        return self.Evals

    def iters(self):
        r"""Get the number of algorithm iteratins made.

        Returns:
                int: Number of generations/iterations made by algorithm.

        """

        return self.Iters

    def nextIter(self):
        r"""Increases the number of algorithm iterations made.

        This function increments number of algorithm iterations/generations counter `self.Iters`.

        """

        self.Iters += 1
