# encoding=utf8
# pylint: disable=anomalous-backslash-in-string, old-style-class
"""Implementations of Step functions."""

import math

__all__ = ['Step', 'Step2', 'Step3']


class Step:
    r"""Implementation of Step function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Step function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor \left |
        x_i \right | \rfloor \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor \left |
                x_i \right | \rfloor \right)$

        Equation:
                \begin{equation} f(\mathbf{x}) = \sum_{i=1}^D \left(
                \lfloor \left | x_i \right | \rfloor \right) \end{equation}

        Domain:
                $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """

    def __init__(self, Lower=-100.0, Upper=100.0):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += math.floor(abs(sol[i]))

            return val

        return evaluate


class Step2:
    r"""Step2 function implementation.

    Date: 2018

    Author: Lucija Brezočnik

    Licence: MIT

    Function: **Step2 function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i + 0.5 \rfloor \right)^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **lobal minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (-0.5,...,-0.5)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i + 0.5 \rfloor \right)^2$

        Equation:
                \begin{equation}f(\mathbf{x}) = \sum_{i=1}^D \left(
                \lfloor x_i + 0.5 \rfloor \right)^2 \end{equation}

        Domain:
                $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """

    def __init__(self, Lower=-100.0, Upper=100.0):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += math.pow(math.floor(sol[i] + 0.5), 2)

            return val

        return evaluate


class Step3:
    r"""Step3 function implementation.

    Date: 2018

    Author: Lucija Brezočnik

    Licence: MIT

    Function: **Step3 function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i^2 \rfloor \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i^2 \rfloor \right)$

        Equation:
                \begin{equation}f(\mathbf{x}) = \sum_{i=1}^D \left(
                \lfloor x_i^2 \rfloor \right)\end{equation}

        Domain:
                $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """

    def __init__(self, Lower=-100.0, Upper=100.0):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += math.floor(math.pow(sol[i], 2))

            return val

        return evaluate
