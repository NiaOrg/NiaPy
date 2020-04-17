# encoding=utf8

"""The module implementing Clustering benchmark."""

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = [
    "Clustering",
    "ClusteringMin",
    "ClusteringMinPenalty"
]

class Clustering(Benchmark):
    r"""Implementation of Clustering function.

    Date:
        2019

    Author:
        Klemen Berkovič

    License:
        MIT

    Function:
        **Clustering function**
        :math:`f(\mathbf{O}, \mathbf{Z}) = \sum_{i=1}^N \sum_{j=1}^K w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2`

        Input domain:
        Depends on dataset used.

        Global minimum:
        Depends on dataset used.

    LaTeX formats:
        Inline:
            $f(\mathbf{O}, \mathbf{Z}) = \sum_{i=1}^N \sum_{j=1}^K w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2$

        Equation:
            \begin{equation} f(\mathbf{O}, \mathbf{Z}) = \sum_{i=1}^N \sum_{j=1}^K w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2 \end{equation}

    Attributes:
        Name (List[str]): List of names for the benchmark
        dataset (numpy.ndarray): Dataset to use for clustering.
        a (int): Number of attirbutes in dataset.

    See Also:
        * :class:`NiaPy.benchmarks.Benchmark`
    """
    Name = ["Clustering", "clustering"]
    dataset = None
    a = 0

    def __init__(self, dataset, **kwargs):
        """Initialize Clustering benchmark.

        Args:
            dataset (numpy.ndarray): Dataset.
            kwargs (Dict[str, Any]): Additional arguments.

        See Also:
            * :func:`NiaPy.benchmarks.Benchmark.__init__`
        """
        Benchmark.__init__(self, np.min(dataset, axis=0), np.max(dataset, axis=0))
        self.dataset, self.a = dataset.astype(float), len(self.Lower)

    @staticmethod
    def latex_code():
        """Return the latex code of the problem.

        Returns:
            str: Latex code.
        """
        return r"""$f(\mathbf{O}, \mathbf{Z}) = \sum_{i=1}^N \sum_{j=1}^K w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2$"""

    def function(self):
        """Return benchmark evaluation function.

        Returns:
            Callable[[int, numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
        """
        dataset, a = self.dataset, self.a
        def fun(k, x, w=None, p=2, **dict):
            k = k if k is not None else int(len(x) / a)  # Number of clusters
            w = w if w is not None else np.ones([k, len(dataset), a], dtype=float)  # Weights
            return np.sum([np.sum((w[i] * (dataset - x[a * i:a * (i + 1)]) ** p)[:, 0]) for i in range(k)])
        return fun

class ClusteringMin(Clustering):
    r"""Implementation of Clustering min function.

    Date:
        2019

    Author:
        Klemen Berkovič

    License:
        MIT

    Function:
        **Clustering min function**
        :math:`f(\mathbf{O}, \mathbf{Z}) = \min_{j=1}^M \left( \sum_{i=1}^N w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2 \right)`

        Input domain:
        Depends on dataset used.

        Global minimum:
        Depends on dataset used.

    LaTeX formats:
        Inline:
            $f(\mathbf{O}, \mathbf{Z}) = \min_{j=1}^M \left( \sum_{i=1}^N w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2 \right)$

        Equation:
            \begin{equation}  f(\mathbf{O}, \mathbf{Z}) = \min_{j=1}^M \left( \sum_{i=1}^N w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \end{equation}

    Attributes:
        Name (List[str]): Names of the benchmark.

    See Also:
        * :class:`NiaPy.benchmark.Clustering`
    """
    Name = ["ClusteringMin", "clusteringmin"]

    def __init__(self, dataset, **kwargs):
        """Initialize Clustering min benchmark.

        Args:
            dataset (numpy.ndarray): Dataset.
            kwargs (Dict[str, Any]): Additional arguments.

        See Also:
            * :func:`NiaPy.benchmarks.Clustering.__init__`
        """
        Clustering.__init__(self, dataset)
        self.a = len(self.Lower)

    @staticmethod
    def latex_code():
        """Return the latex code of the problem.

        Returns:
            str: latex code.
        """
        return r"""$f(\mathbf{O}, \mathbf{Z}) = \sum_{i=1}^N \sum_{j=1}^K w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2$"""

    def function(self):
        """Return benchmark evaluation function.

        Returns:
            Callable[[int, numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
        """
        dataset, a = self.dataset, self.a
        def fun(k, x, w=None, p=2, **dict):
            k = k if k is not None else int(len(x) / a)  # Number of clusters
            w = w if w is not None else np.ones([k, len(dataset), a], dtype=float)  # Weights
            A = np.stack([(w[i] * (dataset - x[a * i:a * (i + 1)]) ** p)[:, 0] for i in range(k)])
            return np.sum(np.min(A, axis=0))
        return fun

class ClusteringMinPenalty(ClusteringMin):
    r"""Implementation of Clustering min function with penalty.

    Date:
        2019

    Author:
        Klemen Berkovič

    License:
        MIT

    Function:
        **Clustering min with penalty function**
        :math:`\mathcal{f} \left(\mathbf{O}, \mathbf{Z} \right) = \mathcal{p} \left( \mathbf{Z} \right) + \sum_{i=0}^{\mathit{N}-1} \min_{\forall j \in \mathfrak{k}} \left( w_{i, j} \times || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \\ \mathcal{p} \left(\mathbf{Z} \right) = \sum_{\forall \mathbf{e} \in \mathfrak{I}} \sum_{j=0}^{\mathit{A}-1} \min \left(\frac{r}{\mathit{K}}, \max \left(0, \frac{r}{\mathit{K}} - || z_{e_0, j} - z_{e_1, j} ||^2 \right) \right)`

        Input domain:
        Depends on dataset used.

        Global minimum:
        Depends on dataset used.

    LaTeX formats:
        Inline:
            $\mathcal{f} \left(\mathbf{O}, \mathbf{Z} \right) = \mathcal{p} \left( \mathbf{Z} \right) + \sum_{i=0}^{\mathit{N}-1} \min_{\forall j \in \mathfrak{k}} \left( w_{i, j} \times || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \\ \mathcal{p} \left(\mathbf{Z} \right) = \sum_{\forall \mathbf{e} \in \mathfrak{I}} \sum_{j=0}^{\mathit{A}-1} \min \left(\frac{r}{\mathit{K}}, \max \left(0, \frac{r}{\mathit{K}} - || z_{e_0, j} - z_{e_1, j} ||^2 \right) \right)$

        Equation:
            \begin{equation} \mathcal{f} \left(\mathbf{O}, \mathbf{Z} \right) = \mathcal{p} \left( \mathbf{Z} \right) + \sum_{i=0}^{\mathit{N}-1} \min_{\forall j \in \mathfrak{k}} \left( w_{i, j} \times || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \\ \mathcal{p} \left(\mathbf{Z} \right) = \sum_{\forall \mathbf{e} \in \mathfrak{I}} \sum_{j=0}^{\mathit{A}-1} \min \left(\frac{r}{\mathit{K}}, \max \left(0, \frac{r}{\mathit{K}} - || z_{e_0, j} - z_{e_1, j} ||^2 \right) \right) \end{equation}

    Attributes:
        Name (List[str]): Names of the benchmark.

    See Also:
        * :class:`NiaPy.benchmark.ClusteringMin`
    """
    Name = ["ClusteringMinPenalty", "clusteringminpen"]

    def __init__(self, dataset, **kwargs):
        """Initialize Clustering min benchmark.

        Args:
            dataset (numpy.ndarray): Dataset.
            kwargs (Dict[str, Any]): Additional arguments.

        See Also:
            * :func:`NiaPy.benchmarks.ClusteringMin.__init__`
        """
        ClusteringMin.__init__(self, dataset)
        self.range = np.abs(self.Upper - self.Lower)

    @staticmethod
    def latex_code():
        """Return the latex code of the problem.

        Returns:
            str: latex code.
        """
        return r"""\mathcal{f} \left(\mathbf{O}, \mathbf{Z} \right) = \mathcal{p} \left( \mathbf{Z} \right) + \sum_{i=0}^{\mathit{N}-1} \min_{\forall j \in \mathfrak{k}} \left( w_{i, j} \times || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \\ \mathcal{p} \left(\mathbf{Z} \right) = \sum_{\forall \mathbf{e} \in \mathfrak{I}} \sum_{j=0}^{\mathit{A}-1} \min \left(\frac{r}{\mathit{K}}, \max \left(0, \frac{r}{\mathit{K}} - || z_{e_0, j} - z_{e_1, j} ||^2 \right) \right)"""

    def penalty(self, x, k):
        r"""Get penelty for inidividual.

        Args:
            x (numpyl.ndarray): Individual.
            k (int): Number of clusters

        Returns:
            float: Penalty for the given individual.
        """
        p, r = 0, self.range / k
        for i in range(k - 1):
            for j in range(k - i - 1):
                if i != k - j - 1: p += np.sum(np.fmin(r, np.fmax(np.zeros(self.a), r - np.abs(x[self.a * i:self.a * (i + 1)] - x[self.a * (k - j - 1):self.a * (k - j)]))))
        return p

    def function(self):
        """Return benchmark evaluation function.

        Returns:
            Callable[[int, numpy.ndarray, Dict[str, Any]], float]: Evaluation function.
        """
        fcm, dataset, a = ClusteringMin.function(self), self.dataset, self.a
        def fun(k, x, w=None, p=2, **kwargs):
            k = k if k is not None else int(len(x) / a)  # Number of clusters
            w = w if w is not None else np.ones([k, len(dataset), a], dtype=float)  # Weights
            return fcm(k, x, w=w, p=p) + self.penalty(x, k)
        return fun
