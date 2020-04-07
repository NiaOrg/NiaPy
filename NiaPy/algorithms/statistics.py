# encoding=utf8

import pkgutil
from io import StringIO

import numpy as np
import pandas as pd
import scipy.stats as stats

import NiaPy.data as pkg_data

__all__ = [
    'BasicStatistics',
    'wilcoxonSignedRanks',
    'wilcoxonTest',
    'friedmanRanks'
]

class BasicStatistics:
    r"""Basic statistics for results of the algorithm rum.

    Attributes:
          Name (List[str]): Name of the unit.
    """
    Name = ['BasicStatistics']

    def __init__(self, array):
        r"""Construct basic statistic unit.

        Args:
            array (Iterable[Union[float, int]]): Array for basic statistics.
        """
        self.array = array if isinstance(array, np.ndarray) else np.asarray(array)

    def min_value(self):
        r"""Get minimum value.

        Returns:
            Union[int, float]: Minimum value.
        """
        return self.array.min()

    def max_value(self):
        r"""Get maximum value.

        Returns:
            Union[int, float]: Maximum value.
        """
        return self.array.max()

    def mean(self):
        r"""Get mean value.

        Returns:
            Union[int, float]: Mean value.
        """
        return self.array.mean()

    def median(self):
        r"""Get median value.

        Returns:
            Union[int, float]: Median value.
        """
        return np.median(self.array)

    def standard_deviation(self):
        r"""Get standard deviation.

        Returns:
            Union[int, float]: Standard deviation.
        """
        return self.array.std(ddof=1)

    def generate_standard_report(self):
        r"""Get standard report.

        Returns:
            str: Standard report.
        """
        return "Min: {0}, Max: {1}, Mean: {2}, Median: {3}, Std. {4}".format(
            self.min_value(),
            self.max_value(),
            self.mean(),
            self.median(),
            self.standard_deviation())

def wilcoxonSignedRanks(a, b):
    r"""Get rank values from signed wilcoxon test.

    Args:
        a (numpy.ndarray): First data.
        b (numpy.ndarray): Second data.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, float]:
            1. Positive ranks.
            2. Negative ranks.
            3. T value
    """
    y = a - b
    y_diff = y[y != 0]
    r = stats.rankdata(np.abs(y_diff))
    r_all = np.sum(r) / 2
    r_p, r_n = r_all + np.sum(r[np.where(y_diff > 0)]), r_all + np.sum(r[np.where(y_diff < 0)])
    return r_p, r_n, np.min([r_p, r_n])

def friedmanRanks(*arrs):
    r"""Calculate friedman ranks for input data.

    Args:
        arrs (Iterable[Any]): TODO.

    Returns:
        numpy.ndarray: TODO.
    """
    r = np.asarray([stats.rankdata([arrs[j][i] for j in range(len(arrs))]) for i in range(len(arrs[0]))])
    return np.asarray([np.sum(r[:, i]) / len(arrs[0]) for i in range(len(arrs))])

def cd(alpha, k, n):
    r"""Get critial distance for friedman test.

    Args:
        alpha (float): Fold value.
        k (int): Number of algorithms.
        n (int): Number of algorithm results.
    """
    nemenyi_df = pd.read_csv(StringIO(pkgutil.get_data(pkg_data.__package__, 'nemenyi.csv').decode('utf-8')))
    q_a = nemenyi_df['%.2f' % alpha][nemenyi_df['k'] == k].values
    return q_a[0] * np.sqrt((k * (k + 1)) / (6 * n))

def wilcoxonTest(data, names, q=None):
    r"""Get p-values or tagged differences between algorithms.

    Args:
        data (numpy.ndarray): Multi dimensional array with algorithms data.
        names (Iterable[str]): Names of algorithms
        q (Optional[float]): TODO.

    Returns:
        pandas.DataFrame: Dataframe with p-values or tagged differences.
    """
    df = pd.DataFrame(np.asarray([[stats.wilcoxon(data[j], data[i])[1] if j != i else 1 for i in range(len(data))] for j in range(len(data))]), index=names, columns=names)
    if q is not None:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]): df.iloc[i, j] = '+' if df.iloc[i, j] <= q else '-'
    return df
