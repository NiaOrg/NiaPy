# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, expression-not-assigned, len-as-condition, no-self-use, unused-argument,no-else-return, old-style-class

import numpy as np

__all__ = ['BasicStatistics']


class BasicStatistics(object):

    Name = ['BasicStatistics']

    def __init__(self, array):
        self.array = array

    def min_value(self):
        return self.array.min()

    def max_value(self):
        return self.array.max()

    def mean(self):
        return self.array.mean()

    def median(self):
        return np.median(self.array)

    def standard_deviation(self):
        return self.array.std(ddof=1)

    def generate_standard_report(self):
        return "Min: {0}, Max: {1}, Mean: {2}, Median: {3}, Std. {4}".format(
            self.min_value(),
            self.max_value(),
            self.mean(),
            self.median(),
            self.standard_deviation())
