import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "numpy/arrayobject.h":
	void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef extern from "cec15_test_func.h":
	double runtest(double*, int, int)

cpdef double run_fun(np.ndarray[double, ndim=1, mode='c'] x, int fnum=1):
	return runtest(&x[0], len(x), fnum)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
