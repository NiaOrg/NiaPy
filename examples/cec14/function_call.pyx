import numpy as np
cimport numpy as np  # for np.ndarray

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef extern from "cec14_test_func.h":
    double runtest(double*, int, int)

cpdef double run_fun(np.ndarray[double, ndim=1, mode='c'] x, int fnum=1):
    # x = np.zeros((10,), dtype=np.double)
    # print (runtest(&x[0], 10, fnum))
    # x = np.full((100,), 1.0, dtype=np.double)
    return runtest(&x[0], len(x), fnum)
