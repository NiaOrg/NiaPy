from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [
    Extension('function_call', ['function_call.pyx', 'cec14_test_func.cpp'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-std=c++14'],
              language='c++'
              ),
]

setup(
    ext_modules=cythonize(extensions),
    extra_compile_args=["-w", '-g'],
)
