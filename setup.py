#!/usr/bin/env python

"""Setup script for the package."""

import io
import os
import sys
import logging

import setuptools


PACKAGE_NAME = 'niapy'
MINIMUM_PYTHON_VERSION = (3, 6)


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info[:2] < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key, filename='__init__.py'):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, filename)
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ', 2)
            if parts[:-1] == [key, '=']:
                return parts[-1].strip("'")
    logging.warning("'%s' not found in '%s'", key, module_path)
    return None


def build_description():
    """Build a description for the project from documentation files."""
    try:
        readme = io.open("README.rst", encoding="UTF-8").read()
    except IOError:
        return "<placeholder>"
    else:
        return readme


check_python_version()

PACKAGE_VERSION = read_package_variable('__version__')

setuptools.setup(
    name=PACKAGE_NAME,
    version="2.0.0rc17",
    description="""
        Python micro framework for building nature-inspired algorithms.
        """,
    url='https://github.com/NiaOrg/NiaPy',
    author='NiaOrg',
    author_email='niapy.organization@gmail.com',
    packages=setuptools.find_packages(),
    long_description=build_description(),
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    tests_require=[
        'flake8 ~= 3.7.7',
        'astroid >= 2.0.4',
        'pytest ~= 3.7.1',
        'coverage ~= 4.4.2',
        'coverage-space ~= 1.0.2'
    ],
    install_requires=[
        'numpy >= 1.17.0',
        'matplotlib >= 2.2.4',
        'pandas >= 0.24.2',
        'openpyxl >= 3.0.3',
    ]
)
