#!/usr/bin/env python

"""Setup script for the package."""

import os
import sys
import logging

import setuptools


PACKAGE_NAME = 'NiaPy'
MINIMUM_PYTHON_VERSION = '2.7'


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {0}+ is required.".format(MINIMUM_PYTHON_VERSION))


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
        readme = open("README.rst").read()  # changelog = open("CHANGELOG.rst").read()
    except IOError:
        return "<placeholder>"
    else:
        return readme  # return readme + '\n' + changelog


check_python_version()

setuptools.setup(
    name=read_package_variable('__project__'),
    version=read_package_variable('__version__'),

    description="Python micro framework for building nature-inspired algorithms.",
    url='https://github.com/NiaOrg/NiaPy',
    author='NiaOrg',
    author_email='niapy.organization@gmail.com',

    packages=setuptools.find_packages(),

    long_description=build_description(),
    license='MIT',
    classifiers=[
        # TODO: update this list to match your application: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],

    install_requires=[
        # TODO: Add your library's requirements here
        'pytest ~= 3.7.1',
        'coverage ~= 4.4.2',
        'coverage-space ~= 1.0.2',
        'numpy ~= 1.14.0',
        'enum34 ~= 1.1.6',
        'click ~= 6.0',
        'scipy ~= 1.0.0',
        'xlsxwriter ~= 1.0.2',
        'matplotlib ~= 2.2.2'
    ]
)
