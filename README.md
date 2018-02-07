[![Unix Build Status](https://img.shields.io/travis/NiaOrg/NiaPy/master.svg)](https://travis-ci.org/NiaOrg/NiaPy) [![Windows Build Status](https://ci.appveyor.com/api/projects/status/l5c0rp04mp04mbtq?svg=true)](https://ci.appveyor.com/project/NiaOrg/NiaPy)<br>Metrics: [![Coverage Status](https://img.shields.io/coveralls/NiaOrg/NiaPy/master.svg)](https://coveralls.io/r/NiaOrg/NiaPy) [![Scrutinizer Code Quality](https://img.shields.io/scrutinizer/g/NiaOrg/NiaPy.svg)](https://scrutinizer-ci.com/g/NiaOrg/NiaPy/?branch=master)<br>Usage: [![PyPI Version](https://img.shields.io/pypi/v/NiaPy.svg)](https://pypi.python.org/pypi/NiaPy)

# Overview

Python micro framework for comparing nature-inspired algorithms.

The micro framework features following algorithms:

- basic:
  - Bat Algorithm [(see example)](examples/run_ba.py)
  - Firefly Algorithm [(see example)](examples/run_fa.py)
- modified:
  - Hybrid Bat Algorithm [(see example)](examples/run_hba.py)

# Setup

## Requirements

* Python 3.6+ (backward compatibility with 2.7.14)

## Installation

Install NiaPy with pip:

```sh
$ pip install NiaPy
```

or directly from the source code:

```sh
$ git clone https://github.com/NiaOrg/NiaPy.git
$ cd NiaPy
$ python setup.py install
```

# Usage

After installation, the package can imported:

```sh
$ python
>>> import NiaPy
>>> NiaPy.__version__
```

For more usage examples please look at *examples* folder.

## About 

## Mission

## Contributors

**License**
This package is distributed under the MIT License. This license can be found online at http://www.opensource.org/licenses/MIT.

**Disclaimer**
This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!
