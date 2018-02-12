[![Unix Build Status](https://img.shields.io/travis/NiaOrg/NiaPy/master.svg)](https://travis-ci.org/NiaOrg/NiaPy)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/l5c0rp04mp04mbtq?svg=true)](https://ci.appveyor.com/project/GregaVrbancic/niapy)
[![Coverage Status](https://img.shields.io/coveralls/NiaOrg/NiaPy/master.svg)](https://coveralls.io/r/NiaOrg/NiaPy) [![Scrutinizer Code Quality](https://img.shields.io/scrutinizer/g/NiaOrg/NiaPy.svg)](https://scrutinizer-ci.com/g/NiaOrg/NiaPy/?branch=master)
[![PyPI Version](https://img.shields.io/pypi/v/NiaPy.svg)](https://pypi.python.org/pypi/NiaPy)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/NiaOrg/NiaPy.svg)](http://isitmaintained.com/project/NiaOrg/NiaPy "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/NiaOrg/NiaPy.svg)](http://isitmaintained.com/project/NiaOrg/NiaPy "Percentage of issues still open")

## About 
Nature-inspired algorithms are a very popular tool for solving optimization problems. Since the beginning of their era, numerous variants of [nature-inspired algorithms were developed](https://arxiv.org/abs/1307.4186). To prove their versatility, those were tested in various domains on various applications, especially when they are hybridized, modified or adapted. However, implementation of nature-inspired algorithms is sometimes difficult, complex and tedious task. In order to break this wall, NiaPy is intended for simple and quick use, without spending a time for implementing algorithms from scratch.

## Mission
Our mission is to build a collection of nature-inspired algorithms and create a simple interface for managing the optimization process along with statistical evaluation. NiaPy will offer:


- numerous benchmark functions implementations,
- use of various nature-inspired algorithms without struggle and effort with a simple interface,
- easy comparison between nature-inspired algorithms and
- statistical tests.

# Overview

Python micro framework for building nature-inspired algorithms.

The micro framework features following algorithms:

- basic:
  - Bat Algorithm ([see example](examples/run_ba.py))
  - Firefly Algorithm ([see example](examples/run_fa.py))
  - Differential evolution algorithm ([see example](examples/run_de.py))
  - Flower Pollination algorithm ([see example](examples/run_fpa.py))
- modified:
  - Hybrid Bat Algorithm ([see example](examples/run_hba.py))

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

For more usage examples please look at **examples** folder.

## Contributing

[![Open Source Helpers](https://www.codetriage.com/niaorg/niapy/badges/users.svg)](https://www.codetriage.com/niaorg/niapy)

We encourage you to contribute to NiaPy! Please check out the [Contributing to NiaPy guide](CONTRIBUTING.md) for guidelines about how to proceed.

Everyone interacting in NiaPy's codebases, issue trackers, chat rooms and mailing lists is expected to follow the NiaPy [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

## Licence
This package is distributed under the MIT License. This license can be found online at http://www.opensource.org/licenses/MIT.

## Disclaimer
This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!
