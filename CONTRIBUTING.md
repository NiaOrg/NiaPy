# Contributing to NiaPy
:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

## Code of Conduct
This project and everyone participating in it is governed by the [NiaPy Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [niapy.organization@gmail.com](mailto:niapy.organization@gmail.com).

## How Can I Contribute?

### Reporting Bugs
Before creating bug reports, please check existing issues list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible. Fill out the [required template](.github/issue_template.md), the information it asks for helps us resolve issues faster.

### Suggesting Enhancements
- Open new issue
- Write in details what enhancement whould you like to see in the future
- If you have technical knowledge, propose solution on how to implement enhancement

### Pull requests

If you are not so familiar with Git or/and GitHub, we suggest you take a look at our [begginer's guide](.github/begginers_guide.md). 

- Fill in the [reqired template](.github/pull_request_template.md)
- Document new code
- Make sure all the code goes through Pylint without problems (run ```make check``` command)
- Make sure PR builds (Travis and AppVeyor) goes through

## Setup development environment

### Requirements

* Make:
    * Windows: [http://mingw.org/download/installer](http://mingw.org/download/installer) [Detailed install instructions](.github/mingw_install_guide.md)
    * Mac: http://developer.apple.com/xcode
    * Linux: http://www.gnu.org/software/make
* pipenv: http://docs.pipenv.org (run ```pip install pipenv``` command)
* Pandoc: http://johnmacfarlane.net/pandoc/installing.html
* Graphviz: http://www.graphviz.org/Download.php

To confirm these system dependencies are configured correctly:

```sh
$ make doctor
```

### Installation

Install project dependencies into a virtual environment:

```sh
$ make install
```

## Development Tasks

### Testing

Manually run the tests:

```sh
$ make test
```

or keep them running on change:

```sh
$ make watch
```

> In order to have OS X notifications, `brew install terminal-notifier`.

### Documentation

Build the documentation:

```sh
$ make docs
```

### Static Analysis

Run linters and static analyzers:

```sh
$ make pylint
$ make pycodestyle
$ make pydocstyle
$ make check  # includes all checks
