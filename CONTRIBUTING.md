# Contributing to NiaPy
:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

## Code of Conduct
This project and everyone participating in it is governed by the [NiaPy Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [niapy.organization@gmail.com](mailto:niapy.organization@gmail.com).

## How Can I Contribute?

### Reporting Bugs
Before creating bug reports, please check existing issues list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible. Fill out the [required template](.github/ISSUE_TEMPLATE/bug_report.md), the information it asks for helps us resolve issues faster.

### Suggesting Enhancements
- Open new issue
- Write in details what enhancement would you like to see in the future
- If you have technical knowledge, propose solution on how to implement enhancement

### Pull requests

If you are not so familiar with Git or/and GitHub, we suggest you take a look at our [beginner's guide](.github/beginners_guide.md).

- Fill in the [required template](.github/pull_request_template.md)
- Document new code
- Make sure all the code goes through Flake8 without problems (run ```make check``` command)
- Make sure PR builds goes through



## Setup development environment

### Requirements

* Make:
    * Windows: [http://mingw.org/download/installer](http://mingw.org/download/installer) [Detailed install instructions](.github/mingw_install_guide.md)
    * Mac: http://developer.apple.com/xcode
    * Linux: http://www.gnu.org/software/make
* pipenv: http://docs.pipenv.org (run ```pip install pipenv``` command)
* Pandoc: [http://johnmacfarlane.net/pandoc/installing.html] (http://johnmacfarlane.net/pandoc/installing.html) * optional
* Graphviz: [http://www.graphviz.org/Download.php](http://www.graphviz.org/Download.php) * optional

### Development dependencies

List of NiaPy's dependencies:

| Package    | Version | Platform |
| ---------- |:-------:|:--------:|
| numpy      | 1.17.0  | All      |
| matplotlib | 2.2.4   | All      |
| pandas     | 0.24.2  | All      |
| openpyxl   | 3.0.3   | All      |


List of development dependencies:

| Package                       | Version | Platform |
| ----------------------------- |:-------:|:--------:|
|flake8                         | Any     | Any      |
|pycodestyle                    | Any     | Any      |
|pydocstyle                     | Any     | Any      |
|pytest                         | ~=3.3   | Any      |
|pytest-describe                | Any     | Any      |
|pytest-expecter                | Any     | Any      |
|pytest-random                  | Any     | Any      |
|pytest-cov                     | Any     | Any      |
|freezegun                      | Any     | Any      |
|coverage-space                 | Any     | Any      |
|docutils                       | Any     | Any      |
|Pygments                       | Any     | Any      |
|wheel                          | Any     | Any      |
|twine                          | Any     | Any      |
|sniffer                        | Any     | Any      |
|sphinx                         | Any     | Any      |
|autopep8                       | Any     | Any      |
|sphinx-autobuild               | Any     | Any      |


To confirm these system dependencies are configured correctly:

```sh
$ make doctor
```

### Installation

Install project dependencies into a virtual environment:

```sh
$ make install
```

To enter created virtual environment with all installed dependencies run:

```sh
$ pipenv shell
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
$ make flake8
$ make pycodestyle
$ make pydocstyle
$ make check  # includes all checks
```

## Support

### Usage Questions

If you have questions about how to use Niapy, or have an issue that isn’t related to a bug, you can place a question on [StackOverflow](https://stackoverflow.com/).

You can also join us at our [Slack Channel](https://join.slack.com/t/niaorg/shared_invite/enQtMzExMTU2MzM1OTg4LTFlYTUxZDcwZTU4ZTBjZDgzZWE3ZTM5MjE3MjVjOTllNTNmYTVjNjE5ZTEzYTU0YTc4OTJiNWI2MDNiZjY2YjQ) or seek support via [email](mailto:niapy.organization@gmail.com)

NiaPy is a community supported package, nobody is paid to develop package nor to handle NiaPy support.

**All people answering your questions are doing it with their own time, so please be kind and provide as much information as possible.**
