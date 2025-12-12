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

* Python >= 3.11
* uv: https://docs.astral.sh/uv/

### Development dependencies

**NiaPy dependencies:**

| Package    | Version  | Platform |
|------------|:--------:|:--------:|
| numpy      | \>=2.3.5 |   All    |
| pandas     | \>=2.3.3 |   All    |
| openpyxl   | >=3.1.5  |   All    |
| matplotlib | >=3.10.8 |   All    |

**Development dependencies:**

| Package         | Version  | Platform |
|-----------------|:--------:|:--------:|
| pytest          | \>=9.0.0 |   All    |
| pytest-cov      | \>=7.0.0 |   All    |
| pytest-randomly | \>=4.0.1 |   All    |

**Documentation dependencies (optional):**

| Package          | Version  | Platform |
|------------------|:--------:|:--------:|
| sphinx           | \>=8.2.3 |   All    |
| sphinx-rtd-theme | \>=3.0.2 |   All    |

### Installation

Install project dependencies into a virtual environment:

```sh
uv sync
```

Install the optional documentation dependencies with:

```sh
uv sync --group docs
```

## Development Tasks

### Testing

Run the tests:

```sh
uv run pytest
```

### Documentation

Build the documentation:

```sh
uv run sphinx-build ./docs ./docs/_build
```

## Support

### Usage Questions

If you have questions about how to use Niapy, or have an issue that isn’t related to a bug, you can place a question on [StackOverflow](https://stackoverflow.com/).

You can also join us at our [Slack Channel](https://join.slack.com/t/niaorg/shared_invite/enQtMzExMTU2MzM1OTg4LTFlYTUxZDcwZTU4ZTBjZDgzZWE3ZTM5MjE3MjVjOTllNTNmYTVjNjE5ZTEzYTU0YTc4OTJiNWI2MDNiZjY2YjQ) or seek support via [email](mailto:niapy.organization@gmail.com)

NiaPy is a community supported package, nobody is paid to develop package nor to handle NiaPy support.

**All people answering your questions are doing it with their own time, so please be kind and provide as much information as possible.**
