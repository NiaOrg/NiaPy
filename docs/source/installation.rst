Installation
============

Setup development environment
-----------------------------

Requirements
~~~~~~~~~~~~

- Python: `download <https://www.python.org/downloads/>`__ (3.11 or greater)
- uv: `docs <https://docs.astral.sh/uv/>`__


Installation of development dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install project dependencies into a virtual environment:

.. code-block:: bash

    uv sync

Install documentation dependencies with:

.. code-block:: bash

    uv sync --group docs

Run tests with:

.. code-block:: bash

    uv run pytest
