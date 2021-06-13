.. NiaPy documentation master file, created by
   sphinx-quickstart on Tue Feb 20 10:56:53 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _reference:

.. image:: _static/NiaPyLogo.png
    :align: center

NiaPy's documentation
=================================

.. image:: http://joss.theoj.org/papers/10.21105/joss.00613/status.svg
        :target: https://doi.org/10.21105/joss.00613
        :alt: Citation

.. automodule:: niapy

Nature-inspired algorithms are a very popular tool for solving optimization problems.
Since the beginning of their era, numerous variants of nature-inspired algorithms were
developed (`paper 1 <https://arxiv.org/abs/1307.4186>`_, `paper 2 <https://www.mdpi.com/2076-3417/8/9/1521>`_).
To prove their versatility, those were tested in various domains on various applications,
especially when they are hybridized, modified or adapted. However, implementation of
nature-inspired algorithms is sometimes difficult, complex and tedious task. In order to break
this wall, NiaPy is intended for simple and quick use, without spending a time for
implementing algorithms from scratch.

* **Free software:** MIT license
* **Github repository:** https://github.com/NiaOrg/NiaPy
* **Python versions:** 3.6.x, 3.7.x, 3.8.x, 3.9.x

The main documentation is organized into a couple sections:

* :ref:`about-docs`
* :ref:`user-docs`
* :ref:`dev-docs`
* :ref:`api-docs`

.. _about-docs:

.. toctree::
   :maxdepth: 2
   :caption: General

   about
   features
   authors
   changelog
   code_of_conduct

.. _user-docs:

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   getting_started
   tutorials/index
   support

.. _dev-docs:

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   guides/index
   contributing_to_niapy
   installation
   testing
   documentation

.. _api-docs:

.. toctree::
  :caption: API Documentation

  api/index
