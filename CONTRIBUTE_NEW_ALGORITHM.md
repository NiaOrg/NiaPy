# Contributing new nature-inspired algorithms to NiaPy framework

## Introduction

Thank you for taking the time to explore this document. The NiaPy framework has experienced over 5 years of growth, thanks to the helpfulness and willingness of our community contributors. Their significant contributions have played a vital role in the success of this project. While designing and implementing key functions for the main framework is crucial, adding new algorithms is an equally important step in keeping the framework dynamic and up-to-date.

Currently, the NiaPy framework hosts more than 30 algorithms, categorized into three families:

- Basic Algorithms: These are vanilla implementations of nature-inspired algorithms, often derived from research papers where authors propose new algorithms.

- Modified Algorithms: This category includes modified, hybrid, and adaptive variants of basic algorithms.

- Other Algorithms: This encompasses algorithms that fit well under the optimization umbrella but cannot be directly classified into the previously mentioned families (e.g., random search).

For a comprehensive overview of the implemented algorithms, please refer to the [following document](https://raw.githubusercontent.com/firefly-cpp/NiaPy/master/Algorithms.md).

## Prerequisites

Before diving into the contribution process, ensure you meet the following prerequisites:

- Basic knowledge of optimization algorithms.
- Familiarity with Python and Numpy.
- An understanding of NiaPy's structure and existing algorithms.

We'll delve into the details of the last prerequisite in this document.

## Adding a New Algorithm

### Algorithm Overview

Given the current influx of metaphor-based nature-inspired algorithms in the research area, it is essential to make wise choices when selecting which algorithm to implement and include in the NiaPy framework.

### Code Structure

#### Main Algorithm File

1. Create a New File: Create a new file in niapy/algorithms/{basic/modified/other}. The file name should correspond to the acronym of the nature-inspired algorithm.

2. Define Algorithm Class: Begin by defining a new class for your algorithm, inheriting from the appropriate base class (e.g., Algorithm). Name your class after the nature-inspired algorithm; avoid using acronyms.

3. Initialization: Implement the __init__ method to initialize necessary parameters and attributes specific to your algorithm. Ensure clarity and consistency in parameter naming.

- Set Parameters Method: Modify the set_parameters method to handle parameters specific to your algorithm. Update the docstring to provide clear information about each parameter.
- Init Population Method: Adjust the init_population method to initialize the initial population for your algorithm. Add any additional parameters required by your algorithm.

4. Run Iteration Method: Customize the run_iteration method, the core function of your algorithm. Implement the main logic, considering the optimization task, population, and iteration-specific parameters.

#### Documentation

- Provide Detailed Information: Document your algorithm, including its reference paper, attributes, and usage instructions.

#### Tests

- Write Test Cases: Write at least two test cases to ensure the correctness of your algorithm's implementation.

#### Examples

- Provide Simple Example: Include a simple example for running your algorithm.

#### Code Style and PEP 8 Compliance

- Follow the PEP 8 style guide for Python.
- Maintain consistency in formatting and naming conventions.

#### License

- MIT License: Each code contribution in this repository is licensed under the MIT license.

## References


