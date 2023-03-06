Testing the library
===================

Prerequisites
-------------
Make sure you have followed :ref:`advanced_installation`.

Important Note
--------------
The tools are run using the ``python`` command to ensure that the current directory is added to the ``sys.path``.

Contents
--------
- :ref:`running_the_unit_tests`
- :ref:`calculating_the_code_coverage`
- :ref:`running_the_linter_static_code_analysis`

Configuration
-------------

The configuration for the:

- Unit tests are located in the pyproject.toml_ file.
- Code coverage is located in the pyproject.toml_ file.
- Linter is located in the pyproject.toml_ file and the prospector.yaml_ file.

.. _pyproject.toml: https://github.com/comp0016-group1/truelearn/blob/main/pyproject.toml
.. _prospector.yaml: https://github.com/comp0016-group1/truelearn/blob/main/prospector.yaml

Running the Unit Tests
----------------------

The unit tests are located in the ``tests`` directory.

Running the tests is as simple as running the following command from the commandline:

``python -m pytest``

Calculating the Code Coverage
-----------------------------

Code coverage is calculated using pytest-cov.

*Note:* We calculate both statement and branch coverage.

To run the tests with coverage, run the following command from the commandline:

``python -m pytest --cov``


Running the Linter (Static Code Analysis)
-----------------------------------------

The linter is run using prospector.

To run the linter, run the following command from the commandline:

``python -m prospector``
