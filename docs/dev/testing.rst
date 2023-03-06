Testing the library
===================

Prerequisites
-------------
Make sure you have followed :ref:`advanced_installation`.

Important Note
--------------
The tools are run using the ``python`` command to ensure that the current directory is added to the ``sys.path``.
This prevents module not found errors.

Contents
--------
- `Configuration`_
- `Running the Unit Tests`_
- `Calculating the Code Coverage`_
- `Running the Linter (Static Code Analysis)`_


Configuration
^^^^^^^^^^^^^

The configuration for the:

- Unit tests are located in the pyproject.toml_ file.
- Code coverage is located in the pyproject.toml_ file.
- Linter is located in the pyproject.toml_ file and the prospector.yaml_ file.

.. _pyproject.toml: https://github.com/comp0016-group1/truelearn/blob/main/pyproject.toml
.. _prospector.yaml: https://github.com/comp0016-group1/truelearn/blob/main/prospector.yaml

Running the Unit Tests
^^^^^^^^^^^^^^^^^^^^^^

Run the following from the commandline:

``python -m pytest``

Output Options
""""""""""""""
Example:

    ``python -m pytest --junitxml=report.xml``

Will output the results to a JUnit XML file called report.
Please see the `pytest documentation`_ for more output options.

.. _pytest documentation: https://docs.pytest.org/en/stable/

Calculating the Code Coverage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Note: We calculate both statement and branch coverage.**

Run the following from the commandline:

``python -m pytest --cov``

Output Options
""""""""""""""
Example:

    ``python -m pytest --cov --cov-report "xml:coverage.xml"``

Will output the results to an XML file called coverage.
Please see the `pytest-cov documentation`_ for more output options.

.. _pytest-cov documentation: https://pytest-cov.readthedocs.io/en/latest/

Running the Linter (Static Code Analysis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following from the commandline:

``python -m prospector``

Output Options
""""""""""""""
Example:

    ``python -m prospector --output-format xunit:lint.xml``

Will output the results to an XML file called lint.xml.
Please see the `prospector documentation`_ for more output options.

.. _prospector documentation: https://prospector.landscape.io/en/master/
