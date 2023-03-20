.. _testing:

Testing the library
===================

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

Run the following from the commandline::

    pytest

Output Options
""""""""""""""
Example::

    pytest --junitxml=report.xml

Will output the results to a JUnit XML file called report.
Please see the `pytest documentation`_ for more output options.

.. _pytest documentation: https://docs.pytest.org/en/stable/


Calculating the Code Coverage
-----------------------------
**Note: We calculate both statement and branch coverage.**

Run the following from the commandline::

    pytest --cov

Output Options
""""""""""""""
Example::

    pytest --cov --cov-report "xml:coverage.xml"

Will output the results to an XML file called coverage.
Please see the `pytest-cov documentation`_ for more output options.

.. _pytest-cov documentation: https://pytest-cov.readthedocs.io/en/latest/


Running the Linter (Static Code Analysis)
-----------------------------------------
Run the following from the commandline::

    prospector truelearn

Output Options
""""""""""""""
Example::

    prospector truelearn --output-format xunit:lint.xml

Will output the results to an XML file called lint.xml.
Please see the `prospector documentation`_ for more output options.

.. _prospector documentation: https://prospector.landscape.io/en/master/


Adding New Tests
----------------
To add a new test, you need to add the test case to the corresponding test in ``truelearn/tests``.

For example, if you want to add a new test for ``KnowledgeClassifier``, you can add the test case to ``truelearn/tests/test_learning.py``.

If it's a new test for a class, we recommend you grouping the test cases into a class. For example, if you want to test
``XClassifier``, you can group all the test cases under ``TestXClassifier``.

If it's a new test for a method, you can optionally group the test cases into a class (if there are many). If it is just a test case,
you do not need to put it into a class.

Also, based on the ``pytest`` rules, you need to make sure that the names of your methods follow the following pattern::

    def test_xxx():
        ...
