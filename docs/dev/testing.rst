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

.. note:: You may need to run the commands listed using ``python -m`` to ensure that
          that the current directory is added to SYS.PATH to prevent module not found errors.

          To do this add ``python -m`` before the command. e.g. ``python -m pytest truelearn``

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

General Strategy
""""""""""""""""
To add a new test, you need to add the test case to the corresponding test in ``truelearn/tests``.

For example, if you want to add a new test for ``KnowledgeClassifier``, you can add the test case to ``truelearn/tests/test_learning.py``.

If it's a new test for a class, we recommend you grouping the test cases into a class. For example, if you want to test
``XClassifier``, you can group all the test cases under ``TestXClassifier``.

If it's a new test for a method, you can optionally group the test cases into a class (if there are many). If it is just a test case,
you do not need to put it into a class.

Also, based on the ``pytest`` rules, you need to make sure that the names of your methods follow the following pattern::

    def test_xxx():
        ...


Writing an visualisation test
"""""""""""""""""""""""""""""

Writing a test for visualisation is a bit more difficult than a simple test.
In tests for visualisation, we typically want to test our generated file is identical or similar to a baseline file.
We call this file comparison test in truelearn.

To write a file comparison test for visualisation, you only need to add a simple class decorator to your class.

For example, this is a simple file comparison test inside ``truelearn/tests/test_utils_visualisations.py``::

    @file_comparison(plotter_type="plotly")
    class TestBarPlotter:
        def test_default(self, resources):
            plotter = visualisations.BarPlotter()
            plotter.plot(resources[0])
            return plotter

        def test_history(self, resources):
            plotter = visualisations.BarPlotter()
            plotter.plot(resources[0], history=True)
            return plotter

The first time this test is run, because there will be no baseline image to compare against, so the test will fail.
But you will find that there is a directory called ``tests`` generated in your current working directory.
Inside it, there will be a directory whose name is lowercase for the test class name (i.e. ``testbarplotter`` for the example above).
You can find out all the generated baseline files in that directory. Your next step is copy this directory to ``truelearn/tests/baseline_files/``.
Then, when you run the test again, if the file you generated matches the baseline file, your test will pass.

Due to the way that file comparison tests work, all visualisation tests must be grouped into classes.
We recommend that you name all tests that make use of file comparison ``TestXXXPlotter``, where ``XXXPlotter`` is the plotter you want to test.
If you do not use file comparisons, you can simply create ``TestXXXPlotterNormal`` and
place all tests that do not use file comparisons in this class.

You can see the documentation of ``file_comparison`` for additional information about its use.
