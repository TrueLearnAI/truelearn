Before/During PR
================

Prior to opening a pull request, ensure that your code has been auto-formatted, and that it passes both the tests and lint checks.


Format
------
We use `black formatter`_ to format our code.

.. _black formatter: https://github.com/psf/black

Before you commit your changes, make sure you run the following command from the root directory::

    black truelearn

It will automatically format your code.


Testing
-------
Make sure you follow the instructions on :ref:`testing` to test your changes.

If you aim at fixing a bug, remember to add at least one new test case for the bug you found.

If you aim at providing some new functionalities, make sure you have fully tested them, which means
you should aim for 100% test coverage.


Linting
-------
Make sure you follow the instructions on :ref:`testing` to lint your changes.


Closing
-------
You are now ready to ship your PR! 🚀

`Note: If you are a member of core maintainers, make sure you check the next section about how to release TrueLearn.`
