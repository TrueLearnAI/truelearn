|PyPi| |License| |Unit tests| |Static analysis| |codecov|

|FOSSA Status| |docs| |Black|

|TrueLearn|

**TrueLearn** is a machine learning library for predicting and modelling learner engagement with educational resources.

.. |PyPi| image:: https://img.shields.io/pypi/pyversions/truelearn?label=Python&style=flat
   :target: https://pypi.org/project/truelearn/
   :alt: PyPI - Python Version

.. |License| image:: https://img.shields.io/badge/License-MIT-blue
   :target: https://github.com/TrueLearnAI/truelearn/blob/main/LICENSE
   :alt: License

.. |Unit tests| image:: https://github.com/TrueLearnAI/truelearn/actions/workflows/unit_tests.yml/badge.svg
   :target: https://github.com/TrueLearnAI/truelearn/actions/workflows/unit_tests.yml
   :alt: Unit tests

.. |Static analysis| image:: https://github.com/comp0016-group1/TrueLearn/actions/workflows/static_analysis.yml/badge.svg
   :target: https://github.com/TrueLearnAI/truelearn/actions/workflows/static_analysis.yml
   :alt: Static analysis

.. |codecov| image:: https://codecov.io/gh/TrueLearnAI/truelearn/branch/main/graph/badge.svg?token=69JZ051NAO
   :target: https://codecov.io/gh/TrueLearnAI/truelearn
   :alt: codecov

.. |FOSSA Status| image:: https://app.fossa.com/api/projects/git%2Bgithub.com%2FTrueLearnAI%2Ftruelearn.svg?type=small
   :target: https://app.fossa.com/projects/git%2Bgithub.com%2FTrueLearnAI%2Ftruelearn?ref=badge_small
   :alt: FOSSA Status

.. |docs| image:: https://readthedocs.org/projects/truelearn/badge/?version=latest
   :target: https://truelearn.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black

.. |TrueLearn| image:: https://raw.githubusercontent.com/truelearnai/truelearn/main/docs/images/TrueLearn_logo.png
   :target: https://truelearnai.github.io/
   :alt: TrueLearn


User Installation
#################

**Install from PyPI:** ::

   pip install -U truelearn

For more information on installation, see the `Getting Started guide <https://truelearn.readthedocs.io/en/stable/tutorial/quickstart.html>`_.

Documentation
#############

**Latest stable release** is available at: https://truelearn.readthedocs.io/en/stable/

**Development version** is available at: https://truelearn.readthedocs.io/en/latest/

Change Log
##########

See the `Change Log <https://truelearn.readthedocs.io/en/stable/index.html#change-log>`_
for a history of all the major changes to the truelearn.

Alternatively you can find it in the ``CHANGELOG.rst`` file found here:
https://github.com/TrueLearnAI/truelearn/blob/main/docs/changelog.rst

Contributing
############

Contributions are welcome, and they are greatly appreciated! Every little bit helps,
and credit will always be given.
Please see: `Contributing Guide <https://truelearn.readthedocs.io/en/stable/dev/index.html>`_ for more information!
We have listed a brief guide below.


Source Code
-----------

You can check out the latest the source code with the following command::

   git clone https://github.com/TrueLearnAI/truelearn.git


Installing TrueLearn from source
--------------------------------

After cloning the repository, you can install TrueLearn locally from source::

   pip install -e .[dev]

   # if you are using zsh
   pip install -e ".[dev]"

See the `Contributing Guide: Getting Started <https://truelearn.readthedocs.io/en/latest/dev/get_started.html>`_
for a more detailed explanation.


Testing
-------

After installation, you can run the tests from the source directory::

   pytest truelearn

See the `Contributing Guide: Testing guide <https://truelearn.readthedocs.io/en/latest/dev/testing.html>`_
for a more detailed explanation.


Before submitting a PR
----------------------

Please make sure you have followed the Guidelines outlined in the
`Contributing Guide: Before/During PR  <https://truelearn.readthedocs.io/en/latest/dev/before_pr.html>`_.
