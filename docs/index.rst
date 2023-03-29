.. TrueLearn documentation master file, created by
   sphinx-quickstart on Tue Feb  7 21:23:20 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TrueLearn's documentation!
=====================================
.. toctree::
   :maxdepth: 2
   :hidden:

   tutorial/quickstart
   examples/index
   modules/api_reference

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Development

   dev/index

|PyPi| |License| |Version| |codecov|

.. |PyPi| image:: https://img.shields.io/pypi/pyversions/truelearn?label=Python&style=flat
   :target: https://pypi.org/project/truelearn/
   :alt: PyPI - Python Version


.. |License| image:: https://img.shields.io/badge/License-MIT-blue
   :target: https://github.com/comp0016-group1/TrueLearn-python-library/blob/main/LICENSE
   :alt: License

.. |Version| image:: https://img.shields.io/pypi/v/truelearn.svg
   :target: https://pypi.org/project/truelearn/
   :alt: PyPI

.. |codecov| image:: https://codecov.io/gh/comp0016-group1/TrueLearn/branch/main/graph/badge.svg?token=69JZ051NAO
   :target: https://codecov.io/gh/comp0016-group1/TrueLearn
   :alt: codecov


TrueLearn is a machine learning library for modelling learner engagement with educational resources.

Get Started
-----------

See the :ref:`installation` for a quick introduction to the package.

Related Papers
--------------

This work is related to the following papers:

- `TrueLearn A Family of Bayesian Algorithms to Match Lifelong Learners to Open Educational Resources`_ published at the Thirty-forth AAAI Conference on Artifical Intelligence, 2020 at New York, NY, USA. BibTex below::

   @inproceedings{truelearn2020,
       author = {Bulathwela, S. and Perez-Ortiz, M. and Yilmaz, E. and Shawe-Taylor, J.},
       title={TrueLearn: A Family of Bayesian Algorithms to Match Lifelong Learners to Open Educational Resources},
       booktitle = {AAAI Conference on Artificial Intelligence},
       year = {2020}
   }

- `Power to the Learner: Towards Human-Intuitive and Integrative Recommendations with Open Educational Resources`_ published in 2022 as part of the Special Issue on AI and Interaction Technologies for Social Sustainability. BibTex below::

   @Article{su141811682,
      author = {Bulathwela, Sahan and Pérez-Ortiz, María and Yilmaz, Emine and Shawe-Taylor, John},
      title = {Power to the Learner: Towards Human-Intuitive and Integrative Recommendations with Open Educational Resources},
      journal = {Sustainability},
      volume = {14},
      year = {2022},
      number = {18},
      article-number = {11682},
      url = {https://www.mdpi.com/2071-1050/14/18/11682},
      ISSN = {2071-1050},
      DOI = {10.3390/su141811682}
   }



- `PEEK: A Large Dataset of Learner Engagement with Educational Videos`_ published in 2021 as part of the 4th workshop on Online Recommender Systems and User Modeling at ACM RecSys 2021. BibTex below::

   @article{DBLP:journals/corr/abs-2109-03154,
       author = {Bulathwela, Sahan and Yilmaz, Emine and Shawe-Taylor, John},
       title={PEEK: A Large Dataset of Learner Engagement with Educational Videos},
       booktitle = {4th workshop on Online Recommender Systems and User Modeling at ACM RecSys 2021},
       year = {2021}
   }


.. _TrueLearn A Family of Bayesian Algorithms to Match Lifelong Learners to Open Educational Resources: https://arxiv.org/abs/2002.00001
.. _Power to the Learner\: Towards Human-Intuitive and Integrative Recommendations with Open Educational Resources: https://www.mdpi.com/2071-1050/14/18/11682
.. _PEEK\: A Large Dataset of Learner Engagement with Educational Videos: https://arxiv.org/abs/2106.00683


.. include:: changelog.rst
.. include:: authors.rst

