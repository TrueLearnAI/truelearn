.. _installation:

Getting Started
===============

Installation
------------

TrueLearn requires Python 3.7+ to run. You can install it from https://www.python.org/downloads/.

It's highly recommended to use a virtual environment to use TrueLearn as it can help avoid potential conflicts.
You can use either `venv <https://docs.python.org/3/tutorial/venv.html>`_ or `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
to manage your environment.

Below is an example to use `venv` to create a virtual environment and activate it::

    python -m venv truelearn-venv # create a virtual environment named truelearn-venv
    truelearn-venv\Scripts\activate # activate the virtual environment

After activating the virtual environment (optional), you can install Truelearn using pip::

    pip install -U truelearn # install truelearn (latest version)
