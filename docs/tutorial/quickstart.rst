.. _installation:

Getting Started
===============

Installation
------------

TrueLearn requires Python 3.7+ to run. You can install it by using::

    pip install -U truelearn

It's highly recommended to use a virtual environment to use TrueLearn as it can help avoid potential conflicts.
You can use either `venv <https://docs.python.org/3/tutorial/venv.html>`_ or `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
to manage your environment.

Below is an example to use `venv` to create a virtual environment and activate it::

    # create a virtual environment named truelearn-venv
    python -m venv truelearn-venv

    # activate the virtual environment
    truelearn-venv\Scripts\activate

After activating the virtual environment (optional), you can install Truelearn using pip::

    # install truelearn (latest version)
    pip install -U truelearn
