# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'TrueLearn'
# pylint: disable=redefined-builtin
copyright = '2023, TrueLearn'
author = 'TrueLearn Team'

# pylint: disable=wrong-import-position
import truelearn

version = truelearn.__version__
release = truelearn.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'sphinx.ext.doctest',
              ]
templates_path = ['templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc extension -------------------------------------------
autodoc_mock_imports = ['trueskill',
                        'sklearn',
                        'mpmath']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Install furo theme with pip install furo
html_theme = 'furo'

# See GitHub issue : https://github.com/readthedocs/readthedocs.org/issues/1776
html_static_path = []
