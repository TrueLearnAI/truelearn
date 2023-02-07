# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import truelearn
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'TrueLearn'
copyright = '2023, KD-7,sahanbull,yuxqiu,deniselezi,aaneelshalman'
author = 'KD-7,sahanbull,yuxqiu,deniselezi,aaneelshalman'

version = truelearn.__version__
release = truelearn.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'sphinx.ext.githubpages']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Install furo theme with pip install furo
html_theme = 'furo'
html_static_path = ['_static']
