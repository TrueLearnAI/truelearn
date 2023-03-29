# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import inspect
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('..'))

# If we are on Read the Docs we need to clean up the generated folder
# before building the docs (it's not done automatically), otherwise
# old files will be left in the generated folder and Sphinx will not
# rebuild them.
if os.environ.get('READTHEDOCS'):
    path = Path('./modules/generated/')
    if path.exists():
        shutil.rmtree(str(path))

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
              'sphinx.ext.linkcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'sphinx.ext.doctest',
              'sphinx_copybutton',
              ]
templates_path = ['templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc extension -------------------------------------------
autodoc_mock_imports = ['trueskill',
                        'sklearn',
                        'mpmath']

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'mpmath': ('https://mpmath.org/doc/current/', None),
    'trueskill': ('https://trueskill.org/', None),
}


# -- Options for linkcode extension ------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html
# Code below from:
# https://github.com/Lasagne/Lasagne/blob/master/docs/conf.py#L114

def linkcode_resolve(domain, info):
    """Determine the URL corresponding to the sourcecode."""

    def find_source():
        # Find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286

        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(truelearn.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != 'py' or not info['module']:
        return None
    try:
        source_info = find_source()
        filename = f"truelearn/{source_info[0]}#L{source_info[1]}-L{source_info[2]}"
    except Exception:
        filename = info['module'].replace('.', '/') + '.py'
    tag = 'main' if 'dev' in version else 'v' + version
    return f"https://github.com/comp0016-group1/truelearn/blob/{tag}/{filename}"


# -- Options for napoleon extension ------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
napoleon_include_special_with_doc = True
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_logo = 'images/TrueLearn_logo.png'


# See GitHub issue : https://github.com/readthedocs/readthedocs.org/issues/1776
html_static_path = []

# -- Extensions to the  Napoleon GoogleDocstring class ---------------------

napoleon_use_ivar = True

