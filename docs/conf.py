# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import inspect
import sys
import subprocess
import re
from typing import Any

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
              'sphinx.ext.linkcode',
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
# https://github.com/DisnakeDev/disnake/blob/7853da70b13fcd2978c39c0b7efa59b34d298186
# /docs/conf.py#L192

def git(*args):
    return subprocess.check_output(["git", *args]).strip().decode()


# Current git reference. Uses branch/tag name if found, otherwise uses commit hash
git_ref = None
try:
    git_ref = git("name-rev", "--name-only", "--no-undefined", "HEAD")
    git_ref = re.sub(r"^(remotes/[^/]+|tags)/", "", git_ref)
except Exception:
    pass

# (if no name found or relative ref, use commit hash instead)
if not git_ref or re.search(r"[\^~]", git_ref):
    try:
        git_ref = git("rev-parse", "HEAD")
    except Exception:
        git_ref = "main"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    try:
        obj: Any = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)

        if isinstance(obj, property):
            obj = inspect.unwrap(obj.fget)  # type: ignore

        path = os.path.relpath(inspect.getsourcefile(obj),
                               start=_disnake_module_path)  # type: ignore
        src, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    path = f"{path}#L{lineno}-L{lineno + len(src) - 1}"
    return f"https://github.com/comp0016-group1/truelearn/blob/{git_ref}/truelearn/" \
           f"{path}"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Install furo theme with pip install furo
html_theme = 'furo'

# See GitHub issue : https://github.com/readthedocs/readthedocs.org/issues/1776
html_static_path = []
