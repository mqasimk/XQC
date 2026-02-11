# docs/conf.py
import os
import sys
# Insert the parent directory of the project into sys.path, so that autodoc can find the xqc package
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'XQC'
author = 'XQC Developers'
# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]

# Generate autosummary pages automatically
autosummary_generate = True
# Enable both Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
