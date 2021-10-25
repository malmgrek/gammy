# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Gammy'
copyright = '2021, Malmgrek'
author = 'Malmgrek'

# The full version, including alpha/beta/rc tags
release = '0.4.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.doctest',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autosummary_generate = True
add_module_names = False
autodoc_default_options = {
    'member-order': 'alphabetical',
    'undoc-members': True,
    'exclude-members': ','.join([
        '__attrs_attrs__',
        '__bases__',
        '__basicsize__',
        '__class__',
        '__delattr__',
        '__doc__',
        '__dict__',
        '__dictoffset__',
        '__dir__',
        '__flags__',
        '__format__',
        '__getattribute__',
        '__init__',
        '__init_subclass__',
        '__instancecheck__',
        '__itemsize__',
        '__module__',
        '__mro__',
        '__name__',
        '__new__',
        '__prepare__',
        '__qualname__',
        '__reduce__',
        '__reduce_ex__',
        '__repr__',
        '__setattr__',
        '__sizeof__',
        '__str__',
        '__subclasscheck__',
        '__subclasses__',
        '__subclasshook__',
        '__text_signature__',
        '__weakref__',
        '__weakrefoffset__',
        'mro',
    ])
}

# Matplotlib plot generation configurations
plot_html_show_formats = False
plot_pre_code = '''
import numpy as np


np.random.seed(42)
'''

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
