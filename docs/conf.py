#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is execfile()d with the current directory set to its
containing dir.
Note that not all possible configuration values are present in this
autogenerated file.
All configuration values have a default; values that are commented out
serve to show the default.
requirements:
"""

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Build apidocs automatically when sphinx is run ----------------------

from sphinx.apidoc import main

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

out_path = 'apidocs'
module_path = os.path.join(ROOT_PATH, 'cell_lists')
main(['--separate',
      '--output-dir', out_path, module_path,
      '--no-toc',
      '--force'])


# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# General information about the project.
project = 'Cell Lists'
author = 'Jaan Tollander de Balsch'
# title = ''
# institute = ''
# department = ''

# Copyright string
copyright = author


# Version and Release
version = '0.1'
release = '0.1'


# Style
language = 'en'
today_fmt = '%Y-%m-%d'
pygments_style = 'sphinx'
todo_include_todos = False

# -- Graphviz --------------------------------------------------------

# extensions += ['sphinx.ext.graphviz']
# graphviz_dot = 'dot'
# graphviz_dot_args = []
# graphviz_output_format = 'png'  # svg


# -- Napoleon --------------------------------------------------------
# http://www.sphinx-doc.org/en/stable/ext/napoleon.html


# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True


# -- Options for HTML output ----------------------------------------------


# import sphinx_rtd_theme
# html_theme = 'sphinx_rtd_theme'
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_theme_options = {}

html_theme = 'alabaster'
html_theme_options = {}

html_logo = None
html_favicon = None
html_static_path = ['_static']
html_last_updated_fmt = '%Y-%m-%d'

# -- Options for HTMLHelp output ------------------------------------------

htmlhelp_basename = project

# -- Options for LaTeX output ---------------------------------------------

latex_engine = 'xelatex'
latex_show_urls = 'footnote'
latex_additional_files = []
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'figure_align': 'htbp',
    'preamble': '',
}

latex_elements['preamble'] += r"""
\usepackage{amsfonts}
\usepackage{parskip}
\usepackage{microtype}
"""

latex_logo = None


"""
Latex documents
- source
- target
- title
- author
- documentclass
"""
latex_documents = [
    (master_doc,
     project + '.tex',
     project,
     author,
     'report'),
]

# -- Options for manual page output ---------------------------------------
"""
One entry per manual page. List of tuples
- source start file
- name
- description
- authors
- manual section
"""
man_pages = [
    (master_doc,
     '',
     '',
     [author],
     1)
]

# -- Options for Texinfo output -------------------------------------------
"""
Grouping the document tree into Texinfo files. List of tuples
- source start file
- target name
- title
- author
- dir menu entry
- description
- category
"""
texinfo_documents = [
    (master_doc,
     '',
     '',
     author,
     '',
     '',
     'Miscellaneous'),
]