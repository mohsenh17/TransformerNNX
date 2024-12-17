# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TransformerNNX'
copyright = '2024, Mohsen Hosseini'
author = 'Mohsen Hosseini'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # For Python code documentation
    'sphinx.ext.napoleon',      # For NumPy/Google style docstrings
    'sphinx.ext.viewcode',      # Links to source code
    'nbsphinx',                 # Render Jupyter Notebooks
    'sphinxcontrib.bibtex',     # For BibTeX citations
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# -- bibtex configuration ----------------------------------------------------
bibtex_bibfiles = ['ref.bib']
bibtex_reference_style = 'label'