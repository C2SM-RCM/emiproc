# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'emiproc'
copyright = '2022-2024, Empa'
author = 'Empa'
release = 'v2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    'enum_tools.autoenum',
]

templates_path = ['_templates']
exclude_patterns = []

#autodoc_typehints = 'description'  # show type hints in doc body instead of signature

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

python_maximum_signature_line_length = 88

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
