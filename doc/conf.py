#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# NeuroLang documentation configuration.

import os
import sys

# -- Path setup ---------------------------------------------------------------
currentdir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath("../"))
sys.path.append(os.path.join(currentdir, "tools"))
sys.path.append(os.path.abspath("sphinxext"))

# -- Project information ------------------------------------------------------
project = "NeuroLang"
copyright = "2017–2026, Demian Wassermann et al."
author = "Demian Wassermann et al."

try:
    from importlib.metadata import version as get_version
    release = get_version("neurolang")
except Exception:
    release = "0.0.1"
source_version = ".".join(release.split(".")[:2])
version = source_version

# -- General configuration ----------------------------------------------------
needs_sphinx = "7.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "numpydoc.numpydoc",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build"]
pygments_style = "friendly"

# -- Sphinx Gallery -----------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "doc_module": ("neurolang",),
    "backreferences_dir": "gen_api",
}

# -- Autosummary / Autodoc ----------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
}

# -- Intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "nilearn": ("https://nilearn.github.io/stable/", None),
}

# -- HTML output --------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.png"
html_favicon = None
html_static_path = ["_static"]
html_css_files = ["neurolang.css"]

html_theme_options = {
    "logo": {
        "text": "NeuroLang",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NeuroLang/NeuroLang",
            "icon": "fa-brands fa-github",
        }
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "header_links_before_dropdown": 5,
    "navbar_links": [
        {"name": "Install", "url": "install"},
        {"name": "Tutorial", "url": "tutorial"},
        {"name": "Concepts", "url": "concepts"},
        {"name": "Examples", "url": "auto_examples/index"},
        {"name": "API", "url": "api"},
    ],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

# No sidebar on the landing page; standard sidebar everywhere else.
html_sidebars = {
    "index": [],
    "**": ["sidebar-nav-bs", "page-toc"],
}

html_domain_indices = False

# -- LaTeX output (kept minimal) ----------------------------------------------
latex_documents = [
    (
        "index",
        "neurolang.tex",
        "NeuroLang Documentation",
        "Demian Wassermann",
        "manual",
    ),
]
