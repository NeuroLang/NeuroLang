# NeuroLang Documentation Modernisation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernise the NeuroLang Sphinx docs from Bootstrap 3 / `sphinx_bootstrap_theme` to PyData Sphinx Theme, with a new hero landing page, uv-based install instructions, three new content pages, and GitHub Actions CI/CD deployment to GitHub Pages.

**Architecture:** All changes live inside the existing `doc/` directory (RST source) and `setup.cfg` (deps). A standalone CSS file supplies brand-colour overrides via PyData CSS custom properties. A new GitHub Actions workflow at `.github/workflows/docs.yml` builds and deploys the docs automatically.

**Tech Stack:** Python, Sphinx ≥ 7, pydata-sphinx-theme ≥ 0.15, sphinx-design ≥ 0.5, sphinx-gallery, numpydoc, GitHub Actions, peaceiris/actions-gh-pages@v3

---

## File Map

| Path | Action | Purpose |
|------|--------|---------|
| `setup.cfg` | Modify lines 59-63 | Replace `sphinx_bootstrap_theme` with modern deps |
| `doc/conf.py` | Modify | Swap theme, add sphinx-design, configure navbar/sidebar |
| `doc/_static/neurolang.css` | Create | Brand colour CSS variables |
| `doc/index.rst` | Rewrite | Hero landing page with feature cards + quickstart |
| `doc/install.rst` | Rewrite | uv-based, sphinx-design tabs, no Anaconda |
| `doc/tutorial.rst` | Modify | Minor prose cleanup |
| `doc/tutorial_logic_programming.rst` | Modify | Fix typos, improve prose (probabilistic stub stays commented) |
| `doc/api.rst` | Modify | Minor cleanup |
| `doc/concepts.rst` | Create | Datalog, probabilistic reasoning, neuroimaging, architecture |
| `doc/changelog.rst` | Create | What's New / release history |
| `doc/contributing.rst` | Create | Dev setup, tests, docs build, PR checklist |
| `.github/workflows/docs.yml` | Create | Build + deploy to GitHub Pages on push to master |

---

## Task 1: Update doc dependencies in setup.cfg

**Files:**
- Modify: `setup.cfg:59-63`

- [ ] **Step 1: Replace the `[doc]` extras block**

Open `setup.cfg`. Find the block starting at line 59:
```
doc =
  sphinx
  sphinx_bootstrap_theme
  sphinx-gallery
  numpydoc
```

Replace it with:
```
doc =
  sphinx>=7.0
  pydata-sphinx-theme>=0.15
  sphinx-design>=0.5
  sphinx-gallery
  numpydoc
```

- [ ] **Step 2: Commit**

```bash
git add setup.cfg
git commit -m "chore(docs): replace sphinx_bootstrap_theme with pydata-sphinx-theme and sphinx-design"
```

---

## Task 2: Rewrite `doc/conf.py` for PyData theme

**Files:**
- Modify: `doc/conf.py`

- [ ] **Step 1: Write the new conf.py**

Replace the entire contents of `doc/conf.py` with the following:

```python
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
```

- [ ] **Step 2: Verify conf.py is valid Python**

```bash
python -c "import ast; ast.parse(open('doc/conf.py').read()); print('conf.py OK')"
```

Expected output: `conf.py OK`

- [ ] **Step 3: Commit**

```bash
git add doc/conf.py
git commit -m "feat(docs): migrate to pydata-sphinx-theme, add sphinx-design"
```

---

## Task 3: Add brand CSS file

**Files:**
- Create: `doc/_static/neurolang.css`

- [ ] **Step 1: Create the CSS file**

Create `doc/_static/neurolang.css` with the following content:

```css
/* NeuroLang brand colour overrides for PyData Sphinx Theme */

/* Light mode */
:root {
    --pst-color-primary: #5C6BC0;
    --pst-color-primary-highlight: #3949AB;
    --pst-color-secondary: #26A69A;
    --pst-color-accent: #26A69A;
}

/* Dark mode */
[data-theme="dark"] {
    --pst-color-primary: #7986CB;
    --pst-color-primary-highlight: #5C6BC0;
    --pst-color-secondary: #4DB6AC;
    --pst-color-accent: #4DB6AC;
}

/* Hero section on the landing page */
.nl-hero {
    padding: 3rem 1rem 2.5rem;
    text-align: center;
    border-bottom: 1px solid var(--pst-color-border);
    margin-bottom: 2rem;
}

.nl-hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--pst-color-primary);
    margin-bottom: 0.5rem;
}

.nl-hero .nl-tagline {
    font-size: 1.2rem;
    color: var(--pst-color-secondary-text);
    margin-bottom: 1.5rem;
}

.nl-hero .nl-cta-group {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
}

/* Feature card grid */
.nl-feature-grid {
    margin: 2rem 0;
}

/* Quickstart block */
.nl-quickstart {
    background: var(--pst-color-surface);
    border-left: 4px solid var(--pst-color-primary);
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    margin: 2rem 0;
}

.nl-quickstart h3 {
    margin-top: 0;
    color: var(--pst-color-primary);
}
```

- [ ] **Step 2: Commit**

```bash
git add doc/_static/neurolang.css
git commit -m "feat(docs): add brand colour CSS overrides for PyData theme"
```

---

## Task 4: Rewrite the landing page (`doc/index.rst`)

**Files:**
- Modify: `doc/index.rst`

- [ ] **Step 1: Write the new index.rst**

Replace the entire contents of `doc/index.rst` with:

```rst
:html_theme.sidebar_secondary.remove: true

.. raw:: html

   <div class="nl-hero">
     <h1>NeuroLang</h1>
     <p class="nl-tagline">
       Probabilistic Logic Programming for Neuroimaging Analysis
     </p>
     <div class="nl-cta-group">
       <a class="sd-sphinx-override sd-btn sd-btn-primary" href="install.html">
         Get Started →
       </a>
       <a class="sd-sphinx-override sd-btn sd-btn-outline-secondary"
          href="auto_examples/index.html">
         View Examples
       </a>
     </div>
   </div>


NeuroLang is a **probabilistic logic programming** system for the analysis of
neuroimaging data. It lets you express complex queries over brain images,
ontologies, and tabular databases in a declarative style — and reason about
them probabilistically.

----

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: nl-feature-grid

   .. grid-item-card:: 🧠  Language
      :text-align: center

      A declarative language built on **Datalog** — a logic rule-based query
      language for relational data. Write what you want, not how to compute it.

   .. grid-item-card:: 🎲  Probabilistic Solver
      :text-align: center

      Supports **discrete, probabilistic, and open-world** reasoning. Query
      over uncertain data with sound probabilistic semantics.

   .. grid-item-card:: 🐍  Python Integration
      :text-align: center

      Embed logic programs directly in Python. Load images, dataframes, and
      ontologies as relations with ``NeurolangDL``.

----

Quick Start
-----------

.. code-block:: bash

   pip install neurolang

.. code-block:: python

   from neurolang.frontend import NeurolangDL

   nl = NeurolangDL()
   nl.add_tuple_set([(0, 1), (1, 2), (2, 3)], name="connected")

   with nl.environment as e:
       e.reachable[e.x, e.y] = e.connected[e.x, e.y]
       e.reachable[e.x, e.y] = e.reachable[e.x, e.z] & e.connected[e.z, e.y]
       result = nl.query((e.x, e.y), e.reachable(e.x, e.y))

   print(result)

See :doc:`install` for full installation instructions and :doc:`tutorial` for
a guided walkthrough.

----

.. toctree::
   :maxdepth: 1
   :hidden:

   install
   tutorial
   concepts
   auto_examples/index
   api
   tutorial_logic_programming
   changelog
   contributing
```

- [ ] **Step 2: Commit**

```bash
git add doc/index.rst
git commit -m "feat(docs): new hero landing page with feature cards and quickstart"
```

---

## Task 5: Rewrite the install page (`doc/install.rst`)

**Files:**
- Modify: `doc/install.rst`

- [ ] **Step 1: Write the new install.rst**

Replace the entire contents of `doc/install.rst` with:

```rst
Installing NeuroLang
====================

NeuroLang requires **Python ≥ 3.8**. The recommended way to install it is
with `uv <https://github.com/astral-sh/uv>`_ — a fast Python package manager —
or with standard ``pip``.

.. tab-set::

   .. tab-item:: Windows

      **1. Install uv**

      Open PowerShell and run:

      .. code-block:: powershell

         powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

      Or use pip directly if you already have Python installed:

      .. code-block:: text

         pip install uv

      **2. Install NeuroLang**

      .. code-block:: text

         uv pip install neurolang

      **3. Verify**

      Open a new terminal and run:

      .. code-block:: text

         python -c "import neurolang; print('NeuroLang installed OK')"

   .. tab-item:: macOS

      **1. Install uv**

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

      **2. Install NeuroLang**

      .. code-block:: bash

         uv pip install neurolang

      **3. Verify**

      .. code-block:: bash

         python -c "import neurolang; print('NeuroLang installed OK')"

   .. tab-item:: Linux

      **1. Install uv**

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

      **2. Install NeuroLang**

      .. code-block:: bash

         uv pip install neurolang

      **3. Verify**

      .. code-block:: bash

         python -c "import neurolang; print('NeuroLang installed OK')"

   .. tab-item:: From source

      Clone the repository and install in editable mode with all dev and doc
      dependencies:

      .. code-block:: bash

         git clone https://github.com/NeuroLang/NeuroLang.git
         cd NeuroLang
         uv pip install -e ".[dev,doc]"

      To update your local copy:

      .. code-block:: bash

         git pull
         uv pip install -e ".[dev,doc]"

      Verify:

      .. code-block:: bash

         python -c "import neurolang; print('NeuroLang installed OK')"


Using pip without uv
---------------------

If you prefer plain pip:

.. code-block:: bash

   pip install neurolang


Dependencies
------------

NeuroLang requires the following libraries (installed automatically):

* `NumPy <https://numpy.org>`_
* `SciPy <https://scipy.org>`_
* `pandas <https://pandas.pydata.org>`_
* `nibabel <https://nipy.org/nibabel/>`_
* `nilearn <https://nilearn.github.io>`_
* `scikit-learn <https://scikit-learn.org>`_
* `matplotlib <https://matplotlib.org>`_
* `rdflib <https://rdflib.readthedocs.io>`_
```

- [ ] **Step 2: Commit**

```bash
git add doc/install.rst
git commit -m "feat(docs): modernise install page with uv, sphinx-design tabs, no Anaconda"
```

---

## Task 6: Update `doc/tutorial.rst` (prose cleanup)

**Files:**
- Modify: `doc/tutorial.rst`

- [ ] **Step 1: Fix the broken reference and improve the intro**

Open `doc/tutorial.rst`. The file currently starts with a blank line before the title. Make these targeted changes:

1. Fix the section title to use proper capitalisation: `Get Started with NeuroLang` → keep as is (already good).

2. Replace the line:
   ```
   which we formalise in python as:
   ```
   with:
   ```
   which we formalise in Python as:
   ```

3. Fix the misspelling on line ~77 — change `Neurlang` → `NeuroLang`:
   ```
   which we formalise in Neurlang in the classical logical programming syntax:
   ```
   becomes:
   ```
   which we formalise in NeuroLang in the classical logical programming syntax:
   ```

4. Add a note at the top (after the title and first paragraph) pointing to the logic programming tutorial:

   After the first paragraph ending with `...probabilistic logic programming language based on Datalog [abiteboul1995]_, [maier2018]_.`, add:

   ```rst
   .. note::

      If you are already familiar with logic programming, see
      :doc:`tutorial_logic_programming` for a more technical walkthrough.
   ```

- [ ] **Step 2: Commit**

```bash
git add doc/tutorial.rst
git commit -m "fix(docs): fix typos and add cross-reference note in tutorial"
```

---

## Task 7: Update `doc/tutorial_logic_programming.rst` (prose fix)

**Files:**
- Modify: `doc/tutorial_logic_programming.rst`

- [ ] **Step 1: Fix broken references and prose**

Open `doc/tutorial_logic_programming.rst`.

1. Fix the broken cross-reference anchors. The first two lines are:
   ```rst
   .. python_embedded_
   
   Using Datalog Embedded in Python
   ```
   Change `.. python_embedded_` to the correct RST label syntax:
   ```rst
   .. _python_embedded:
   ```

2. The section `.. datalog_` is referenced but never defined. Add it above the section heading for the Datalog frontend (if present), or remove the reference in line 5. Since the Datalog frontend section does not exist in the file, remove the broken reference from line 5. Change:
   ```rst
   NeuroLang is implemented over the basis of Datalog+/- with probabilistic extensions. In that there are two main frontend which might came useful: the python_embedded_ frontend and the datalog_ frontend
   ```
   to:
   ```rst
   NeuroLang is implemented on top of Datalog+/- with probabilistic extensions.
   The primary way to use it is the :ref:`python_embedded` Python-embedded frontend.
   ```

3. Fix the doctest syntax for the aggregation example — `>>>` lines should not use `>>>` inside a `::` block. The section "Including Aggregations and Builtin Functions" uses inconsistent `>>>` and `>>>` indenting. Leave the content as-is (it is a known in-progress section), but add a note:

   After the heading `Including Aggregations and Builtin Functions`, add:
   ```rst
   .. note::

      The aggregation API is under active development. The examples below
      illustrate the intended syntax.
   ```

- [ ] **Step 2: Commit**

```bash
git add doc/tutorial_logic_programming.rst
git commit -m "fix(docs): fix RST label syntax and prose in logic programming tutorial"
```

---

## Task 8: Update `doc/api.rst` (minor cleanup)

**Files:**
- Modify: `doc/api.rst`

- [ ] **Step 1: Add a brief intro and fix the title**

The file currently starts with `.. _api_ref:` then `User Guide` as the section title. Rename the title from `User Guide` to `API Reference` and add a one-line description:

Replace:
```rst
.. _api_ref:

.. currentmodule:: neurolang

User Guide
==========
```

with:
```rst
.. _api_ref:

.. currentmodule:: neurolang

API Reference
=============

Complete API documentation for all public NeuroLang modules.
Generated automatically from docstrings.

```

- [ ] **Step 2: Commit**

```bash
git add doc/api.rst
git commit -m "fix(docs): rename 'User Guide' to 'API Reference' with intro text"
```

---

## Task 9: Create `doc/concepts.rst`

**Files:**
- Create: `doc/concepts.rst`

- [ ] **Step 1: Write concepts.rst**

Create `doc/concepts.rst` with the following content:

```rst
.. _concepts:

Concepts
========

This page explains the core ideas behind NeuroLang. You do not need to read
this before the :doc:`tutorial` — it is here as a reference once you want to
understand *why* things work the way they do.


Logic Programming & Datalog
----------------------------

Logic programming is a style of programming where you declare *what* is true,
not *how* to compute it. A **Datalog** program consists of two kinds of
statements:

**Facts** — ground truths about the world:

.. code-block:: python

   with nl.environment as e:
       e.region["V1"] = True
       e.region["V2"] = True
       e.adjacent["V1", "V2"] = True

**Rules** — derived truths (if A and B are true, then C is true):

.. code-block:: python

   with nl.environment as e:
       # x is reachable from y if x is adjacent to y
       e.reachable[e.x, e.y] = e.adjacent[e.x, e.y]
       # x is reachable from y if x is adjacent to z and z is reachable from y
       e.reachable[e.x, e.y] = e.adjacent[e.x, e.z] & e.reachable[e.z, e.y]

The Datalog *engine* (the solver) computes all facts that can be derived from
the rules. NeuroLang extends standard Datalog with:

* **Aggregations** — ``COUNT``, ``MAX``, ``SUM`` over sets of tuples
* **Built-in functions** — register any Python callable as a Datalog symbol
* **Tuple-generating dependencies (TGDs)** — open-world reasoning rules


Probabilistic Reasoning
------------------------

Standard Datalog is deterministic: a fact is either derived or it is not. In
many neuroimaging applications we need to reason about *uncertain* data — for
example, whether a brain region is functionally connected to another region
given noisy fMRI data.

NeuroLang adds **independent probabilistic facts**: each tuple in a
probabilistic relation is an independent Bernoulli random variable with an
associated probability. For example, a term-to-region mapping derived from
meta-analysis might state that the term "memory" is associated with region
"hippocampus" with probability 0.87.

Queries over probabilistic data use **possible worlds semantics**: the answer
to a query is the probability that the query holds in a randomly sampled world.

.. note::

   The probabilistic extensions are implemented in
   :mod:`neurolang.probabilistic`. See the example gallery for concrete
   neuroimaging use cases.


Neuroimaging Integration
-------------------------

NeuroLang treats neuroimaging data as relational data:

* **Volumetric images** (NIfTI, loaded via nibabel) are represented as sets of
  ``(voxel_id, intensity)`` or ``(x, y, z, intensity)`` tuples.
* **Atlas labels** (e.g., Destrieux, AAL) become ``(label, region_id)``
  relations.
* **Ontologies** (OWL/RDF via rdflib) are loaded as triple stores and
  queried with Datalog rules.
* **Coordinate activations** (NeuroSynth) become ``(pmid, x, y, z)``
  probabilistic relations.

The :class:`~neurolang.frontend.deterministic_frontend.NeurolangDL` and
:class:`~neurolang.frontend.NeurolangPDL` frontends expose helper methods
such as :meth:`add_tuple_set` and :meth:`add_atlas_set` to load these
data sources.


Architecture
-------------

NeuroLang is structured in three layers::

   ┌─────────────────────────────────────────────────────────┐
   │                    User / Python                        │
   │     NeurolangDL  /  NeurolangPDL  frontend              │
   │           (neurolang/frontend/)                         │
   └───────────────────────┬─────────────────────────────────┘
                           │  Datalog/probabilistic program
                           ▼
   ┌─────────────────────────────────────────────────────────┐
   │          Intermediate Representation (IR)               │
   │   Expressions, symbol table, type system                │
   │   (neurolang/expressions.py, neurolang/logic/)          │
   └───────────────────────┬─────────────────────────────────┘
                           │  Relational algebra plan
                           ▼
   ┌─────────────────────────────────────────────────────────┐
   │                     Solver                              │
   │   Chase algorithm, relational algebra, SDD/WMC          │
   │   (neurolang/datalog/, neurolang/probabilistic/)        │
   └─────────────────────────────────────────────────────────┘

The **frontend** layer translates Python expressions into an internal IR. The
**IR layer** performs type inference and expression normalisation. The
**solver** executes the query using a chase-based fixpoint algorithm (for
deterministic queries) or a weighted model counter (for probabilistic queries).
```

- [ ] **Step 2: Commit**

```bash
git add doc/concepts.rst
git commit -m "feat(docs): add Concepts page covering Datalog, probabilistic reasoning, architecture"
```

---

## Task 10: Create `doc/changelog.rst`

**Files:**
- Create: `doc/changelog.rst`

- [ ] **Step 1: Write changelog.rst**

Create `doc/changelog.rst` with the following content:

```rst
.. _changelog:

Changelog
=========

All notable changes to NeuroLang are documented here.

Contributors: when adding a change, add an entry under *Unreleased* using the
format below, then move it to the appropriate version section when releasing.


Unreleased
----------

*No unreleased changes yet.*


v0.0.1 (Alpha)
--------------

Initial alpha release of NeuroLang.

**New features:**

* Datalog-based logic programming frontend (:class:`~neurolang.frontend.deterministic_frontend.NeurolangDL`)
* Probabilistic extensions with independent probabilistic facts
  (:class:`~neurolang.frontend.NeurolangPDL`)
* Tuple-generating dependencies (TGDs) for open-world reasoning
* Aggregation support (``COUNT``, ``MAX``, ``SUM``, custom callables)
* Neuroimaging integration: nibabel image loading, Destrieux atlas, NeuroSynth
* Example gallery: Destrieux gyri, sulcal queries, NeuroSynth implementation


How to add a changelog entry
-----------------------------

Add entries under **Unreleased** in the following format:

.. code-block:: rst

   Unreleased
   ----------

   **New features:**

   * Short description of the feature (`:class:` or `:func:` link if applicable)

   **Bug fixes:**

   * Short description of the bug and fix

   **Breaking changes:**

   * What changed and how to migrate
```

- [ ] **Step 2: Commit**

```bash
git add doc/changelog.rst
git commit -m "feat(docs): add Changelog page with v0.0.1 entry and contribution instructions"
```

---

## Task 11: Create `doc/contributing.rst`

**Files:**
- Create: `doc/contributing.rst`

- [ ] **Step 1: Write contributing.rst**

Create `doc/contributing.rst` with the following content:

```rst
.. _contributing:

Contributing to NeuroLang
=========================

Thank you for considering a contribution! This page explains how to set up
a development environment and submit changes.


Prerequisites
-------------

* **Python ≥ 3.8**
* **git**
* **uv** (fast Python package manager) — install with:

  .. code-block:: bash

     curl -LsSf https://astral.sh/uv/install.sh | sh

  Or on Windows (PowerShell):

  .. code-block:: powershell

     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"


Development Install
-------------------

.. code-block:: bash

   git clone https://github.com/NeuroLang/NeuroLang.git
   cd NeuroLang
   uv pip install -e ".[dev,doc]"

This installs NeuroLang in *editable* mode together with all development
(``pytest``, ``pytest-cov``) and documentation (Sphinx) dependencies.


Running the Tests
-----------------

.. code-block:: bash

   pytest neurolang/

To run with coverage:

.. code-block:: bash

   pytest --cov=neurolang neurolang/

The test suite should pass fully before you submit a pull request.


Building the Documentation
--------------------------

.. code-block:: bash

   make -C doc html

This will build the HTML docs into ``doc/_build/html/``. Open
``doc/_build/html/index.html`` in your browser to preview.

.. note::

   Building the full example gallery requires all neuroimaging packages
   (nibabel, nilearn, neurosynth) and can take several minutes. To skip
   the gallery during a quick doc build, set:

   .. code-block:: bash

      SPHINXOPTS="-D sphinx_gallery_conf.run_stale_examples=False" make -C doc html


Submitting a Pull Request
--------------------------

1. Fork the repository on GitHub.
2. Create a feature branch: ``git checkout -b my-feature``
3. Make your changes.
4. Ensure all tests pass: ``pytest neurolang/``
5. Add a changelog entry under *Unreleased* in ``doc/changelog.rst``.
6. Update docstrings for any changed public API.
7. Push your branch and open a pull request against ``master``.

**PR checklist:**

* [ ] Tests pass (``pytest neurolang/``)
* [ ] New code has docstrings (NumPy format)
* [ ] Changelog entry added to ``doc/changelog.rst``
* [ ] No new Sphinx build warnings (``make -C doc html``)


Code Style
----------

NeuroLang follows `PEP 8 <https://peps.python.org/pep-0008/>`_ with
NumPy-style docstrings. Run ``flake8`` before committing:

.. code-block:: bash

   pip install flake8
   flake8 neurolang/


Reporting Bugs
--------------

Open an issue on the
`GitHub issue tracker <https://github.com/NeuroLang/NeuroLang/issues>`_.
Include:

* NeuroLang version (``python -c "import neurolang; print(neurolang.__version__)"`` if available)
* Python version
* Operating system
* Minimal reproducible example
```

- [ ] **Step 2: Commit**

```bash
git add doc/contributing.rst
git commit -m "feat(docs): add Contributing guide with uv setup, tests, docs build, PR checklist"
```

---

## Task 12: Create GitHub Actions docs workflow

**Files:**
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Write the workflow file**

Create `.github/workflows/docs.yml` with the following content:

```yaml
name: Deploy Documentation

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

permissions:
  contents: write

jobs:
  build-docs:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install uv
        run: pip install uv

      - name: Install doc dependencies
        run: uv pip install --system ".[doc]"

      - name: Build HTML documentation
        run: make -C doc html SPHINXOPTS="-W --keep-going"

      - name: Deploy to GitHub Pages
        if: github.event_name == 'push' && github.ref == 'refs/heads/master'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./doc/_build/html
          force_orphan: true
```

**Note:** The `SPHINXOPTS="-W --keep-going"` flag turns Sphinx warnings into
errors and continues past the first error so all issues are reported in one
run. Remove `-W` if the existing docs have too many warnings to fix at once.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/docs.yml
git commit -m "ci: add GitHub Actions workflow to build and deploy docs to GitHub Pages"
```

---

## Task 13: Final integration check

**Files:**
- No new files — verification only

- [ ] **Step 1: Install the new doc dependencies locally**

```bash
pip install uv
uv pip install --system -e ".[doc]"
```

Expected: installs `pydata-sphinx-theme`, `sphinx-design`, and other deps
without errors.

- [ ] **Step 2: Build the docs (skip gallery for speed)**

```bash
make -C doc html SPHINXOPTS="-D sphinx_gallery_conf.run_stale_examples=False"
```

Expected: build completes with no errors. Warnings about missing
`sphinx-gallery` examples are acceptable. The output should be in
`doc/_build/html/`.

- [ ] **Step 3: Spot-check the rendered output**

Open `doc/_build/html/index.html` and verify:

1. PyData theme header with "NeuroLang" logo text and GitHub icon link
2. Light/dark mode toggle visible in navbar
3. Hero section with "Get Started →" and "View Examples" buttons
4. Three feature cards: Language, Probabilistic Solver, Python Integration
5. Quickstart code block visible

Open `doc/_build/html/install.html` and verify:
1. Four tabs: Windows / macOS / Linux / From source
2. `uv pip install neurolang` commands visible
3. No Anaconda references

Open `doc/_build/html/concepts.html` and verify:
1. All four sections render (Logic Programming, Probabilistic, Neuroimaging, Architecture)
2. Architecture diagram ASCII art renders in a `pre` block

- [ ] **Step 4: Commit any fixes discovered during spot-check**

If you found rendering issues (e.g. a broken RST directive, a missing `.. toctree::` entry), fix them and commit:

```bash
git add doc/
git commit -m "fix(docs): resolve rendering issues found in integration check"
```

- [ ] **Step 5: Final commit confirming build passes**

```bash
git add .
git commit -m "chore(docs): confirm full docs build passes after modernisation"
```

---

## Self-Review Against Spec

| Spec requirement | Covered by task |
|-----------------|----------------|
| Replace `sphinx_bootstrap_theme` with PyData Sphinx Theme | Task 1, Task 2 |
| Brand colour `#5C6BC0` indigo via CSS vars | Task 3 |
| Hero landing page with feature cards + quickstart | Task 4 |
| `sphinx_design` tabs on install page | Task 5 |
| Replace Anaconda with uv everywhere | Task 5, Task 11 |
| Fix broken URLs / encoding in install | Task 5 (full rewrite) |
| New `concepts.rst` with 4 sections | Task 9 |
| New `changelog.rst` | Task 10 |
| New `contributing.rst` | Task 11 |
| GitHub Actions docs.yml (build + deploy) | Task 12 |
| `setup.cfg` doc extras updated | Task 1 |
| `doc/conf.py` updated | Task 2 |
| Tutorial prose fixes | Task 6 |
| Logic programming tutorial RST fixes | Task 7 |
| API page cleanup | Task 8 |
| Probabilistic stub left as-is | ✓ (not touched) |
| Integration smoke-test | Task 13 |
