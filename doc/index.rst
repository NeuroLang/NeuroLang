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
   authors
