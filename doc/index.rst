:html_theme.sidebar_secondary.remove: true

.. raw:: html

   <div class="nl-hero">
     <h1 class="nl-hero-logo">
       <img class="only-light" alt="NeuroLang"
            src="_static/logo/svg/neurolang-lockup-stacked.svg">
       <img class="only-dark" alt="NeuroLang"
            src="_static/logo/svg/neurolang-lockup-stacked-dark.svg">
     </h1>
     <p class="nl-tagline">
       Probabilistic Logic Programming for Neuroimaging Analysis
     </p>
     <div class="nl-cta-group">
       <a class="sd-sphinx-override sd-btn sd-btn-primary"
          href="_static/repl/index.html">
         Try it in the browser →
       </a>
       <a class="sd-sphinx-override sd-btn sd-btn-outline-secondary"
          href="tutorial.html">
         Get Started
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

   .. grid-item-card:: ⌨️  Interactive
      :text-align: center

      Type queries in a REPL, in the browser or in a terminal, on engines of
      ready-made data: Neurosynth, NeuroStore, atlases, ontologies, tracts.
      Or embed NeuroLang in Python (:doc:`tutorial_python`).

----

Quick Start
-----------

Open `the REPL in the browser <_static/repl/index.html>`_, or install
NeuroLang and start it in a terminal:

.. code-block:: bash

   pip install "neurolang[apps]"
   neurolang-query -i --engine neurosynth

Then ask which studies of Neurosynth mention pain:

.. nl-repl::

   ans(s) :- mentions(s, "pain")

and which brain regions their peaks fall in most often, in the Destrieux
atlas:

.. nl-repl::

   pain_region(l, count(s)) :- destrieux_activated(l, s) & mentions(s, "pain")
   ans(l, n) :- pain_region(l, n)

See :doc:`repl` for the REPL, :doc:`tutorial` for a first walkthrough, and
:doc:`tutorial_datalog` for the language.

.. toctree::
   :maxdepth: 1
   :hidden:

   install
   repl
   tutorial
   tutorial_datalog
   tutorial_squall
   tutorial_text_embeddings
   tutorial_neurosynth_tests
   tutorial_cognitive_atlas
   concepts
   ontologies
   probabilistic_lifting
   inclusion_exclusion
   engine_configuration
   query_assistant
   repl_server
   tutorial_python
   performance
   auto_examples/index
   api
   changelog
   contributing
   authors
