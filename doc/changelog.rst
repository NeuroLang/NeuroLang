.. _changelog:

Changelog
=========

All notable changes to NeuroLang are documented here.

NeuroLang is in alpha.  Pre-release versions (``0.1aN``) are published on
`PyPI <https://pypi.org/project/neurolang/#history>`_; the *Unreleased*
section below lists the changes on the current ``master`` branch that are
not yet part of a numbered release.

Contributors: when adding a change, add an entry under *Unreleased* using the
format below, then move it to the appropriate version section when releasing.


Unreleased
----------

**New features:**

* The interactive query REPL runs in the browser, with nothing to
  install: https://neurolang.github.io/_static/repl/index.html runs
  ``neurolang-query -i --engine neurosynth --resolution 2`` on Pyodide,
  with the REPL's commands and output, and ``.view`` shows its viewer
  below the output.  ``?ref=BRANCH`` in its address runs the NeuroLang
  of that branch, tag or commit, read from GitHub (``&repo=OWNER/NAME``
  for a fork).  It is :class:`neurolang.utils.tui.web.WebRepl`
  driven by a page (:file:`doc/_static/repl/`), whose bundle, NeuroLang's
  wheel and the data of the engine, :file:`doc/make_repl_bundle.py`
  builds with the documentation.  To run in Pyodide, ``pysdd``, which has
  no WebAssembly build, became optional, as the queries that lifted
  inference solves do not need it, and queries run in the calling thread
  where threads cannot start.

* Engines in :file:`engines.yaml` can inherit from other engines with
  ``extends`` and from ``abstract`` base entries, so settings common to
  several engines are declared once.  ``neurosynth`` now extends a shared
  ``coordinate_meta_analysis`` base.  See :doc:`engine_configuration`.
* A new ``neurostore`` engine, built on the same base, loads the NeuroStore
  database (as fetched by NiMARE) with the predicates of ``neurosynth``,
  TF-IDF computed from the abstracts (words and sequences of up to three
  words), the titles, abstracts and LLM-extracted concepts and domains of
  the studies, and SPECTER2 ``query_similarity`` over its abstracts
  (a ``text_embeddings`` source can now be engine predicates,
  ``source: {predicate: ...}``).  A ``brain_mask(i, j, k)`` predicate gives
  the whole brain (MNI152 brain mask) next to ``cortical_mask``.
* The ``neurosynth`` and ``neurostore`` engines define the lift, relative
  risk and uniformity ratio of a term, Neurosynth's association and
  uniformity tests, as predicates: ``uniformity('working memory', i, j, k, u)``,
  ``lift``, ``relative_risk`` and ``ratios`` (over the whole brain, with a new
  ``selected_voxel`` choice); see :ref:`tutorial-neurosynth-tests`.
  With SPECTER2, ``uniformity_llm('a description', i, j, k, u)``, ``lift_llm``,
  ``relative_risk_llm`` and ``ratios_llm`` do the same for a free text: the
  studies of the term are the 5% closest to it (``study_similarity``,
  ``similar_study``).
* ``neurosynth`` and ``neurostore`` load the Cognitive Atlas ontology
  (``ontologies``, now loaded before ``datalog_init``) for open-world queries
  over its concepts, tasks and contrasts, joined with the studies through
  ``cogat_subclass_of``, ``cogat_measures`` and ``cogat_term``; see
  :doc:`tutorial_cognitive_atlas`.
* NeuroLang has a logo (:file:`doc/_static/logo/`): the brain written as
  a probabilistic program. It heads the README and the documentation,
  whose colours become those of the logo, and is the favicon of the
  documentation, of the web application and of the ``.view`` viewer.

* The ``.view`` viewer has the controls of NiiVue's display: the view
  (a slice, the three slices in a row, a column or a grid, with or
  without the 3D render, or the 3D render alone), the drag mode of the
  right button (contrast, pan and zoom, measure, angle, 3D slicer), the
  crosshair's colour, width and 3D drawing, the gamma of the background,
  NiiVue's colour bars, labels, orientation cube and ruler, the
  radiological convention, the interpolation and a clip plane; the hot
  keys V and C keep the controls in step, and Reset restores them.
* Documentation: the REPL example of the Neurosynth tests
  (:ref:`tutorial-neurosynth-tests`) gives the term in the query,
  ``ratios("vision", i, j, k, ...)``, rather than in a
  ``term_of_interest`` relation joined in every rule, with a ``%``
  comment before each line.

* The ``.view`` viewer draws its slices with NiiVue (0.69.0, inlined in
  the page, which still needs no network but a browser with WebGL2): the
  order its overlays are drawn in can be changed with ▲ and ▼, and the
  mouse wheel scrolls through the slices.

* Documentation: Neurosynth's association and uniformity tests written as
  ratios of probabilities (lift, relative risk, and the ratio to the
  brain average), computed in the REPL of the ``neurosynth`` engine, see
  :ref:`tutorial-neurosynth-tests`.

* Documentation: the mutual information of a term and the activation of a
  voxel, a rule over the engine's ``p_term``, ``p_a`` and ``p_a_given_t``,
  ranks the voxels as Neurosynth's association test does, which the lift
  does not: for *prefrontal*, the Dice coefficient of its top 5% of the
  brain with that of Neurosynth's map is 0.83, against 0.51 for the lift.
  See :ref:`neurosynth-mutual-information`.

* ``.view`` of a result with several value columns shows an overlay per
  column, each with its own threshold, colormap, symmetry, opacity and
  visibility, a colour bar and its value at the cursor; ``--fill V`` sets
  the value of the voxels without one (default 0, hidden). An infinite
  value is drawn as the largest finite one.

* The ``neurosynth`` engine has a ``cognitive_task(task_id, name,
  description)`` relation: the tasks of the Cognitive Atlas, downloaded
  from its API (``cognitiveatlas.org/api/v-alpha/task``), with their
  descriptions, e.g.
  ``ans(d) :- cognitive_task(t, "stop signal task", d)``.

* The ``neurosynth`` engine has a ``cortical_mask(i, j, k)`` relation: the
  voxels of the MNI template grid in the cortex, labelled by the
  Harvard–Oxford cortical atlas (maximum probability, 25% threshold),
  resampled by nearest neighbour, for instance to restrict
  ``voxel_given_term(t, i, j, k, p)`` to the cortex.

* Text embeddings work offline: SPECTER2, its tokenizer and its adapters
  are downloaded once into ``{data_dir}/<engine>/embeddings/models/``
  (by the first query or by
  ``python -m neurolang.utils.text_embeddings``) and then loaded from
  there without network.

* The ``.view`` brain viewer of the interactive REPL starts with the
  ``plasma`` colormap and shows the top 5% of the voxels (threshold at the
  95th percentile of the nonzero values) unless ``--cmap`` or
  ``--threshold`` say otherwise.

* ``percentile(v, q)`` aggregation: the ``q``-th percentile of a column,
  ignoring NaN values, to keep for instance the top ``p``% of a column
  with ``v >= percentile(v, 100 - p)``.  The built-in aggregations
  (``count``, ``max``, ``mean``, ``sum``, ``std``, ``percentile``) are now
  also available in :class:`~neurolang.frontend.NeurolangDL`.

* Semantic search over document texts with SPECTER2
  (:mod:`neurolang.utils.text_embeddings`).  Engines declare a
  ``text_embeddings`` section in ``engines.yaml`` to register a function
  ``f(query, doc)`` returning the cosine similarity between a document and
  a free-text query.  The ``neurosynth`` engine provides
  ``query_similarity``, used as
  ``study(s) & (sim == query_similarity('visual word form reading', s))``,
  over PubMed titles and abstracts; see the new gallery example
  :ref:`sphx_glr_auto_examples_plot_text_embeddings_forward_inference.py`.
  Requires ``pip install "neurolang[embeddings]"``.

* Declarative engine registry for the ``neurolang-query`` CLI.
  Engines are defined in YAML and initialised by pluggable Python modules
  under :file:`neurolang/utils/engines/`.  Includes ``--list-engines`` for
  discovery and ``--list-predicates`` showing declarative metadata without
  data download.  (:mod:`neurolang.utils.engine_registry`,
  :mod:`neurolang.utils.engines`)

* The existing NeuroSynth and Destrieux engines were refactored into the
  new declarative system.  Engine init code moved from
  :mod:`neurolang.utils.cli` to :mod:`neurolang.utils.engines.neurosynth.init`
  and :mod:`neurolang.utils.engines.destrieux.init`.

* ``neurolang-query --list-predicates`` no longer requires downloading data;
  predicate metadata is read from the YAML engine config.

* **Datalog init** — engines can declare inline Datalog rules that run after
  the Python init via the ``datalog_init`` YAML field.  Derived predicates can
  be defined without writing Python.

* **CSV/TSV relations** — engines can load tabular data files as extensional
  predicates with the ``relations`` YAML field.  Supported formats:
  ``.csv``, ``.tsv``, ``.csv.gz``, ``.tsv.gz``.
  Relation entries may include an optional ``description`` field for
  predicate metadata shown by ``--list-predicates``.  The ``file`` value
  may be a URL for automatic download.

* **Builtins** — engines can declare ``builtins: [exp, log, startswith]``
  in YAML to register known functions as callable symbols without Python.

* **Downloads** — the ``downloads`` section fetches files from URLs
  (with optional archive extraction) before other init phases.

* **Probabilistic choices** — engines can declare uniform probabilistic
  choices declaratively with ``probabilistic_choice:`` instead of calling
  ``add_uniform_probabilistic_choice_over_set`` in Python.

* **Ontology loading** — the ``ontologies`` section loads OWL/RDF
  ontologies from URLs or local paths.

* **Atlases** — engines can declare ``atlases:`` with nilearn atlas names
  (``destrieux``, ``schaefer``, ``difumo``) and parameters.  Atlas regions
  are loaded as ``ExplicitVBR`` predicates before Datalog init rules.

* **Declarative migration** — ``exp``, ``log``, ``startswith`` symbols
  now come from YAML ``builtins`` rather than ``base.py``; the
  ``selected_study`` choice for the neurosynth engine is now declared
  in the engine YAML instead of Python init code; the ``destrieux``
  predicate is now declared in the YAML ``atlases:`` section.

* **Base symbols** — engines can set ``use_base_symbols: true`` in YAML
  to register common neuroimaging symbols (``agg_count``,
  ``agg_create_region``, ``agg_create_region_overlay``,
  ``principal_direction``, ``region_union``) automatically, without a
  Python init script.  ``region_union`` was moved into
  :func:`~neurolang.utils.engines.base.init_base_engine` as part of this
  change.

* **YAML-only engines** — the ``destrieux`` engine no longer needs a
  ``python_init`` module.  It is now fully declarative, using
  ``use_base_symbols`` and the ``atlases:`` section.  The
  :file:`neurolang/utils/engines/destrieux/init.py` module was removed.

**Bug fixes:**

* The REPL block of :ref:`tutorial-neurosynth-tests` names the rules
  restricted to the cortex with ``_cortex`` (``ratios_cortex``,
  ``selected_voxel_cortex``, …): pasted in the ``neurosynth`` engine,
  which now predefines ``selected_voxel``, ``p_a``, ``p_a_given_t`` and
  ``ratios`` over the brain, it failed on its second definition of
  ``selected_voxel`` and added its rules to those of the others.

* The REPL skips the ``%`` comment lines between statements, e.g. before
  a dot-command of a pasted block or in a piped batch, which it submitted
  as programs that failed to parse; a comment line within a statement of
  several lines no longer ends it.

* Magic sets leave out of a magic rule the atoms sharing no variable with
  its bound arguments, whose product with the magic set was only
  projected out: the Neurosynth tests of :ref:`tutorial-neurosynth-tests`
  with the term in the query, ``ratios("vision", i, j, k, ...)``, take
  13 s instead of 63 s.

* Magic sets pass the values of a stored relation with a constant to the
  rules after it in a query: ``ans(i, j, k, p) :- cognitive_task(_,
  "color-word stroop task", d) & voxel_given_sim(d, i, j, k, p)`` gives
  the description of the task to the rules computing the similarities to
  it, which raised an ``UnboundVariablesError``. A recursive component of
  a program is now a stratum of its own, before the rules depending on
  it, such as an aggregation, which raised a
  ``NoValidChaseClassForStratumException``.

* Magic sets missed the initial magic fact of a constant when the same
  predicate, with the same adornment, was called with another constant in
  a rule rewritten earlier, e.g. ``a(y) :- r(1, y)`` and
  ``b(y) :- r(2, y)``, so one of the calls gave no rows.

* A rule variable defined by an equality over head variables, e.g.
  ``sim`` in ``close(q, s) :- study(s) & (sim == f(q, s)) & (sim > .75)``,
  was rejected as unbound when the rule was defined, although a query
  constant, e.g. ``ans(s) :- close('math', s)``, binds ``q`` through magic
  sets.

* The magic sets SIPS passed bindings from the predicates computed from
  within-language probabilistic queries, e.g. a conditional (MARG, ``//``)
  rule ``p``, which are solved after the magic predicates:
  ``both(i, a, b) :- p('math', i, a) & p('lang', i, b)`` failed.


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
