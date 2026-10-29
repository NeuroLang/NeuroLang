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

* ``uv run --extra apps neurolang-repl-serve`` serves the REPL in the browser
  of a checkout on this machine: it builds the page's bundle the first time
  (or with ``--rebuild``) and serves it uncached at
  ``http://localhost:8766/_static/repl/index.html``. It is the command of the
  workspace's root project, whose only package is :file:`tools/serve_repl`.

**Internal:**

* Helpers reused instead of repeated: the conjuncts of a formula and the
  selection of a plan by comparisons in the probabilistic lifting, the
  aggregate types of the magic sets, the pandas reductions table, the
  Euclidean join's index type, the export, the insert, the ``extends`` and
  the missing optional dependency of the apps; docstrings of the apps name
  ``neurolang_apps``.

**Bug fixes:**

* The REPL in the browser could not start: its page and the script that builds
  its bundle (``doc/make_repl_bundle.py``) imported modules from
  ``neurolang.utils`` that the split into packages moved to
  ``neurolang_apps.utils``.

**Performance:**

* A probability over the choice of a study with a peak within a radius of
  each voxel, ``PROB[brain_mask(i, j, k), (euclidean(i, j, k, i1, j1, k1) <
  3), peak_reported(i1, j1, k1, s), selected_study(s)]``, takes 13 s and half
  the memory, instead of 47 s and close to 10 GB, and now runs in a browser
  tab. The natural join with a relation unique on the only common column, as
  the probabilities of a choice are, is a look-up rather than a pandas merge;
  Python's ``sum``, ``max``, ``min`` and ``len`` in an aggregation run as
  pandas reductions on columns of numbers without missing values; groups are
  not sorted; and the Euclidean join keeps no distances it is not asked for,
  numbers its pairs of rows with 32-bit integers and does not copy its result.

**New features:**

* Confidence intervals by resampling the studies.
  ``nl.add_resampling("boot_studies", choice="selected_study")``, or
  ``resamplings`` in an engine's YAML, adds replicates of a probabilistic
  choice, the Bayesian, classical or Poisson bootstrap of its outcomes.
  ``RESAMPLE[boot_studies](pred(x)) = b`` is then ``pred(x)``, any rule
  over ``PROB`` of the choice including the measures computed from the
  probabilities, such as Bayes factors, on replicate ``b``, and
  ``percentile(x, 2.5)`` and ``percentile(x, 97.5)`` over the replicates
  give the interval. Every query sees the same replicates, so that two
  measures are compared on the same ``b``. It rewrites the rules into
  sums over the outcomes of the choice, weighted by each replicate, and
  needs no other solver; on Neurosynth, the interval of the Bayes factor
  of a term for a region takes 15 s. The ``neurosynth`` and
  ``neurostore`` engines have ``boot_studies``, 500 and 200 replicates of
  ``selected_study``, and the query assistant has examples and guidance
  for questions on how certain an answer is. See the tutorial,
  :doc:`tutorial_datalog`, section 10.6.
* The REPL in the browser shows the work of the AI assistant as it happens, in
  a section of the page you can open: the model's calls, the tool calls of
  the ReAct agent, its queries and what they gave back. The code box colours
  the query as the terminal does, in Datalog and in SQUALL. Each result has a
  Download menu: all its rows as CSV or TSV, and its NIfTI image, the file
  ``.save`` writes, for each value column; the brain tab offers the images
  too. In the terminal REPL and in the page, a *Summary* of the query the
  assistant wrote, its explanation and the query read back in English, is
  above the query, apart from it, instead of below it. The *Docs* menu and a hint under the box link the keyword at the
  cursor (``PROB``, ``AGGREGATE``, ``:-``…) to its section of the tutorial;
  the examples and the relations of the engine, and the rules you define,
  are listed, to put them in the box with a click; and *Copy link* shares the
  query in the address (after the ``#``, which no server receives).

* The example LLM configuration has the providers ``albert-deepseek``,
  ``openrouter-gemma`` and ``openrouter-deepseek``: DeepSeek V4 Flash and
  Gemma 4 31B on Albert and on OpenRouter.

* The query assistant can share its questions and answers with a feedback
  server, where people rate them correct, incorrect or unsure and correct
  their query, to train models and grow the banks of examples. It is off
  by default: ``feedback`` in ``llm.yaml`` or
  ``NEUROLANG_LLM_FEEDBACK_URL`` turns it on. In the REPL, ``.rate`` and
  ``.refine`` rate and correct the last answer. ``neurolang-feedback-server``
  (no new dependency) collects them, shows a review page and exports
  them as JSON lines or as a bank of examples. See
  :ref:`assistant-feedback`.

* The query assistant searches the studies by a description or a
  Cognitive Atlas concept: the ``neurosynth`` and ``neurostore`` engines
  share a bank of examples (``coordinate_meta_analysis.yaml``, which banks
  now inherit through ``extends``) with questions answered with
  ``similar_study``, ``p_a_given_similar``, ``uniformity_llm`` and the
  new ``cogat_definition(term, definition)``; ``neurostore.yaml`` adds
  questions on its LLM-extracted concepts and domains.  ``neurosynth``
  gets ``study_title(study, title)``, from its metadata, as ``neurostore``
  had.

* The engine of the REPL in the browser leaves out the precomputed analyses
  (``lift``, ``relative_risk``, ``uniformity``, ``ratios``): building them took
  almost 4 GB, which Safari does not give a tab, and made the page loop on
  "Building the engine". The page builds them with ``?engine=full``.
* The page of the REPL in the browser has a computer, light or dark theme
  switch, says when the browser stops its worker, shows how long a slow step
  lasts, and reads ``bundle.json`` afresh, so that a new deploy is not met by
  the name of the old wheel.
* NeuroLang is split into four packages, in a `uv workspace
  <https://docs.astral.sh/uv/concepts/projects/workspaces/>`_ under
  :file:`packages/`: ``nl-expr`` (the expression engine, imported as
  ``nl_expr``), ``nl-core`` (relational algebra, Datalog and probabilistic
  Datalog, ``nl_core``), ``neurolang`` (the languages, frontends and brain
  regions) and ``neurolang-apps`` (the ``neurolang-query`` command line, its
  interactive mode, the query assistant and the engines,
  ``neurolang_apps``). ``neurolang[apps]`` installs them all; ``neurolang``
  alone is the library without the applications. The modules keep their
  names under ``neurolang``, which are the same modules:
  ``neurolang.datalog`` is ``nl_core.datalog`` and ``python -m
  neurolang.utils.cli`` still runs the command line. The tests of a package
  use only that package and those before it, and CI runs the tests of the
  packages a pull request changes, on every Python version, and those of the
  packages built on them, on Python 3.12: a change to the terminal interface
  no longer runs the tests of the engine. A clone installs with ``uv sync
  --all-packages --all-extras --no-extra embeddings``, or ``uv pip install
  -e ".[apps,dev,doc]"``; see :doc:`contributing`.

* ``neurolang-apps`` depends on ``pyarrow``, which the ``neurostore``
  engine needs to read the parquet files of its releases and which
  NeuroLang never declared.

**Bug fixes:**

* Two aggregations with arguments in one head, such as
  ``percentile(v, 2.5)`` and ``percentile(v, 97.5)``, gave the result of
  the last one for both: their functions shared one global name.

* A query whose head aggregates over probabilities, such as
  ``ans(max(p)) :- p_term(t, p)``, failed with
  ``NoValidChaseClassForStratumException``: it was solved by the
  deterministic chase, which does not compute the probabilities.

* A probability over a predicate of two rules, one of which binds a
  variable of the head only through an atom both share, as
  ``a(s, r) :- region(r), selected_study(s), activates(s, r)`` and
  ``a(s, r) :- region(r), selected_study(s), mentions(s, 't1')``, failed
  with ``NeuroLangTypeException``: the lifted plan was the union of the
  relations of ``activates(s, r)`` and ``mentions(s, 't1')``, without the
  same columns. Such a union is declined, see
  :doc:`probabilistic_lifting`.


v0.1a21 (2026-10-21)
--------------------

**New features:**

* The page of the REPL in the browser is a small interface: a box to ask
  the AI assistant, a box for the Datalog or SQUALL code with its history
  in a dropdown, and the result as a table or in the brain viewer.
* The REPL in the browser has the AI assistant (``.ask``, ``.llm``) with
  the visitor's own API key, given in the page for Anthropic, OpenAI,
  OpenRouter or any OpenAI-compatible server. The key stays in the tab
  (or on the device, if asked), goes only from the page to the provider,
  and the query the assistant writes is put in the input. The LLM clients
  send their requests with the browser's ``XMLHttpRequest``
  (``neurolang.utils.llm.browser``), and queries run inline where there
  are no threads nor processes.
* The coordinate meta-analysis engines (``neurosynth``, ``neurostore``)
  have the peaks in MNI millimetres, ``peak_mni(x, y, z, study)``, the
  Destrieux atlas of the cortex and the Tian atlas of the subcortex beside
  Schaefer, and for each atlas ``{atlas}_activated(label, study)``: the
  studies with a peak within 3 mm of a voxel of a region.  The LLM
  assistant answers their questions with probabilistic query answering
  (an engine's ``assistant_guidance`` goes into its prompt), with
  Destrieux for the classical anatomy and Schaefer and Tian for the
  functional anatomy; the neurosynth example bank is rewritten so, with
  examples of the co-activation of regions and of points.

* The LLM assistant writes the query it accepted with a comment line
  explaining each rule, from one more request to the model (``%`` lines
  in Datalog, which runs as it is; ``comment_rules: false`` turns it
  off).  ``.llm --steps N``, ``.ask --steps N``, ``.llm steps N`` and
  ``--steps N`` on the command line set the most steps of the ReAct
  agent.

* In the REPL, the rules of a submission with a query (``ans`` or
  ``obtain``) are local to it: they are discarded after it, so that two
  queries defining ``active`` do not join their rules.  ``.scope off``
  keeps them in the session, as before.  Comments are shown in a lighter
  grey.

* ``.view`` and ``.save`` (and ``--view``, ``--save``) paint a result on
  the regions of an atlas, ``ans(l, p)``, region by region: a sort of
  region labels names its atlas with ``atlas`` in ``engines.yaml``.
  ``.view`` shows a result of pairs of regions or points with a value,
  such as their co-activation, as a network drawn by NiiVue's connectome
  view, over the brain.

* The page of the REPL in the browser has the logo, a header with links to
  the documentation, the tutorial and the source, the command it runs to
  copy, and a status that shows when it is busy; the choice of the
  NeuroLang version is folded away.

* The REPL in the browser shows a result as a table to sort by clicking
  a column, to filter (``stroop`` keeps the rows with the word,
  ``name:stroop`` those whose column ``name`` has it), to page through, to
  copy and to download as CSV, instead of the text table of the terminal;
  ``Ctrl+R`` searches the history of the inputs, which lasts between
  visits, as in a shell (``Ctrl+R`` again for an older match, ``Ctrl+S``
  for a newer one, ``Enter`` to take it, ``Esc`` to cancel), and Up and
  Down browse it.  :class:`neurolang.utils.tui.web.WebRepl` returns a
  result's rows as data (``Reply.tables``).
  
* An LLM query assistant writes a Datalog or SQUALL query from a
  question in natural language: ``.ask QUESTION`` in the interactive
  mode, which puts the query in the next prompt (``.llm QUESTION`` is
  the shortcut for the ReAct agent writing Datalog), and
  ``neurolang-query --ask QUESTION`` on the command line, where ``--run``
  also runs the query and ``-v`` and ``-vv`` show the reasoning and tool
  calls of its steps, and the messages sent to the model.  Two
  strategies are offered: in-context learning, whose few-shot examples
  are the closest to the question (BM25), with validation and repair on
  the engine, and a ReAct agent that explores the engine with tools
  (``search_values``, ``run_query``, ``check_query``…), by native tool
  calls or a text protocol.  The providers, OpenRouter, Albert,
  Anthropic, any OpenAI- or Anthropic-compatible server or a local
  command, are set in a YAML file, their keys read from the environment
  (``neurolang/utils/llm/llm.example.yaml``).  The queries are shown on
  their lines, highlighted.  It needs the ``llm`` extra and is described
  in :doc:`query_assistant`.  The benchmark that measures it is in
  `neurolang-text2query-bench
  <https://github.com/NeuroLang/neurolang-text2query-bench>`_.
  
* ``neurolang-query --save PATH`` writes the result as CSV or TSV, or as
  a NIfTI image on the grid of the engine, and ``--view`` shows its image
  in the viewer of the REPL, as ``.save`` and ``.view`` do (they need
  the ``tui`` extra).

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

* NeuroLang is layered as the packages it is to be split into: the
  expression engine (expressions, walkers, types, logic, config), the core
  (relational algebra and its sets, Datalog, probabilistic), the language
  (parsers, frontends, regions) and the applications (command line,
  terminal interface, engines).  Each layer imports only those below it,
  which ``.importlinter`` checks in CI: ``import neurolang.expressions``
  needs neither pandas nor the relational algebra sets, and the core
  needs neither the frontends nor lark, nibabel or nilearn.
  ``import neurolang`` imports the frontends when one is first used.
  The FOL to Datalog translation moved from ``neurolang.logic.horn_clauses``
  to :mod:`neurolang.datalog.horn_clauses`, and the Datalog pretty-printer
  to :mod:`neurolang.datalog.pretty_printer`
  (``neurolang.frontend.datalog.pretty_printer`` still imports it).
  Modules that choose the classes of the relational algebra backend when
  imported register with ``config.reload_on_backend_switch``, rather than
  being listed in the configuration.  The tests of the core that ran
  queries through a frontend moved to :file:`neurolang/frontend/tests`.

**Bug fixes:**

* The REPL block of :ref:`tutorial-neurosynth-tests` names the rules
  restricted to the cortex with ``_cortex`` (``ratios_cortex``,
  ``selected_voxel_cortex``, …): pasted in the ``neurosynth`` engine,
  which now predefines ``selected_voxel``, ``p_a``, ``p_a_given_t`` and
  ``ratios`` over the brain, it failed on its second definition of
  ``selected_voxel`` and added its rules to those of the others.
  
* The badges of the README: the coverage badge, deployed to ``gh-pages``,
  was deleted by each deployment of the benchmarks' dashboard there (and
  deleted it in turn): it has its own branch, ``badges``. The Python badge
  reads the versions from ``pyproject.toml``, as the release on PyPI
  declares none, and the static "code style: flake8" badge, which nothing
  checked, gives way to those of the documentation and spelling workflows.

* A Euclidean distance bound to a point given by numbers,
  ``peak_mni(x, y, z, s), (euclidean(x, y, z, -24, -4, -20) <= 3)``, was
  computed for every row (14 s on the peaks of neurosynth), and the
  reserved ``EUCLIDEAN`` raised an error: the point is now ranged by a
  stored relation of its own, so the points within the distance are
  precomputed as for two relations (well under a second).

* A probabilistic query with a variable occurring only in deterministic
  relations, ``q(x, PROB) :- r(x, y) & selected(x)``, gave the sum of the
  probabilities over the values of ``y`` rather than their probability,
  e.g. 1.0 rather than 0.5 for an ``x`` of two rows of ``r``: those rows
  are the same event.  The lifted plan of the deterministic subquery
  ``∃y r(x, y)`` keeps its tuples with a probability of 1, and the small
  dichotomy theorem solver leaves such a query to the lifted solver.
  
* A probabilistic query no longer fails with ``AttributeError: type object
  'Unknown' has no attribute '__origin__'`` when the program has a
  query-based probabilistic choice, such as ``selected_voxel(i, j, k) ^ (1 /
  m) :- ...``, that the query does not use, or whose rule's body is empty:
  the choice is then over no tuple, as a query-based probabilistic fact
  over no tuple already was.

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

**Removed:**

* The legacy web application, ``neurolang/utils/server`` (2.3 MB, with
  ``papaya.js``), and the hatch build hook that built its frontend with
  npm, ``hatch_build.py``; the wheel no longer ships it.
* ``siibra`` and ``seaborn`` are no longer dependencies: NeuroLang never
  imports them.  The example
  :ref:`sphx_glr_auto_examples_plot_squall_bayes_factor_decoding.py` needs
  ``pip install siibra``.
* Unreferenced files: ``unused/neurosynth/cogat.v2.owl`` (1.4 MB), the
  pictures of an old Datalog grammar in
  ``neurolang/frontend/datalog/doc/`` (0.7 MB) and ``MANIFEST.in``.


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
