.. _neurolang-engine-registry:

Engine Configuration (engines.yaml)
===================================

Engines are declaratively configured in
``neurolang/utils/engines/engines.yaml`` and built by
:func:`~neurolang.utils.engine_registry.build_engine`.  Each engine entry
may specify the following keys, all of them optional:

- ``description`` — human-readable summary shown by ``--list-engines``
- ``requires_mni_mask`` — whether the engine needs an MNI brain mask (it is
  downloaded, optionally resampled with ``--resolution``, and passed to the
  init phases)
- ``use_base_symbols`` — when ``true`` (and an MNI mask is available),
  registers common neuroimaging symbols (``agg_count``,
  ``agg_create_region``, ``agg_create_region_overlay``,
  ``principal_direction``, ``region_union``)
- ``builtins`` — known Python functions exposed to the Datalog solver; the
  available names are ``exp``, ``log`` and ``startswith``
- ``python_init`` — dotted path of a Python *module* (not of a function)
  which must define ``init_engine(nl, mask, data_dir)``; it registers
  extensional data (predicates, regions, atlases) into the engine
- ``downloads`` — list of ``{url, dest, extract}`` entries fetched into the
  data directory (``extract: true`` uncompresses an archive, a list of
  member names extracts only those members)
- ``templates`` — MNI templates fetched with nilearn or TemplateFlow,
  optionally registered as a ``predicate(i, j, k)`` of non-zero voxels
- ``atlases`` — declarative atlas loading (``destrieux``, ``schaefer``,
  ``difumo``) as region predicates
- ``datalog_init`` — inline Datalog rules that are loaded at engine startup
- ``relations`` — CSV/TSV files (local paths or URLs) loaded as extensional
  predicates
- ``probabilistic_choice`` — uniform probabilistic choices over an existing
  predicate, declared as YAML mappings (``source`` and an optional
  ``description``)
- ``ontologies`` — list of ``{url}`` entries for OWL/RDF ontologies to load
- ``predicates`` — schema metadata (arity, column names, description) used
  by ``--list-predicates`` and the web UI
- ``dimension_type_predicates`` — list of predicate names that SQUALL
  queries may use as dimension-type annotations; they are stripped from the
  rules before solving (defaults to ``probability`` and ``value``)

The init phases run in the order described in :doc:`concepts`: builtins,
base symbols, Python init, downloads, templates, atlases, Datalog init,
relations, probabilistic choices and ontologies.

Example — the ``neurosynth`` engine (abridged)::

    engines:
      neurosynth:
        description: >
          Neurosynth database — forward and reverse inference over
          reported activation peaks, TF-IDF term frequencies, and a
          probabilistic choice over studies.
        requires_mni_mask: true
        use_base_symbols: true
        builtins: [exp, log, startswith]
        python_init: "neurolang.utils.engines.neurosynth.init"
        atlases:
          schaefer:
            predicate_name: schaefer
            n_rois: 100
            resolution_mm: 2
            yeo_networks: 17
        datalog_init: |
          study_with_peaks(S) :- peak_reported(I, J, K, S)
          mentions(s, t) :- term_in_study_tfidf(s, t, v), (v > 0.03)
        predicates:
          peak_reported:
            arity: 4
            columns: [i, j, k, study_id]
            description: "Reported activation peaks in voxel coordinates"
          study:
            arity: 1
            columns: [study_id]
            description: "All study identifiers"
          term_in_study_tfidf:
            arity: 3
            columns: [study_id, term, tfidf]
            description: "Term frequency–inverse document frequency values"

In this engine ``term_in_study_tfidf`` has columns ``(study_id, term,
tfidf)``, and the derived ``mentions(s, t)`` relation keeps the same
order: ``(study_id, term)``.  The ``selected_study`` uniform choice over studies is
registered by the Python init module.

A note on the ``datalog_init`` syntax: the text Datalog parser does not
distinguish variables from predicate names by case, so ``S`` and ``s`` are
both variables, and the final ``.`` of a rule is optional.  Comparisons must
be written between parentheses, as in ``(v > 0.03)``.  See
:doc:`tutorial_datalog` for the full syntax.

To add a new engine, append an entry to ``engines.yaml`` and (if needed)
provide a ``python_init`` module whose ``init_engine`` function populates the
engine's symbol table.  The ``neurolang-query`` CLI and the web server will
automatically pick up the new engine.
