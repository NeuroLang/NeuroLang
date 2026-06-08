.. _neurolang-engine-registry:

Engine Configuration (engines.yaml)
==================================

Engines are declaratively configured in
``neurolang/utils/engines/engines.yaml``.  Each engine entry specifies:

- ``description`` — human-readable summary shown by ``--list-engines``
- ``requires_mni_mask`` — whether the engine needs an MNI brain mask
- ``builtins`` — Python builtins exposed to the Datalog solver (e.g. ``exp``,
  ``log``, ``startswith``)
- ``python_init`` — dotted path to a Python callable that registers
  extensional data (predicates, regions, atlases) into the engine
- ``atlases`` — declarative atlas loading (nilearn fetchers, region
  predicates)
- ``datalog_init`` — inline Datalog rules that are loaded at engine startup
- ``probabilistic_choice`` — uniform probabilistic choices declared as
  YAML mappings
- ``predicates`` — schema metadata (arity, column names, description) used
  by ``--list-predicates`` and the web UI

Example — the ``neurosynth`` engine::

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
          mentions(t, s) :- term_in_study_tfidf(t, v, s), (v > 0.03)
        probabilistic_choice:
          selected_study:
            source: study
            description: "Uniform probabilistic choice over studies"
        predicates:
          peak_reported:
            arity: 4
            columns: [i, j, k, study_id]
            description: "Reported activation peaks in voxel coordinates"
          study:
            arity: 1
            columns: [study_id]
            description: "All study identifiers"

To add a new engine, append an entry to ``engines.yaml`` and (if needed)
provide a ``python_init`` function that populates the engine's symbol table.
The ``neurolang-query`` CLI and the web server will automatically pick up the
new engine.
