.. _neurolang-engine-registry:

Engine Configuration (engines.yaml)
===================================

Engines are declaratively configured in
``packages/neurolang_apps/neurolang_apps/utils/engines/engines.yaml`` and built by
:func:`~neurolang.utils.engine_registry.build_engine`.  Each engine entry
may specify the following keys, all of them optional:

- ``extends`` — name (or list of names) of engines whose configuration is
  inherited, see `Sharing configuration between engines`_
- ``abstract`` — when ``true`` the entry only serves as a base for other
  engines: it is neither listed nor buildable
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
  ``tian``, ``difumo``) as region predicates ``name(label, region)``.  A
  deterministic atlas with ``activated: {predicate_name, radius_mm,
  peaks}`` also registers ``predicate_name(label, study)``: the studies
  with a peak within ``radius_mm`` (3 by default) of a voxel of the
  region, the peaks being the millimetre coordinates of the relation
  ``peaks`` (``x, y, z, study``, ``peak_mni`` by default).  The distances
  are measured in millimetres through the atlas's affine, so its grid need
  not be the template's, and the pairs are computed once, when the engine
  is built.  ``tian`` is the Melbourne subcortex atlas (Tian et al.,
  2020, Nat Neurosci, doi:10.1038/s41593-020-00711-6, whose licence asks
  for this citation), 3T, ``scale`` 1 to 4 (16, 32, 50 or 54 regions),
  downloaded from the authors' repository
- ``datalog_init`` — inline Datalog rules that are loaded at engine startup
- ``relations`` — CSV/TSV files (local paths or URLs) loaded as extensional
  predicates
- ``probabilistic_choice`` — uniform probabilistic choices over an existing
  predicate, declared as YAML mappings (``source`` and an optional
  ``description``)
- ``ontologies`` — list of OWL/RDF ontologies to load: ``url`` (a URL or
  a local path) or ``urls`` (several files loaded together), and the
  ``format``, ``individuals`` and ``names`` options of
  :meth:`~neurolang.frontend.NeurolangPDL.load_ontology`
- ``text_embeddings`` — semantic search over document texts, registered as
  a similarity function (see :ref:`text-embeddings-section` below)
- ``predicates`` — schema metadata (arity, column names, description) used
  by ``--list-predicates``, the interactive REPL and the web UI; see
  :ref:`predicate-arguments` below
- ``hidden_predicates`` — predicates the interactive REPL does not offer
  in SQUALL, because their argument order breaks sentences (they remain
  usable when typed)
- ``dimension_type_predicates`` — list of predicate names that SQUALL
  queries may use as dimension-type annotations; they are stripped from the
  rules before solving (defaults to ``probability`` and ``value``)
- ``assistant_guidance`` — how the LLM query assistant (``.ask``,
  ``neurolang-query --ask``) is to answer questions on the engine: added
  to its system prompt.  The coordinate meta-analysis engines ask for
  probabilistic answers over ``selected_study``, activation as a peak
  within 3 mm of a point in MNI millimetres, the Destrieux atlas for the
  classical anatomy of the cortex and the Schaefer and Tian atlases for
  the functional anatomy
- ``examples`` — example queries, as lists under ``datalog`` and
  ``squall``; the interactive REPL suggests the first one of its language
  at startup and lists them all in ``.help`` (by default it suggests
  obtaining the first declared predicate)

The init phases run in the order described in :doc:`concepts`: builtins,
base symbols, Python init, downloads, templates, atlases, Datalog init,
relations, text embeddings, probabilistic choices and ontologies.

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
          weighted_mention(s, t, w) :- term_in_study_tfidf(s, t, w)
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
order: ``(study_id, term)``.  ``weighted_mention(study_id, term, tfidf)``
holds the same rows as ``term_in_study_tfidf`` under a name that reads
better in SQUALL, where an n-ary relation is a noun followed by a tuple of
variables, in the order of its columns::

    obtain every Weighted_mention (?s; ?t; ?w)
        such that ?t is 'pain' and ?w is greater than 0.1.

SQUALL verbs relate two arguments (``a Study mentions a Term``), so a
relation of three columns cannot be a verb.  The ``selected_study`` uniform choice over studies is
registered by the Python init module.

.. _predicate-arguments:

Predicate arguments and sorts
-----------------------------

A predicate may declare its ``arguments``: the entities it relates, in
column order, each of a ``sort`` spanning one or more ``columns`` of a
``type``.  The arity of a predicate, in the sense of the entities it
relates, is the number of its arguments, not of its columns: a voxel
``(i, j, k)`` is one argument, so ``reports(study_id, i, j, k)`` relates
a study and a voxel::

    reports:
      description: "Studies reporting an activation peak at a voxel"
      arguments:
        - {sort: study, columns: [study_id], type: int}
        - {sort: voxel, columns: [i, j, k], type: int}

``columns`` and ``arity`` (the number of columns) are then derived from
the arguments, and ``entities`` holds the number of arguments; declared
``columns`` or ``arity`` must agree with them.  A sort spans the same
columns everywhere.  ``probability: true`` marks a probabilistic symbol,
whose probability column is not an argument.
:func:`~neurolang.utils.engine_registry.get_signatures` returns the
arguments of each predicate.

A sort of points may declare its ``space``: ``space: voxel`` marks the
indices ``(i, j, k)`` of a grid, and ``space: mm`` coordinates
``(x, y, z)`` in millimetres in the space of the engine's template, the
MNI image of an engine with ``requires_mni_mask`` (``nl.template`` on the
engine :func:`~neurolang.utils.engine_registry.build_engine` returns).
The grid of a sort of voxels is that of the template unless the sort
names another with ``grid``: that of an atlas of the engine, by its key
under ``atlases``.  The Schaefer atlas of ``neurosynth`` has its own
2 mm grid, which ``labels`` and ``schaefer_voxels`` index::

    atlases:
      schaefer: {n_rois: 100, resolution_mm: 2, yeo_networks: 17}
    sorts:
      voxel: {columns: [i, j, k], type: int, space: voxel}
      mni: {columns: [x, y, z], type: float, space: mm}
      atlas_voxel:
        columns: [i, j, k]
        type: int
        space: voxel
        grid: schaefer

A grid is ``template`` or the key of an atlas of the engine, and only a
sort of ``space: voxel`` names one; the configuration is rejected
otherwise.  The engine holds the image of each grid in ``nl.grids``, by
name: the template's, as ``template``, and each atlas's, registered
when the atlas is loaded.

The REPL's ``.save out.nii.gz`` (and ``.view``) writes a result on a
grid, with its shape and affine.  Each column of the result has the sort
of the argument of the predicate its variable is in, in the last query:
``ans(x, y, z) :- voxel(x, y, z)`` holds voxel indices of the template
grid, whatever the names of its columns, ``ans(l, i, j, k) :- labels(l,
i, j, k)`` indices of the Schaefer atlas grid.  The sorts follow from
the predicates declaring their ``arguments``, then from the rules defined
from them, in Datalog as in SQUALL (``obtain every Voxel in 3D``).  The
columns of a ``space: voxel`` sort are written as they are on its grid,
those of an ``mm`` sort converted to the nearest voxel of the template.
A column of unknown sort, such as one of a relation loaded with
``.load``, is of the sort it is named as.  ``--voxel I J K`` and ``--mm
X Y Z`` give the columns, on the template's grid (see ``.help save``).

A sort of region labels may name, with ``atlas``, the atlas whose
regions it labels, by its key under ``atlases``; it has one column and
no ``space``::

    destrieux_label:
      columns: [label]
      type: str
      atlas: destrieux

A result with a column of such a sort is then a result on the regions of
the atlas: ``.save out.nii.gz`` and ``.view`` paint each region, on the
atlas's grid, with the value of its row (``ans(l, p) :- PROB[...]``),
and ``.view`` shows a result with two such columns, pairs of regions and
a value, as a network over the brain, a node at the centre of each
region and an edge per pair (see ``.help view``).  The engine keeps the
values labelling each region in the image of the atlas, by name, in
``nl.atlas_labels`` (see
:func:`~neurolang.utils.engine_registry.atlas_label_values`).

The interactive REPL completes SQUALL from the sorts: after ``every`` it
offers the nouns, capitalised; after ``every Study that`` the verbs whose
first entity is a study (``mentions``, ``reports``); after ``every Term
that`` the object clause ``a Study mentions``; after a noun of several
columns ``in 3D`` and the tuple ``(?i; ?j; ?k)``, and ``with Probability``
after a relation defined ``with inferred probability``.  In Datalog,
the columns of an entity are completed together (``i, j, k``).  The
inverse verb form ``~verb`` is not offered, as SQUALL does not invert
its arguments in a relative clause; ``every Term that a Study mentions``
reads the same.

A note on the ``datalog_init`` syntax: the text Datalog parser does not
distinguish variables from predicate names by case, so ``S`` and ``s`` are
both variables, and the final ``.`` of a rule is optional.  Comparisons must
be written between parentheses, as in ``(v > 0.03)``.  See
:doc:`tutorial_datalog` for the full syntax.

.. _text-embeddings-section:

Text embeddings
---------------

A ``text_embeddings`` entry embeds the title and abstract of every
document of a relation with SPECTER2 and registers a function
``function(query, doc)`` returning the cosine similarity between a
free-text query and a document, ``nan`` for an id that is not in the
relation (see :doc:`tutorial_text_embeddings`)::

    text_embeddings:
      study_abstracts:
        ids_from: study                 # first column holds the document ids
        source: pubmed                  # ids are PubMed ids
        model: allenai/specter2_base
        document_adapter: allenai/specter2
        query_adapter: allenai/specter2_adhoc_query
        function: query_similarity
        description: "Similarity between a query and a study abstract"

``source`` is ``pubmed`` (titles and abstracts fetched from the NCBI
E-utilities), a table of texts, local or remote::

        source:
          file: https://example.org/documents.tsv
          id_column: id
          text_columns: [title, abstract]   # first column is the title

or texts the engine already holds as predicates ``p(id, text)``, which is
how the ``neurostore`` engine embeds the abstracts that come with its
database, without any download::

        source:
          predicate: study_abstract
          title_predicate: study_title      # optional

``model``, ``document_adapter`` and ``query_adapter`` default to the values
above.

``encoder: albert-qwen`` embeds with Qwen3-VL-Embedding-8B through the
`Albert API <https://albert.api.etalab.gouv.fr>`_ instead, without any
local model; it needs an API key in ``ALBERT_API_KEY``.  Its settings, all
optional, go in an ``albert`` mapping, and the SPECTER2 keys are then
ignored::

        encoder: albert-qwen
        albert:
          model: qwen3-vl-embedding-8b
          query_instruction: "Given a description of a cognitive process,
            task or topic, retrieve abstracts of neuroimaging studies about it."

The ``NEUROLANG_TEXT_ENCODER`` environment variable (``specter2`` or
``albert-qwen``) overrides ``encoder`` in every engine, e.g. to use Albert
with the built-in ``neurosynth`` and ``neurostore`` engines.  Each encoder
has its own cache, ``{name}.npz`` for SPECTER2 and ``{name}.albert-qwen.npz``
for Albert.  The engine build only reads the ids.  Texts are fetched and
embedded on the first query that uses the function, and cached under
``{data_dir}/{engine}/embeddings/``.  The embeddings can be computed ahead
of time with ``python -m neurolang.utils.text_embeddings --engine
neurosynth --data-dir neurolang_data``.

Sharing configuration between engines
-------------------------------------

Settings common to several engines live in an ``abstract`` engine that the
concrete ones pull in with ``extends``.  The built-in ``neurosynth`` and
``neurostore`` engines both extend ``coordinate_meta_analysis``, which
provides the MNI mask, the base symbols, the Schaefer atlas, the sorts, the
predicates and their Datalog rules, and the text embeddings; each engine
adds its ``python_init`` and what is specific to its database::

    engines:
      coordinate_meta_analysis:
        abstract: true
        requires_mni_mask: true
        use_base_symbols: true
        datalog_init: |
          study_with_peaks(s) :- peak_reported(i, j, k, s).
        text_embeddings:
          study_abstracts: {ids_from: study, source: pubmed,
                            function: query_similarity}

      neurostore:
        extends: coordinate_meta_analysis
        python_init: "neurolang.utils.engines.neurostore.init"
        text_embeddings:
          study_abstracts:
            source: {predicate: study_abstract}   # replaces ``pubmed``

With several bases (``extends: [a, b]``) later ones override earlier ones,
and the engine's own entries override all of them.  Merging works per key:

- ``datalog_init`` texts are concatenated, inherited rules first;
- ``builtins``, ``downloads``, ``ontologies`` and
  ``dimension_type_predicates`` are concatenated without duplicates;
- mappings (``atlases``, ``templates``, ``relations``, ``sorts``,
  ``predicates``, ``text_embeddings``, ...) are merged recursively, so an
  engine can change one parameter of an inherited entry; setting an entry
  to ``null`` removes the inherited one;
- every other value, lists included (``python_init``, ``description``,
  ``hidden_predicates``, ...), is replaced.

Unknown bases and cyclic ``extends`` chains raise ``ValueError``.

The ``neurostore`` engine
-------------------------

``neurostore`` loads the NeuroStore database (the one behind Neurosynth
Compose, which NiMARE fetches with ``fetch_neurostore``).  It downloads a
release archive from ``neurostore.org``, verifies its checksum and caches it
under ``<data_dir>/neurostore/<version>``; ``NEUROLANG_NEUROSTORE_RELEASE``
selects the release (``latest``, ``nightly`` or e.g. ``2026-10``).  It has
all the predicates of ``neurosynth``, with the same names and columns.
Study ids are NeuroStore's alphanumeric strings.  NeuroStore ships no term
table, so ``term_in_study_tfidf`` is computed from the study abstracts with
scikit-learn (words and sequences of two or three, so that terms such as
``working memory`` exist; English stop words, terms in at least 30 studies,
values above ``1e-3``) and is not comparable with Neurosynth's
curated-vocabulary values.  Peaks in spaces other than MNI or Talairach are
dropped.  It adds ``study_title(study_id, title)``,
``study_abstract(study_id, abstract)``, and
``study_concept(study_id, concept)`` / ``study_domain(study_id, domain)``,
the cognitive concepts and domains that an LLM extracted from the tasks of
each study.

Like ``neurosynth``, it defines the ``lift``, ``relative_risk``,
``uniformity`` and ``ratios`` of a term (:doc:`tutorial_neurosynth_tests`).
With SPECTER2 over its abstracts it also defines them for a free text,
``uniformity_llm('a description', i, j, k, u)`` and the like.

Its SPECTER2 ``query_similarity(query, study)`` embeds those titles and
abstracts; precompute the embeddings with ``python -m
neurolang.utils.text_embeddings --engine neurostore --data-dir
neurolang_data``.

The Cognitive Atlas
-------------------

Both ``neurosynth`` and ``neurostore`` load the Cognitive Atlas ontology
with ``ontologies`` (see :doc:`tutorial_cognitive_atlas`), next to the
``cognitive_task`` relation: its classes, labels, definitions and axioms are
facts and constraints that queries use with the engine's data, in the
open-world manner of NeuroLang.  Because ``ontologies`` is loaded before
``datalog_init``, the rules of an engine can use the ontology's predicates,
as ``cogat_subclass_of`` and ``cogat_measures`` do.

To add a new engine, append an entry to ``engines.yaml`` and (if needed)
provide a ``python_init`` module whose ``init_engine`` function populates the
engine's symbol table.  The ``neurolang-query`` CLI and the web server will
automatically pick up the new engine.
