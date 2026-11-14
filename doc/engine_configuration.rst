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
  extensional data (predicates, regions, atlases) into the engine.  It may
  be a list of modules, whose ``init_engine`` run in order; an engine
  extending others runs theirs first (the lists are concatenated, parent
  first)
- ``solver_mixins`` — classes, as ``"module:Class"`` strings, put before
  the solver of the engine, e.g. to add Datalog commands ``.name(...)``
  (the ``tracts`` engine, below)
- ``python_init_options`` — keyword arguments given to every
  ``init_engine`` of ``python_init`` (one that has no use for the options
  of another takes them with ``**options``). The ``neurosynth`` and
  ``neurostore`` engines take ``unknown_space``, set once in
  ``coordinate_meta_analysis``: the peaks whose space is neither MNI nor
  Talairach (Neurosynth's ``UNKNOWN``, 10.6% of its peaks; NeuroStore's
  other or missing spaces) are taken as MNI (``mni``, the default, as
  Neurosynth and NiMARE do), converted as Talairach (``tal``) or left out
  (``drop``); only the Talairach peaks are converted to MNI otherwise
- ``downloads`` — list of ``{url, dest, extract, name, md5}`` entries
  fetched into the data directory (``extract: true`` uncompresses an
  archive, a list of member names extracts only those members; ``name`` is
  the file's name when the URL has none, ``md5`` its checksum)
- ``tractogram`` — ``{path, subsample, step, seed}``, for an engine whose
  ``datalog_init`` loads the whole-brain tractogram file ``path`` (relative
  to the data directory; usually one of its ``downloads``) with
  ``.load_tractogram(id, path)``: how the bundle of the browser's REPL
  (``doc/make_repl_bundle.py``) reduces it to ship it.  It replaces the file
  in the data it ships by its subsample, at the same path and in the same
  format (``.tck`` or ``.trk``), ``subsample`` streamlines drawn at random
  with the seed ``seed`` (default 0), each thinned to every ``step``-th point
  (default 1) and its last, so that the engine's program is the same there.
  The engine itself loads the file whole, wherever it runs; the entry is read
  by the bundle only (``neurosynth_iit``: 15,000 of the 37,980 streamlines of
  the IIT Human Brain Atlas v3's ``IIT3mean.tck``, ``step: 3``)
- ``templates`` — MNI templates fetched with nilearn or TemplateFlow,
  optionally registered as a ``predicate(i, j, k)`` of non-zero voxels
- ``atlases`` — declarative atlas loading (``destrieux``, ``schaefer``,
  ``tian``, ``difumo``) as region predicates ``name(label, region)``.  A
  deterministic atlas is a :class:`~neurolang.atlases.Atlas`, fetched and
  read as ``.load_atlas(name, kind="destrieux")`` (``"schaefer"``,
  ``"tian"``) reads it, its label names normalised (``-`` and ``_`` are
  spaces), and recorded in the engine's ``atlases`` by its key, where
  ``.index_tracts`` of the ``tracts`` engine finds it (the probabilistic
  DiFuMo is thresholded into regions).  A deterministic atlas with ``activated: {predicate_name, radius_mm,
  peaks}`` also registers ``predicate_name(label, study)``: the studies
  with a peak within ``radius_mm`` (by default the engine's
  ``kernel_radius_mm``, else 3) of a voxel of the
  region, the peaks being the millimetre coordinates of the relation
  ``peaks`` (``x, y, z, study``, ``peak_mni`` by default).  The distances
  are measured in millimetres through the atlas's affine, so its grid need
  not be the template's, and the pairs are kept in the data directory once
  found (see :ref:`engine-cache`).  With ``points: {predicate_name}`` a
  deterministic atlas also registers ``predicate_name(x, y, z, label)``:
  the centre of each voxel of each region, in millimetres through the
  atlas's affine, a stored relation, so that a rule bounding the distance
  of these points to those of another stored relation,
  ``EUCLIDEAN(x, y, z, x2, y2, z2) <= r``, finds the pairs with a k-d
  tree instead of comparing every pair; the engines extending
  ``mni_atlases`` have ``schaefer_points``, ``destrieux_points`` and
  ``tian_points`` (236,000 rows, 13 MB).
  ``tian`` is the Melbourne subcortex atlas (Tian et al.,
  2020, Nat Neurosci, doi:10.1038/s41593-020-00711-6, whose licence asks
  for this citation), 3T, ``scale`` 1 to 4 (16, 32, 50 or 54 regions),
  downloaded from the authors' repository
- ``euclidean_join`` — settings of the Euclidean join that finds the pairs
  of points of a ``EUCLIDEAN(...) < r`` bound that differ from the profile
  in use (``kdtree_chunk_pairs: 500000``): the thresholds, the sizes of the
  chunks and the other values of
  ``neurolang/utils/euclidean_join.yaml``, whose profiles (``default``,
  and ``memory``, that of the interactive REPL) hold the others; see
  :ref:`euclidean-join-settings`.  The mappings of an engine and of the
  engines it extends are merged
- ``datalog_init`` — inline Datalog rules that are loaded at engine startup,
  run from the engine's data directory: the paths of its commands
  (``.load_tractogram``, ``.load_atlas``...) are relative to it, where
  ``downloads`` puts the files.  ``{kernel_radius_voxels}`` in them is
  ``kernel_radius_mm`` in voxels of the engine's grid
- ``kernel_radius_mm`` — the radius, in millimetres, of the sphere around a
  reported peak that a study activates, the same at every resolution: the
  bound of ``activated`` (``{kernel_radius_voxels}`` in ``datalog_init``,
  3 voxels on the 2 mm grid, 6 on the 1 mm one), the ``radius_mm`` of the
  ``activated`` atlases that do not set theirs, and the number written
  for ``{kernel_radius_mm}`` in the texts of the engine (descriptions,
  ``assistant_guidance``, examples).  The coordinate meta-analysis
  engines take 6 mm, the radius of the spheres of Neurosynth's activation
  maps (its software's default, ``r=6``; see
  :doc:`tutorial_neurosynth_tests`); ALE instead blurs each peak with a
  Gaussian whose width depends on the number of subjects of the study
  (Eickhoff et al., 2009)
- ``relations`` — CSV/TSV files (local paths or URLs) loaded as extensional
  predicates
- ``probabilistic_choice`` — uniform probabilistic choices over an existing
  predicate, declared as YAML mappings (``source`` and an optional
  ``description``)
- ``resamplings`` — replicates of a probabilistic choice for confidence
  intervals, ``name: {choice: selected_study, n: 500, scheme: bayesian,
  seed: 0}``, queried with ``RESAMPLE[name](pred(x)) = b`` (see the
  tutorial :doc:`tutorial_datalog`, section 10.7)
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
  probabilistic answers over ``selected_study`` (counts for "how
  many"), activation as a peak within ``kernel_radius_mm`` of a point in MNI
  millimetres, the Destrieux atlas for the classical anatomy of the
  cortex and the Schaefer and Tian atlases for the functional anatomy
- ``examples`` — example queries, as lists under ``datalog`` and
  ``squall``; the interactive REPL suggests the first one of its language
  at startup and lists them all in ``.help`` (by default it suggests
  obtaining the first declared predicate)
- ``questions`` — questions in English that the REPL in the browser lists
  first, a click putting one in the box of the query assistant; each is
  one of the questions of ``assistant_examples``, of which the tests check
  the queries.  The query examples are folded under them
- ``assistant_examples`` — the bank of questions with their ``datalog``
  or ``squall`` queries from which the LLM query assistant takes its
  few-shot examples (see :doc:`query_assistant`)

The long entries, ``examples``, ``assistant_examples`` and
``assistant_guidance``, are kept out of ``engines.yaml``, in files of the
engine's directory next to it (see `Files of an engine`_).

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
i, j, k)`` indices of the Schaefer atlas grid.  With a template, each
atlas is resampled to it (nearest neighbour) when it is loaded, so the
voxel (i, j, k) of an atlas is that of ``voxel`` and of the peaks.  The
sorts follow from
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
region and an edge per pair (see ``.help view``).  The values labelling
each region in the image of the atlas, by name, are those of the atlas
the engine records under that key, ``nl.atlases`` (see
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
how the meta-analysis engines embed the titles and abstracts of their
studies, ``study_title`` and ``study_abstract`` (see *The publications of
the studies*)::

        source:
          predicate: study_abstract
          title_predicate: study_title      # optional

``model``, ``document_adapter`` and ``query_adapter`` default to the values
above.  The engine build only reads the ids.  Texts are fetched and
embedded on the first query that uses the function, and cached under
``{data_dir}/{engine}/embeddings/``.  An engine whose entry is that of an
engine it extends shares its caches: ``{engine}`` is the most basic
non-abstract engine of its ``extends`` chain with the same resolved entry
(``neurosynth`` for ``neurosynth_iit``, ``neurostore`` for
``neurostore_iit``; see ``engine_registry.embeddings_owner``).  The engine
keeps embeddings in its own directory when the shared ones are of other
documents (its study set differs), or when only its own directory has
them.  The embeddings can be computed ahead
of time with ``python -m neurolang.utils.text_embeddings --engine
neurosynth --data-dir neurolang_data``.

.. _engine-cache:

Starting an engine without finding its tables again
---------------------------------------------------

Most of the build of the ``neurosynth`` engine, 58 s natively and 112 s in a
browser, finds what never changes for the same data: the studies activating
the regions of the atlases (48 s natively), and, from nilearn, the MNI mask
resampled to the grid of the engine, the voxels of the brain and of the cortex
on it, and where the atlases are, for which nilearn and scikit-learn are
imported, which takes more than a second and several in a browser.
:func:`~neurolang_apps.utils.engine_registry.build_engine` keeps all of them in
``DATA_DIR/ENGINE/engine_cache``, in files named by their kind and by what they
depend on, and reads them when they are there. Then nilearn is not imported at
all, the build takes 6 s natively and 8 s in a browser, and the page starts in
23 s instead of 127 s. A change of the atlas, the peaks, the radius or the
resolution makes other files; a file that is missing, damaged or cannot be
written means only that the table is found again. The MNI mask is named by
the resolution alone, so remove the files (``--clear``) if the template
changes.

The files hold arrays, an image and text, nothing of Python or of a version of
numpy or pandas, so the terminal REPL and the REPL in the browser read the same
ones, whichever implementation of Python wrote them. The engine itself cannot
be kept: its expressions cannot be pickled, and a snapshot of the memory of
Pyodide is 780 MB (270 MB compressed).

``python -m neurolang_apps.utils.engine_cache ENGINE -d DATA_DIR -r MM`` builds
the engine at the resolution it will be used with, so that the tables are
kept; ``--list`` lists them and ``--clear`` removes them;
``neurolang-query --no-engine-cache`` builds the engine without reading or
writing them. The build of the browser's bundle (``doc/make_repl_bundle.py``,
run by the documentation workflow) keeps them in the data zip, 3.6 MB, builds
the engine from them and without them to check that the trimmed data suffices
for both, and stops if it kept none.

Sharing configuration between engines
-------------------------------------

Settings common to several engines live in an ``abstract`` engine that the
concrete ones pull in with ``extends``.  The built-in ``neurosynth`` and
``neurostore`` engines both extend ``coordinate_meta_analysis``, which
provides the MNI mask, the base symbols, the sorts, the predicates and
their Datalog rules, and the text embeddings; it extends ``mni_atlases``,
the atlases Destrieux, Schaefer and Tian with the sorts, predicates and
rules of their labels and regions, and adds to each atlas the studies that
activate its regions (``activated``), which need the peaks.  The ``tracts``
engine extends ``mni_atlases`` too, and so has the same atlases, loaded the
same way.  Each engine adds its ``python_init`` and what is specific to its
database::

    engines:
      mni_atlases:
        abstract: true
        atlases:
          destrieux:
            predicate_name: destrieux
            points: {predicate_name: destrieux_points}
          tian:
            predicate_name: tian
            scale: 2
            points: {predicate_name: tian_points}

      coordinate_meta_analysis:
        abstract: true
        extends: mni_atlases
        atlases:      # merged into those of mni_atlases, atlas by atlas
          destrieux:
            activated: {predicate_name: destrieux_activated}
          tian:
            activated: {predicate_name: tian_activated}
        kernel_radius_mm: 6
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
- ``builtins``, ``downloads``, ``ontologies``,
  ``dimension_type_predicates``, ``assistant_examples``,
  ``hidden_predicates``, ``solver_mixins`` and ``python_init`` (a single
  module is a list of one) are concatenated without duplicates;
- mappings (``atlases``, ``templates``, ``relations``, ``sorts``,
  ``predicates``, ``text_embeddings``, ...) are merged recursively by
  OmegaConf, so an engine can change one parameter of an inherited entry;
  setting an entry to ``null`` removes the inherited one (a list cannot
  replace a mapping, nor a mapping a list);
- every other value is replaced (``description``, ``requires_mni_mask``,
  ...).

A base reached by two paths (``extends: [a, b]`` where both ``a`` and
``b`` extend ``c``) is merged once, where it is first reached: its rules
are not defined twice.  Unknown bases and cyclic ``extends`` chains raise
``ValueError``.

Files of an engine
------------------

Every file ``<engine>/*.yaml`` in the directory of the built-in
``engines.yaml`` is merged into the entry of its engine, in the order of
the file names, with OmegaConf (``OmegaConf.merge``, which merges
mappings and replaces lists: a key is best set in one file only).  The
built-in engines keep there

- ``<engine>/examples.yaml``: the queries the REPL shows (``examples``)
  and the questions of the query assistant (``assistant_examples``);
- ``<engine>/guidance.yaml``: the ``assistant_guidance``.

``coordinate_meta_analysis/`` holds the examples and the guidance shared by
``neurosynth`` and ``neurostore``, and ``neurostore/examples.yaml`` adds the
questions on NeuroStore's own predicates: an engine inherits
``assistant_examples`` as it inherits ``builtins``, those of its bases
first.  The OmegaConf interpolation ``${engines.<engine>.<key>}`` can refer
to any value of the composed configuration.  An engines file given to
the functions of :mod:`~neurolang_apps.utils.engine_registry` as
``yaml_path`` is read alone.

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

The publications of the studies
-------------------------------

``neurosynth`` and ``neurostore``, and the engines extending them, have the
title, abstract, year and journal of each study: ``study_title(study_id,
title)``, ``study_abstract(study_id, abstract)``, ``study_year(study_id,
year)`` and ``study_journal(study_id, journal)``, so that a query, and the
query assistant, restricts the studies to years or journals and conditions
on them as on a term::

    recent(s) :- study_year(s, y), (y >= 2010), (y <= 2015)
    active(l, s) :- destrieux_activated(l, s), selected_study(s)
    cond(s) :- recent(s), mentions(s, 'pain'), selected_study(s)
    ans(l, p) :- PROB[active(l, s) // cond(s)] = p

Neurosynth's metadata holds the titles, years and journals; the abstracts
are PubMed's, asked for at the first build of the engine (about 70
requests of two hundred studies, four to five minutes; less with an
``NCBI_API_KEY``) and kept in
``<data_dir>/neurosynth/pubmed_abstracts.json``. A NeuroStore release holds
the titles, abstracts and journals but not the years, PubMed ids and DOIs:
they are those NeuroStore lists for its studies, about twenty requests of ten thousand
studies (700 MB, a few minutes) at the first build, kept in
``<data_dir>/neurostore/<version>-studies.parquet``. Both also have the identifiers of each study,
``study_pmid(study_id, pmid)``, its PubMed id as an integer, and
``study_doi(study_id, doi)``, its DOI in lower case without
``https://doi.org/``, the same in every engine: a Neurosynth study's id is
its PubMed id, and NeuroStore lists those of its studies (97% have one, 99%
a DOI; 14,358 of the 14,371 Neurosynth studies are in NeuroStore's 2026-11
release). The PubMed ids match the studies of the two databases: the
studies of a NeuroStore query saved with ``.save`` and loaded in
Neurosynth, joined on ``study_pmid``. Match on them rather than on the
DOIs, which the two databases give differently for 1.8% of their common
studies (one of them wrong: Neurosynth gives two studies the same DOI). The next builds ask for
nothing; without the network, the studies not kept have no abstract or
year, with a warning, until the next build. The journals are named as each
database names them (``Neuron``, but ``The Journal of neuroscience : the
official journal of the Society for Neuroscience`` in Neurosynth). In the
browser the tables are those of the page's data.

The Cognitive Atlas
-------------------

Both ``neurosynth`` and ``neurostore`` load the Cognitive Atlas ontology
with ``ontologies`` (see :doc:`tutorial_cognitive_atlas`), next to the
``cognitive_task`` relation: its classes, labels, definitions and axioms are
facts and constraints that queries use with the engine's data, in the
open-world manner of NeuroLang.  Because ``ontologies`` is loaded before
``datalog_init``, the rules of an engine can use the ontology's predicates,
as ``cogat_subclass_of`` and ``cogat_measures`` do.

The ``tracts`` engine
---------------------

``tracts`` is tractography, ``neurolang_apps.utils.engines.tracts`` (the extra
``tracts`` of ``neurolang-apps``; without it the engine fails to build,
with a message that says so).  It extends ``mni_atlases``: its atlases are
those of ``neurosynth`` (Destrieux, Schaefer, Tian, about 270 KB downloaded
on the first build), without the studies that activate them and without
the MNI mask.  A program loads its tractograms with the commands
``.load_tractogram`` and ``.index_tracts``, which its ``solver_mixins``
entry,
``neurolang_apps.utils.engines.tracts.frontend:TractsCommandsMixin``, adds
to the solver;
``.index_tracts(t, "destrieux")`` indexes a tractogram in MNI152 by an
atlas of the engine, whose label names are those of its relation:
``label("destrieux", l, name, h)`` joins ``destrieux(name, region)``.  An
atlas of its own is loaded as any engine loads one, ``.load_atlas("fs",
"wmparc.mgz", "FreeSurferColorLUT.txt", kind="freesurfer",
space="MNI152")``, and indexed by its name; ``.index_tracts`` gives its
labels the terms of the vocabulary by the term mapping of its kind
(FreeSurfer, Destrieux, Tian) or by ``term_mapping=...``.  It
asks where the streamlines go in the vocabulary of
tractography, Datalog rules that its ``python_init``
(``neurolang_apps.utils.engines.tracts.frontend``) gives the engine:
``traverses(x, term, h)``,
``endpoint_in(x, e, term, h)``, ``anterior_of(x, term, h)`` and the like
(the engine's ``DESIGN.md``, section 16).  The same ``init_engine``
gives it the hooks with which ``.view`` draws a result of streamline ids
``x`` on the label image of the atlas.  The ends of the streamlines are
also points in millimetres, ``endpoint_point(x, e, px, py, pz)``, stored
as the points of the atlases are, so that the parcels near the ends are
found with a tree::

   near(x, e, n) :- endpoint_point(x, e, ex, ey, ez),
       schaefer_points(mx, my, mz, n),
       (EUCLIDEAN(ex, ey, ez, mx, my, mz) <= 3)

The uncinate fasciculus of
tract_querier is the engine's ``examples/uncinate.dl``::

   neurolang-query --engine tracts --file uncinate.dl
   neurolang-query --engine tracts -i --file uncinate.dl   # then .view

The ``neurosynth_tracts`` engine
--------------------------------

``neurosynth_tracts`` is both, ``extends: [neurosynth,
meta_analysis_tracts]``, the abstract engine ``meta_analysis_tracts``
extending ``coordinate_meta_analysis`` and ``tracts``: the
studies, terms, peaks and Cognitive Atlas of ``neurosynth``, its atlases
with the studies that activate their regions (``schaefer_activated(label,
study)``...) and the centres of their voxels in millimetres
(``schaefer_points(x, y, z, label)``...), and the commands, vocabulary and
``.view`` of ``tracts``.  Their common base ``mni_atlases`` is merged once,
and both ``python_init`` run, ``neurosynth``'s then
``neurolang_apps.utils.engines.tracts.frontend``.
It needs the extra ``tracts``, and ``embeddings`` for ``similar_study``
(SPECTER2; without it a query that uses it fails with a message saying what
to install).  The ends of the streamlines of a bundle, ``endpoint_point(x,
e, x, y, z)``, join the points of the parcels within a distance, and the
parcels the studies that activate them:
The engine's ``examples/uncinate_cogat.dl`` asks which Cognitive Atlas
concepts the studies of the parcels the uncinate fasciculus connects are
about::

   neurolang-query --engine neurosynth_tracts --file uncinate_cogat.dl

``neurosynth_iit`` adds the abstract engine ``iit_tractogram``: the
downloads and the commands that load the whole-brain tractogram of the IIT
Human Brain Atlas v3, index it by its wmparc and import tract_querier's
bundles.  ``neurostore_tracts`` (``extends: [neurostore,
meta_analysis_tracts]``) and ``neurostore_iit`` (``extends:
[neurostore_tracts, iit_tractogram]``) are the same over the studies of
``neurostore``, with the same questions of the query assistant.

To add a new engine, append an entry to ``engines.yaml`` and (if needed)
provide a ``python_init`` module whose ``init_engine`` function populates the
engine's symbol table.  The ``neurolang-query`` CLI and the web server will
automatically pick up the new engine.
