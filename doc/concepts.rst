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


Declarative Engine Registry & CLI
----------------------------------

NeuroLang ships a command-line interface, ``neurolang-query``, that provides
a quick way to run Datalog queries against pre-configured datasets without
writing Python:

.. code-block:: bash

   # List available dataset engines
   neurolang-query --list-engines

   # List predicates for the NeuroSynth engine
   neurolang-query --engine neurosynth --list-predicates

   # Run a Datalog query
   neurolang-query "ans(t) :- term_in_study_tfidf(t, w, s)"

   # Use the Destrieux atlas engine
   neurolang-query --engine destrieux "ans(name) :- destrieux(name, region)"

Engines are declared declaratively in :file:`engines/engines.yaml`:

.. code-block:: yaml

   engines:
     neurosynth:
       description: "Neurosynth database — forward and reverse inference"
       requires_mni_mask: true
       python_init: "neurolang.utils.engines.neurosynth.init"
       datalog_init: |
         study_with_peaks(S) :- peak_reported(I, J, K, S)
       relations:
         extra_data: "shared/my_data.csv"
       predicates:
         peak_reported:
           arity: 4
           columns: [i, j, k, study_id]
           description: "Reported activation peaks in voxel coordinates"
         ...

Each engine references a Python module that exports
``init_engine(nl, mask, data_dir)``.  The :mod:`neurolang.utils.engine_registry`
module handles YAML loading, MNI mask retrieval, and delegating to the
engine's init script.

**Init phases.**  When :func:`~neurolang.utils.engine_registry.build_engine`
builds an engine, it runs up to nine phases in order:

#. **Builtins** (``builtins``) — registers known functions (``exp``,
   ``log``, ``startswith``) as callable symbols.  Simple list in YAML::

       builtins: [exp, log, startswith]

#. **Base symbols** (``use_base_symbols``) — when ``use_base_symbols: true``
   is set (and ``requires_mni_mask`` is also true), the engine registry
   automatically registers common neuroimaging symbols (``agg_count``,
   ``agg_create_region``, ``agg_create_region_overlay``,
   ``principal_direction``, ``region_union``) without needing a custom
   Python init script.  This replaces calling
   ``init_base_engine()`` from within a Python init module::

       use_base_symbols: true

#. **Python init** (``python_init``) — imports the module and calls
   ``init_engine(nl, mask, data_dir)``.  Use this for downloading data,
   registering complex symbols, or loading neuroimaging atlases.  If
   ``use_base_symbols`` already covers your needs, this phase can be
   omitted entirely — see the ``destrieux`` engine entry for an example.

#. **Downloads** (``downloads``) — fetches files from URLs before the
   other phases.  Supports archive extraction::

       downloads:
         - url: "https://example.com/data.tar.gz"
           dest: my_engine/
           extract: true

#. **Templates** (``templates``) — downloads neuroimaging templates via
   **nilearn** or **TemplateFlow** (if the ``templateflow`` package is
   installed).  Each template can optionally register a
   ``voxel(i, j, k)`` predicate from its non-zero mask::

       templates:
         # Nilearn brain mask → voxel(i, j, k) predicate
         mni_brain:
           source: nilearn
           variant: brain_mask
           predicate: voxel
         # Nilearn T1 template (2 mm) — just download, no predicate
         mni_t1:
           source: nilearn
           variant: template
           resolution: 2
         # TemplateFlow T1 template at 1 mm
         t1_highres:
           source: templateflow
           template: MNI152NLin2009cAsym
           resolution: 1
           suffix: T1w

   Supported **nilearn** variants:

   * ``brain_mask`` — MNI152 brain mask (``load_mni152_brain_mask``)
   * ``gm_mask`` — grey-matter mask (``load_mni152_gm_mask``)
   * ``wm_mask`` — white-matter mask (``load_mni152_wm_mask``)
   * ``template`` — MNI152 T1 template (``load_mni152_template``;
     pass ``resolution: 1`` or ``resolution: 2``)
   * ``gm_template`` — grey-matter template
   * ``wm_template`` — white-matter template

   **TemplateFlow** templates require ``pip install templateflow`` and
   are identified by a ``template`` name from the
   `TemplateFlow repository <https://www.templateflow.org/>`_
   (e.g. ``MNI152NLin2009cAsym``, ``MNIInfant``).  The ``suffix``
   and ``resolution`` fields select the specific image variant.

   When ``predicate`` is set, the template is loaded as a NIfTI image
   and all non-zero voxels are registered as a
   ``predicate(i, j, k)`` tuple set — useful as a coordinate-space
   reference in Datalog queries.

#. **Atlases** (``atlases``) — downloads brain atlases via nilearn and
   registers each region as a predicate with
   :class:`~neurolang.regions.ExplicitVBR` geometry::

       atlases:
         destrieux:
           predicate_name: destrieux
         schaefer:
           n_rois: 400
           yeo_networks: 7
           resolution_mm: 2
           predicate_name: schaefer_400
         difumo:
           dimension: 64
           threshold: 0.5
           predicate_name: difumo_64

   For probabilistic atlases (``difumo``) a second predicate is registered
   automatically — ``{name}_prob(component, i, j, k, prob)`` — with the raw
   probability value at each voxel for each component (filtered by
   ``prob_threshold``, default 0.01)::

       atlases:
         difumo:
           dimension: 64
           threshold: 0.5
           prob_threshold: 0.01
           predicate_name: difumo_64
           prob_predicate_name: difumo_64_prob

   Supported atlases: ``destrieux`` (deterministic), ``schaefer``
   (deterministic), ``difumo`` (probabilistic).

#. **Datalog init** (``datalog_init``) — an optional YAML multiline string
   of Datalog rules.  These are evaluated after templates and atlases
   so they can reference template-derived voxel predicates and atlas
   predicates::

       datalog_init: |
         left_region(N, R) :- destrieux(N, R), startswith('lh', N)
         right_region(N, R) :- destrieux(N, R), startswith('rh', N)

#. **Relations** (``relations``) — loads CSV/TSV files as extensional
   predicates.  Each entry maps a relation name to a file path relative to
   the :file:`engines/` directory.  The ``file`` value may be a URL for
   automatic download::

       relations:
         my_table: "my_engine/data.csv"
         extra_refs: "shared/references.tsv"

   For richer metadata that appears in ``--list-predicates`` output, use a
   dict with an optional ``description`` field::

       relations:
         my_table:
           file: "https://example.com/data.csv"
           description: "Experiment metadata loaded from CSV"

   Supported formats: ``.csv``, ``.tsv``, ``.csv.gz``, ``.tsv.gz``.  The
   file is read with ``pandas.read_csv`` and registered via
   :meth:`~neurolang.frontend.query_resolution_datalog.QueryBuilderDatalog.add_tuple_set`.
   Because relations are loaded after the Datalog init, your derived rules
   can already reference them.

#. **Probabilistic choices** (``probabilistic_choice``) — declares uniform
   probabilistic choices over existing predicates::

       probabilistic_choice:
         selected_study:
           source: study
           description: "Uniform choice over studies"

#. **Ontologies** (``ontologies``) — loads OWL/RDF ontologies from URLs
   or local paths::

       ontologies:
         - url: "https://example.com/ontology.owl"

All phases are optional — include only what your engine needs.
If ``use_base_symbols`` and the declarative YAML fields (``atlases``,
``templates``, ``relations``, ``datalog_init``, ``probabilistic_choice``)
cover your engine, you can omit ``python_init`` entirely — the
``destrieux`` engine is a complete example of this pattern.

To add a new engine, create a YAML entry.  If you need a custom Python
init script, add a corresponding :mod:`init` module under
:file:`neurolang/utils/engines/`.  See
:mod:`neurolang.utils.engines.neurosynth.init` for a complete Python init
example, and the ``destrieux`` engine entry for a purely YAML-driven
example.

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
