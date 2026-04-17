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
