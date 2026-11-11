.. _tutorial-cognitive-atlas:

Querying the Cognitive Atlas Ontology
=====================================

The ``neurosynth`` and ``neurostore`` engines load the `Cognitive Atlas
<https://www.cognitiveatlas.org>`_ ontology (``cogat.v2.owl`` of the
`cogat-ontology <https://github.com/CognitiveAtlas/cogat-ontology>`_
repository: about 1,800 classes, the concepts, tasks and task contrasts of
the field, with their labels, definitions and relations) next to the
studies, terms and peaks.  Nothing is copied into tables: the ontology is
read by :meth:`~neurolang.frontend.NeurolangPDL.load_ontology` and stays an
ontology.  Its triples are facts that rules and queries join with the data of
the engine, and its axioms (a class ``rdfs:subClassOf`` another) are
constraints that rewrite the queries, which is NeuroLang's open-world
reasoning (see :ref:`neurolang-rdf-data`).  The file is downloaded once into
``{data_dir}/{engine}/``, and loading it takes about two seconds.

This page shows how to pull information out of it.  The queries are Datalog,
for the REPL (:doc:`repl`) on the ``neurostore`` engine: in the browser, or
``neurolang-query -i --engine neurostore``.  Python runs them too
(:doc:`tutorial_python`).

What is in the ontology
-----------------------

The ontology is loaded with ``individuals: true`` and ``names: prefixed``,
so that a triple ``s p o`` is the fact ``p(s, o)``, with classes and
properties named after their prefix, and the IRIs of the classes
(``http://www.cognitiveatlas.org/ontology/cogat.owl#CAO_00587``) are their
values.  The ones to know:

``rdfs:label(class, text)``
    the name of a concept, task or contrast, capitalised
    (``'Working memory'``); contrasts are named ``Task_name-contrast``
    (``'Digit_span_task-backward'``).
``skos:definition(class, text)``
    its definition.
``neurolang:subClassOf(class, super_class)``
    the class hierarchy.  ``rdfs:subClassOf`` also holds the axioms that
    are not a class, such as the restrictions below.
``owl:onProperty(restriction, property)``, ``owl:someValuesFrom(restriction, class)``
    the existential restrictions.  A contrast that measures a concept is a
    subclass of the restriction ``measures some <concept>``.

and, defined by the engine with rules over those:

``cogat_subclass_of(class, ancestor)``
    the hierarchy at any depth: the closure of ``neurolang:subClassOf``.
``cogat_measures(contrast, concept)``
    the contrasts (and tasks) that measure a concept.
``cogat_term(class, term)``
    the label in lower case, to join with the terms of ``mentions`` and
    ``term_in_study_tfidf``.

They are listed with ``--list-predicates``, and the sorts are
``cogat_class`` (an IRI) and ``cogat_term``.

Looking up concepts
-------------------

A concept is found by its label, and what is known of it by its IRI:

.. nl-repl::
    :engine: neurostore

    ans(d) :- rdfs:label(c, 'Working memory'), skos:definition(c, d)

    ans(parent) :- rdfs:label(c, 'Working memory'),
        neurolang:subClassOf(c, p), rdfs:label(p, parent)

The first returns the definition of *working memory*, "active maintenance and
flexible updating of goal-task relevant information…", the second its parent,
*Cognitive control*.

Below a concept
---------------

The classes below *Cognitive control*, at any depth (16 of them, among which
*Attention*, *Auditory working memory*, *Divided attention* and
*Listening*):

.. nl-repl::
    :engine: neurostore

    ans(l) :- rdfs:label(c, 'Cognitive control'),
        cogat_subclass_of(x, c), rdfs:label(x, l)

and the contrasts that measure *working memory*, e.g.
``Digit_span_task-backward``:

.. nl-repl::
    :engine: neurostore

    ans(l) :- rdfs:label(c, 'Working memory'),
        cogat_measures(x, c), rdfs:label(x, l)

Any other relation of the ontology is read the same way, through its
restrictions.  For instance ``cogat_measures`` is defined as:

.. code-block:: neurolang

    cogat_measures(x, c) :- rdfs:subClassOf(x, r),
        owl:onProperty(r, <http://www.cognitiveatlas.org/ontology/cogat.owl#measures>),
        owl:someValuesFrom(r, c)

and the properties used by the restrictions of the ontology
(``descended_from``, ``has_contrast``, ``has_part``, ``is_related_to``,
``measured_by``, ``measures``, ``part_of``) are listed by:

.. nl-repl::
    :engine: neurostore

    ans(p) :- owl:onProperty(r, p)

From the ontology to the studies
--------------------------------

The ontology becomes useful with the data: its labels are the terms of the
studies.  The studies that mention *working memory* or any concept below it
(531 studies below it in the 2026-11 NeuroStore release, and 3,297
that mention *working memory* itself):

.. nl-repl::
    :engine: neurostore

    ans(s) :- rdfs:label(c, 'Working memory'),
        cogat_subclass_of(x, c), cogat_term(x, t), mentions(s, t)

    ans(s) :- mentions(s, 'working memory')

and the voxels they report, to map the concept with all its sub-concepts:

.. nl-repl::
    :engine: neurostore

    ans(i, j, k, s) :- rdfs:label(c, 'Cognitive control'),
        cogat_subclass_of(x, c), cogat_term(x, t), mentions(s, t),
        reports(s, i, j, k)

The opposite join, which Cognitive Atlas labels the studies talk about and
how often (349 of them in the 2026-11 release, the first being *Activation*,
*Memory* and *Attention*):

.. nl-repl::
    :engine: neurostore

    ans(l, agg_count(s)) :- rdfs:label(x, l), cogat_term(x, t),
        mentions(s, t)

``cogat_term`` lowercases the labels first.  Writing the comparison in the
query itself, ``(lcase(l) == t)`` between ``rdfs:label`` and ``mentions``,
gives the same answer but compares every label with every row of
``mentions``, which is about ten times slower.

``mentions`` keeps the TF-IDF above 0.03.  To weigh the studies instead, use
``weighted_mention(s, t, w)`` or ``term_in_study_tfidf(s, t, w)`` in the same
rule.  In the ``neurostore`` engine the terms are words and sequences of up
to three words of the abstracts, so multi-word concepts such as *working
memory* or *spatial working memory* are terms; the ``neurosynth`` vocabulary
has its own multi-word terms.

Using the ontology without an engine
------------------------------------

Any engine, or a frontend of your own in Python, can load the file and run
the same queries: see :ref:`python-ontology` in :doc:`tutorial_python`.

To use the ontology in an engine of your own, add it to the ``ontologies``
of its entry in ``engines.yaml``, or extend ``coordinate_meta_analysis``
(see :doc:`engine_configuration`)::

    ontologies:
      - url: https://raw.githubusercontent.com/CognitiveAtlas/cogat-ontology/master/ontology/cogat.v2.owl
        individuals: true
        names: prefixed
