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
for the ``neurolang-query`` REPL (``neurolang-query --engine neurostore``),
the ``--engine`` option of its one-shot mode, or
:meth:`~neurolang.frontend.NeurolangPDL.execute_datalog_program`.

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

A concept is found by its label, and what is known of it by its IRI::

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
*Listening*)::

    ans(l) :- rdfs:label(c, 'Cognitive control'),
        cogat_subclass_of(x, c), rdfs:label(x, l)

and the contrasts that measure *working memory*, e.g.
``Digit_span_task-backward``::

    ans(l) :- rdfs:label(c, 'Working memory'),
        cogat_measures(x, c), rdfs:label(x, l)

Any other relation of the ontology is read the same way, through its
restrictions.  For instance ``cogat_measures`` is defined as::

    cogat_measures(x, c) :- rdfs:subClassOf(x, r),
        owl:onProperty(r, <http://www.cognitiveatlas.org/ontology/cogat.owl#measures>),
        owl:someValuesFrom(r, c)

and the properties used by the restrictions of the ontology
(``descended_from``, ``has_contrast``, ``has_part``, ``is_related_to``,
``measured_by``, ``measures``, ``part_of``) are listed by::

    ans(p) :- owl:onProperty(r, p)

From the ontology to the studies
--------------------------------

The ontology becomes useful with the data: its labels are the terms of the
studies.  The studies that mention *working memory* or any concept below it
(301 studies below it in the 2026-10 NeuroStore release, among the 2,031
that mention *working memory* itself)::

    ans(s) :- rdfs:label(c, 'Working memory'),
        cogat_subclass_of(x, c), cogat_term(x, t), mentions(s, t)

    ans(s) :- mentions(s, 'working memory')

and the voxels they report, to map the concept with all its sub-concepts::

    ans(i, j, k, s) :- rdfs:label(c, 'Cognitive control'),
        cogat_subclass_of(x, c), cogat_term(x, t), mentions(s, t),
        reports(s, i, j, k)

The opposite join, which Cognitive Atlas labels the studies talk about and
how often (282 of them in the 2026-10 release, the first being *Activation*,
*Memory* and *Attention*)::

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

Linking a definition to the studies with SPECTER2
-------------------------------------------------

A term does not need to be mentioned: the ontology defines the concept it
names with a text, and SPECTER2 (see :ref:`tutorial-text-embeddings`) measures
how close the title and abstract of every study are to that text.  With the
term as the first argument of every rule, the definition of ``'working
memory'`` is the query of ``query_similarity``, and the 5% of the studies
closest to it are the studies of the term, whose uniformity map is computed
as in :ref:`tutorial-neurosynth-tests`::

    definition(t, d) :- cogat_term(c, t) & skos:definition(c, d)
    similarity(t, s, v) :- study(s) & definition(t, d) & (v == query_similarity(d, s))
    cutoff(t, percentile(v, 95)) :- similarity(t, s, v)
    similar(t, s) :- similarity(t, s, v) & cutoff(t, c) & (v >= c)

``examples/plot_cognitive_atlas_uniformity.py`` (in the example gallery)
computes the map for ``'working memory'`` in the ``neurostore`` engine, over
the whole brain (``brain_mask``), and compares it with the map of the studies
that mention the term.  A study without abstract has a similarity of 0 and is
left out of the cutoff.  In the 2026-10 release the 35,880 studies with an
abstract have a mean similarity of 0.68 (standard deviation 0.02); 1,794
studies are in the top 5%, 478 of them among the 2,031 that mention *working
memory*, the others being about its neighbours, for instance cognitive control
and goal maintenance.  Both uniformity maps show the dorsolateral prefrontal
cortex, the supplementary motor area and dorsal anterior cingulate, the
parietal cortex and the anterior insula; their voxels in the brain correlate
at 0.94 (the top 5% of their voxels overlap with a Dice coefficient of 0.81),
and the largest ratios are 10.5 for the studies close to the definition and
11.6 for those that mention the term.  The basal ganglia and the thalamus
are above the average of the brain (a mean ratio of 1.4 to 1.7, up to 3.5 to
4.3 in the caudate, the pallidum and the thalamus), whereas the hippocampus
and the amygdala are barely above it (a mean ratio of 1.2 to 1.3, and at most
2.5), for both selections of studies: below the 95th percentile of the map
that the example plots.  Averaging over ``cortical_mask`` instead of
``brain_mask`` would leave them out of the map altogether.

The abstracts are embedded once (about 70 minutes on 32 CPU cores for the
36,884 studies, minutes on a GPU), then cached; see
:ref:`tutorial-text-embeddings` to compute them ahead of time.

Using the ontology without an engine
------------------------------------

The same queries run on any frontend that loaded the file::

    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.load_ontology(
        "cogat.v2.owl", individuals=True, names="prefixed"
    )
    nl.execute_datalog_program("""
        cogat_subclass_of(x, y) :- neurolang:subClassOf(x, y)
        cogat_subclass_of(x, z) :-
            cogat_subclass_of(x, y), neurolang:subClassOf(y, z)
    """)
    below = nl.execute_datalog_program(
        "ans(l) :- rdfs:label(c, 'Working memory'),"
        " cogat_subclass_of(x, c), rdfs:label(x, l)"
    )

To use the ontology in an engine of your own, add it to the ``ontologies``
of its entry in ``engines.yaml``, or extend ``coordinate_meta_analysis``
(see :doc:`engine_configuration`)::

    ontologies:
      - url: https://raw.githubusercontent.com/CognitiveAtlas/cogat-ontology/master/ontology/cogat.v2.owl
        individuals: true
        names: prefixed
