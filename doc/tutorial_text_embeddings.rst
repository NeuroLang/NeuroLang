.. _tutorial-text-embeddings:

Semantic Search Over Study Abstracts
====================================

NeuroSynth links studies to terms through TF-IDF counts over a fixed
vocabulary.  The ``neurosynth`` engine also lets you rank studies by how
close their title and abstract are to any free-text query.  Studies are
embedded with SPECTER2 [singh2023]_ (document adapter ``allenai/specter2``),
queries with its ad-hoc query adapter (``allenai/specter2_adhoc_query``),
and the score is the cosine similarity between the two embeddings.

The feature needs extra packages::

    pip install "neurolang[embeddings]"

.. note::

   The SPECTER2 weights are stored as PyTorch ``.bin`` files, which recent
   versions of ``transformers`` only load with ``torch>=2.6``.  On
   platforms where an older ``torch`` is the latest available (e.g. Intel
   macOS), install ``adapters==1.1.1`` and ``transformers~=4.48.3``.

Titles and abstracts come from PubMed.  The first query that uses the
similarity downloads them (about three minutes for the 14,371 NeuroSynth
studies), downloads SPECTER2 and its adapters (about 430 MB), embeds the
studies (minutes on a GPU, longer on a CPU), and caches all of it under
``{data_dir}/neurosynth/embeddings/``, the model and its tokenizer in
``models/``.  To do it ahead of time::

    python -m neurolang.utils.text_embeddings -e neurosynth -d neurolang_data

After that, queries need no network: new query texts are embedded with
the local copy of the model.  To work offline on another machine, e.g. a
compute node without internet, run this command once on a machine with
internet and copy the data directory.

NCBI asks for a contact e-mail with E-utilities requests.  It is optional;
set it, and optionally an API key raising the rate limit from 3 to 10
requests per second, through the environment::

    export NEUROLANG_ENTREZ_EMAIL=you@example.org
    export NCBI_API_KEY=...

The engine configuration is described in :ref:`text-embeddings-section`.


Similarity of every study to a query
------------------------------------

``query_similarity(query, s)`` returns the cosine similarity between the
title and abstract of study ``s`` and the text ``query``, or ``nan`` if
``s`` is not a NeuroSynth study.  It is a function, so a rule gets the
similarity with an equality and ranges over the studies with ``study``.
Here with the ``neurolang-query`` command line
(``python -m neurolang.utils.cli``)::

    python -m neurolang.utils.cli -e neurosynth --sort sim:desc \
        "ans(s, sim) :- study(s) & \
            (sim == query_similarity('visual word form reading', s)) \
            & (sim > 0.78)"

The query text can also be a variable of a rule, as long as the rule is
queried, or called, with a constant for it: magic sets pass the constant
to the rule.  In the ``neurolang-query`` interactive mode::

    close(w, s) :- study(s) & (sim == query_similarity(w, s)) & (sim > .78)
    ans(s) :- close('visual word form reading', s)

Querying ``close(w, s)`` without a constant for ``w`` fails with an
:class:`~neurolang.exceptions.UnboundVariablesError`: the similarity
cannot be computed without a query text.

The query text can also flow through a probabilistic rule.  Here ``q`` is
a variable of the conditional rule ``voxel_given_sim``, bound by the
constant of the query and passed to ``close``::

    close(q, s) :- study(s) & (sim == query_similarity(q, s)) & (sim > .75)
    voxel_given_sim(q, i, j, k, PROB) :- (peak_reported(i, j, k, s) & selected_study(s)) // (close(q, s) & selected_study(s))
    ans(i, j, k, p) :- voxel_given_sim('math', i, j, k, p)

Each query text, e.g. ``'math'`` and ``'reading'`` in
``both(i, j, k, p, r) :- voxel_given_sim('math', i, j, k, p) &
voxel_given_sim('reading', i, j, k, r)``, gets its own conditional
probabilities.  A constant written in a rule works as well, e.g.
``math_study(s) :- close('math', s)`` used in the condition
``// (math_study(s) & selected_study(s))``.  The query texts can also
come from a stored relation with a constant joined in the query, e.g.
``ans(i, j, k, p) :- topic("maths", q) & voxel_given_sim(q, i, j, k, p)``:
magic sets pass the texts of ``topic`` with ``"maths"`` to ``close``, see
below. A relation without a constant, as ``topic(q)`` in
``ans(q, i, j, k, p) :- topic(q) & voxel_given_sim(q, i, j, k, p)``,
does not pass its values to ``close``: it must bind ``q`` in the body of
``close`` itself, e.g. ``close(q, s) :- topic(q) & study(s) & ...``.

SPECTER2 similarities fall in a narrow range: for this query they average
0.71 with a standard deviation of 0.03 over NeuroSynth, and 0.78 keeps
about the top 1% of the studies.  Pick thresholds from the distribution
rather than as absolute values: the ``percentile`` aggregation computes,
for each query text, the similarity above which the top 1% of the studies
lie, and ``close`` keeps them, whatever the query::

    cutoff(q, percentile(sim, 99)) :- study(s) & (sim == query_similarity(q, s))
    close(q, s) :- study(s) & (sim == query_similarity(q, s)) & cutoff(q, t) & (sim >= t)
    voxel_given_sim(q, i, j, k, PROB) :- (peak_reported(i, j, k, s) & selected_study(s)) // (close(q, s) & selected_study(s))
    ans(i, j, k, p) :- voxel_given_sim('math', i, j, k, p)

Each query text is embedded once and the similarities of all the studies
to it are cached (the 16 most recent query texts), so each call is a
lookup.

Activations within a radius of the peaks
----------------------------------------

A study rarely reports a peak exactly at a voxel: the activation of a
study can be taken as the voxels within a radius of its peaks, with the
``EUCLIDEAN`` distance bound (section 4.8 of :doc:`tutorial_squall`
describes it in SQUALL; the pairs within the distance are computed once,
when the rule is defined). Combined with
the percentile cutoff above, this gives the probability that a voxel is
within 2 voxels of a peak of a study, given that the study is among the
top 1% closest to the query text::

    cutoff(q, percentile(sim, 99)) :- study(s) & (sim == query_similarity(q, s))
    close(q, s) :- study(s) & (sim == query_similarity(q, s)) & cutoff(q, c) & (sim >= c)
    near_peak(i, j, k, s) :- voxel(i, j, k) & peak_reported(i1, j1, k1, s) & (EUCLIDEAN(i, j, k, i1, j1, k1) < 2)
    voxel_given_sim(q, i, j, k, PROB) :- (near_peak(i, j, k, s) & selected_study(s)) // (close(q, s) & selected_study(s))
    ans(i, j, k, p) :- voxel_given_sim('math', i, j, k, p)

The radius is in voxels of the engine's grid, so 2 is 2 times its
resolution in millimetres.

Query texts from the Cognitive Atlas
------------------------------------

The query text can be the description of a task of the Cognitive Atlas,
from ``cognitive_task(task_id, name, description)``, joined with
``voxel_given_sim`` in the query: the descriptions of the tasks named in
the query are passed by magic sets to ``cutoff`` and ``close``::

    cutoff(q, percentile(sim, 95)) :- study(s) & (sim == query_similarity(q, s))
    close(q, s) :- study(s) & (sim == query_similarity(q, s)) & cutoff(q, c) & (sim >= c)
    near_peak(i, j, k, s) :- voxel(i, j, k) & peak_reported(i1, j1, k1, s) & (EUCLIDEAN(i, j, k, i1, j1, k1) < 3)
    voxel_given_sim(q, i, j, k, PROB) :- (near_peak(i, j, k, s) & selected_study(s)) // (close(q, s) & selected_study(s))
    ans(i, j, k, p) :- cognitive_task(_, "color-word stroop task", d) & voxel_given_sim(d, i, j, k, p)

The map is that of the studies among the top 5% closest to the
description of the task. The constant ``"color-word stroop task"`` is
what lets magic sets pass the description: with ``cognitive_task(_, _,
d)``, every description would be a query text, and the query fails with
an :class:`~neurolang.exceptions.UnboundVariablesError` (``In cutoff:
the variables sim, q appear only in ...``). To map several tasks, name
them in a relation of query texts which binds ``q`` in the bodies of
``cutoff`` and ``close``::

    task_query(d) :- cognitive_task(_, "color-word stroop task", d)
    task_query(d) :- cognitive_task(_, "stop signal task", d)
    cutoff(q, percentile(sim, 95)) :- task_query(q) & study(s) & (sim == query_similarity(q, s))
    close(q, s) :- task_query(q) & study(s) & (sim == query_similarity(q, s)) & cutoff(q, c) & (sim >= c)

Each description is embedded as a query text, and the similarities to
the 16 most recent ones are cached: list the tasks of interest rather
than every task of the Cognitive Atlas.

The term-based counterpart replaces the similarity by the TF-IDF of a term.
Rather than keeping the studies whose TF-IDF passes a threshold
:math:`\tau`, Iovene et al. [iovene2021]_ make the association of a term
to a study a probabilistic fact, whose probability is a soft threshold of
the TF-IDF :math:`x`,

.. math::

   \omega(x; \alpha, \tau) = \sigma(\alpha (x - \tau)) =
   \frac{1}{1 + e^{-\alpha (x - \tau)}},

which tends to the hard threshold :math:`\mathbf{1}[x > \tau]` as
:math:`\alpha` grows. They report :math:`\alpha = 300`, with
:math:`\tau = 0.1` in their experiments; the ``mentions`` relation of the
engine uses a hard threshold of 0.03 instead. A query-based probabilistic
fact, ``head :: probability :- body``, defines it::

    term_in_study(s, t) :: 1 / (1 + exp(-300 * (w - 0.1))) :- term_in_study_tfidf(s, t, w)
    near_peak(i, j, k, s) :- voxel(i, j, k) & peak_reported(i1, j1, k1, s) & (EUCLIDEAN(i, j, k, i1, j1, k1) < 2)
    voxel_given_term(t, i, j, k, PROB) :- (near_peak(i, j, k, s) & selected_study(s)) // (term_in_study(s, t) & selected_study(s))
    ans(i, j, k, p) :- voxel_given_term('pain', i, j, k, p)

On NeuroSynth, this query takes about 9 s: 435 of the 516 studies with a
TF-IDF for 'pain' are above :math:`\tau = 0.1`, and the probabilities
reach 0.025.


Forward inference from free text
--------------------------------

The studies closest to a query can replace the term association of the
usual NeuroSynth forward-inference query.  The rules below keep as many
studies as are tagged with the term 'reading', with an aggregation
``nth_largest`` defined in Python (see the example
:ref:`sphx_glr_auto_examples_plot_text_embeddings_forward_inference.py`):

.. code-block:: python

    from pathlib import Path
    from typing import Iterable

    import numpy as np
    from neurolang.utils.engine_registry import build_engine

    nl = build_engine("neurosynth", Path("neurolang_data"), resolution=4)

    @nl.add_symbol
    def nth_largest(values: Iterable, n: Iterable) -> float:
        return float(np.sort(np.asarray(values))[-int(np.asarray(n)[0])])

    text_map = nl.execute_datalog_program(
        "similarity(s, v) :- study(s) & "
        "(v == query_similarity('visual word form reading', s)).\n"
        "reading_study(s) :- term_in_study_tfidf(s, 'reading', w).\n"
        "n_reading(count(s)) :- reading_study(s).\n"
        "cutoff(nth_largest(v, n)) :- similarity(s, v) & n_reading(n).\n"
        "text_study(s) :- similarity(s, v) & cutoff(c) & (v >= c).\n"
        "activation(i, j, k) :- selected_study(s) & peak_reported(i, j, k, s).\n"
        "relevant(s) :- selected_study(s) & text_study(s).\n"
        "marg(i, j, k, PROB) :- activation(i, j, k) // relevant(s).\n"
        "ans(i, j, k, p) :- marg(i, j, k, p)."
    )

``text_map`` gives, for each voxel, the probability that a study reports a
peak there given that it is among the 521 studies closest to the query.
On NeuroSynth v7 at 4 mm, 17 of the 20 closest studies and 207 of the 521
are tagged 'reading', and the map correlates with the 'reading' term map
after smoothing both with an 8 mm Gaussian kernel, with r = 0.97 (Dice
overlap of the top 5% voxels 0.84).

Other engines can use the same machinery on their own documents, from
PubMed or from a table of texts, through the ``text_embeddings`` section of
``engines.yaml``.

.. [iovene2021] Iovene, V., Zanitti, G. E., & Wassermann, D. (2021).
   Complex Coordinate-Based Meta-Analysis with Probabilistic Programming.
   *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(1),
   223–231.

.. [singh2023] Singh, A., D'Arcy, M., Cohan, A., Downey, D., & Feldman, S.
   (2023). SciRepEval: A Multi-Format Benchmark for Scientific Document
   Representations. *EMNLP 2023*.
