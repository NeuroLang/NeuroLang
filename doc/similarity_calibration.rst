.. _similarity-calibration:

Calibrating the similarity gate of ``similar_study``
====================================================

The meta-analysis engines (``neurosynth``, ``neurostore``) define the
studies about a free text, ``similar_study(q, s)``, from the SPECTER2
similarity of the text to each study's title and abstract
(:doc:`tutorial_text_embeddings`).  This page explains how the rule is
built, why it is built this way, and how its threshold is calibrated for
an engine or an embedding model.


What ``similar_study`` does
---------------------------

The rule has two steps, both computed once per text by
``query_similarity_cutoff(q)``::

    similar_study(q, s) :- study(s) & (v == query_similarity(q, s)) & (c == query_similarity_cutoff(q)) & (v >= c)

1. **A gate on the text.**  The mean similarity of the text's 20 closest
   studies (the *top-20 mean*) must reach the ``similarity_gate`` of the
   engine's ``text_embeddings`` entry (:ref:`text-embeddings-section`).
   If it does not, no study is about the text: the cutoff is ``inf`` and
   ``similar_study`` is empty.
2. **A cutoff on the studies.**  If the gate passes, the cutoff is the
   95th percentile of the text's positive similarities: the text gets the
   top 5% of the studies closest to it.

A text that passes the gate gets its whole top 5%, a text that fails it
gets nothing.


Why not a floor, or standardised similarities
----------------------------------------------

Every text has a top 5%, even one that no study is about: 'stock market
crash' would get 719 studies of NeuroSynth.  A floor on each study's
similarity, ``v >= 0.75``, does not fix this well.  SPECTER2 similarities
have a baseline that depends on the text: over NeuroSynth, the median
similarity of a text to the studies ranges from 0.57 to 0.76.  So a floor
measures how much the wording of a text looks like neuroscience more than
how many studies are about it.  Within the top 5%, it keeps a fragment
decided by a few outlying studies: 5 studies for 'tapping fingers in
sequence', 32 for 'imagining future events', and all 719 for the
Cognitive Atlas definition of 'memory'.

Standardising the similarities of each text, as ``z = (v - median) /
MAD`` or ``(v - mean) / sd``, does not tell the two kinds of texts apart
either.  A text off the domain also has a few studies standing out from
its own distribution, so the largest ``z``, the mean ``z`` of the 20
closest studies and the number of studies above ``z >= k`` have a ROC AUC
of 0.44 to 0.65 against texts off the domain.  What separates them is how
close the closest studies are, a statistic of the text.  ROC AUC of each
statistic, in-domain texts against the null, on the fit half (see the
procedure below), SPECTER2:

==============================  ==========  ==========  ==========  ==========
statistic                       NeuroSynth              NeuroStore
------------------------------  ----------------------  ----------------------
\                               all         topics,     all         topics,
                                            phrases                 phrases
==============================  ==========  ==========  ==========  ==========
mean of the 20 largest          0.934       0.962       0.905       0.953
largest                         0.920       0.959       0.886       0.950
median                          0.855       0.866       0.818       0.837
mean of the 20 largest, ``z``   0.62        0.83        0.54        0.78
largest ``z``                   0.54        0.76        0.47        0.70
studies with ``z >= 5``         0.53        0.63        0.50        0.67
==============================  ==========  ==========  ==========  ==========

"All" is every in-domain text; "topics, phrases" leaves out the Cognitive
Atlas definitions.  The mean of the 20 largest similarities is the gate's
statistic: the best AUC, and a simple one.


The calibration procedure
-------------------------

To calibrate the gate for an engine or an embedding model:

1. **A null set.**  Write a few hundred texts that no study is about, of
   varied topics (cooking, sport, finance, history, law...) and lengths,
   from short phrases to definition-length paragraphs.
2. **An in-domain set.**  Take the definitions of the Cognitive Atlas
   (``cogat_definition(t, d)``), the topics users ask about, and short
   task phrases such as 'tapping fingers in sequence'.
3. **A fit/check split.**  Split each set into two halves at random: a fit
   half, to choose the statistic and its threshold, and a check half, to
   measure them.
4. **The statistics.**  Score every study against each text with
   :meth:`~neurolang_apps.utils.text_embeddings.EmbeddingIndex.scores`, and
   compute the candidate statistics of each text: the largest score, the
   mean of the 20 largest, the median, and their standardised versions.
5. **ROC AUC.**  On the fit half, compare the in-domain texts with the null
   by the ROC AUC of each statistic, and keep the simplest statistic with
   a high AUC.
6. **The threshold.**  Set it at a high percentile of the statistic over
   the null texts of the fit half.  Its false-positive rate is about one
   minus that percentile: 10% at the 90th, 5% at the 95th, 1% at the 99th.
7. **The check.**  On the check half, measure the share of null texts that
   pass (the false-positive rate) and the share of in-domain texts kept.
8. **The setting.**  Set the threshold as ``similarity_gate`` in the
   engine's ``text_embeddings`` entry of ``engines.yaml``.  An engine that
   extends ``coordinate_meta_analysis`` inherits its 0.75 and overrides it
   in its own entry, as NeuroStore does::

       neurostore:
         extends: coordinate_meta_analysis
         text_embeddings:
           study_abstracts: {similarity_gate: 0.76}

In Python, for one text:

.. code-block:: python

    from pathlib import Path

    import numpy as np
    from neurolang_apps.utils.engine_registry import build_engine

    nl = build_engine("neurosynth", Path("neurolang_data"))
    index = nl.text_embedding_indexes["study_abstracts"]
    scores = index.scores("medieval castle architecture")
    print(np.sort(scores[scores > 0])[-20:].mean())   # 0.730: below the gate


Running the calibration script
------------------------------

The script ``scripts/similarity_gate.py`` of
`neurolang-text2query-bench <https://github.com/NeuroLang/neurolang-text2query-bench>`_
(branch ``neurosynth-v2``) runs these steps::

    python scripts/similarity_gate.py --engine neurostore \
        --data-dir neurolang_data \
        --cache runs/similarity_gate_scores_neurostore.npz \
        > runs/similarity_gate_report_neurostore.md

Its texts are in ``scripts/similarity_gate_texts/``: ``null.txt``, 328
texts off the domain, and ``domain_phrases.txt``, 94 task phrases.  The
script adds the 1138 Cognitive Atlas definitions of the engine and the 27
topics and definitions used by the benchmark's ``similar_study``
questions.  ``--seed`` (0) draws the fit/check split, and ``--gate``
(0.75) is a threshold whose check-half rates the report gives.  ``--cache`` keeps
the scores of every text against every study, so that a rerun only
recomputes the statistics; it is rebuilt when the texts change.

The report, in Markdown, has the AUC and the 95th and 99th percentile
thresholds of every statistic with their check-half rates, the
statistics of the short phrases, the benchmark topics and the null texts
closest to the domain, the top-20 mean gate at the null's 90th, 95th and
99th percentiles, and the AUC by text length.  The reports of the two
engines are ``runs/similarity_gate_report.md`` (NeuroSynth) and
``runs/similarity_gate_report_neurostore.md``.


Reference results
-----------------

With SPECTER2, 328 null texts and 1259 in-domain ones (1138 Cognitive
Atlas definitions, 27 benchmark topics, 94 task phrases), seed 0.

NeuroSynth has 14,371 studies, NeuroStore 64,920.  The thresholds of the
top-20 mean at percentiles of the null's fit half, and on the check half
the share of the null texts that pass and of the in-domain texts kept
(all; Cognitive Atlas definitions / benchmark topics / task phrases):

========  =========  =====  ==============  =========  =====  ==============
gate      NeuroSynth                        NeuroStore
--------  --------------------------------  --------------------------------
\         threshold  null   kept            threshold  null   kept
========  =========  =====  ==============  =========  =====  ==============
null p90  0.739      5.5%   78% (76/86/94)  0.754      3.7%   68% (66/86/91)
null p95  0.750      2.4%   63% (60/79/85)  0.762      3.0%   58% (56/71/87)
null p99  0.763      0%     48% (46/64/74)  0.772      1.8%   46% (43/64/77)
========  =========  =====  ==============  =========  =====  ==============

The gates are the 95th percentiles, rounded: 0.75 for NeuroSynth (2.4%
of the check half's null passes, 63% of its in-domain texts are kept)
and 0.76 for NeuroStore (3.7%, 62%: 59% of the definitions, 71% of the
benchmark topics, 87% of the task phrases), set in its own
``text_embeddings`` entry.  NeuroStore needs a higher gate: with 4.5
times as many studies, the 20 closest to any text are closer, and 0.75
would let 9.1% of the check half's null through.  The statistic also
separates the texts a little less well (AUC 0.905 against 0.934).  The
99th percentile drops more in-domain texts on both engines, among them
the benchmark topics 'disgust at rotten food' and 'gambling with risky
odds' (top-20 means 0.751 and 0.755 on NeuroSynth, 0.765 and 0.763 on
NeuroStore).  A text that passes gets its whole top 5%: 719 studies of
NeuroSynth, 3208 of NeuroStore.

Short texts are the hardest.  On NeuroSynth, 'tasting bitter drinks'
(top-20 mean 0.730) and 'tapping fingers in sequence' (0.746) get no
study, 'socially excluded' (0.763) gets its top 5%.  On NeuroStore the
three are higher, 0.740, 0.757 and 0.777, and so are the gate's
outcomes: the first two fail 0.76, the third passes.  A fuller
description of the task, or the definition of a Cognitive Atlas
concept, does better.  A few texts outside the domain pass the gate,
the highest being 'golf handicap calculation' (0.772 on NeuroSynth,
0.783 on NeuroStore).

Some studies are close to many texts: one is in the top 5% of 69% of
the in-domain texts on NeuroSynth, of 77% on NeuroStore.  Subtracting
each study's mean similarity to the null texts improves the AUC only a
little (on the check half 0.955 against 0.939 on NeuroSynth, 0.941
against 0.912 on NeuroStore), so the gate does not correct for it.


Recalibrating for another encoder or engine
-------------------------------------------

The gate is a similarity, so it belongs to an encoder and a corpus: a
different embedding model (e.g. a Qwen embedding served by the Albert
API) has another scale of cosine similarities, and a different corpus
moves the closest studies.  To recalibrate:

1. Make the engine's index use the encoder: any object with the
   interface of :class:`~neurolang_apps.utils.text_embeddings.Specter2Encoder`
   (``encode_documents``, ``encode_query``, ``cache_key``) behind an
   :class:`~neurolang_apps.utils.text_embeddings.EmbeddingIndex`.
2. Run the script on the engine with a new ``--cache``: it only needs
   ``index.scores``.  Keep the same texts and seed, so that the results
   compare with the ones above.
3. Check that the top-20 mean still has the best AUC, read its
   thresholds at the null's 90th, 95th and 99th percentiles, and choose
   one by its check-half false-positive rate and in-domain share kept.
4. Set ``similarity_gate`` in the engine's ``text_embeddings`` entry, and
   note the numbers on this page.

A new engine over the same encoder needs the same: the null and the
in-domain sets do not change, the studies do.
