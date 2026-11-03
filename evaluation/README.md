# Evaluations

Experiments around the free-text queries of the `neurosynth` and `neurostore`
engines (SPECTER2 `query_similarity`), with their write-ups, scripts and outputs:

* `specter2_retrieval/`: retrieval quality of SPECTER2 against titles, MeSH
  headings, LLM-extracted concepts and Cognitive Atlas definitions, with a TF-IDF
  baseline.
* `text_meta_analysis/`: do free-text queries without the vocabulary term give
  the meta-analytic map of the term? Identification of 36 Neurosynth terms from
  108 paraphrases, and demos composing free text with other predicates.

The scripts are the ones that were run; they use working directories under
`/tmp` and need the data and embeddings described in each README.
