# Retrieval evaluation of SPECTER2 (document adapter + ad hoc query adapter)

How `query_similarity` works: a study is embedded as `title [SEP] abstract` with
the document adapter (`allenai/specter2`), a query with the ad hoc query adapter
(`allenai/specter2_adhoc_query`), and the score is the cosine. Evaluated on the
14,370 Neurosynth studies (PubMed abstracts) and the 35,880 NeuroStore studies
with an abstract (2026-10), the same embeddings as the maps. TF-IDF cosine
(unigrams+bigrams, sublinear tf) on title+abstract is the lexical baseline.
`scripts/` recompute everything (as run: they work in `/tmp/stage*` directories, to be adapted); `output/` holds the raw output: `eval_retrieval.txt` (sections 1 to 3), `eval_retrieval2.txt` (fusion and lexical-free positives, related studies), `eval_defs.txt` (section 4).

## 1. A title as the query, the study as the target

| target | query | R@1 | R@5 | R@10 | R@50 | MRR |
|---|---|---|---|---|---|---|
| `title [SEP] abstract` (Neurosynth) | ad hoc adapter | 96.3 | 99.5 | 99.7 | 100.0 | 0.977 |
| same | document adapter | 96.4 | 99.5 | 99.7 | 100.0 | 0.978 |
| same, TF-IDF | | 98.6 | 99.9 | 100.0 | 100.0 | 0.992 |
| `title [SEP] abstract` (NeuroStore) | ad hoc adapter | 95.1 | 99.2 | 99.6 | 99.9 | 0.970 |
| abstract only (Neurosynth) | ad hoc adapter | 62.9 | 83.2 | 88.3 | 95.7 | 0.720 |
| same | document adapter | 67.7 | 86.9 | 91.3 | 97.2 | 0.763 |
| same, TF-IDF | | 65.7 | 86.8 | 91.3 | 97.0 | 0.751 |

The first rows are a sanity check, not a quality measure: the title is part of
the target text. Without it (abstract-only targets) the ad hoc adapter does
no better than TF-IDF, and the document adapter is better than the ad hoc one
for a title, which is document-like.

## 2. Topical retrieval, with independent labels

| ground truth | model | MAP | nDCG@10 | P@10 | R@100 |
|---|---|---|---|---|---|
| Neurosynth, MeSH heading (526 queries) | SPECTER2 ad hoc | 0.175 | 0.360 | 0.339 | 0.255 |
| | SPECTER2 document adapter | 0.175 | 0.350 | 0.333 | 0.260 |
| | TF-IDF | 0.264 | 0.497 | 0.474 | 0.339 |
| | fusion (z-scores) | 0.296 | 0.516 | 0.496 | 0.375 |
| NeuroStore, LLM concept (536 queries) | SPECTER2 ad hoc | 0.077 | 0.178 | 0.170 | 0.171 |
| | TF-IDF | 0.192 | 0.359 | 0.346 | 0.345 |
| | fusion (z-scores) | 0.196 | 0.361 | 0.349 | 0.347 |
| NeuroStore, 13 domains | SPECTER2 ad hoc | 0.270 | 0.565 | 0.577 | 0.019 |
| | TF-IDF | 0.270 | 0.491 | 0.500 | 0.017 |

On keyword-like labels SPECTER2 is below TF-IDF (it wins on 28% of the MeSH
queries and 6% of the concept queries), and the ad hoc adapter does not beat
the document adapter. These labels are assigned from words of the text, which
favours lexical matching. Where the positives share no content word with the
query (43% of MeSH positives, 21% of concept positives), TF-IDF cannot find
them (0.1% and 0.0% in the top 100) and SPECTER2 finds some: 9.7% and 2.0%
(chance 0.7% and 0.3%). Fusing both scores helps on MeSH (+0.032 MAP).

## 3. Related studies, judged by shared MeSH headings

The 10 nearest studies of 3,000 Neurosynth studies: mean MeSH Jaccard 0.197
(SPECTER2 on title+abstract), 0.194 (abstract only), 0.193 (TF-IDF), 0.107
(random): the three are equivalent; most of the shared headings are generic
neuroimaging ones.

## 4. A Cognitive Atlas definition as the query (what the example does)

124 NeuroStore concepts that have a Cognitive Atlas definition (median 58 studies):

| query | MAP | nDCG@10 | P@10 | R@100 |
|---|---|---|---|---|
| SPECTER2 ad hoc, label | 0.111 | 0.275 | 0.263 | 0.191 |
| SPECTER2 ad hoc, definition | 0.056 | 0.127 | 0.126 | 0.099 |
| SPECTER2 document adapter, definition | 0.055 | 0.138 | 0.126 | 0.095 |
| TF-IDF, label | 0.248 | 0.465 | 0.460 | 0.367 |
| TF-IDF, definition | 0.043 | 0.078 | 0.078 | 0.082 |
| RRF, SPECTER2 label + TF-IDF label | 0.218 | 0.432 | 0.421 | 0.335 |

The definition is a worse query than the label, with SPECTER2 (better on 11%
of the concepts) and with TF-IDF. For *working memory* (1,944 annotated
studies): the top 1,794 studies by the definition contain 393 annotated ones
(precision 22%, recall 20%), by the label 643 (36%), by TF-IDF of the label
1,138 (63%, recall 59%).

Caveat: the concept labels come from an LLM extraction that uses the same
vocabulary as the labels, which favours label queries.
