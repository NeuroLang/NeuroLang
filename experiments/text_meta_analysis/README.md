# Free-text meta-analytic queries with NeuroLang and SPECTER2

**Question.** Can a user type a description of a topic, without the vocabulary term,
and get from NeuroLang + SPECTER2 the meta-analytic map the term would have given?

**Scope of the claim.** Earlier retrieval checks (`../specter2_retrieval_eval`) showed
that SPECTER2 is *not* better than TF-IDF when the query contains the words of the
target (keywords, labels), and that a Cognitive Atlas definition is a worse query than
the label. This experiment tests the case where SPECTER2 should matter: the query
does **not** contain the term or its morphological variants.

## Design

* **Data.** Neurosynth: 14,371 studies, 444,098 peaks, PubMed abstracts embedded with
  SPECTER2 (document adapter; queries with the ad hoc query adapter).
* **36 terms** of the Neurosynth vocabulary with 103 to 1,252 studies (TF-IDF > 0.001),
  from many domains (pain, face, reading, reward, fear, speech, music, working memory,
  episodic memory, semantic, empathy, decision making, inhibition, conflict, vision,
  auditory, somatosensory, sleep, anxiety, depression, schizophrenia, addiction,
  default mode, ...). `paraphrases.json`.
* **108 paraphrases**, 3 per term, written without the term or any variant (Porter-stem
  check in `scripts/check_para.py`: 0 violations), e.g. *pain* : "cortical response to
  noxious heat applied to the skin". Only 7 of 108 contain another vocabulary term whose
  studies overlap the target's (Jaccard >= 0.2).
* **Reference map** of a term: the uniformity map (P(A_v|T)/P(A_V|T), whole brain, 3-voxel
  radius) of the studies with the term. **Text-driven map**: the same map for the top-N
  studies for the paraphrase, N = the number of studies of the term, with SPECTER2, with
  TF-IDF cosine on the paraphrase (title+abstract, unigrams+bigrams), their z-score fusion,
  and N random studies.
* **Metric: identification.** Rank the 36 reference maps by correlation with the
  text-driven map; accuracy@1/3/5 of the right term (chance 2.8% / 8.3% / 13.9%). Raw
  correlation, and centered (the mean reference map, the network common to all fMRI
  studies, removed).
* **No shared studies.** The literature is split at random in two halves, 10 times: the
  reference maps use the studies of the term in one half, the selections come from the
  other half, with N the number of studies of the term in that half. The first version of
  this experiment (`logs/experiment.log`, reference maps and selections from the same pool)
  gave 64.8% vs 38.0% but is inflated: 28% of the SPECTER2 selection are studies of the
  reference map itself. Its split-half "ceiling" (44%) being below SPECTER2's accuracy
  showed it.
* The maps come from a sparse numpy computation, validated against the Datalog map of the
  engine (Neurosynth *working memory*: r = 1.000000, max difference 1e-6). The demos below
  run in the NeuroLang engine itself.

## Result

Identification among 36 terms, 10 random splits of the literature (`logs/experiment2.log`):

| selection from the paraphrase | acc@1 | acc@3 | acc@5 | acc@1 95% CI | precision of the selection |
|---|---|---|---|---|---|
| **SPECTER2** | **32.9%** | **60.6%** | **72.0%** | [20.6, 45.4] | 0.28 |
| TF-IDF on the paraphrase | 20.6% | 45.4% | 56.2% | [11.5, 30.3] | 0.18 |
| fusion (z-score) | 25.7% | 54.8% | 65.6% | [15.6, 36.9] | 0.23 |
| random studies | 3.0% | 8.2% | 13.2% | [0.0, 8.7] | 0.03 |
| *context: the term itself, SPECTER2* | 40.8% | 64.4% | 73.1% | | |
| *context: the term itself, TF-IDF* | 50.6% | 73.9% | 81.9% | | |
| *ceiling: the term's own studies of the query half* | 51.4% | 74.7% | 81.7% | | |

SPECTER2 - TF-IDF: **+12.3 points** acc@1 (95% CI [+5.0, +19.8], bootstrap over terms).
With the common network removed (centered correlation): SPECTER2 36.9%, TF-IDF 29.8%,
fusion 33.9%, ceiling 55.0%, difference +7.1 points (CI [+1.1, +13.1]). Per term, `per_paraphrase_splits.csv`.

Reading: from a description with no vocabulary term, SPECTER2 gets the right map first
among 36 in a third of the cases, a third more often than TF-IDF, and reaches about 64% of
what the term's own studies reach (TF-IDF 40%). The ceiling is 51% because maps of
half of the literature are noisy and some terms are near-synonyms (speech~auditory
r = 0.92, reading~semantic 0.86). Fusion does not help here.

## Demos in the NeuroLang engine (`logs/demo.log`, `demo_maps.png`, `maps/`)

1. **Free text, end to end in Datalog.** "cortical response to noxious heat applied to the
   skin": `query_similarity` + the `percentile` aggregation + the uniformity rules select
   516 studies; the map correlates r = 0.916 (Dice 0.76) with the Neurosynth *pain* map.
2. **Set difference** (text AND NOT term): 2,182 studies are close to "holding information
   in mind over a delay"; 1,869 do not mention *working memory*; their map correlates
   r = 0.856 (Dice 0.65) with the *working memory* map (the untagged studies include
   attention and inhibition studies).
3. **Region constraint and a conditional probability.** "acquisition and extinction of
   conditioned shock expectancy": 36.6% of the studies close to the text report an amygdala
   peak against 13.7% of all studies (lift 2.7; 3.2 for the studies tagged *fear*); the
   map of the 133 studies with an amygdala peak is stronger (largest ratio 27 vs 13)
   and correlates r = 0.87 with that of all the text-selected studies.

These compose the free-text selection with a term, a negation, an atlas region and a
conditional probability in one program, which a term-based interface does not offer.

## Limits

* 36 terms: wide confidence intervals; one dataset (Neurosynth), the studies of which
  also define the reference maps through their TF-IDF.
* The paraphrases were written by one author (an LLM), not collected from users, and the
  3 per term are not independent. Terms that are near-synonyms limit identification.
* SPECTER2 does not beat TF-IDF for keyword queries (see the retrieval evaluation);
  the claim is limited to queries without the term. The accuracy of 33% is far from the
  ceiling; the selection precision (28%) shows that most of the top studies are not
  tagged with the term, though their maps are often close.
* Maps correlate highly in general (a network common to fMRI studies): identification
  and the centered correlation are the measures to use, not the raw correlation.
