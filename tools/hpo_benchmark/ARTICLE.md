# Exact probabilistic diagnosis of rare diseases from noisy phenotype lists with lifted inference

*Draft technical report, 2026-11-23. Phases 1 (deterministic) and 2 (language-model arms, 60 patients). Numbers are copied from the run outputs in `results/`; none are rounded in the authors' favour. Not peer reviewed.*

## Abstract

We ask whether an exact probabilistic posterior over diseases, written as one conditional query in a probabilistic Datalog engine (NeuroLang) and solved by lifted inference alone (no weighted model counting), is a useful and tractable model for ranking rare diseases from a list of observed Human Phenotype Ontology (HPO) terms. The model is a uniform prior over 12,867 diseases and tuple-independent disease-phenotype probabilities taken from HPO annotations, plus a "leak" parameter that lets a phenotype be observed from outside the disease. On 354 real phenopackets from 139 held-out diseases, filtered to remove cases whose publication HPO already cites for the same disease, the leak model ranks the true disease first in 19.3% and within the top ten in 42.3% of 4,248 perturbed patients (mean reciprocal rank, MRR, 0.282), against 0.223 for a non-probabilistic overlap baseline (paired difference +0.059, 95% CI [+0.035, +0.088]) and 0.021 for the same model without the leak. The lifted posterior agrees with an independent numpy computation to 3.4e-13 over 300 queries covering all 12,867 diseases, each in about half a second. Propagating matches through the HPO hierarchy increased the share of matched terms (49.5% to 58.6%) but did not improve ranking on real data (paired MRR difference -0.007, [-0.017, +0.005]) and worsened it on simulated data (-0.035, [-0.045, -0.025]). In Phase 2, on 60 real patients (one per disease), a ReAct agent (Claude Sonnet 5.5) wrote the posterior query itself in 60 of 60 cases, with probabilities equal to the exact posterior (worst difference 1.4e-13), so the agent arm reproduces the symbolic ranking exactly. **Sonnet answering directly from the phenotype labels, with no database, ranked the true disease better than the symbolic system: strict-match MRR 0.430 against 0.217 (paired difference -0.213 for the agent, 95% CI [-0.322, -0.091]).** The language model may have seen the published cases, and it receives term labels while the symbolic system matches identifiers only; neither effect was separated. The whole of Phase 2 cost $2.60. Real-data accuracy of the symbolic system is modest in absolute terms, and the results are conditional on several assumptions listed in Section 7.

## 1. Introduction

Rare-disease diagnosis from a clinician's list of phenotypes is a ranking problem over thousands of candidate diseases, with uncertain evidence on both sides: annotations record how often a phenotype occurs in a disease (often from a handful of case reports), and a patient's list contains omissions and incidental findings. A probabilistic treatment is natural, but the engineering question we address is narrower: can the posterior be expressed declaratively and computed exactly by a general probabilistic logic engine, within the fragment where inference is polynomial (lifted inference), at the scale of the full HPO annotation corpus?

This report covers a benchmark comparing a symbolic system (the NeuroLang posterior) with a ReAct language-model agent that writes the query and with a language model answering without a database. Phase 1 is deterministic; Phase 2 runs the language-model arms on a 60-patient subsample, so its intervals are wide. We did not compare against existing ontology-based phenotype-matching tools; the baseline we do use (Section 3.3) is deliberately simple.

Contributions, as supported by the evidence below:
1. A declarative posterior over 12,867 diseases that lifted inference solves exactly, with measured agreement to a reference implementation (Section 5.4).
2. A leakage-controlled evaluation on real phenopackets with a development/test split by disease (Section 4).
3. A negative result on ontology-based ancestor matching (Section 5.3).
4. A comparison, on 60 real patients, with a language-model agent that writes the query and with a language model answering directly, in which the direct language model is better (Section 5.5).

## 2. Data

**Disease model.** `phenotype.hpoa` (HPO annotations, version 2026-09-02) lists disease-phenotype associations with an optional frequency. We keep phenotypic-abnormality annotations (aspect P) that are not negated and drop annotations marked as excluded, giving 267,580 disease-phenotype probabilities over 12,867 diseases (8,465 OMIM, 4,355 ORPHA, 47 DECIPHER) and 11,560 phenotypes; the median disease has 15 annotated phenotypes. Frequencies are HPO frequency classes mapped to the midpoint of their stated range, fractions n/m clipped to [0.01, 0.99], or percentages. **19% of annotations carry no frequency and are assigned 0.5; this is an assumption, not data.** Fractions such as 1/2 are small case counts and not population frequencies.

**Ontology.** `hp.json`, release 2026-09-01: 20,532 nodes and 24,436 `is_a` edges (median 9 ancestors per term). Patient terms are remapped through `replaced_by` links for obsolete terms (588 obsolete, 507 with replacements); no retained case contained an obsolete term after remapping.

**Real patients.** The phenopacket-store (release 0.1.27, BSD-3-Clause) holds 10,377 single-disease OMIM phenopackets. Each has observed (non-excluded) HPO terms.

**Licences.** The HPO licence page could not be read by our tools and `hp.json` only points to it, so **the HPO licence has not been verified** and must be checked before publication.

## 3. Methods

### 3.1 Probabilistic model

Let D be a disease drawn uniformly from the N = 12,867 candidates, and let a patient show terms O = {t_1, ..., t_k}. For each disease d and term t let p(d, t) be the annotation probability (0 if unannotated). Assuming terms are conditionally independent given the disease (naive Bayes),

P(d | O) = prod_j q(d, t_j) / sum_e prod_j q(e, t_j),

where q(d, t) = 1 - (1 - p(d, t))(1 - ε). The **leak** ε is the probability that a phenotype appears although the disease does not explain it; ε = 0 is the plain model.

In NeuroLang, `D` is a probabilistic choice (exactly one disease is true) and `H(d, s)` a tuple-independent relation holding q. The posterior is the conditional query

```
post(d, PROB) :- (D(d) & H(d, s1) & ... & H(d, sk)) // (D(e) & H(e, s1) & ... & H(e, sk))
```

Because `H` is independent of `D`, the numerator is the product above divided by N, and the denominator sums the same quantity over e; the query therefore equals the formula, and it is solved by the small-dichotomy and Dalvi-Suciu lifted solvers, with the weighted-model-counting solver removed from the engine so that any query lifting cannot solve raises an error instead of falling back.

*Implementation note.* The leak is applied per patient: a relation holding q for that patient's k observed terms over all diseases (12,867 x k tuples) is registered before the query. A purely declarative formulation, a union rule `explained(d,s) :- H(d,s)` and `explained(d,s) :- Dis(d), Leak(s)`, timed out (180 s) or raised a recursion error in 4 of 4 attempts in an earlier probe and was abandoned. The leak therefore lives in a data-preparation step, not in the logic program.

### 3.2 Ancestor propagation

A patient term t is matched to an annotation s when s is an `is_a` ancestor of t within distance k_up (the annotation is more general) or a descendant within k_down (the patient term is more general), excluding the root terms HP:0000001 and HP:0000118. The matched probability is γ · p(d, s), and q takes the maximum over exact and matched values before the leak is applied. Parameters (k_up, k_down, γ) are searched jointly with ε (below).

### 3.3 Overlap baseline

A non-probabilistic ranking: score(d) = sum over observed terms of the annotation frequency p(d, t) (0 if absent), with ties broken by the number of matched terms. Ties among diseases are resolved by their mean rank. This is the simplest control for "what do the probabilities add", not a state-of-the-art matcher.

### 3.4 Noise protocol

Each base patient is perturbed in a 4 x 3 grid, with the same seeds for every system: **spurious terms** (0, 1, 2 or 4 terms drawn uniformly from the 11,560 annotated phenotypes) crossed with **dropped true terms** (0%, 25% or 50% of the observed terms, rounded, always keeping at least two). Real cases already contain unannotated terms (Section 4), so the "0 spurious, 0% dropped" cell is not noise-free.

### 3.5 Metrics and statistics

For each patient we record the rank of the true disease (ties share their mean rank). We report top-1, top-10, mean reciprocal rank (MRR) and the **empty rate**, the share of patients for which a system returns no supported disease (all posteriors zero). An empty posterior and a true disease absent from the candidate list both count as a miss (reciprocal rank 0). Confidence intervals are percentile bootstrap intervals that resample diseases, not patients, because cases of one disease are correlated (1,000 resamples for paired differences, 300 for single-system MRR). Paired differences use identical perturbed patients.

### 3.6 Language-model arms (Phase 2)

Both arms use Claude Sonnet 5.5 (`claude-sonnet-5-5`) through the Inria API key, with no temperature setting (the model rejects it).

**ReAct agent with NeuroLang.** The repository's `ReactAgent` (Datalog, at most 10 steps, no few-shot examples, no verification step) works on a fresh engine per patient holding two relations: `Prior(disease)`, a uniform probabilistic choice over the 12,867 diseases, and `Evidence(disease, term)`, the smoothed probabilities q(d, t) (leak ε = 0.005, the exact-model setting chosen on development) of that patient's observed terms for every disease. The agent is not given the raw annotation corpus. The question lists the patient's terms (identifier and label) and asks it to rank diseases by posterior probability. The guidance text explains the two relations and the syntax of probability markers, including a **generic** conditional-query example over unrelated predicates (`ans(x, PROB) :- (R(x) & S(x)) // (T(y))`); it does not give the posterior query. The leak is applied by the harness, not by the agent. The agent's final query is re-run by us, and its output is ranked by the last numeric column.

**Sonnet direct.** The model receives the patient's terms as `HP:id: label` lines and is asked for at most 10 most likely diseases as JSON objects with a name and an OMIM identifier. No database, no annotation data. An answer counts as correct under a **strict** rule when its OMIM identifier equals the true disease or its normalised name equals the true disease's HPOA name. We also report a **lenient** rule (name similarity >= 0.85, Python `difflib`), which turned out to credit sibling subtypes (Section 5.5), and apply the same lenient rule to the symbolic systems for a like-for-like comparison.

**Subsample and cost control.** Sixty test cases were drawn, one per disease from distinct candidate diseases (seed 0), and each assigned one of four noise cells (spurious terms 0 or 2 crossed with 0% or 50% dropped), 15 patients per cell. The perturbation seeds are those of Phase 1, so the symbolic and baseline ranks come from the same patients. A hard spend cap of $60 was coded (upper-bound cost, billing uncached input at the cache-write rate), after a three-patient pilot measured $0.04 per patient; the cap was never approached.

## 4. Evaluation design

### 4.1 Leakage control

HPO frequencies are partly derived from published case reports, which the phenopacket-store also contains. A phenopacket is **removed if its PMID (from the file name) is cited by HPOA for the same disease**. This removes 7,720 of 10,377 (74%), leaving 2,656 with phenotypic features, over 201 diseases, with strong skew (one disease has 462 cases, another 333). Residual leakage is possible: annotations with other PMIDs from the same authors, or with no PMID.

We then keep cases with at least 3 observed terms and at most 5 cases per disease (selected by a fixed seed), giving 538 cases over 198 diseases. Measured on the 2,656 filtered cases before this selection (median 8 observed terms), **on average 51% of a case's observed terms are absent from its own disease's annotation under exact matching, and 89% of cases have at least one such term** (16% absent, on the 10,377 unfiltered cases): even before perturbation, the evidence overlaps the model only partly. We did not recompute these two figures on the final 538.

### 4.2 Development and test split

The 198 diseases are split once, by a fixed seed, into a development set (59 diseases, 184 cases) and a test set (139 diseases, 354 cases). ε, γ, k_up and k_down were chosen on development only; every number in Section 5 is on the test set. **All 10 test cases of two diseases (5 each) have a true disease with no usable annotation, so it is not among the 12,867 candidates; they are counted as misses and cap test top-k at 97.2%.** The development split contains no such case.

### 4.3 Simulated set

Two hundred patients are drawn from HPOA itself: a disease uniformly among those with at least 6 annotated phenotypes, each phenotype present with its annotation probability, then 2, 3 or 4 present terms are observed, with the same noise grid. This set has known truth at every noise level but is **circular**: the model is built from the same data, so it checks the machinery and shows how noise degrades it, and says nothing about diagnostic accuracy.

### 4.4 Parameter selection

On the development set (184 cases x 12 noise cells = 2,208 perturbed patients) we searched ε in {0.005, 0.02, 0.05} and, for the ancestor variant, (k_up, k_down) in {(1,0), (1,1), (2,2)} with γ in {0.25, 0.5, 1.0}. The criterion was MRR. The best exact model used ε = 0.005 (development MRR 0.2420); the best ancestor model used k_up = 1, k_down = 1, γ = 0.25, ε = 0.02 (0.2421). The two are indistinguishable on development; the whole grid is small and its differences between neighbouring settings range from about 0.001 to 0.01. Without a leak the exact plain model is also reported (ε = 0).

## 5. Results (held-out test diseases)

Real set: 354 base patients, 139 diseases, 4,248 perturbed patients. Simulated: 200 base patients (199 diseases), 2,400 perturbed. Intervals in brackets are 95% bootstrap intervals over diseases.

### 5.1 Real phenopackets

| Condition (n patients) | System | Top-1 | Top-10 | MRR [95% CI] | Empty |
|---|---|---|---|---|---|
| All noise cells (4,248) | Exact, no leak | 1.7% | 2.8% | 0.021 [0.013, 0.030] | 91.7% |
| | Exact, leak | 19.3% | 42.3% | 0.282 [0.220, 0.345] | 0% |
| | Ancestor, leak | 18.8% | 42.6% | 0.275 [0.213, 0.335] | 0% |
| | Overlap baseline | 14.8% | 35.0% | 0.223 [0.165, 0.281] | 0% |
| 0 spurious, 0% dropped (354) | Exact, no leak | 5.4% | 8.2% | 0.065 [0.035, 0.100] | 78.0% |
| | Exact, leak | 22.6% | 47.2% | 0.323 [0.258, 0.392] | 0% |
| | Ancestor, leak | 22.3% | 47.5% | 0.315 [0.245, 0.385] | 0% |
| | Overlap baseline | 17.2% | 40.7% | 0.254 [0.191, 0.317] | 0% |
| 4 spurious (354) | Exact, no leak | 0.0% | 0.0% | 0.000 | 100% |
| | Exact, leak | 22.3% | 46.9% | 0.319 [0.254, 0.391] | 0% |
| | Ancestor, leak | 20.9% | 45.5% | 0.300 [0.232, 0.373] | 0% |
| | Overlap baseline | 16.9% | 39.0% | 0.250 [0.185, 0.314] | 0% |
| 50% dropped (354) | Exact, no leak | 7.1% | 13.8% | 0.096 [0.062, 0.131] | 57.6% |
| | Exact, leak | 15.3% | 36.4% | 0.227 [0.172, 0.282] | 0% |
| | Ancestor, leak | 15.8% | 36.7% | 0.233 [0.172, 0.291] | 0% |
| | Overlap baseline | 11.6% | 29.9% | 0.184 [0.132, 0.236] | 0% |
| 4 spurious + 50% dropped (354) | Exact, no leak | 0.0% | 0.0% | 0.000 | 100% |
| | Exact, leak | 12.1% | 38.4% | 0.220 [0.168, 0.275] | 0% |
| | Ancestor, leak | 12.7% | 38.4% | 0.217 [0.164, 0.272] | 0% |
| | Overlap baseline | 10.2% | 30.2% | 0.179 [0.123, 0.237] | 0% |

Paired MRR differences over all cells: **leak minus overlap +0.059 [+0.035, +0.088]; leak minus no leak +0.261 [+0.208, +0.319]; ancestor minus exact leak -0.007 [-0.017, +0.005].**

Observations:
- Without a leak the model is unusable on real data: 91.7% of patients receive an empty posterior, because a disease missing any single observed term gets probability zero. This follows from the model and is expected; real cases already carry unannotated terms.
- With a leak, adding up to four random terms barely changes the real-data ranking (MRR 0.323 to 0.319 with four spurious terms, 354 patients each), while dropping half of the true terms lowers it to 0.227. Our interpretation, which we did not test, is that the leak absorbs unexplained terms equally for every disease, and that random terms add little to a list in which about half of the terms are already unexplained; dropped true terms remove evidence.
- The overlap baseline is 0.059 MRR lower. It is a simple control, so this does not bound what other matchers achieve.
- Absolute accuracy is modest: the true disease is first in about one patient in five and in the top ten in about two in five, in a test set where 2.8% of cases cannot be ranked at all.

### 5.2 Simulated patients

| Condition (n patients) | System | Top-1 | Top-10 | MRR [95% CI] | Empty |
|---|---|---|---|---|---|
| All noise cells (2,400) | Exact, no leak | 12.1% | 20.8% | 0.154 [0.141, 0.165] | 74.0% |
| | Exact, leak | 44.1% | 81.6% | 0.581 [0.533, 0.623] | 0% |
| | Ancestor, leak | 41.2% | 78.0% | 0.546 [0.497, 0.591] | 0% |
| | Overlap baseline | 35.6% | 70.9% | 0.488 [0.434, 0.534] | 0% |
| 0 spurious, 0% dropped (200) | Exact, no leak | 62.5% | 90.0% | 0.730 | 0% |
| | Exact, leak | 61.5% | 90.0% | 0.723 | 0% |
| 4 spurious + 50% dropped (200) | Exact, no leak | 0.0% | 0.0% | 0.000 | 100% |
| | Exact, leak | 25.0% | 74.0% | 0.435 [0.386, 0.487] | 0% |
| | Ancestor, leak | 21.0% | 64.5% | 0.373 [0.324, 0.425] | 0% |
| | Overlap baseline | 23.0% | 64.0% | 0.386 [0.338, 0.437] | 0% |

Paired MRR differences: leak minus overlap +0.093 [+0.068, +0.117]; leak minus no leak +0.427 [+0.393, +0.459]; ancestor minus exact leak -0.035 [-0.045, -0.025]. In the noise-free simulated cell the leak is not needed (no leak MRR 0.730, leak 0.723): the leak matters only once terms outside the disease appear. Remember that these patients are generated from the model's own data.

### 5.3 Ancestor propagation

Ancestor matching raises the share of observed terms that match an annotation from 49.5% (exact) to 54.9% (ancestors only) and 58.6% (both directions, unlimited distance), and the share of cases with every term matched from 9.1% to 14.9% (538 cases, before the split). Ranking does not follow. On held-out real data the ancestor model's MRR is statistically indistinguishable from the exact leak model (-0.007, CI spanning zero); on simulated data it is worse (-0.035, CI excluding zero). On the development grid, MRR fell as γ and the distances grew (for example 0.2420 for exact at ε = 0.005 against 0.1747 at k_up = k_down = 2 and γ = 1.0, also at ε = 0.005). A plausible reason, untested, is that general ancestor terms match many diseases and add support that does not discriminate. The result is conditional on the discount form (a single γ per match), the exclusion of the two root terms, and the small search grid. Different ontology-aware schemes, for example matching weighted by information content, were not tried.

### 5.4 Engine check and cost

For 150 randomly drawn test patients, each in the exact-leak and the ancestor-leak model (300 queries), the lifted posterior over all 12,867 diseases was compared with the numpy computation of the same formula: all 300 queries succeeded, with the same support, and the **maximum absolute difference was 3.4e-13**. Queries took a median of 0.55 s and at most 1.1 s, including registering the per-patient relation. Resident memory grew from about 0.4 GB (data loaded) to 2.3 GB over the 300 registered relations, because the per-patient relations are not released; a long-running service would need to drop them. The comparison validates the engine against our own reference implementation of the same model; it does not validate the model.

### 5.5 Language-model arms (60 patients, 60 diseases)

Ranks of the true disease on the same 60 perturbed patients. Intervals are 95% bootstrap intervals over patients (equal to diseases here, one patient per disease). The ReAct agent's rows are identical to the NeuroLang posterior's, because the agent wrote an equivalent query (below), so the two are shown once.

| Condition (n) | System | Top-1 | Top-10 | MRR [95% CI] |
|---|---|---|---|---|
| All (60) | NeuroLang posterior = ReAct agent | 11.7% | 41.7% | 0.217 [0.137, 0.304] |
| | Overlap baseline | 10.0% | 36.7% | 0.194 [0.117, 0.279] |
| | Sonnet direct, strict match | 33.3% | 61.7% | 0.430 [0.316, 0.541] |
| | Sonnet direct, lenient match | 45.0% | 73.3% | 0.549 [0.435, 0.656] |
| 0 spurious, 0% dropped (15) | NeuroLang posterior = agent | 40.0% | 80.0% | 0.532 [0.323, 0.737] |
| | Sonnet direct, strict | 53.3% | 80.0% | 0.650 [0.433, 0.850] |
| 2 spurious, 0% dropped (15) | NeuroLang posterior = agent | 0.0% | 26.7% | 0.100 [0.031, 0.184] |
| | Sonnet direct, strict | 40.0% | 73.3% | 0.497 [0.280, 0.717] |
| 0 spurious, 50% dropped (15) | NeuroLang posterior = agent | 0.0% | 33.3% | 0.096 [0.031, 0.180] |
| | Sonnet direct, strict | 26.7% | 60.0% | 0.376 [0.171, 0.589] |
| 2 spurious, 50% dropped (15) | NeuroLang posterior = agent | 6.7% | 26.7% | 0.139 [0.033, 0.288] |
| | Sonnet direct, strict | 13.3% | 33.3% | 0.197 [0.043, 0.397] |

Paired MRR differences over all 60: ReAct agent minus Sonnet direct (strict) **-0.213 [-0.322, -0.091]**; agent minus Sonnet direct (lenient) -0.332 [-0.444, -0.212]; agent minus overlap baseline +0.023 [-0.009, +0.065].

**The agent arm.** In 60 of 60 patients the agent produced a valid final answer, within 10 steps (median 4 calls, maximum 9; median 20.6 s per patient), with a probability column; its ranking of the true disease equalled the reference rank in 60 of 60, and its probabilities matched the exact posterior with a worst absolute difference of 1.4e-13 (60 of 60 below 1e-6). In the three queries we inspected in the pilot, the agent wrote either the conditional form `PROB[match(d) // any_match()]` or the joint conditioned on the evidence, i.e. the posterior, after being shown only the generic syntax; we did not inspect all 60 queries. The agent therefore adds no error here and no accuracy: its result is the engine's. Its cost is in tokens: 973,801 input (793,729 served from the cache) and 156,103 output over 60 patients.

**Strict versus lenient matching.** The lenient rule changes the result for 15 of 60 patients, and in all 15 listed cases the model's matched answer is a **sibling subtype of the true disease, not the true disease** (for example "Noonan syndrome 1" for "Noonan syndrome 2", "Retinitis pigmentosa 4" for "Retinitis pigmentosa 18", "Loeys-Dietz syndrome 1" for 4). It therefore measures recovering the syndrome family, and we use the strict rule as the headline. For a like-for-like family-level comparison we applied the same lenient rule to the symbolic systems (any candidate disease with name similarity >= 0.85 to the true one counts, best rank taken; median family size 4.5, 11 of 60 patients have a family of one): posterior MRR 0.351 (top-1 23.3%, top-10 60.0%), overlap 0.328, Sonnet direct 0.549 (top-1 45.0%, top-10 73.3%); paired difference Sonnet minus posterior **+0.197 [+0.073, +0.317]**. The gap narrows slightly and stays significant. The language model's answer list is capped at 10 while the symbolic ranking is not, which if anything favours the symbolic side in MRR.

**Sonnet direct** returned a parseable answer for all 60 patients (median 10 answers); 17 of its 584 answered names matched neither a candidate disease name nor identifier, which does not distinguish between a hallucinated disease and a naming difference.

**Cost.** The whole of Phase 2 cost $2.60 as an upper estimate (agent about $2.09, direct about $0.51): $0.04 per patient. Prices used: $2 per million input tokens, $0.10 per million cached tokens, $10 per million output tokens (Anthropic's Sonnet 5.5 price page); uncached input was billed at the cache-write price of $2.50, so the figure is slightly high.

### 5.6 Does more evidence close the gap? (post hoc)

Asked after Phase 2, so this is an exploratory analysis, not part of the pre-specified design. The disease model is enriched with all phenopackets that are usable as training data (10,345 cases over 778 candidate diseases, including the cases the leakage filter removes from the *test* set): p(d, t) = (n_dt + a * p_HPOA(d, t)) / (N_d + a), where N_d cases of d show n_dt times the term t. For a held-out case, **every case from its own publication (PMID) is removed** from its disease's counts, and the disease falls back to HPOA if none remain. a = 8 and eps = 0.005 were chosen on the development diseases (grid a in {0.5, 2, 8, 32} x eps in {0.005, 0.02}; development MRR 0.302 against 0.242 for HPOA alone). Test: the same 354 cases and perturbed patients as Section 5.1.

| Condition | HPOA only, MRR [95% CI] | Enriched, MRR [95% CI] |
|---|---|---|
| All cells (4,248 patients) | 0.282 [0.220, 0.345] | **0.342 [0.284, 0.402]** (top-1 25.2%, top-10 50.8%) |
| 0 spurious, 0% dropped | 0.323 | 0.409 |
| 50% dropped | 0.227 | 0.280 |

Paired difference over all cells: **+0.060 [+0.034, +0.088]**. The gain comes from diseases that have other publications: for the 208 of 354 base cases with at least one other-publication case, MRR rises from 0.337 to 0.431; for the 146 without any, from 0.203 to 0.214 (HPOA fallback; the small change is due to enriched competitors).

On the 60 Phase 2 patients: enriched MRR 0.287 [0.192, 0.381] (HPOA only 0.217; paired +0.070 [+0.017, +0.131]), against **Sonnet direct 0.430; paired difference -0.143 [-0.258, -0.028]. The language model still leads overall.** Split by the training evidence of the true disease (n of the 60 patients): 0 other-publication cases (34): enriched 0.169, Sonnet 0.416; 1-9 cases (15): 0.382 against 0.483; at least 10 cases (11): 0.522 against 0.400. The last row is 11 patients and no interval was computed; it is an observation, not a result.

Limits of this analysis: it tests only the case where the true disease has phenopackets (every test disease does, 778 of 12,867 candidate diseases do), so the other 12,000 candidates keep HPOA-only models and the evaluation is not representative of a disease without case data; residual leakage remains possible because 7,203 distinct subject identifiers appear across 10,377 phenopackets, so the same individual may be described under different PMIDs; unlisted terms are treated as absent in the counts; and the design (a, eps) was selected after seeing Phase 2.

### 5.7 Is the posterior calibrated? (post hoc)

Everything above scores *ranks*. Rank is unaffected by calibration, so we measured it separately. Test set, perturbed patients whose true disease is a candidate (4,128 patients; `hpo_calibration.py`). Top-label calibration: among patients where the top-ranked disease has posterior probability p, how often is it correct?

| Posterior of the top disease | Patients | Mean confidence | Top-1 correct (HPOA only) | Top-1 correct (enriched) |
|---|---|---|---|---|
| [0.00, 0.10) | 363 | 0.048 | 0.028 | 0.054 |
| [0.10, 0.25) | 490 | 0.179 | 0.047 | 0.078 |
| [0.25, 0.50) | 829 | 0.374 | 0.138 | 0.141 |
| [0.50, 0.75) | 773 | 0.619 | 0.140 | 0.184 |
| [0.75, 0.90) | 467 | 0.833 | 0.191 | 0.278 |
| [0.90, 1.00) | 1,206 | 0.979 | 0.453 | 0.520 |

(Bin counts are for the HPOA-only model; the enriched model has similar counts.) Overall, HPOA only: mean top-1 confidence 0.597 against top-1 accuracy 0.216, **expected calibration error 0.381**; enriched: 0.597 against 0.261, ECE 0.337. **The posterior is strongly overconfident**: a top disease given probability above 0.9 (29% of patients) is right 45% of the time (52% enriched), and for 22.5% of patients (HPOA only; 18.3% enriched) the true disease gets a probability below 1e-6, a catastrophic assignment for a probability a clinician might read. Mean log-loss of the true disease is 8.55 (7.21 enriched). Calibration is no better in the clean cell (ECE 0.442, HPOA only), so noise is not the cause.

*Control.* On 2,000 simulated clean patients with the plain model, which approximately matches how those patients were generated, calibration is much better (ECE 0.047, top-1 accuracy 0.652 against mean confidence 0.699), though not perfect: the 0.75-0.90 bin is right 60% of the time. We believe, without testing it, that the simulator's choice of observed terms among present ones differs slightly from the model's independence assumption. The contrast supports that the miscalibration on real patients comes from model misspecification (independence between terms, small-count and defaulted frequencies, unannotated terms, uniform prior), not from the engine.

*Simple repair.* A single temperature T (posterior proportional to the likelihood to the power 1/T), fitted on the development diseases by log-loss (T = 4), gives on the test set mean confidence 0.097 against accuracy 0.216, ECE 0.134 and log-loss 6.22, against 0.597, 0.216, 0.381 and 8.55 at T = 1. It is better but now underconfident, and the log-loss remains high; one global temperature does not fix it, because the miscalibration depends on how many terms a patient has and how many are unannotated. We did not try richer recalibration.

*What this does and does not say.* Two layers must be separated. **Propagation** is exact: given the input probabilities, the lifted posterior equals an independent computation to 3.4e-13 (Section 5.4), and in UKGE (probe report, section 2) it matches direct enumeration to 2e-16. **Calibration** is not: the output probabilities are only as meaningful as the inputs and the model, and here they are not meaningful as frequencies. Two related earlier observations point the same way: on the planted-noise MetaQA data the best answer threshold was 0.2-0.5, never uniformly 0.5; we did not test calibration there or on UKGE. Sonnet direct returns a ranked list without probabilities, so no calibration comparison with it is possible, and the ReAct agent reproduces the same (miscalibrated) posterior.

## 6. Discussion

The central finding is that the posterior is cheap and exact in the lifted-only fragment, even with all 12,867 diseases as candidates, and that the leak parameter, not the ontology, is what makes it usable on real data. The real-data accuracy is, however, limited by the data and not only by the model: half of the observed terms of a typical case are absent from its own disease's annotation, so even a correct model sees partial evidence, and ontology propagation did not repair it.

The language-model comparison gives a result opposite to the hypothesis that motivated it. On 60 real patients a language model with no database ranked the true disease better than the exact symbolic posterior (strict MRR 0.430 against 0.217, top-1 33.3% against 11.7%), and the gap was largest where noise was added (for example two spurious terms: 0.497 against 0.100). We did not test why. Three candidate explanations are all consistent with the data: (i) the model brings clinical knowledge beyond the HPO annotations, which matters because about half of a case's terms are not in its disease's annotation; (ii) the model has seen the published cases, which the leakage filter cannot control for since the filter concerns the annotation corpus, not the model's training data; (iii) the model receives term labels while the symbolic system matches identifiers exactly, so it can generalise across related terms in a way the ancestor variant did not. The comparison therefore does not separate reasoning from recall, and should not be read as showing that language models diagnose better.

Two further points. The ReAct agent matched the exact posterior in 60 of 60 patients, so for this task the open question is not whether an agent can write the query (it can, given a syntax description and the harness-applied leak) but whether the posterior is the right model. And with 60 patients, differences between cells (15 patients each) are mostly unresolvable; the overall differences are what the data support.

## 7. Limitations and threats to validity

- **Leakage** is controlled only through documented PMIDs; related annotations may remain, which would inflate real-data accuracy.
- **Exact term matching** is the main modelling restriction; the ancestor variant is a single simple scheme.
- **Defaulted frequencies** (19% of annotations at 0.5) and small-count fractions affect q directly. We did not run a sensitivity analysis on them.
- **Independence and uniform prior.** Terms are treated as conditionally independent, and all diseases as equally likely a priori; real prevalence differs by orders of magnitude.
- **Narrow test slice.** The test set covers 139 diseases (of 12,867 candidates), with at most 5 cases per disease, and 10 of 354 cases are unrankable.
- **Parameter selection** used a small grid on 59 development diseases; ε was chosen between 0.005 and 0.02 on differences of about 0.001 MRR.
- **Noise protocol.** Spurious terms are uniformly random and unrelated to any disease, which is easy to absorb; clinically plausible confounders (terms of a co-occurring disease) were not tested.
- **Simulated set** is circular and tests the machinery only.
- **No comparison with established phenotype-matching tools**, and the overlap baseline is deliberately simple.
- **Phase 2 is small and uncontrolled for memorisation.** Sixty patients (15 per cell, one per disease), one run each, one model, one prompt per arm, no repeated sampling. Sonnet may have encountered the published cases; we have no test for it (for example cases published after its training cutoff, which we did not identify).
- **Asymmetric inputs.** The direct arm sees term labels; the symbolic arm and the agent see identifiers only. The agent's guidance includes a generic conditional-query example; a different amount of guidance might change the agent's results.
- **Name matching.** The strict rule can undercount correct answers phrased differently, and the lenient rule credits sibling subtypes (Section 5.5). Neither is a clinical judgement.
- **Calibration** was measured post hoc and only on HPO (Section 5.7); a non-monotone repair or per-patient-size temperature was not tried.
- **A hybrid** (the language model re-ranking the symbolic posterior's top candidates) was not tested.
- **Licensing:** HPO licence unverified; phenopacket-store is BSD-3-Clause.
- Engine memory grows with each registered per-patient relation (Section 5.4).

## 8. Reproducibility

Code is in `tools/hpo_benchmark/` (run every script from that directory); the data are not in the repository (see `README.md`).

| Step | Script | Output |
|---|---|---|
| Data, ontology, case filtering | `hpo_lib.py`, `hpo_onto.py`, `hpo_cases.py` | n/a |
| Models, noise, scoring | `hpo_bench.py` | n/a |
| Development grid | `hpo_dev.py` | `results/hpo_split.json`, `results/hpo_chosen.json` |
| Held-out run | `hpo_phase1.py` | `results/hpo_phase1.jsonl` (6,648 records) |
| Tables and bootstrap | `hpo_summary.py` | printed |
| Engine check | `hpo_lifted_check.py` | `results/hpo_lifted_check.jsonl` (300 records) |
| Phase 2 (agent + direct) | `step9_hpo_phase2.py`, `hpo_names.py` | `results/hpo_phase2.jsonl` (60 records), `results/hpo_phase2_plan.json`; 3-patient pilot in `results/hpo_phase2_pilot.jsonl` |
| Phase 2 tables | `hpo_phase2_summary.py`, `hpo_phase2_family.py` | printed; `results/hpo_phase2_family.json` |
| Enriched model (post hoc) | `hpo_enriched.py`, `hpo_enriched_test.py` | `results/hpo_enriched_chosen.json`, `results/hpo_enriched_test.json` |
| Calibration (post hoc) | `hpo_calibration.py`, `hpo_calibration_T.py` | `results/hpo_calib_real_*.json`; temperature results printed |

Phase 2 needs `ANTHROPIC_API_INRIA_KEY` and runs with `uv run --package neurolang-apps --extra llm`. All random choices use fixed seeds (split seed 0, case selection seed 0, simulated patients seed 123, perturbation seeds derived from set, patient and noise cell). Data: HPO annotations 2026-09-02, `hp.json` 2026-09-01, phenopacket-store 0.1.27.
