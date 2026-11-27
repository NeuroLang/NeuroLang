# HPO hybrid: does Sonnet do better with the symbolic posterior's candidates? (written before the run, 2026-11-24)

Context: on 60 real patients, Sonnet answering from the phenotypes alone ranked the true disease better than the exact symbolic posterior
(strict MRR 0.430 against 0.217). Hypothesis to test: the two are complementary: the posterior can propose candidates from all 12,867 diseases,
Sonnet can rank them using its clinical knowledge.

## Patients
100 fresh patients from the 354 held-out test cases of the article (no phenopacket among the 60 patients of Phase 2), one per disease, chosen at
random (seed 11) with the true disease a candidate of the 12,867; 25 in each of the 4 noise cells of Phase 2 (0 or 2 spurious terms x 0% or 50% dropped),
with Phase 1's perturbation seeds.

## Candidates
The K = 30 diseases with the highest posterior of the exact model with leak (the configuration chosen on the development diseases:
`results/hpo_chosen.json`, computed by the numpy reference, which the lifted NeuroLang query matched to 3.4e-13 on 300 queries). Recall@30 is reported.

## Arms (all Sonnet 5.5, Inria key, one answer per patient; the answer is a ranked list of at most 10)
| Arm | What Sonnet gets |
|---|---|
| H0 direct | the patient's terms (id and label), no candidates; the answer is names and OMIM ids (strict match, as before) |
| H1 candidates | H0 plus the 30 candidates (OMIM id and name), in a random order, no numbers |
| H2 candidates + probabilities | H1 plus each candidate's posterior probability (percent), same random order |
| H3 candidates ranked | the 30 candidates ordered by posterior, no numbers |
References: the posterior's own ranking; recall@30 (the ceiling of H1 to H3).

## Metrics
Rank of the true disease in the answer (a miss if absent), top-1, top-10, MRR; strict matching by OMIM id or normalised name; an answer outside the candidates in
H1 to H3 is ignored. 95% intervals by bootstrap over patients (10,000), paired differences between arms.

## Reading
- H1 (or H2, H3) better than H0 with an interval excluding 0: the candidates help; the posterior is a useful retrieval step.
- H2 against H1: the value of the probabilities; H3 against H1: the value of the rank order. The posterior is overconfident (ECE 0.38): if H2 is worse than H1,
  the numbers misled the model, which is a result about calibration, not about the engine.
- The hybrid beats the posterior alone if its MRR is higher with an interval excluding 0.
- No arm is rerun; all are reported. The previous 60 patients are not mixed in.
