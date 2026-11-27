# HPO hybrid: results (protocol: `PROTOCOL_HYBRID.md`, written before the run)

100 fresh real patients (100 diseases, 25 in each of 4 noise cells), K = 30 candidates from the exact posterior with leak (numpy reference).
**Recall@30: 54%** (54 true diseases among the candidates, 46 outside). All answers parsed. Code: `step14_hpo_hybrid.py`; per-patient results `results/hybrid_results.json`.

| Arm | Top-1 | Top-10 | MRR [95% CI over patients] |
|---|---|---|---|
| posterior alone (its ranking) | 25.0% | 43.0% | 0.328 [0.248, 0.411] |
| H0 Sonnet direct | **37.0%** | **58.0%** | **0.428** [0.340, 0.519] |
| H1 candidates, random order, no numbers | 21.0% | 47.0% | 0.293 [0.219, 0.373] |
| H2 candidates + posterior probabilities | 27.0% | 50.0% | 0.353 [0.271, 0.438] |
| H3 candidates in posterior order, no numbers | 27.0% | 51.0% | 0.349 [0.269, 0.436] |

Paired MRR differences: H1 - H0 -0.135 [-0.221, -0.052]; H2 - H0 -0.075 [-0.166, +0.012]; H3 - H0 -0.079 [-0.168, +0.008];
**H2 - H1 +0.060 [+0.001, +0.120]; H3 - H1 +0.056 [+0.006, +0.112]**; H2 - posterior +0.025 [+0.002, +0.056]; H1 and H3 against the posterior include 0.

Reading, by the rules fixed beforehand:
- **The candidates do not help Sonnet**: no hybrid arm beats direct answering (H1 is significantly worse). The hypothesis that the posterior is a useful retrieval step for Sonnet is not supported.
- **The probabilities and the rank order do help the hybrid** (H2 and H3 over H1, intervals just excluding 0): it is the order of the posterior, not the numbers, that
  carries the information (H2 about H3). The overconfidence of the posterior (ECE 0.38) did not mislead the model here.
- The hybrid with probabilities improves slightly on the posterior alone (+0.025), by reordering the candidates.
- The ceiling is recall: with 54% of the true diseases among the 30 candidates, a hybrid restricted to them cannot reach the 58% top-10 of direct answering.
  Of the 46 diseases outside the candidates, Sonnet's direct answer has 19 in its top 10 (10 at rank 1); there H1 to H3 score 0 by construction.

Exploratory (not in the protocol; the split depends on the posterior only, not on the LLM): among the 54 patients whose disease is a candidate, MRR is posterior 0.598,
H0 0.566, H1 0.543, **H2 0.654, H3 0.647**: restricted to its candidates, Sonnet with the posterior's information is the best of the arms, by amounts (0.05 to 0.09)
that 54 patients cannot separate.

Not tested: a hybrid that unions Sonnet's own list with the posterior's candidates. The two retrievals are complementary: 73 of the 100 true diseases are in the
posterior's 30 candidates or in Sonnet's direct top 10 (54 and 58 each); a larger K. Limits: 100 patients, one model, one prompt per arm, no repeats; Sonnet may have seen the published cases.

## Post hoc: where does Sonnet beat the posterior? (`post_hoc_coverage.py`, `results/post_hoc_coverage.txt`)

Decided after the results above, so exploratory. The 160 patients of phase 2 and of this hybrid. **Coverage** is the share of the patient's observed
terms (before perturbation) that the true disease's HPO annotation lists: how much of the evidence the posterior's model can use. It depends on the
patient and the disease only, not on either system's output, but it uses the true disease, so it cannot be computed at test time.

| Coverage tertile | Patients | Posterior MRR | Sonnet direct MRR | Sonnet - posterior [95% CI] |
|---|---|---|---|---|
| low (0.00-0.38) | 56 | 0.040 | 0.295 | +0.255 [+0.148, +0.368] |
| middle (0.40-0.65) | 51 | 0.221 | 0.424 | +0.203 [+0.070, +0.332] |
| high (0.67-1.00) | 53 | 0.609 | 0.575 | -0.035 [-0.162, +0.093] |

Coverage correlates 0.62 with the posterior's reciprocal rank, and 0.22 with Sonnet's.

Reading: Sonnet's advantage is concentrated where the annotation covers little of the patient's evidence. Where it covers most of it, the two are
equal. This is a knowledge-coverage gap, not a reasoning gap. Sonnet brings knowledge (or memory of published cases) that the annotation lacks,
and these data cannot separate the two.
