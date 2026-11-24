# Does the probability help? Replication prediction on the SMR evidence

Written before the run (2026-11-24). BiomedSQL's questions are deterministic ("is there a significant SMR signal for gene X?"),
so they cannot show a benefit of uncertainty. This experiment asks the question a scientist asks next: *a gene is significant in
some QTL datasets; will it be in a new one?* The answer is known for every item: the held-out dataset's own test.

## Items

240 items (g, d, h): a gene g, a disease d and a QTL dataset h such that h and at least one other dataset tested g for d
and at least one other dataset found it significant (the 3,960 predictions with support > 0 of `tools/biomedsql_calibration.py`,
whose base rate is 38%). Drawn at random, seed 0, one item per (gene, disease), the held-out dataset drawn uniformly among
those that tested it. Outcome y = 1 if h's test of g for d is significant (p_SMR_multi < 2.95e-6, BiomedSQL's threshold).
Items are clustered by gene; intervals resample genes.

Question given to every system: "Gene G was tested as a mediator of DISEASE in several QTL datasets. Using only the datasets
other than H (a TISSUE KIND dataset), what is the probability that H finds G significant (p_SMR_multi < 2.95e-6)?"

## Systems (each outputs a probability, or a yes/no for D1)

| System | What it has | How it answers |
|---|---|---|
| D1 any signal | the other datasets | BiomedSQL's yes/no: 1 if some other dataset found g significant (so 1 on every item) |
| D2 majority | the other datasets | 1 if at least half of those that tested it found it significant, else 0 |
| L1 LLM, no data | its own knowledge | Sonnet 5.5, a probability |
| L2 LLM + evidence | the table of the other datasets' tests (dataset, tissue, kind, p_SMR_multi, significant?) | Sonnet 5.5, a probability |
| P NeuroLang | the QTL tokens without h (`qtl_tested`, `qtl_significant`, `selected_qtl`), the engine's guidance | a ReAct agent (Sonnet 5.5) writes the conditional query; its number is the answer |

P's number is the share of the other datasets that tested g for d and found it significant, computed exactly by lifted
inference (no model counting). The agent is checked against the direct computation (`tools/biomedsql_calibration.py`).
L2 sees strictly more than P (p-values, tissues), so it is the strong neural competitor, not a straw man.

## Metrics (fixed)

Probabilities are clipped to [0.02, 0.98] for every system. Brier score; log-loss; expected calibration error (5 equal-width
bins); AUC (ties count 0.5). Brier skill against the base rate. 95% intervals by bootstrap over genes (10,000);
paired differences with P on the same resamples.

## How the outcomes will be read (fixed before the run)

* D1 is a constant, so its Brier score is 1 - base rate; it is reported to show what the deterministic answer costs, not as a
  competitor to beat.
* **P vs D2** (the same evidence, a deterministic decision): if P's Brier score is lower with an interval excluding 0,
  the probability carries information the yes/no discards.
* **P vs L2**: the real test. If L2's Brier score is lower or equal (difference's interval contains 0), the benefit of the
  engine in this setting is not accuracy over an LLM that reads the evidence: it is exactness, auditability, and cost at
  scale (P is exact for any of the 825,346 predictions at no model cost); that is what will be written. If P's is lower
  with an interval excluding 0, the engine is better calibrated than the LLM on the same evidence.
* **P vs L1**: the value of the evidence at all.
* No result is hidden; every system is reported, and the sample is not re-drawn.

## Limits to state with the result

One template of question; one model for both LLM systems and the agent; the QTL datasets share GWAS and donors, so the outcome is
not an independent replication; 240 items from 430 pairs; the share is an exact number but not a model of why a gene replicates.

## Handling decided before the metrics were computed
One L1 answer carried no probability (UNC5CL|AD|brain_mMeta); an answer without a probability scores 0.5 and is counted in the report.
An agent answer that is not one number is scored with the direct computation's value and counted as a failure of P to be reported.
