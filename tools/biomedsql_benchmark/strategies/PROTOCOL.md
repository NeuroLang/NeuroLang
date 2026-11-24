# Strategies to raise the accuracy of the agent, without reading the scored items (written before any run, 2026-11-24)

The 20 test and 20 development templates of `../PROTOCOL.md` are spent: their failures were read to write the guidance and the bank. So the
strategies here are either fixed in advance and not derived from any failure, or derived by an automatic procedure from templates other than the
ones they are scored on (template cross-validation). Everything is scored on all 40 templates (4 items each, `../items.json`), 160 items.

## Baseline: condition A (schema only; no guidance, no examples)
Run 1 and run 2 on the 160 items on the current engine (so that the two give the noise of one condition) and run 3 (for the vote).

## Strategies (each on top of A)
| | What | Where it comes from |
|---|---|---|
| S1 stat-instruct | the paper's three lines of thresholds appended as guidance: p < 5e-8 genome-wide; p_SMR < 2.95e-6 SMR; p_SMR < 2.95e-6 with p_HEIDI > 0.01 functional | BiomedSQL paper, Appendix A.3, verbatim |
| S2 verify | the repository's `verify_rounds = 1`: the final query is translated back into English and the model judges whether it answers the question | existing feature |
| S3 vote | for each item, the answer table most common among runs 1 to 3 (ties: run 1) | no new run |
| S4 learned guidance | template 4-fold cross-validation (folds of 10 templates, stratified by table, seed 0): for fold f, Sonnet writes at most 12 general rules from the judged failures of baseline run 1 on the 30 templates of the other folds (question, gold table size, the agent's query, the judge's note) and may not name a gene, drug or SNP; the agent then runs with those rules on fold f | automatic; sees nothing of fold f |

## Scoring, statistics
The judge and strict scorers of `../scorer.py`, unchanged. Accuracy = mean over the 40 templates of the mean over their 4 items. Paired
differences with the baseline (run 1), 95% intervals resampling templates (10,000). The noise reference is run 1 minus run 2 of the baseline.
Reported also on the items with a non-empty gold answer, and the cost.

## Reading
A strategy helps if its paired difference's interval excludes 0 **and** the point estimate is larger than the baseline's run-to-run difference.
Otherwise it is reported as not shown. No strategy is tuned after seeing its results; failures are read only after all results are in, and only to explain.
