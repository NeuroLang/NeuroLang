# Strategies to raise the agent's accuracy: results (`PROTOCOL.md`, written before any run)

160 items = the 40 templates of BiomedSQL, 4 items each, scored by the judge of `../scorer.py` (unchanged), on the current engine
(with `support`, `support_prior` and their tokens in the schema). Accuracy = mean over the 40 templates; 95% intervals resample templates.
The baseline A (schema only, no guidance, no examples) was run three times; **S4 is template 4-fold cross-validated** (its rules for a
fold are written from the baseline's failures on the other 30 templates); S1 to S3 are fixed in advance.

| Strategy | Judge accuracy [95% CI] | Strict | Paired difference with A run 1 [95% CI] | Non-empty gold only | Cost per 160 |
|---|---|---|---|---|---|
| A run 1 | 70.0 [56.9, 82.5] | 10.6 | | 75.0 | $2.18 |
| A run 2 | 71.2 [58.1, 83.1] | 10.0 | +1.2 [-1.9, +4.4] (the noise reference) | 75.7 | $2.21 |
| A run 3 | 69.4 [56.9, 81.2] | 10.0 | -0.6 [-4.4, +3.8] | 74.3 | $2.29 |
| S3 vote of runs 1 to 3 | 71.2 [58.1, 83.1] | 11.2 | +1.2 [-1.2, +3.8] | 75.7 | $2.24 (no new run) |
| S1 stat-instruct (the paper's thresholds) | 73.8 [61.3, 85.0] | 12.5 | +3.8 [-1.9, +10.6] | 74.3 | $2.26 |
| S2 verify (`verify_rounds = 1`) | 75.0 [62.5, 85.6] | 15.0 | +5.0 [0.0, +10.6] | 75.7 | $3.55 |
| S4 learned guidance, template CV | 79.4 [69.4, 88.1] | 28.1 | **+9.4 [-1.2, +21.2]** | 79.7 | $2.75 + induction |

**By the rule fixed beforehand (the interval excludes 0 and the estimate exceeds the baseline's run-to-run difference, +1.2): no strategy is shown to help.**
S2's interval touches 0; S4 has the largest estimate, with an interval that is 22 points wide.

## What the numbers do show
- **The baseline is stable**: three runs within 1.8 points; the vote adds nothing (+1.2).
- **S4's estimate is the largest and is concentrated**: per fold, A -> S4: 70.0 -> 80.0, 57.5 -> 80.0, 77.5 -> 82.5, 75.0 -> 75.0 (three of four folds improve, one is unchanged).
  Its strict score nearly triples (10.6 -> 28.1): the induced rules tell the agent which columns to return, as the in-template examples of condition C did.
  Per template: large gains on Q1, Q5, Q23, Q35 (0 -> 50 to 100) and Q14, Q22 (50 -> 100), losses on Q7, Q10, Q12, Q40 (100 -> 50) and Q17 (100 -> 75).
- **Where the gains and losses come from** (read after all results were in, to explain, not to tune): the rules induced from the other folds describe
  question kinds ("Is X a druggable gene?", "Is drug D approved to treat Z?"); they help on a sibling template in the held-out fold and hurt where a
  held-out template differs from the kind they describe. The twin structure of BiomedSQL's 40 templates (18 of the 20 test templates had a close development twin)
  is what the cross-validation can exploit; it says nothing about a new kind of question.
- **S1 does what the paper's thresholds should**: Q45 (genome-wide significance of a gene's variants) goes from 0 to 100, Q26 50 -> 75, Q22 25 -> 75; elsewhere
  it changes little or slightly lowers (Q9, Q40, Q59 100 -> 75). The net is +3.8, not distinguishable from noise.
- **S2 mostly helps on the empty-gold items** (the dominant error): 12/29 correct for A, 16/29 for S1, **18/29 for S2**, 14/29 for S4; on the 131 items with an answer it changes nothing (+0.7).
  Its price is +63% in cost and more model calls.
- Three templates stay at 0 under every strategy: Q19, Q50 and Q52 (SMR and GWAS "top" and "relevance" lookups with conventions of their own). Q1 and Q5 are at 0 except under S4; Q16 only reaches 50 under S2 and S4.

## Cost, limits
Agent cost of the experiment $15.24 (baseline x3, S1, S2, S4; the judge and the guidance induction are a few dollars more, not measured). One run per strategy; intervals over 40 templates are wide
(a difference below about 6 points cannot be told from noise here); the judge is the agent's own model; S4's rules were induced by the same model;
S1 to S3 were fixed in advance, S4 by a procedure fixed in advance and run once; nothing was tuned after seeing a result.

## What would settle it
More templates than BiomedSQL has (40): the scored unit is the template, so a stronger test needs a second benchmark of independent templates (for instance
BiomedSQL's 20-table extension, which the paper mentions, or a new set written by people who did not write the engine).
