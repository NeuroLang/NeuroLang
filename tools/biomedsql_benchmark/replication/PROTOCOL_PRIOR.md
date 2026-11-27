# Prior on the units of evidence, in the engine (written before the run, 2026-11-24)

Question: the post hoc prior of `RESULTS.md` (smoothing the share with imaginary datasets) closed the Brier gap to an LLM that reads the
evidence. Does it hold on **fresh items**, as an engine predicate, with parameters fitted without them?

## Engine
Two imaginary units of evidence, `prior_significant` and `prior_none`, are added to the 28 QTL datasets. They are tested, for every
(gene, disease) pair, the first one always significant and the second never. A probabilistic choice `selected_unit` over the 30 units has
weights 1 for each dataset, `a` for `prior_significant` and `b` for `prior_none` (normalised). The predicate

    support_prior(g, d, PROB) :- (unit_significant(u, g, d) & selected_unit(u)) // (unit_tested(u, g, d) & selected_unit(u))

equals (k + a) / (n + a + b), k datasets significant among the n that tested the pair: a conditional probability over a choice of unit of
evidence, solved by lifted inference. Required: the engine's value equals this closed form to 1e-9 on every item, with the lifted solvers only.

## Fitting (no fresh item, no gene of either item set)
`a` in {0.01, 0.05, 0.1, 0.25, 0.5, 1, 2}, `b` in {0.1, 0.25, 0.5, 1, 2, 3, 5}, chosen by log-loss (probabilities clipped to [0.02, 0.98])
on the leave-one-dataset-out predictions with support > 0 of all genes that are in neither item set. The pair is written in `engines.yaml`.

## Fresh items
240 items built as `replication/build_items.py` builds them (seed 1) from the pairs whose gene is not among the 222 genes of the first item set.

## Systems and metrics
The unsmoothed share (the engine's `support`), `support_prior`, and L2 (Sonnet 5.5 given the evidence table: the same prompt as before, one answer per item).
Brier (primary), log-loss, ECE, AUC; skill against the base rate; 95% intervals resampling genes; paired differences.

## Reading
- `support_prior` better than the share with an interval excluding 0: the prior generalises.
- `support_prior` against L2: equal if the Brier difference's interval contains 0; otherwise the sign says which. The earlier tie is not assumed.
- If `a` or `b` falls on the edge of the grid, the grid was too small and the result is reported as such.

## Amendment, before any run of this protocol (2026-11-24)
Drawing the fresh items showed that only 145 pairs exist with a gene outside the 222 genes of the first item set (131 genes), and that these 131 genes
are exactly the ones on which the earlier post hoc prior (a = 0.05, b = 1) was fitted: no gene is left to fit on. So:
- **Fit** on all the predictions with support > 0 of the 222 genes of the first item set (the genes on which the failure was observed), grid as above.
- **Evaluate** on all the predictions with support > 0 of the 131 other genes (every held-out dataset of every pair, not one per pair: 1,197 predictions,
  `build_eval.py`), intervals resampling genes. The earlier post hoc fit saw these genes' data, but its values are not reused; the choice of the form
  of the prior was made after seeing the failure on the first 222 genes. Both facts are stated with the result.
- L2 is run on these predictions (one answer each).
