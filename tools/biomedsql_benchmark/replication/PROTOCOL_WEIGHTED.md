# Graded units of evidence in the engine (written before the evaluation, 2026-11-25)

Question: `support_prior` counts a dataset's test as evidence only when it passes the threshold, whatever its p-value. A logistic regression that sees
how strong each test is matches L2 (`post_hoc_logit.py`, Brier 0.131 against 0.129 and 0.147 for `support_prior`). Does an engine predicate that weighs
each test by its strength, while staying an exact conditional probability over a choice of unit of evidence, close part of that gap on held-out genes?

## Choice of the form (done, on the fitting genes only)
`prototype_weighted.py`, output `results/prototype_weighted.txt`. It uses 5-fold cross-validation grouped by gene, on the 2,763 predictions with support > 0
of the 222 genes of `items.json`. The 131 evaluation genes were not read. The rule, fixed in the script before it ran, is lowest CV log-loss, with the
simpler form winning within 0.002.

| Form | CV log-loss | CV Brier |
|---|---|---|
| P `support_prior`: (k + a) / (n + a + b) | 0.4126 | 0.1319 |
| **G graded**: (Σ π(bin of -log10 p) + a) / (n + a + b), π monotone | **0.4084** | 0.1312 |
| O: G restricted to the held-out dataset's organ | 0.4644 | 0.1544 |
| GL: G with datasets of other organs weighted λ | 0.4085 | 0.1312 |
| W: one weight per dataset in the choice | 0.4081 | 0.1305 |
| GW: G and W | 0.4125 | 0.1314 |
| *reference, not eligible*: logistic of `post_hoc_logit.py`, all 8 features / strength features only | 0.3937 / 0.4044 | 0.1253 / 0.1297 |

GL was added after the first run of the script had shown O failing (still on the fitting genes only). G is chosen (W is within 0.002 and less simple).
With the engine's existing prior fixed (a = 0.01, b = 0.5, `selected_unit` unchanged), G scores CV log-loss 0.4081 and Brier 0.1307, no worse. So the
engine keeps `selected_unit` and adds only graded evidence. π is fitted on all 2,763 predictions with that prior:

| -log10 p | [0, 1) | [1, 2) | [2, 3) | [3, 5.53) | [5.53, 8) | [8, 12) | >= 12 |
|---|---|---|---|---|---|---|---|
| π | 0.0025 | 0.0025 | 0.0025 | 0.0874 | 0.8172 | 1 | 1 |

The floor of 0.0025 is the lower bound of the parametrisation. 5.53 is BiomedSQL's threshold (2.95e-6).

## Engine
- `unit_evidence(u, g, d)`: a probabilistic fact per test, with probability π of the bin of its strongest probe. `prior_significant` has π = 1, and
  `prior_none` gives no evidence. π is a `python_init_options` entry in `engines.yaml`.
- The predicate is
  `support_graded(g, d, PROB) :- (unit_evidence(u, g, d) & selected_unit(u)) // (unit_tested(u, g, d) & selected_unit(u))`.
  It equals (Σ_u π_u + a) / (n + a + b), the sum over the datasets that tested the pair.
- Required: the engine's value equals this closed form to 1e-9 on every evaluation prediction, with the lifted solvers only, and `support` and
  `support_prior` are unchanged.

## Evaluation
- **Primary**: the 1,197 predictions with support > 0 of the 131 genes of `items_eval.json`, each with its held-out dataset left out of the units.
  These are paired Brier differences, with 95% intervals over 2,000 gene resamples:
  - `support_graded` - `support_prior`;
  - `support_graded` - L2 (Sonnet 5.5 with the evidence table: `results/eval_answers.jsonl`, not rerun);
  - `support_graded` - the logistic of `post_hoc_logit.py`.

  Also log-loss, ECE and AUC.
- **These genes are not fresh.** This is their third use, after `PROTOCOL_PRIOR.md` and `post_hoc_logit.py`, and the logistic's coefficients pointed to the
  strength of the tests, which this form uses. A gain here is weaker evidence than a gain on unseen genes would be.
- **Secondary (fresh for this comparison)**: the 5,285 predictions with support = 0 of the same 131 genes. There the share says 0, and graded evidence
  need not. These are the Brier and log-loss of `support_graded` against `support_prior`, paired, with intervals over genes. No L2 is available (no new
  API calls).
- **False alarms**: the 806,883 predictions of the other genes, none of which is ever significant (y = 0 everywhere). Their mean and maximum
  probability, for both predicates.

## Reading
- `support_graded` better than `support_prior` on the primary set, with an interval excluding 0: the graded evidence generalises.
- Against L2 and the logistic: equal if the interval contains 0, otherwise the sign says which is better.
- The secondary set is read the same way against `support_prior`. A worse result there is reported as a cost of the form.
- Every result is reported, whichever way it comes out.
