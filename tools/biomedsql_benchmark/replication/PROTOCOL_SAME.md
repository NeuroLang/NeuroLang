# Can the engine beat an LLM given the same evidence? (written before any analysis of it, 2026-11-25)

The claim "the engine beats an LLM reading the same evidence" was never tested as worded. L2 (Sonnet with the evidence table) saw p-values,
tissues, QTL kinds and the gene's name. The engine's predicates see only which datasets tested the pair and which found it significant
(`support_prior`), or that and binned p-values (`support_graded`). This protocol tests the claim in three parts, with free steps before paid ones.

## Data and splits
- **Items:** all 3,960 leave-one-dataset-out predictions with support > 0, from 353 genes in 65 loci.
- **Loci:** genes within 1 Mb (`leak_checks.py`, `ProbeChr` / `Probe_bp`, single linkage).
- **Folds:** every engine or fitted model is scored out of fold, with 5 outer folds grouped by locus (seed 0). Any choice among forms is made by
  an inner 4-fold locus-grouped CV on the outer training loci (log-loss), never on the outer test fold.
- **Comparison set:** the 1,437 items of `items.json` and `items_eval.json`, where the LLM arms run.
- **Not fresh:** all these predictions have been seen in earlier analyses. That is stated with every result.
- **Intervals:** 95%, resampling loci (2,000 draws, seed 0).
- **Metric:** Brier with probabilities clipped to [0.02, 0.98] (primary), then log-loss, and AUC on unclipped values.

## Part 1: ceiling (free)
How well can anything fitted on the full table do?
- **F1** best function of (k, n): a smoothed rate per (k, n) cell (beta-binomial shrinkage towards the fitted prior);
- **F2** the logistic of `post_hoc_logit.py` (8 features);
- **F3** gradient-boosted trees (sklearn `HistGradientBoostingClassifier`, default settings, max_iter 200) on the full table: the −log10 p of each
  of the 28 datasets for the pair (missing when untested), the held-out dataset's identity, organ and kind, k, n and the disease.

## Part 2: same evidence (paid, Sonnet 5.5 through OpenRouter)
- **L2-full:** the L2 prompt of `run_llm.py` unchanged, rerun on one route.
- **L2-flags:** the evidence of `support_prior` only. The gene's name, the held-out dataset's name, tissue and kind are withheld; the disease is
  kept. The other datasets are listed as "dataset 1 ... dataset n", each with significant yes/no.
- Both arms: `max_tokens` 4,000 (reasoning is on by default through this route) and their own answer cache. 1,437 items each.
- The old direct L2 answers are reported beside the new ones, as a check of the route.

## Part 3: parity on the full table (free)
Exact lifted engine forms, in closed form, scored out of fold:
- **E1** `support_prior` (a, b refitted per fold);
- **E2** `support_graded` (π refitted per fold, prior fixed as shipped);
- **E3** noisy-OR: P = 1 − (1 − ε) Π over the other datasets of (1 − τ · π(bin of the dataset's −log10 p)). This is a disjunction of independent
  probabilistic facts, so it is liftable, and it rises with the number of strong tests, as the logistic's log n weight says a weighted average
  cannot. ε, τ and π (monotone, the 7 bins of `prototype_weighted.py`) are fitted per fold.
- **E\*** the form chosen per outer fold by the inner CV among E1 to E3: the primary engine.

## Readings, fixed now
1. **Ceiling.** If none of F1 to F3 beats L2-full on the comparison set with an interval excluding 0, the result is: on this data no model fitted
   on the table beats the LLM, so "the engine beats an LLM reading the full evidence" is **not the case here**.
2. **Same evidence.** E\* minus L2-flags, and E1 minus L2-flags, on Brier:
   - interval below 0: the engine beats the LLM given the same evidence;
   - interval containing 0: equal;
   - interval above 0: the LLM is better even with the engine's evidence only.
3. **Parity.** E\* minus L2-full on Brier: **non-inferior** if the upper bound of the interval is below **0.010**. Otherwise L2-full is better and
   the gap is reported.
4. E\* is also compared with F2 (the logistic) to say how much of the table's information exact lifted forms recover.
5. Every part is reported, whichever way it comes out. A positive Part 2 does not support any claim about the full table.
